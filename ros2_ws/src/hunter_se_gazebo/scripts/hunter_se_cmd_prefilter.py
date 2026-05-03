#!/usr/bin/env python3
"""
Unity-like command prefilter for Hunter SE.

This node sits between ROS publishers on /cmd_vel and Gazebo's Ackermann plugin.
It preserves the public Twist interface while adding three behaviors that are
closer to the Unity controller:

  1. Command delay + first-order longitudinal lag
  2. Brake-first handling before reversing direction
  3. Steering angle rate limiting in front-wheel Ackermann space

Input:
  /cmd_vel           geometry_msgs/msg/Twist

Output:
  /cmd_vel_filtered  geometry_msgs/msg/Twist
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


@dataclass
class TimedCommand:
    available_at: float
    linear_x: float
    angular_z: float


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def move_towards(current: float, target: float, max_delta: float) -> float:
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + math.copysign(max_delta, delta)


class HunterSECmdPrefilter(Node):
    def __init__(self) -> None:
        super().__init__("hunter_se_cmd_prefilter")

        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("command_timeout_sec", 0.5)
        self.declare_parameter("input_delay_sec", 0.05)
        self.declare_parameter("wheelbase_m", 0.55)
        self.declare_parameter("steering_limit_deg", 30.0)
        self.declare_parameter("steering_rate_deg_s", 315.789)
        self.declare_parameter("min_speed_for_steering_mps", 0.15)
        self.declare_parameter("speed_lag_tau_sec", 0.25)
        self.declare_parameter("accel_limit_mps2", 2.0)
        self.declare_parameter("brake_decel_mps2", 3.0)
        self.declare_parameter("deadband_speed_mps", 0.02)
        self.declare_parameter("max_speed_mps", 1.5)
        self.declare_parameter("odom_topic", "/odometry")
        self.declare_parameter("overspeed_deadband_mps", 0.01)
        self.declare_parameter("speed_governor_gain", 2.5)

        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.command_timeout_sec = float(self.get_parameter("command_timeout_sec").value)
        self.input_delay_sec = float(self.get_parameter("input_delay_sec").value)
        self.wheelbase_m = float(self.get_parameter("wheelbase_m").value)
        self.steering_limit_rad = math.radians(
            float(self.get_parameter("steering_limit_deg").value)
        )
        self.steering_rate_rad_s = math.radians(
            float(self.get_parameter("steering_rate_deg_s").value)
        )
        self.min_speed_for_steering_mps = float(
            self.get_parameter("min_speed_for_steering_mps").value
        )
        self.speed_lag_tau_sec = float(self.get_parameter("speed_lag_tau_sec").value)
        self.accel_limit_mps2 = float(self.get_parameter("accel_limit_mps2").value)
        self.brake_decel_mps2 = float(self.get_parameter("brake_decel_mps2").value)
        self.deadband_speed_mps = float(self.get_parameter("deadband_speed_mps").value)
        self.max_speed_mps = float(self.get_parameter("max_speed_mps").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.overspeed_deadband_mps = float(
            self.get_parameter("overspeed_deadband_mps").value
        )
        self.speed_governor_gain = float(
            self.get_parameter("speed_governor_gain").value
        )

        self._pending: deque[TimedCommand] = deque()
        self._active_linear = 0.0
        self._active_angular = 0.0
        self._last_input_time = self._now_sec()
        self._filtered_speed = 0.0
        self._filtered_steering = 0.0
        self._last_tick_time = self._now_sec()
        self._measured_speed = 0.0

        self._sub = self.create_subscription(Twist, "/cmd_vel", self._cmd_cb, 10)
        self._odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_cb,
            10,
        )
        self._pub = self.create_publisher(Twist, "/cmd_vel_filtered", 10)
        self._timer = self.create_timer(1.0 / self.publish_rate_hz, self._tick)

        self.get_logger().info(
            "hunter_se_cmd_prefilter enabled: RWD-targeted Unity-like speed/steering shaping"
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _cmd_cb(self, msg: Twist) -> None:
        now = self._now_sec()
        self._last_input_time = now
        self._pending.append(
            TimedCommand(
                available_at=now + self.input_delay_sec,
                linear_x=clamp(float(msg.linear.x), -self.max_speed_mps, self.max_speed_mps),
                angular_z=float(msg.angular.z),
            )
        )

    def _odom_cb(self, msg: Odometry) -> None:
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self._measured_speed = math.hypot(vx, vy)

    def _apply_pending_commands(self, now: float) -> None:
        while self._pending and self._pending[0].available_at <= now:
            cmd = self._pending.popleft()
            self._active_linear = cmd.linear_x
            self._active_angular = cmd.angular_z

    def _target_steering_from_twist(self, speed: float, yaw_rate: float) -> float:
        if abs(yaw_rate) < 1e-6:
            return 0.0

        direction = -1.0 if speed < 0.0 else 1.0
        speed_for_steering = max(abs(speed), self.min_speed_for_steering_mps)
        steering = math.atan((self.wheelbase_m * yaw_rate) / (direction * speed_for_steering))
        return clamp(steering, -self.steering_limit_rad, self.steering_limit_rad)

    def _filter_speed(self, dt: float, target_speed: float) -> float:
        if abs(target_speed) < self.deadband_speed_mps:
            return move_towards(
                self._filtered_speed,
                0.0,
                self.brake_decel_mps2 * dt,
            )

        if self._filtered_speed != 0.0 and math.copysign(1.0, self._filtered_speed) != math.copysign(1.0, target_speed):
            return move_towards(
                self._filtered_speed,
                0.0,
                self.brake_decel_mps2 * dt,
            )

        if self.speed_lag_tau_sec <= 1e-6:
            lagged_target = target_speed
        else:
            alpha = 1.0 - math.exp(-dt / self.speed_lag_tau_sec)
            lagged_target = self._filtered_speed + (target_speed - self._filtered_speed) * alpha

        speed_error = lagged_target - self._filtered_speed
        accel_limit = self.accel_limit_mps2 if abs(lagged_target) >= abs(self._filtered_speed) else self.brake_decel_mps2
        max_step = accel_limit * dt

        if abs(speed_error) < self.deadband_speed_mps:
            return lagged_target

        return move_towards(self._filtered_speed, lagged_target, max_step)

    def _govern_target_speed(self, target_speed: float) -> float:
        if abs(target_speed) < self.deadband_speed_mps:
            return target_speed

        if self._measured_speed <= self.max_speed_mps + self.overspeed_deadband_mps:
            return target_speed

        overspeed = self._measured_speed - self.max_speed_mps
        reduced_mag = max(
            0.0,
            abs(target_speed) - self.speed_governor_gain * overspeed,
        )
        return math.copysign(reduced_mag, target_speed)

    def _tick(self) -> None:
        now = self._now_sec()
        dt = max(now - self._last_tick_time, 1e-4)
        self._last_tick_time = now

        self._apply_pending_commands(now)

        if now - self._last_input_time > self.command_timeout_sec:
            target_speed = 0.0
            target_yaw_rate = 0.0
        else:
            target_speed = self._active_linear
            target_yaw_rate = self._active_angular

        target_steering = self._target_steering_from_twist(target_speed, target_yaw_rate)
        governed_target_speed = self._govern_target_speed(target_speed)
        self._filtered_steering = move_towards(
            self._filtered_steering,
            target_steering,
            self.steering_rate_rad_s * dt,
        )
        self._filtered_speed = self._filter_speed(dt, governed_target_speed)

        if abs(self._filtered_speed) < self.deadband_speed_mps:
            self._filtered_speed = 0.0

        if abs(self._filtered_steering) < 1e-4:
            self._filtered_steering = 0.0

        out = Twist()
        out.linear.x = self._filtered_speed
        if abs(self._filtered_speed) < 1e-6:
            out.angular.z = 0.0
        else:
            out.angular.z = (
                self._filtered_speed * math.tan(self._filtered_steering) / self.wheelbase_m
            )
        self._pub.publish(out)


def main() -> None:
    rclpy.init()
    node = HunterSECmdPrefilter()
    try:
        rclpy.spin(node)
    finally:
        stop = Twist()
        node._pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
