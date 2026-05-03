#!/usr/bin/env python3
"""
Print Hunter SE speed, wheel rim speed, and slip indicators for empty-map tests.
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class HunterSEDynamicsMonitor(Node):
    def __init__(self) -> None:
        super().__init__("hunter_se_dynamics_monitor")

        self.declare_parameter("joint_states_topic", "/hunter_se/joint_states")
        self.declare_parameter("odom_topic", "/odometry")
        self.declare_parameter("cmd_topic", "/cmd_vel_filtered")
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("wheel_radius_m", 0.136)

        self._joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._cmd_topic = str(self.get_parameter("cmd_topic").value)
        self._publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self._wheel_radius_m = float(self.get_parameter("wheel_radius_m").value)

        self._vehicle_speed = None
        self._cmd_speed = None
        self._cmd_yaw_rate = None
        self._rear_left_w = None
        self._rear_right_w = None
        self._joint_state_count = 0

        self.create_subscription(
            JointState,
            self._joint_states_topic,
            self._joint_state_cb,
            10,
        )
        self.create_subscription(
            Odometry,
            self._odom_topic,
            self._odom_cb,
            10,
        )
        self.create_subscription(
            Twist,
            self._cmd_topic,
            self._cmd_cb,
            10,
        )
        self.create_timer(1.0 / self._publish_rate_hz, self._print_status)

    def _joint_state_cb(self, msg: JointState) -> None:
        if not msg.name:
            return

        self._joint_state_count += 1

        try:
            left_index = msg.name.index("rear_left_wheel")
            right_index = msg.name.index("rear_right_wheel")
        except ValueError:
            return

        if msg.velocity:
            if left_index < len(msg.velocity):
                self._rear_left_w = float(msg.velocity[left_index])
            if right_index < len(msg.velocity):
                self._rear_right_w = float(msg.velocity[right_index])

    def _odom_cb(self, msg: Odometry) -> None:
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self._vehicle_speed = math.hypot(vx, vy)

    def _cmd_cb(self, msg: Twist) -> None:
        self._cmd_speed = float(msg.linear.x)
        self._cmd_yaw_rate = float(msg.angular.z)

    def _slip_ratio(self, rim_speed: float, vehicle_speed: float) -> float:
        denom = max(abs(vehicle_speed), 0.05)
        return (rim_speed - abs(vehicle_speed)) / denom

    def _print_status(self) -> None:
        if self._joint_state_count == 0:
            print("[dynamics] waiting for /hunter_se/joint_states ...", flush=True)
            return

        if self._vehicle_speed is None or self._rear_left_w is None or self._rear_right_w is None:
            print("[dynamics] waiting for odom / wheel velocities ...", flush=True)
            return

        rear_left_rim = abs(self._rear_left_w) * self._wheel_radius_m
        rear_right_rim = abs(self._rear_right_w) * self._wheel_radius_m
        rear_avg_rim = 0.5 * (rear_left_rim + rear_right_rim)

        slip_left = self._slip_ratio(rear_left_rim, self._vehicle_speed)
        slip_right = self._slip_ratio(rear_right_rim, self._vehicle_speed)
        slip_avg = self._slip_ratio(rear_avg_rim, self._vehicle_speed)

        cmd_text = "cmd=waiting"
        if self._cmd_speed is not None and self._cmd_yaw_rate is not None:
            cmd_text = (
                f"cmd_v={self._cmd_speed:+.3f} m/s  "
                f"cmd_w={self._cmd_yaw_rate:+.3f} rad/s"
            )

        print(
            "[dynamics] "
            f"body_v={self._vehicle_speed:+.3f} m/s   "
            f"rear_l_rim={rear_left_rim:+.3f} m/s   "
            f"rear_r_rim={rear_right_rim:+.3f} m/s   "
            f"rear_avg_rim={rear_avg_rim:+.3f} m/s   "
            f"slip_l={100.0 * clamp(slip_left, -5.0, 5.0):+6.1f}%   "
            f"slip_r={100.0 * clamp(slip_right, -5.0, 5.0):+6.1f}%   "
            f"slip_avg={100.0 * clamp(slip_avg, -5.0, 5.0):+6.1f}%   "
            f"{cmd_text}",
            flush=True,
        )


def main() -> None:
    rclpy.init()
    node = HunterSEDynamicsMonitor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
