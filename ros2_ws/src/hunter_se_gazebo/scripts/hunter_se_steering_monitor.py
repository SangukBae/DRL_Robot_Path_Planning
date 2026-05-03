#!/usr/bin/env python3
"""
Print Hunter SE steering and speed against configured maxima.
"""

from __future__ import annotations

import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class HunterSESteeringMonitor(Node):
    def __init__(self) -> None:
        super().__init__("hunter_se_steering_monitor")

        self.declare_parameter("joint_states_topic", "/hunter_se/joint_states")
        self.declare_parameter("odom_topic", "/odometry")
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("wheelbase_m", 0.547696)
        self.declare_parameter("track_m", 0.503404)
        self.declare_parameter("center_steering_limit_deg", 18.81684653381576)
        self.declare_parameter("max_speed_mps", 1.333)

        self._joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self._wheelbase_m = float(self.get_parameter("wheelbase_m").value)
        self._track_m = float(self.get_parameter("track_m").value)
        self._center_limit_rad = math.radians(
            float(self.get_parameter("center_steering_limit_deg").value)
        )
        self._max_speed_mps = float(self.get_parameter("max_speed_mps").value)

        self._left_angle = None
        self._right_angle = None
        self._joint_state_count = 0
        self._speed_mps = None

        self._max_inner_rad, self._max_outer_rad = self._compute_wheel_angle_limits()

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
        self.create_timer(1.0 / self._publish_rate_hz, self._print_status)

    def _compute_wheel_angle_limits(self) -> tuple[float, float]:
        turn_radius = self._wheelbase_m / math.tan(self._center_limit_rad)
        inner = math.atan(
            self._wheelbase_m / (turn_radius - self._track_m / 2.0)
        )
        outer = math.atan(
            self._wheelbase_m / (turn_radius + self._track_m / 2.0)
        )
        return inner, outer

    def _joint_state_cb(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return

        self._joint_state_count += 1
        try:
            left_index = msg.name.index("front_left_steering")
            right_index = msg.name.index("front_right_steering")
        except ValueError:
            return

        if left_index < len(msg.position):
            self._left_angle = float(msg.position[left_index])
        if right_index < len(msg.position):
            self._right_angle = float(msg.position[right_index])

    def _odom_cb(self, msg: Odometry) -> None:
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self._speed_mps = math.hypot(vx, vy)

    def _wheel_maxima_for_turn(self) -> tuple[float, float]:
        if self._left_angle is None or self._right_angle is None:
            return self._max_inner_rad, self._max_inner_rad

        avg = 0.5 * (self._left_angle + self._right_angle)
        if avg > 1e-5:
            return self._max_inner_rad, self._max_outer_rad
        if avg < -1e-5:
            return self._max_outer_rad, self._max_inner_rad
        return self._max_inner_rad, self._max_inner_rad

    def _center_steering_from_wheels(self) -> float:
        if self._left_angle is None or self._right_angle is None:
            return 0.0

        if abs(self._left_angle) < 1e-6 and abs(self._right_angle) < 1e-6:
            return 0.0

        sign = 1.0 if (self._left_angle + self._right_angle) >= 0.0 else -1.0
        inner = max(abs(self._left_angle), abs(self._right_angle))
        outer = min(abs(self._left_angle), abs(self._right_angle))

        if inner < 1e-6:
            return 0.0

        radius_from_inner = self._wheelbase_m / math.tan(inner) + self._track_m / 2.0
        if outer < 1e-6:
            center = math.atan(self._wheelbase_m / radius_from_inner)
            return math.copysign(center, sign)

        radius_from_outer = self._wheelbase_m / math.tan(outer) - self._track_m / 2.0
        radius = 0.5 * (radius_from_inner + radius_from_outer)
        center = math.atan(self._wheelbase_m / radius)
        return math.copysign(center, sign)

    def _format_wheel(self, name: str, current_rad: float, max_rad: float) -> str:
        current_deg = math.degrees(current_rad)
        max_deg = math.degrees(max_rad)
        saturation = 100.0 * clamp(abs(current_rad) / max(abs(max_rad), 1e-6), 0.0, 1.0)
        return (
            f"{name}={current_deg:+6.2f} deg ({current_rad:+.4f} rad)  "
            f"max={math.copysign(max_deg, current_deg):+6.2f} deg  "
            f"sat={saturation:5.1f}%"
        )

    def _format_speed(self) -> str:
        if self._speed_mps is None:
            return "speed=waiting"

        saturation = 100.0 * clamp(
            self._speed_mps / max(self._max_speed_mps, 1e-6), 0.0, 1.0
        )
        return (
            f"speed={self._speed_mps:+.3f} m/s  "
            f"max={self._max_speed_mps:.3f} m/s  "
            f"sat={saturation:5.1f}%"
        )

    def _print_status(self) -> None:
        if self._left_angle is None or self._right_angle is None:
            if self._joint_state_count == 0:
                print("[steering] waiting for /hunter_se/joint_states ...", flush=True)
            return

        left_max_rad, right_max_rad = self._wheel_maxima_for_turn()
        center_rad = self._center_steering_from_wheels()
        center_deg = math.degrees(center_rad)
        center_limit_deg = math.degrees(self._center_limit_rad)
        avg_saturation = 100.0 * clamp(
            abs(center_rad) / max(self._center_limit_rad, 1e-6), 0.0, 1.0
        )

        print(
            "[steering] "
            f"{self._format_wheel('front_left', self._left_angle, left_max_rad)}   "
            f"{self._format_wheel('front_right', self._right_angle, right_max_rad)}   "
            f"center={center_deg:+6.2f} deg  "
            f"center_max={math.copysign(center_limit_deg, center_deg):+6.2f} deg  "
            f"center_sat={avg_saturation:5.1f}%   "
            f"{self._format_speed()}",
            flush=True,
        )


def main() -> None:
    rclpy.init()
    node = HunterSESteeringMonitor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
