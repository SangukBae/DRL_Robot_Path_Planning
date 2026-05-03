#!/usr/bin/env python3
"""
Write Hunter SE validation data to a CSV file.
"""

from __future__ import annotations

import csv
import math
import os
from datetime import datetime

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class HunterSEValidationLogger(Node):
    def __init__(self) -> None:
        super().__init__("hunter_se_validation_logger")

        self.declare_parameter("joint_states_topic", "/hunter_se/joint_states")
        self.declare_parameter("odom_topic", "/odometry")
        self.declare_parameter("cmd_topic", "/cmd_vel_filtered")
        self.declare_parameter("phase_topic", "/hunter_se/validation_phase")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("wheelbase_m", 0.547696)
        self.declare_parameter("track_m", 0.503404)
        self.declare_parameter("wheel_radius_m", 0.136)
        self.declare_parameter("center_steering_limit_deg", 18.81684653381576)
        self.declare_parameter("max_speed_mps", 1.333)
        self.declare_parameter("output_dir", os.path.expanduser("~/.ros/hunter_se_validation_logs"))

        self._joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._cmd_topic = str(self.get_parameter("cmd_topic").value)
        self._phase_topic = str(self.get_parameter("phase_topic").value)
        self._publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self._wheelbase_m = float(self.get_parameter("wheelbase_m").value)
        self._track_m = float(self.get_parameter("track_m").value)
        self._wheel_radius_m = float(self.get_parameter("wheel_radius_m").value)
        self._center_limit_rad = math.radians(
            float(self.get_parameter("center_steering_limit_deg").value)
        )
        self._max_speed_mps = float(self.get_parameter("max_speed_mps").value)
        self._output_dir = str(self.get_parameter("output_dir").value)

        self._left_angle = None
        self._right_angle = None
        self._rear_left_w = None
        self._rear_right_w = None
        self._vehicle_speed = None
        self._cmd_speed = None
        self._cmd_yaw_rate = None
        self._phase = ""

        os.makedirs(self._output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = os.path.join(self._output_dir, f"hunter_se_validation_{stamp}.csv")
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow([
            "wall_time_iso",
            "sim_time_sec",
            "phase",
            "cmd_speed_mps",
            "cmd_yaw_rate_rad_s",
            "body_speed_mps",
            "front_left_deg",
            "front_right_deg",
            "center_deg",
            "center_saturation_pct",
            "rear_left_rim_mps",
            "rear_right_rim_mps",
            "rear_avg_rim_mps",
            "slip_left_pct",
            "slip_right_pct",
            "slip_avg_pct",
        ])
        self._csv_file.flush()
        self.get_logger().info(f"writing validation CSV to {self._csv_path}")

        self.create_subscription(JointState, self._joint_states_topic, self._joint_state_cb, 10)
        self.create_subscription(Odometry, self._odom_topic, self._odom_cb, 10)
        self.create_subscription(Twist, self._cmd_topic, self._cmd_cb, 10)
        self.create_subscription(String, self._phase_topic, self._phase_cb, 10)
        self.create_timer(1.0 / self._publish_rate_hz, self._write_row)

    def _joint_state_cb(self, msg: JointState) -> None:
        if not msg.name:
            return
        try:
            fl_index = msg.name.index("front_left_steering")
            fr_index = msg.name.index("front_right_steering")
            rl_index = msg.name.index("rear_left_wheel")
            rr_index = msg.name.index("rear_right_wheel")
        except ValueError:
            return

        if fl_index < len(msg.position):
            self._left_angle = float(msg.position[fl_index])
        if fr_index < len(msg.position):
            self._right_angle = float(msg.position[fr_index])
        if msg.velocity:
            if rl_index < len(msg.velocity):
                self._rear_left_w = float(msg.velocity[rl_index])
            if rr_index < len(msg.velocity):
                self._rear_right_w = float(msg.velocity[rr_index])

    def _odom_cb(self, msg: Odometry) -> None:
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self._vehicle_speed = math.hypot(vx, vy)

    def _cmd_cb(self, msg: Twist) -> None:
        self._cmd_speed = float(msg.linear.x)
        self._cmd_yaw_rate = float(msg.angular.z)

    def _phase_cb(self, msg: String) -> None:
        self._phase = msg.data

    def _center_steering_from_wheels(self) -> float:
        if self._left_angle is None or self._right_angle is None:
            return 0.0
        if abs(self._left_angle) < 1e-6 and abs(self._right_angle) < 1e-6:
            return 0.0

        sign = 1.0 if (self._left_angle + self._right_angle) >= 0.0 else -1.0
        inner = max(abs(self._left_angle), abs(self._right_angle))
        outer = min(abs(self._left_angle), abs(self._right_angle))
        radius_from_inner = self._wheelbase_m / math.tan(inner) + self._track_m / 2.0
        if outer < 1e-6:
            center = math.atan(self._wheelbase_m / radius_from_inner)
            return math.copysign(center, sign)
        radius_from_outer = self._wheelbase_m / math.tan(outer) - self._track_m / 2.0
        center = math.atan(self._wheelbase_m / (0.5 * (radius_from_inner + radius_from_outer)))
        return math.copysign(center, sign)

    def _slip_ratio(self, rim_speed: float, vehicle_speed: float) -> float:
        denom = max(abs(vehicle_speed), 0.05)
        return (rim_speed - abs(vehicle_speed)) / denom

    def _write_row(self) -> None:
        sim_time_sec = self.get_clock().now().nanoseconds * 1e-9
        wall_time_iso = datetime.now().isoformat(timespec="milliseconds")

        center_rad = self._center_steering_from_wheels()
        center_sat = 100.0 * clamp(
            abs(center_rad) / max(self._center_limit_rad, 1e-6), 0.0, 1.0
        )

        rear_left_rim = (
            abs(self._rear_left_w) * self._wheel_radius_m
            if self._rear_left_w is not None else ""
        )
        rear_right_rim = (
            abs(self._rear_right_w) * self._wheel_radius_m
            if self._rear_right_w is not None else ""
        )
        rear_avg_rim = (
            0.5 * (rear_left_rim + rear_right_rim)
            if rear_left_rim != "" and rear_right_rim != "" else ""
        )

        slip_left = (
            100.0 * self._slip_ratio(rear_left_rim, self._vehicle_speed)
            if rear_left_rim != "" and self._vehicle_speed is not None else ""
        )
        slip_right = (
            100.0 * self._slip_ratio(rear_right_rim, self._vehicle_speed)
            if rear_right_rim != "" and self._vehicle_speed is not None else ""
        )
        slip_avg = (
            100.0 * self._slip_ratio(rear_avg_rim, self._vehicle_speed)
            if rear_avg_rim != "" and self._vehicle_speed is not None else ""
        )

        self._writer.writerow([
            wall_time_iso,
            f"{sim_time_sec:.6f}",
            self._phase,
            "" if self._cmd_speed is None else f"{self._cmd_speed:.6f}",
            "" if self._cmd_yaw_rate is None else f"{self._cmd_yaw_rate:.6f}",
            "" if self._vehicle_speed is None else f"{self._vehicle_speed:.6f}",
            "" if self._left_angle is None else f"{math.degrees(self._left_angle):.6f}",
            "" if self._right_angle is None else f"{math.degrees(self._right_angle):.6f}",
            f"{math.degrees(center_rad):.6f}",
            f"{center_sat:.3f}",
            "" if rear_left_rim == "" else f"{rear_left_rim:.6f}",
            "" if rear_right_rim == "" else f"{rear_right_rim:.6f}",
            "" if rear_avg_rim == "" else f"{rear_avg_rim:.6f}",
            "" if slip_left == "" else f"{slip_left:.3f}",
            "" if slip_right == "" else f"{slip_right:.3f}",
            "" if slip_avg == "" else f"{slip_avg:.3f}",
        ])
        self._csv_file.flush()

    def destroy_node(self):
        try:
            self._csv_file.close()
        finally:
            super().destroy_node()


def main() -> None:
    rclpy.init()
    node = HunterSEValidationLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
