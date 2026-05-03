#!/usr/bin/env python3
"""
Automatic Hunter SE validation driver for empty-world tests.

Publishes a fixed drive sequence to /cmd_vel so steering, speed, and slip can
be inspected without manual keyboard input.
"""

from __future__ import annotations

from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String


@dataclass(frozen=True)
class Phase:
    name: str
    duration_sec: float
    linear_x: float
    angular_z: float


class HunterSEValidationDriver(Node):
    def __init__(self) -> None:
        super().__init__("hunter_se_validation_driver")

        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("max_speed_mps", 1.333)
        self.declare_parameter("max_turn_yaw_rate_rad_s", 1.0)
        self.declare_parameter("stop_hold_sec", 1.0)

        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        max_speed_mps = float(self.get_parameter("max_speed_mps").value)
        max_turn_yaw_rate_rad_s = float(
            self.get_parameter("max_turn_yaw_rate_rad_s").value
        )
        self._stop_hold_sec = float(self.get_parameter("stop_hold_sec").value)

        self._phases = [
            Phase("settle", 2.0, 0.0, 0.0),
            Phase("straight_accel", 5.0, max_speed_mps, 0.0),
            Phase("max_right_turn", 6.0, max_speed_mps, -max_turn_yaw_rate_rad_s),
            Phase("straight_recover", 5.0, max_speed_mps, 0.0),
            Phase("max_left_turn", 6.0, max_speed_mps, max_turn_yaw_rate_rad_s),
            Phase("brake_to_stop", 4.0, 0.0, 0.0),
        ]

        self._phase_index = 0
        self._phase_elapsed = 0.0
        self._stop_elapsed = 0.0
        self._dt = 1.0 / publish_rate_hz
        self._announced_phase_index = -1
        self._completion_announced = False

        self._pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._phase_pub = self.create_publisher(String, "/hunter_se/validation_phase", 10)
        self._timer = self.create_timer(self._dt, self._tick)

    def _publish_phase_command(self, phase: Phase) -> None:
        msg = Twist()
        msg.linear.x = phase.linear_x
        msg.angular.z = phase.angular_z
        self._pub.publish(msg)
        self._phase_pub.publish(String(data=phase.name))

    def _publish_stop(self) -> None:
        self._pub.publish(Twist())
        self._phase_pub.publish(String(data="stopped"))

    def _tick(self) -> None:
        if self._phase_index >= len(self._phases):
            self._publish_stop()
            self._stop_elapsed += self._dt
            if not self._completion_announced:
                self._completion_announced = True
                self.get_logger().info(
                    f"validation complete, holding stop for {self._stop_hold_sec:.1f}s"
                )
            if self._stop_elapsed + 1e-9 >= self._stop_hold_sec:
                self.get_logger().info("validation driver exiting")
                rclpy.shutdown()
            return

        phase = self._phases[self._phase_index]
        if self._announced_phase_index != self._phase_index:
            self._announced_phase_index = self._phase_index
            self.get_logger().info(
                f"phase={phase.name} duration={phase.duration_sec:.1f}s "
                f"cmd_v={phase.linear_x:+.3f} cmd_w={phase.angular_z:+.3f}"
            )

        self._publish_phase_command(phase)
        self._phase_elapsed += self._dt

        if self._phase_elapsed + 1e-9 >= phase.duration_sec:
            self._phase_index += 1
            self._phase_elapsed = 0.0


def main() -> None:
    rclpy.init()
    node = HunterSEValidationDriver()
    try:
        rclpy.spin(node)
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
