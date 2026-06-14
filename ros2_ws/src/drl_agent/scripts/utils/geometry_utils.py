#!/usr/bin/env python3
"""Pure 2D geometry helpers shared across the DRL package.

This module intentionally has NO ROS / numpy / torch dependency — only the
standard-library ``math`` — so it imports and runs in a plain Python process and
is covered by fast ROS-free unit tests (see ``tests/test_geometry_utils.py``).

It collects small angle/distance helpers that were previously inlined (and
duplicated) across ``environment.py`` and the policy scripts. The behaviour is
identical to the original inline expressions; this is an extraction, not a
behaviour change. Heavier, stateful logic (sensor noise, human motion, map
sampling) is deliberately left in ``environment.py`` for now.
"""

import math

TWO_PI = 2.0 * math.pi


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle (radians) into the half-open interval [-pi, pi).

    Byte-for-byte the ``(a + math.pi) % (2*math.pi) - math.pi`` idiom used
    throughout the codebase (note this maps exactly +pi to -pi).
    """
    return (float(angle) + math.pi) % TWO_PI - math.pi


def angle_to(x0: float, y0: float, x1: float, y1: float) -> float:
    """Absolute world heading (radians) of the vector (x0,y0) -> (x1,y1)."""
    return math.atan2(y1 - y0, x1 - x0)


def heading_error(target_angle: float, current_yaw: float) -> float:
    """Signed smallest-rotation error from ``current_yaw`` to ``target_angle``,
    wrapped into (-pi, pi]."""
    return wrap_to_pi(target_angle - current_yaw)


def euclidean_distance(x0: float, y0: float, x1: float, y1: float) -> float:
    """Planar Euclidean distance between two points."""
    return math.hypot(x1 - x0, y1 - y0)


def goal_distance_and_heading(robot_x: float, robot_y: float, robot_yaw: float,
                              goal_x: float, goal_y: float):
    """Return (distance, heading_error) from a robot pose to a goal point.

    ``heading_error`` is the signed bearing of the goal relative to the robot's
    current yaw, wrapped to (-pi, pi]. This mirrors ``Environment._goal_metrics``.
    """
    dx = goal_x - robot_x
    dy = goal_y - robot_y
    dist = math.hypot(dx, dy)
    theta = wrap_to_pi(math.atan2(dy, dx) - robot_yaw)
    return dist, theta


def to_robot_frame(rel_x: float, rel_y: float, robot_yaw: float):
    """Rotate a world-frame offset (rel_x, rel_y) into the robot body frame."""
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    return (c * rel_x + s * rel_y, -s * rel_x + c * rel_y)
