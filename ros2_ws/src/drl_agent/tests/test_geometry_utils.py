"""ROS-free unit tests for utils/geometry_utils.py."""

import math

import pytest

import drl_agent.common.geometry_utils as g


# Behaviour matches the codebase idiom exactly: the interval is [-pi, pi),
# so +pi (and any odd multiple of pi) maps to -pi.
@pytest.mark.parametrize("a,expected", [
    (0.0, 0.0),
    (math.pi, -math.pi),
    (-math.pi, -math.pi),
    (math.pi / 2, math.pi / 2),
    (3 * math.pi, -math.pi),
    (-3 * math.pi, -math.pi),
    (2 * math.pi, 0.0),
    (4 * math.pi + 0.3, 0.3),
])
def test_wrap_to_pi(a, expected):
    assert g.wrap_to_pi(a) == pytest.approx(expected, abs=1e-9)


def test_wrap_to_pi_range_is_bounded():
    for k in range(-50, 51):
        v = g.wrap_to_pi(k * 0.37)
        assert -math.pi - 1e-12 <= v < math.pi


def test_heading_error_is_signed_shortest_rotation():
    # target just CCW of current → small positive error
    assert g.heading_error(0.1, -0.1) == pytest.approx(0.2, abs=1e-9)
    # wrap-around: target = -179deg, current = +179deg → +2deg, not -358deg
    err = g.heading_error(math.radians(-179), math.radians(179))
    assert err == pytest.approx(math.radians(2), abs=1e-6)


def test_angle_to_and_distance():
    assert g.angle_to(0, 0, 1, 0) == pytest.approx(0.0)
    assert g.angle_to(0, 0, 0, 1) == pytest.approx(math.pi / 2)
    assert g.euclidean_distance(0, 0, 3, 4) == pytest.approx(5.0)


def test_goal_distance_and_heading_matches_inline_formula():
    # Reference implementation = the exact expression environment.py used.
    def ref(rx, ry, ryaw, gx, gy):
        dx, dy = gx - rx, gy - ry
        dist = math.hypot(dx, dy)
        theta = (math.atan2(dy, dx) - ryaw + math.pi) % (2 * math.pi) - math.pi
        return dist, theta

    cases = [
        (0, 0, 0, 5, 0),
        (1, 2, 0.5, -3, 4),
        (-2, -2, -1.2, 2, 2),
        (0, 0, math.pi, -1, 0),
    ]
    for c in cases:
        d, t = g.goal_distance_and_heading(*c)
        rd, rt = ref(*c)
        assert d == pytest.approx(rd, abs=1e-9)
        assert t == pytest.approx(rt, abs=1e-9)


def test_to_robot_frame_rotation():
    # A point straight ahead in the world, robot yawed +90deg → it is on the
    # robot's RIGHT (negative body-y).
    bx, by = g.to_robot_frame(1.0, 0.0, math.pi / 2)
    assert bx == pytest.approx(0.0, abs=1e-9)
    assert by == pytest.approx(-1.0, abs=1e-9)


def test_pure_functions_are_deterministic():
    # Same inputs → byte-identical outputs across repeated calls (no hidden RNG).
    for _ in range(3):
        assert g.wrap_to_pi(12.3456) == g.wrap_to_pi(12.3456)
        assert g.goal_distance_and_heading(1, 1, 0.3, 4, 5) == \
            g.goal_distance_and_heading(1, 1, 0.3, 4, 5)
