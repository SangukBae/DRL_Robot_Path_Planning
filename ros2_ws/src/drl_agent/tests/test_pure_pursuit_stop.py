"""ROS-free tests for the Pure-Pursuit stop/yield speed ramp.

Pin two contracts:
  1. Backward compatibility — with ``low_speed_distance_m`` left at its default
     (0.0) the speed/steering output is byte-identical to the pre-stop function.
  2. Stop capability — once the ramp is enabled AND a short waypoint is
     commanded, the forward speed ramps continuously down to a true 0 m/s, so a
     policy can actually choose to stop while keeping the waypoint contract.
"""

import math

import pytest

import drl_agent.common.pure_pursuit as pp

# Hunter SE controller geometry used in the configs.
WB = 0.547696
STEER_LIM = math.radians(21.58)
CRUISE = 2.0
FACTOR = 0.6


def _cmd(x_wp, y_wp, min_speed=0.0, ramp=0.0):
    return pp.waypoint_to_command(
        x_wp, y_wp, WB, STEER_LIM, CRUISE, min_speed, FACTOR,
        low_speed_distance_m=ramp,
    )


def test_default_ramp_is_noop_backward_compatible():
    # Straight-ahead 1.5 m waypoint: full cruise, no steering.
    v, steer = _cmd(1.5, 0.0, min_speed=0.3, ramp=0.0)
    assert v == pytest.approx(CRUISE)
    assert steer == pytest.approx(0.0)


def test_old_cruising_subrange_unchanged_by_ramp():
    # With ramp = 0.8 m, any waypoint distance >= 0.8 m is in the unchanged
    # cruising regime — the ramp must not touch it.
    for r in (0.8, 1.2, 2.0):
        v_no_ramp, s_no_ramp = _cmd(r, 0.0, min_speed=0.0, ramp=0.0)
        v_ramp, s_ramp = _cmd(r, 0.0, min_speed=0.0, ramp=0.8)
        assert v_ramp == pytest.approx(v_no_ramp)
        assert s_ramp == pytest.approx(s_no_ramp)


def test_short_waypoint_ramps_speed_down():
    # Inside the ramp zone the speed scales linearly with the distance.
    full, _ = _cmd(0.8, 0.0, min_speed=0.0, ramp=0.8)
    half, _ = _cmd(0.4, 0.0, min_speed=0.0, ramp=0.8)
    assert half == pytest.approx(0.5 * full, rel=1e-6)


def test_true_zero_speed_reachable():
    # A zero-length waypoint (policy commands action[0] at its lowered floor)
    # yields a genuine 0 m/s stop regardless of the ramp.
    v, steer = _cmd(0.0, 0.0, min_speed=0.0, ramp=0.8)
    assert v == 0.0
    assert steer == 0.0


def test_ramp_overrides_min_speed_floor():
    # The ramp is applied AFTER the min_speed floor, so even a positive floor
    # cannot prevent stopping inside the ramp zone.
    v, _ = _cmd(0.05, 0.0, min_speed=0.3, ramp=0.8)
    assert v < 0.3
