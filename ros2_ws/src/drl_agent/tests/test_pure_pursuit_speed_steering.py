"""ROS-free tests for the PHASE3 continuous speed/steering action mode
(pure_pursuit.speed_steering_action_to_command / ackermann_rollout) and a
backward-compatibility pin on the pre-existing action-decoding functions this
change did NOT modify (hybrid_action_to_command / action_to_waypoint /
waypoint_to_command), so a future edit near this file trips a regression
immediately instead of silently drifting phase2/both's numerics.
"""

import math

import pytest

import drl_agent.common.pure_pursuit as pp

# Hunter SE controller geometry used in the configs.
WB = 0.547696
STEER_LIM = math.radians(21.58)
CRUISE = 2.0


# --------------------------------------------------------------------------- #
#  speed_steering_action_to_command: new-mode action contract
# --------------------------------------------------------------------------- #
def test_speed_axis_minus_one_is_zero():
    v, _ = pp.speed_steering_action_to_command([-1.0, 0.0], CRUISE, STEER_LIM)
    assert v == pytest.approx(0.0)


def test_speed_axis_zero_is_half_cruise():
    v, _ = pp.speed_steering_action_to_command([0.0, 0.0], CRUISE, STEER_LIM)
    assert v == pytest.approx(CRUISE / 2.0)


def test_speed_axis_plus_one_is_full_cruise():
    v, _ = pp.speed_steering_action_to_command([1.0, 0.0], CRUISE, STEER_LIM)
    assert v == pytest.approx(CRUISE)


def test_speed_axis_is_linear_in_between():
    v, _ = pp.speed_steering_action_to_command([-0.5, 0.0], CRUISE, STEER_LIM)
    assert v == pytest.approx(0.25 * CRUISE)


def test_steering_axis_minus_one_is_negative_limit():
    _, steer = pp.speed_steering_action_to_command([0.0, -1.0], CRUISE, STEER_LIM)
    assert steer == pytest.approx(-STEER_LIM)


def test_steering_axis_zero_is_zero():
    _, steer = pp.speed_steering_action_to_command([0.0, 0.0], CRUISE, STEER_LIM)
    assert steer == pytest.approx(0.0)


def test_steering_axis_plus_one_is_positive_limit():
    _, steer = pp.speed_steering_action_to_command([0.0, 1.0], CRUISE, STEER_LIM)
    assert steer == pytest.approx(STEER_LIM)


def test_action_is_clipped_outside_normalized_range():
    v, steer = pp.speed_steering_action_to_command([5.0, -5.0], CRUISE, STEER_LIM)
    assert v == pytest.approx(CRUISE)
    assert steer == pytest.approx(-STEER_LIM)


def test_speed_zero_means_actual_command_is_a_full_stop():
    """A -1 speed action must translate into a genuine 0 m/s command that would
    be published straight to the prefilter (linear.x=speed) -- no waypoint /
    yield indirection required to reach a stop, unlike the legacy modes."""
    v, steer = pp.speed_steering_action_to_command([-1.0, 0.3], CRUISE, STEER_LIM)
    assert v == 0.0
    # Steering is still meaningful even at zero speed (center steering angle
    # is a valid command regardless of forward speed).
    assert steer == pytest.approx(0.3 * STEER_LIM)


# --------------------------------------------------------------------------- #
#  ackermann_rollout: Ackermann-arc heading/distance used for the directional-
#  risk sector lookup (reflects BOTH steering AND commanded speed).
# --------------------------------------------------------------------------- #
def test_rollout_zero_speed_collapses_to_forward_sector():
    theta, dist = pp.ackermann_rollout(0.0, STEER_LIM, WB, 1.0)
    assert theta == 0.0
    assert dist == 0.0


def test_rollout_straight_ahead_zero_steering():
    theta, dist = pp.ackermann_rollout(1.0, 0.0, WB, 1.0)
    assert theta == pytest.approx(0.0)
    assert dist == pytest.approx(1.0)


def test_rollout_positive_steering_gives_positive_theta():
    # Positive (left) steering must roll out to a positive (left) heading.
    theta, dist = pp.ackermann_rollout(1.0, STEER_LIM, WB, 1.0)
    assert theta > 0.0
    assert dist > 0.0


def test_rollout_distance_shrinks_with_speed_same_steering():
    """The whole point of using an Ackermann rollout instead of a bare
    steering-angle proxy: at low speed the robot barely moves, so a
    near-stationary command must not pretend to be heading confidently into a
    tight-turn sector far away."""
    theta_fast, dist_fast = pp.ackermann_rollout(2.0, STEER_LIM, WB, 1.0)
    theta_slow, dist_slow = pp.ackermann_rollout(0.1, STEER_LIM, WB, 1.0)
    assert dist_slow < dist_fast
    # Direction is still consistent (same steering sign) even if magnitude differs.
    assert (theta_slow > 0) == (theta_fast > 0)


def test_rollout_negative_horizon_is_a_noop():
    theta, dist = pp.ackermann_rollout(1.0, STEER_LIM, WB, 0.0)
    assert (theta, dist) == (0.0, 0.0)


# --------------------------------------------------------------------------- #
#  ackermann_swept_path: physically-accurate swept path used by the
#  action-conditioned (speed_steering) risk target -- unlike ackermann_rollout
#  above (a single-arc heading/distance proxy for SECTOR selection), this
#  models hunter_se_cmd_prefilter's own accel/brake/steering-rate limits so a
#  stop/evasive action's predicted risk reflects real residual motion instead
#  of assuming the robot teleports to the commanded speed/steering.
# --------------------------------------------------------------------------- #
ACCEL = 6.0
BRAKE = 6.0
STEER_RATE = math.radians(200.0)


def test_swept_path_zero_horizon_or_samples_is_empty():
    assert pp.ackermann_swept_path(1.0, 0.0, 0.0, 0.0, WB, 0.0) == []
    assert pp.ackermann_swept_path(1.0, 0.0, 0.0, 0.0, WB, 1.0, num_samples=0) == []


def test_swept_path_returns_num_samples_points_even_when_fully_stopped():
    """A stationary robot commanding to stay stopped must still get
    intermediate-time samples (all at the origin) -- callers rely on this to
    catch a pedestrian who is only close at some mid-horizon instant."""
    pts = pp.ackermann_swept_path(0.0, 0.0, 0.0, 0.0, WB, 1.0, num_samples=5)
    assert len(pts) == 5
    assert pts[-1][0] == pytest.approx(1.0)
    for _t, x, y in pts:
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)


def test_swept_path_constant_arc_matches_analytic_rollout_endpoint():
    """Regression for the reviewed end-heading integration bug: when current
    speed/steering already equal the target (no ramp at all -- a pure
    constant-(v, steering) arc for the WHOLE horizon), ackermann_swept_path's
    final point must match ackermann_rollout's CLOSED-FORM circular-arc
    endpoint (radius*sin/cos of the total turn angle) to tight numerical
    precision. Using the end-of-substep heading for the position update
    (instead of the substep-midpoint heading) measurably overshoots this at
    full steering lock -- ~8.8cm off at 2 m/s / STEER_LIM / 1s / 15 samples
    before the fix -- since each substep's heading is itself changing
    linearly (at rate omega) over that substep, exactly like v_mid/steer_mid.
    """
    v, steer, horizon = 2.0, STEER_LIM, 1.0
    theta_analytic, dist_analytic = pp.ackermann_rollout(v, steer, WB, horizon)
    x_analytic = dist_analytic * math.cos(theta_analytic)
    y_analytic = dist_analytic * math.sin(theta_analytic)

    pts = pp.ackermann_swept_path(v, steer, v, steer, WB, horizon, num_samples=15)
    _t_last, x_last, y_last = pts[-1]

    assert x_last == pytest.approx(x_analytic, abs=1e-3)
    assert y_last == pytest.approx(y_analytic, abs=1e-3)


def test_swept_path_already_at_target_is_constant_velocity_straight_line():
    """No ramp transient when current speed/steering already equal the
    target -- degenerates to the same constant-velocity arc a naive
    "instant application" model would compute."""
    pts = pp.ackermann_swept_path(2.0, 0.0, 2.0, 0.0, WB, 1.0, num_samples=10)
    t_last, x_last, y_last = pts[-1]
    assert t_last == pytest.approx(1.0)
    assert x_last == pytest.approx(2.0)
    assert y_last == pytest.approx(0.0)


def test_swept_path_braking_uses_current_speed_and_reaches_analytic_stopping_point():
    """The core regression: braking from a nonzero ACTUAL speed to a stop
    target must cover the real ground the vehicle physically needs (governed
    by brake_decel_mps2), not teleport to rest at the origin. At 2.0 m/s
    braking at 6.0 m/s^2, the analytic stopping time is 2.0/6.0 s and the
    analytic stopping distance is 2.0^2/(2*6.0) m; the trapezoidal
    (midpoint-velocity) integration recovers this almost exactly."""
    horizon = 1.0
    pts = pp.ackermann_swept_path(
        2.0, 0.0, 0.0, 0.0, WB, horizon,
        accel_limit_mps2=ACCEL, brake_decel_mps2=BRAKE, num_samples=15,
    )
    stopping_time = 2.0 / BRAKE
    stopping_distance = 2.0 ** 2 / (2 * BRAKE)
    # The robot must have stopped (x no longer advancing) by the END of the
    # horizon, at (approximately) the analytic stopping distance -- NOT at
    # the origin (which an instant-stop model would have given) and NOT
    # still advancing at 2.0 m/s (which ignoring the brake limit entirely
    # would have given: 2.0 m in 1.0s).
    t_last, x_last, y_last = pts[-1]
    assert stopping_time < horizon  # sanity: robot actually stops within the horizon
    assert x_last == pytest.approx(stopping_distance, abs=1e-3)
    assert x_last < 1.0  # nowhere near "kept driving at 2.0 m/s the whole time"
    assert x_last > 0.0  # nowhere near "teleported to a dead stop instantly"


def test_swept_path_braking_from_off_grid_speed_still_close_to_analytic():
    """The 2.0 m/s braking case above happens to reach the target speed
    exactly on a substep boundary (2.0/6.0 s is an exact multiple of
    horizon/num_samples), which is a favorable case for the discretized
    integration. Pin a speed that does NOT align with the sample grid (1.0
    m/s -> stops mid-substep) to confirm the residual discretization error
    stays small (a few mm, not comparable to the ~8cm-scale errors this
    module's other fixes address) instead of being masked by the aligned
    case above."""
    horizon = 1.0
    pts = pp.ackermann_swept_path(
        1.0, 0.0, 0.0, 0.0, WB, horizon,
        accel_limit_mps2=ACCEL, brake_decel_mps2=BRAKE, num_samples=15,
    )
    stopping_distance = 1.0 ** 2 / (2 * BRAKE)  # ~0.0833 m
    _t_last, x_last, _y_last = pts[-1]
    assert x_last == pytest.approx(stopping_distance, abs=0.005)


def test_swept_path_accelerating_from_rest_is_rate_limited_not_instant():
    """Symmetric check: an accelerating command from rest must also ramp up
    (at accel_limit_mps2) rather than instantly reaching the target speed."""
    pts = pp.ackermann_swept_path(
        0.0, 0.0, 2.0, 0.0, WB, 1.0, accel_limit_mps2=ACCEL, num_samples=15,
    )
    accel_time = 2.0 / ACCEL
    accel_distance_during_ramp = 2.0 ** 2 / (2 * ACCEL)
    # An instant-application model would have covered 2.0 m in 1.0s; the
    # rate-limited ramp must cover strictly less (the first accel_time
    # seconds are spent below cruise speed).
    t_last, x_last, _y_last = pts[-1]
    naive_instant_distance = 2.0 * 1.0
    assert x_last < naive_instant_distance
    expected = accel_distance_during_ramp + 2.0 * (1.0 - accel_time)
    assert x_last == pytest.approx(expected, abs=1e-3)


def test_swept_path_steering_is_rate_limited_towards_target():
    """A large steering command from straight-ahead must curve GRADUALLY
    (steering_rate_rad_s-limited), not snap the heading instantly -- checked
    indirectly via the swept path curving progressively more over time
    (lateral offset grows super-linearly early on as steering ramps in)."""
    pts = pp.ackermann_swept_path(
        1.0, 0.0, 1.0, STEER_LIM, WB, 1.0,
        steering_rate_rad_s=STEER_RATE, num_samples=20,
    )
    # Time to reach full steering lock from zero.
    steer_ramp_time = STEER_LIM / STEER_RATE
    assert steer_ramp_time < 1.0
    # The very first sample (well before the steering ramp completes) must
    # be nearly straight ahead -- a negligible lateral (y) offset -- since
    # steering has barely begun turning in.
    early_t, _early_x, early_y = pts[0]
    assert early_t < steer_ramp_time
    assert early_y < 0.001
    # By the end of the horizon (steering reached full lock long ago and has
    # been holding it), the path has curved substantially.
    _t_last, _x_last, y_last = pts[-1]
    assert y_last > 0.05


# --------------------------------------------------------------------------- #
#  Backward-compatibility pin: existing waypoint_yield (3D hybrid) and legacy
#  2D waypoint decoding are BYTE-IDENTICAL to before this change (no lines
#  inside these functions were touched -- only NEW functions were added
#  alongside them -- but pin known numeric outputs so any future edit near
#  this module trips immediately).
# --------------------------------------------------------------------------- #
FACTOR = 0.6
ACTIONS_LOW = [0.0, -0.524, -1.0]
ACTIONS_HIGH = [2.0, 0.524, 1.0]


def test_hybrid_action_to_command_move_mode_unchanged():
    v, steer, theta, ctl = pp.hybrid_action_to_command(
        [0.5, 0.0, -1.0], ACTIONS_LOW, ACTIONS_HIGH, WB, STEER_LIM, CRUISE, FACTOR,
        yield_enabled=True, yield_threshold=0.3,
        lookahead_min_m=0.8, v_move_min_mps=0.35, yield_creep_speed_mps=0.0,
    )
    assert ctl["yielding"] is False
    assert theta == pytest.approx(0.0)
    assert steer == pytest.approx(0.0)
    assert v == pytest.approx(CRUISE)   # straight ahead, no steering penalty


def test_hybrid_action_to_command_yield_mode_unchanged():
    v, steer, theta, ctl = pp.hybrid_action_to_command(
        [0.5, 0.0, 1.0], ACTIONS_LOW, ACTIONS_HIGH, WB, STEER_LIM, CRUISE, FACTOR,
        yield_enabled=True, yield_threshold=0.3,
        lookahead_min_m=0.8, v_move_min_mps=0.35, yield_creep_speed_mps=0.0,
    )
    assert ctl["yielding"] is True
    assert v == pytest.approx(0.0)   # yield creep speed 0.0 -> full stop


def test_action_to_waypoint_unchanged():
    r, theta, x_wp, y_wp = pp.action_to_waypoint([1.0, 0.0], [0.8, -0.524], [2.0, 0.524])
    assert r == pytest.approx(2.0)
    assert theta == pytest.approx(0.0)
    assert x_wp == pytest.approx(2.0)
    assert y_wp == pytest.approx(0.0)
