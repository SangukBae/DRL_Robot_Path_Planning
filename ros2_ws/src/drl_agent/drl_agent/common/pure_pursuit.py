#!/usr/bin/env python3
"""Shared Pure-Pursuit action→waypoint→command helpers.

Extracted from environment.py so the simulator (environment.py) and the
real-robot inference node (real_policy_runner.py) use *exactly* the same
action→cmd_vel convention. Pure functions only — no ROS, no state.

Convention (matches the trained policy):
  action[0] → waypoint distance r  ∈ [actions_low[0], actions_high[0]] m (forward)
  action[1] → waypoint angle theta ∈ [actions_low[1], actions_high[1]] rad
              (robot frame, positive = left / CCW)
  waypoint (robot frame): x_wp = r·cos(theta) (forward), y_wp = r·sin(theta) (left)
  command: center steering angle [rad] (clipped) + forward speed [m/s].
"""

import math
import numpy as np


def action_to_waypoint(action, actions_low, actions_high):
    """Map a normalized action in [-1, 1]^2 to (r, theta, x_wp, y_wp)."""
    a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
    low = np.asarray(actions_low, dtype=np.float32)
    high = np.asarray(actions_high, dtype=np.float32)
    cmd = 0.5 * (a + 1.0) * (high - low) + low  # [-1, 1] → [low, high]
    r = float(np.clip(cmd[0], low[0], high[0]))
    theta = float(np.clip(cmd[1], low[1], high[1]))
    return r, theta, r * math.cos(theta), r * math.sin(theta)


def waypoint_to_command(x_wp, y_wp, wheelbase_m, steering_limit_rad,
                        cruise_speed_mps, min_speed_mps, speed_steer_factor,
                        low_speed_distance_m=0.0):
    """Pure Pursuit: robot-frame waypoint → (speed [m/s], steering [rad]).

    Speed is a function of the commanded waypoint *geometry* only (the
    steering ratio), so the original contract is preserved exactly when
    ``low_speed_distance_m == 0.0`` (the default).

    STOP/YIELD capability (opt-in, ``low_speed_distance_m > 0``): the forward
    speed is additionally ramped DOWN toward 0 as the commanded waypoint
    distance ``L`` drops below ``low_speed_distance_m``
    (``speed *= L / low_speed_distance_m``). This is applied AFTER the
    ``min_speed_mps`` floor so it can pull the speed all the way to 0 — i.e. a
    policy that commands a short waypoint (small ``action[0]``) can creep or
    fully stop. The steering geometry is unchanged. With the default the only
    floor is ``min_speed_mps`` exactly as before, so to truly reach 0 m/s the
    caller must BOTH lower the action floor (``actions_low[0]``) so a short
    waypoint is reachable AND set ``low_speed_distance_m > 0`` (and/or
    ``min_speed_mps = 0``).
    """
    L = math.hypot(x_wp, y_wp)
    if L < 1e-3:
        return 0.0, 0.0
    steering = math.atan2(2.0 * y_wp * wheelbase_m, L * L)
    steering = float(np.clip(steering, -steering_limit_rad, steering_limit_rad))
    steer_ratio = abs(steering) / max(steering_limit_rad, 1e-6)
    speed = cruise_speed_mps * (1.0 - speed_steer_factor * steer_ratio)
    speed = max(speed, min_speed_mps)
    if low_speed_distance_m > 0.0 and L < low_speed_distance_m:
        speed *= L / low_speed_distance_m
    return speed, steering


def speed_steering_action_to_command(action, cruise_speed_mps, steering_limit_rad):
    """Continuous speed/steering action -> (speed [m/s], steering [rad]).

    New ``action_mode: speed_steering`` contract (2-D, no waypoint, no binary
    yield channel):
      action[0] in [-1, 1] -> speed in [0, cruise_speed_mps] (linear map;
        -1 -> 0 m/s so the policy can bring the robot to a genuine continuous
        stop, without the waypoint/yield indirection ``hybrid_action_to_command``
        needs to reach 0).
      action[1] in [-1, 1] -> center steering angle in [-steering_limit_rad,
        +steering_limit_rad].

    Output is fed straight to the prefilter as linear.x=speed,
    angular.z=steering -- the same Twist contract every other action mode
    already uses (see hybrid_action_to_command / waypoint_to_command).
    """
    a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
    speed = float(0.5 * (float(a[0]) + 1.0) * cruise_speed_mps)
    steering = float(np.clip(float(a[1]) * steering_limit_rad,
                              -steering_limit_rad, steering_limit_rad))
    return speed, steering


def ackermann_rollout(speed_mps, steering_rad, wheelbase_m, horizon_sec):
    """Bicycle-model forward rollout: robot-frame heading/distance to where the
    robot will actually be after ``horizon_sec`` at constant (speed, steering).

    Returns (theta, distance):
      theta    = atan2(dy, dx), the robot-frame angle (0 = straight ahead)
                 toward the rolled-out point.
      distance = hypot(dx, dy), how far the robot actually gets in that time.

    Used (speed_steering mode only) to pick the directional-risk sector for
    the Action-Risk Head / risk_map_reward target instead of a steering-angle
    -only proxy: a near-zero commanded speed collapses distance -> ~0 and
    theta -> 0 (defaults to the forward sector) regardless of the steering
    angle, so the target reflects the actual commanded DECELERATION, not just
    which way the wheels are pointed. At speed 0 (or non-positive horizon) the
    robot goes nowhere, so (0.0, 0.0) is returned directly.
    """
    v = float(speed_mps)
    if abs(v) < 1e-6 or horizon_sec <= 0.0:
        return 0.0, 0.0
    omega = v * math.tan(steering_rad) / max(wheelbase_m, 1e-6)
    if abs(omega) < 1e-6:
        dx, dy = v * horizon_sec, 0.0
    else:
        radius = v / omega
        dx = radius * math.sin(omega * horizon_sec)
        dy = radius * (1.0 - math.cos(omega * horizon_sec))
    return math.atan2(dy, dx), math.hypot(dx, dy)


def _move_towards(current, target, max_delta):
    """Symmetric rate limiter -- mirrors hunter_se_cmd_prefilter.py's
    `move_towards`: steps `current` toward `target` by at most `max_delta`,
    snapping to `target` if already within range."""
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + math.copysign(max_delta, delta)


# Defaults mirror hunter_se_gazebo/config/hunter_se_cmd_prefilter.yaml --
# keep these two files in sync (the prefilter is the ground truth; these are
# a forward-PREDICTION of what it will do, not a second source of truth).
_DEFAULT_ACCEL_LIMIT_MPS2 = 6.0
_DEFAULT_BRAKE_DECEL_MPS2 = 6.0
_DEFAULT_STEERING_RATE_RAD_S = math.radians(200.0)
_DEFAULT_PATH_SAMPLES = 15


def ackermann_swept_path(
    current_speed_mps, current_steering_rad,
    target_speed_mps, target_steering_rad,
    wheelbase_m, horizon_sec,
    *,
    accel_limit_mps2=_DEFAULT_ACCEL_LIMIT_MPS2,
    brake_decel_mps2=_DEFAULT_BRAKE_DECEL_MPS2,
    steering_rate_rad_s=_DEFAULT_STEERING_RATE_RAD_S,
    num_samples=_DEFAULT_PATH_SAMPLES,
):
    """Midpoint-integrated bicycle-model rollout of the robot's ACTUAL
    short-horizon swept path, accounting for hunter_se_cmd_prefilter's control
    dynamics --
    NOT an instant jump to (target_speed, target_steering) on this same tick.
    Used to build the directional-risk / Action-Risk Head target (see
    environment.py::_compute_directional_risk) so a stop/evasive action's
    predicted risk reflects the real residual motion before the robot
    actually reaches that speed/steering, instead of assuming it teleports
    there -- e.g. braking from 2 m/s to a stop at the default 6 m/s^2 still
    covers ~0.33 m over ~0.33 s, which a naive "instant stop" rollout would
    have scored as zero displacement.

    Modeled dynamics (see hunter_se_cmd_prefilter.py):
      - speed: rate-limited ramp toward target_speed_mps, using
        accel_limit_mps2 while |target| >= |current speed| (speeding up,
        either direction) or brake_decel_mps2 while slowing down --
        matching the prefilter's own magnitude-based selection.
      - steering: rate-limited ramp toward target_steering_rad at
        steering_rate_rad_s (the prefilter applies no lag to steering, so
        this part is exact, not an approximation).
    Deliberately NOT modeled: the prefilter's additional first-order lag on
    speed (speed_lag_tau_sec) before its own rate limit. For any speed step
    bigger than roughly `accel_limit_mps2 * dt / (1 - exp(-dt/tau))` (a
    fraction of a m/s at this vehicle's tuned tau=0.05s and 50 Hz tick) the
    rate limit binds almost the entire time regardless -- the lag only
    shapes the last sliver of a small correction -- and a genuine near-zero
    target speed bypasses the lag entirely in the prefilter's own deadband
    branch, ramping straight to 0 at brake_decel_mps2. So a pure rate-limited
    ramp is a close approximation for exactly the safety-relevant regime
    (large, sudden speed changes -- emergency stops, sudden accelerations),
    at a fraction of the cost of a full per-tick lag+ratelimit simulation.

    ``current_speed_mps``/``current_steering_rad``: the robot's ACTUAL
    measured state right now (e.g. self.latest_actual_signed_speed /
    self.latest_center_steering) -- the rollout's initial condition. Passing
    the COMMANDED (not yet realized) v/steering here instead would silently
    reintroduce the "instant application" assumption this function exists to
    remove.

    ``num_samples`` (default 15, up from the previous fixed 5): finer
    resolution reduces how much a fast crossing's true closest approach (which
    can fall strictly between two samples) is overestimated -- see
    environment_curriculum.yaml's directional_risk.rollout_path_samples.

    Returns a list of (t, x, y) tuples, t in (0, horizon_sec], t ascending,
    robot-LOCAL frame (x forward / y left) relative to the robot's CURRENT
    pose/heading. Empty list only at horizon_sec/num_samples <= 0 (degenerate
    config) -- NOT at current/target speed == 0: a stopped robot still needs
    intermediate-time samples so a pedestrian who is only close to it at some
    time STRICTLY BETWEEN now and the horizon is still caught by the caller's
    matching-time comparison, not just an endpoint check.
    """
    if horizon_sec <= 0.0 or num_samples <= 0:
        return []
    dt = horizon_sec / num_samples
    v = float(current_speed_mps)
    steer = float(current_steering_rad)
    x = y = heading = 0.0
    pts = []
    for i in range(1, int(num_samples) + 1):
        v_prev, steer_prev = v, steer
        rate = accel_limit_mps2 if abs(target_speed_mps) >= abs(v) else brake_decel_mps2
        v = _move_towards(v, float(target_speed_mps), rate * dt)
        steer = _move_towards(steer, float(target_steering_rad), steering_rate_rad_s * dt)
        # Trapezoidal (midpoint-velocity) integration, not the substep's raw
        # post-ramp v/steer: within a substep the ramp is linear (constant
        # rate) right up until it snaps onto the target, so the AVERAGE of
        # the substep's start/end values recovers that substep's true
        # integral (exactly, for the common case where the ramp doesn't
        # cross onto its target mid-substep) instead of systematically
        # under/over-shooting a braking/accelerating trajectory -- e.g. a
        # 2 m/s -> 0 brake at 6 m/s^2 should cover exactly v0^2/(2*decel), and
        # a plain post-ramp-value Euler step undershoots that by ~20%.
        v_mid = 0.5 * (v_prev + v)
        steer_mid = 0.5 * (steer_prev + steer)
        omega = v_mid * math.tan(steer_mid) / max(wheelbase_m, 1e-6)
        # Use the SUBSTEP-MIDPOINT heading for the position update, not the
        # end-of-substep heading -- position is integrated from a heading
        # that is itself changing linearly (at rate omega) over the substep,
        # so (like v_mid/steer_mid above) the midpoint value is what recovers
        # the substep's true arc integral instead of systematically curving
        # the reported point a bit further than the real midpoint-of-turn
        # position (e.g. ~8.8cm off at 2 m/s, full steering lock, 1s horizon,
        # 15 samples -- a fixed-speed/fixed-steering circular arc test pins
        # this against the closed-form arc endpoint).
        heading_mid = heading + 0.5 * omega * dt
        x += v_mid * math.cos(heading_mid) * dt
        y += v_mid * math.sin(heading_mid) * dt
        heading += omega * dt
        pts.append((dt * i, x, y))
    return pts


def hybrid_action_to_command(
    action, actions_low, actions_high,
    wheelbase_m, steering_limit_rad,
    cruise_speed_mps, speed_steer_factor,
    *,
    yield_enabled=True,
    yield_threshold=0.0,
    lookahead_min_m=0.8,
    v_move_min_mps=0.35,
    yield_creep_speed_mps=0.0,
):
    """3D hybrid action → (speed [m/s], steering [rad], theta [rad], info).

    Keeps the existing axis convention and ADDS a dedicated stop/yield channel:
      action[0] → waypoint distance r  ∈ [actions_low[0], actions_high[0]] m
      action[1] → waypoint angle theta ∈ [actions_low[1], actions_high[1]] rad
      action[2] → yield scalar         (raw normalized value in [-1, 1])

    This REPLACES the old "short waypoint = implicit stop" contract (the
    ``low_speed_distance_m`` ramp in :func:`waypoint_to_command`). Stopping is
    now possible ONLY in yield mode, so avoidance (steer/drive) and yielding
    (stop/creep) live on separate axes.

    Mode is decided by the yield channel:
      * ``yield_enabled and action[2] >= yield_threshold`` → YIELD mode: the
        forward speed is capped at ``yield_creep_speed_mps`` (0 → full stop).
      * otherwise → MOVE mode: the policy CANNOT stop here — the lookahead is
        floored to ``lookahead_min_m`` (so a tight/short waypoint can't collapse
        to 0) AND the forward speed is floored to ``v_move_min_mps``.

    The steering geometry is identical to :func:`waypoint_to_command`. Legacy 2D
    actions (len < 3) are treated as MOVE mode (a2 = -inf), so this stays usable
    for the non-curriculum path if ever wired there.
    """
    a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
    low = np.asarray(actions_low, dtype=np.float32)
    high = np.asarray(actions_high, dtype=np.float32)
    cmd = 0.5 * (a + 1.0) * (high - low) + low
    r = float(np.clip(cmd[0], low[0], high[0]))
    theta = float(np.clip(cmd[1], low[1], high[1]))
    a_yield = float(a[2]) if a.shape[0] > 2 else -1.0
    yielding = bool(yield_enabled and a_yield >= yield_threshold)

    if not yielding:
        r = max(r, lookahead_min_m)          # MOVE: lookahead floor (no collapse)
    x_wp, y_wp = r * math.cos(theta), r * math.sin(theta)
    L = math.hypot(x_wp, y_wp)
    if L < 1e-3:
        return 0.0, 0.0, theta, {"yielding": yielding, "r": r}
    steering = math.atan2(2.0 * y_wp * wheelbase_m, L * L)
    steering = float(np.clip(steering, -steering_limit_rad, steering_limit_rad))
    steer_ratio = abs(steering) / max(steering_limit_rad, 1e-6)
    speed = cruise_speed_mps * (1.0 - speed_steer_factor * steer_ratio)
    if yielding:
        speed = min(speed, yield_creep_speed_mps)   # YIELD: stop / creep allowed
    else:
        speed = max(speed, v_move_min_mps)          # MOVE: HARD speed floor
    return speed, steering, theta, {"yielding": yielding, "r": r}
