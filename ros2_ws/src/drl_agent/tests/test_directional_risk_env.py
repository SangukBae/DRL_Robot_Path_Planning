"""Tests for the env-side PHASE2 plumbing on Environment
(`_compute_directional_risk`, `_append_aux_labels`'s action-risk prepend).

These are pure-Python methods on the ROS Node class but live in a module that
imports ROS at load time. Follows the project's established stub-import
pattern (see test_open_map_safe_start_yaw.py / test_episode_active_counts_
wiring.py): import `environment` with ROS deps stubbed, then drive the REAL
unbound methods on a minimal fake node (types.SimpleNamespace).

Covers:
  * _compute_directional_risk reads the pre-step human_states/GT pose and
    returns (risk_dir, min_dist_dir) sliced at the sector matching theta.
  * _append_aux_labels prepends the action-risk sentinel block ONLY when
    action_risk_head_env_enabled AND a target is supplied; absent (byte-
    identical to before PHASE2) when either is false/None.
  * reset() never gets this block (the trainer always calls _append_aux_labels
    with action_risk_target=None there — verified via the same default).
"""

import math
import sys
import threading
import types

import numpy as np
import pytest


_STUB_NAMES = (
    "rclpy", "rclpy.node", "rclpy.parameter", "rclpy.executors",
    "rclpy.callback_groups", "rclpy.qos",
    "rcl_interfaces", "rcl_interfaces.msg", "squaternion",
    "geometry_msgs", "geometry_msgs.msg", "nav_msgs", "nav_msgs.msg",
    "sensor_msgs", "sensor_msgs.msg", "visualization_msgs", "visualization_msgs.msg",
    "std_msgs", "std_msgs.msg",
    "drl_agent_interfaces", "drl_agent_interfaces.msg", "drl_agent_interfaces.srv",
    "ros_gz_interfaces", "ros_gz_interfaces.msg", "ros_gz_interfaces.srv",
)


def _import_environment_with_temp_stubs():
    class _Meta(type):
        _n = [0]

        def __getattr__(cls, name):
            _Meta._n[0] += 1
            return _Meta._n[0]

    saved = {n: sys.modules.get(n) for n in _STUB_NAMES}
    try:
        for n in _STUB_NAMES:
            if n not in sys.modules:
                mod = types.ModuleType(n)
                mod.__getattr__ = lambda name: _Meta(name, (), {})  # noqa: B023
                sys.modules[n] = mod
        import drl_agent.env.simulation.environment as env_mod
        return env_mod
    finally:
        for n, orig in saved.items():
            if orig is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = orig


env_mod = _import_environment_with_temp_stubs()
Environment = env_mod.Environment
aux_labels = env_mod.aux_labels


def _directional_risk_cfg(**over):
    base = dict(enabled=True, num_sectors=16, horizons_sec=[1.0],
                risk_distance_scale=3.0, min_speed_for_motion=0.05)
    base.update(over)
    return aux_labels.AuxLabelConfig(base)


def _node(**overrides):
    node = types.SimpleNamespace(
        human_states={},
        _human_lock=threading.Lock(),
        gt_x=0.0, gt_y=0.0, gt_yaw=0.0,
        _directional_risk_cfg=_directional_risk_cfg(),
        action_risk_head_env_enabled=False,
        _aux_pred_enabled=False,
        # Default to a non-speed_steering mode so every EXISTING test below
        # (none of which sets action_mode) keeps exercising the ORIGINAL,
        # current-pose-only per-sector path (compute_directional_risk_map) --
        # matching phase2/waypoint_yield's real, unchanged behaviour.
        action_mode="waypoint_yield",
        # speed_steering-only rollout inputs/dynamics (only touched by that
        # branch -- see _compute_directional_risk). Defaults mirror
        # hunter_se_cmd_prefilter.yaml. Tests exercising speed_steering set
        # latest_actual_signed_speed/latest_center_steering explicitly.
        latest_actual_signed_speed=0.0,
        latest_center_steering=0.0,
        _dr_rollout_accel_mps2=6.0,
        _dr_rollout_brake_decel_mps2=6.0,
        _dr_rollout_steering_rate_rad_s=math.radians(200.0),
        _dr_rollout_path_samples=15,
        # TRAJ_RISK: default OFF -> every existing test above (none of which
        # sets this) keeps exercising the original per-sector lookup for
        # waypoint_yield, unaffected by the swept-path extension.
        _waypoint_trajectory_risk_enabled=False,
    )
    for k, v in overrides.items():
        setattr(node, k, v)
    return node


# --------------------------------------------------------------------------- #
#  _compute_directional_risk
# --------------------------------------------------------------------------- #
def test_no_humans_zero_risk_far_min_dist():
    node = _node()
    risk_dir, min_dist_dir = Environment._compute_directional_risk(node, 0.0)
    assert risk_dir == 0.0
    assert min_dist_dir == 1.0


def test_human_directly_ahead_high_risk_in_forward_sector():
    # Human 1m straight ahead (theta=0, robot frame), well inside D_c=3.0.
    node = _node(human_states={"h0": {"x": 1.0, "y": 0.0, "yaw": math.pi, "v": 0.0}})
    risk_dir, min_dist_dir = Environment._compute_directional_risk(node, 0.0)
    assert risk_dir > 0.5
    assert min_dist_dir < 1.0


def test_human_behind_low_risk_for_forward_theta():
    # Human directly BEHIND the robot must not raise the FORWARD sector's risk.
    node = _node(human_states={"h0": {"x": -1.0, "y": 0.0, "yaw": 0.0, "v": 0.0}})
    risk_dir, _ = Environment._compute_directional_risk(node, 0.0)
    assert risk_dir == 0.0


def test_min_dist_dir_is_sector_specific_not_global():
    # Regression for the reviewed bug: a human very close but BEHIND (a
    # different sector) must NOT pull the FORWARD sector's min_dist_dir down.
    # min_dist_dir must be the nearest human WITHIN theta's sector, not a
    # horizon-global nearest-human-anywhere distance.
    node = _node(human_states={"h0": {"x": -0.2, "y": 0.0, "yaw": 0.0, "v": 0.0}})
    risk_dir, min_dist_dir = Environment._compute_directional_risk(node, 0.0)  # forward
    assert risk_dir == 0.0
    # The old (buggy) global-min behaviour would have given ~0.2/3.0 = 0.067 here.
    assert min_dist_dir == 1.0


def test_far_human_no_risk():
    node = _node(human_states={"h0": {"x": 20.0, "y": 0.0, "yaw": 0.0, "v": 0.0}})
    risk_dir, min_dist_dir = Environment._compute_directional_risk(node, 0.0)
    assert risk_dir == 0.0
    assert min_dist_dir == 1.0


def test_phase2_waypoint_yield_ignores_v_and_cmd_steering():
    """Isolation regression: for any action_mode OTHER than speed_steering
    (e.g. phase2's waypoint_yield), _compute_directional_risk must be
    COMPLETELY unaffected by v/cmd_steering -- the speed_steering-only
    swept-path fix must never change phase2's reward/target semantics,
    whatever v/cmd_steering happen to be computed at the call site (they
    always are, for every action_mode -- see _step_callback_impl)."""
    node = _node(
        action_mode="waypoint_yield",
        human_states={"h0": {"x": 1.0, "y": 0.0, "yaw": math.pi, "v": 0.3}},
    )
    baseline = Environment._compute_directional_risk(node, 0.0, target_v=0.0, target_cmd_steering=0.0)
    moving = Environment._compute_directional_risk(node, 0.0, target_v=1.5, target_cmd_steering=0.3)
    assert baseline == moving


def test_speed_steering_swept_path_catches_crossing_human():
    """Regression for the reviewed counter-example: a robot driving STRAIGHT
    AHEAD and a pedestrian crossing perpendicular whose paths nearly collide
    at the horizon's end.

    The pedestrian's OWN bearing sector (from the robot's CURRENT pose) is a
    DIFFERENT sector (6) than the action's own heading sector (8, straight
    ahead == num_sectors // 2). A per-sector lookup at the action's sector
    would read risk 0 (nothing maps to sector 8) even though the swept paths
    actually meet -- action_mode=="speed_steering" must use the
    action-conditioned GLOBAL swept-path risk (no sector re-filtering)
    instead, and must report the true collision.
    """
    cfg = _directional_risk_cfg(horizons_sec=[1.0])
    action_sector = aux_labels.sector_index_for_theta(0.0, cfg.num_sectors)
    human_bearing_sector = aux_labels.sector_index_for_theta(
        math.atan2(-2.0, 2.0), cfg.num_sectors)
    assert action_sector == 8
    assert human_bearing_sector == 6
    assert action_sector != human_bearing_sector

    node = _node(
        action_mode="speed_steering",
        vehicle_wheelbase_m=1.0,
        _directional_risk_cfg=cfg,
        # Robot is ALREADY cruising at the target speed/steering (no ramp-up
        # transient) so the rollout is a clean constant-velocity straight
        # line -- isolating this test to the sector-filter fix, not the
        # separate accel/brake-ramp dynamics (see test_pure_pursuit_speed_
        # steering.py for those).
        latest_actual_signed_speed=2.0,
        latest_center_steering=0.0,
        # Pedestrian starting at (2, -2), walking straight toward +y at 2 m/s
        # -> reaches (2, 0) at t=1.0s.
        human_states={"h0": {"x": 2.0, "y": -2.0, "yaw": math.pi / 2, "v": 2.0}},
    )
    # Robot drives straight ahead (theta=0, cmd_steering=0) at 2 m/s -> also
    # reaches (2, 0) at t=1.0s: a genuine collision at the swept-path endpoint.
    risk_dir, min_dist_dir = Environment._compute_directional_risk(
        node, 0.0, target_v=2.0, target_cmd_steering=0.0)
    assert risk_dir == pytest.approx(1.0)
    assert min_dist_dir == pytest.approx(0.0)


def test_speed_steering_stopped_robot_catches_intermediate_time_close_pass():
    """A STOPPED robot (already at rest AND commanding a stop) must still
    catch a pedestrian who is only close to it at some time STRICTLY BETWEEN
    now and the horizon -- not just the endpoints -- since
    ackermann_swept_path now samples intermediate times even at v==0 (the
    robot just stays at the origin for each sample).

    Pinning _dr_rollout_path_samples=5 (rather than the current default 15)
    keeps this test's numbers independent of the default sample count: with
    5 samples over a 1.0s horizon, sample times are 0.2/0.4/0.6/0.8/1.0. The
    pedestrian here passes exactly through the robot's (stationary) position
    at t=0.6s and is well outside D_c=3.0 at every other sample and at t=0 --
    so this can ONLY be caught by actually sampling t=0.6, not by an
    endpoint-only (t=0 vs t=horizon) check.
    """
    cfg = _directional_risk_cfg(horizons_sec=[1.0])
    vx, vy = 3.0, -5.0  # crosses (x=-1.8, y=3.0) -> (x=0, y=0) at t=0.6s
    v = math.hypot(vx, vy)
    yaw = math.atan2(vy, vx)
    node = _node(
        action_mode="speed_steering",
        vehicle_wheelbase_m=1.0,
        _directional_risk_cfg=cfg,
        _dr_rollout_path_samples=5,
        latest_actual_signed_speed=0.0,
        latest_center_steering=0.0,
        human_states={"h0": {"x": -1.8, "y": 3.0, "yaw": yaw, "v": v}},
    )
    risk_dir, min_dist_dir = Environment._compute_directional_risk(
        node, 0.0, target_v=0.0, target_cmd_steering=0.0)
    # Nearest approach is at (0, 0) -- distance 0 from the stationary robot.
    assert risk_dir == pytest.approx(1.0)
    assert min_dist_dir == pytest.approx(0.0)


def test_speed_steering_braking_from_speed_reaches_true_stopping_point():
    """Regression for the reviewed prefilter-dynamics gap: a robot moving at
    2 m/s that commands a full stop does NOT teleport to rest at its current
    position -- hunter_se_cmd_prefilter brakes it at brake_decel_mps2,
    covering real ground first. The rollout must use the robot's CURRENT
    actual speed (self.latest_actual_signed_speed) as the swept-path's
    initial condition, not silently assume the target speed applies
    instantly from t=0.

    At accel/brake=6.0 m/s^2 (this fixture's default, matching
    hunter_se_cmd_prefilter.yaml), braking from 2.0 m/s to 0 takes
    2.0/6.0 ~= 0.333s and covers EXACTLY v0^2/(2*decel) ~= 0.333m (the
    trapezoidal/midpoint-velocity integration in ackermann_swept_path
    recovers this analytic value almost exactly). Placing a stationary human
    exactly at that true stopping point must register as an actual
    near-collision (min_dist ~= 0, risk ~= 1.0) -- an "instant stop" model
    would instead have the robot never leave the origin, scoring this same
    human as merely ~0.333 m away (min_dist ~= 0.111, risk ~= 0.889).
    """
    cfg = _directional_risk_cfg(horizons_sec=[1.0])
    stopping_distance = 2.0 ** 2 / (2 * 6.0)  # exact analytic braking distance
    node = _node(
        action_mode="speed_steering",
        vehicle_wheelbase_m=1.0,
        _directional_risk_cfg=cfg,
        latest_actual_signed_speed=2.0,
        latest_center_steering=0.0,
        human_states={"h0": {"x": stopping_distance, "y": 0.0, "yaw": 0.0, "v": 0.0}},
    )
    risk_dir, min_dist_dir = Environment._compute_directional_risk(
        node, 0.0, target_v=0.0, target_cmd_steering=0.0)
    assert min_dist_dir == pytest.approx(0.0, abs=1e-6)
    assert risk_dir == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
#  TRAJ_RISK: waypoint_yield swept-path risk (opt-in extension)
# --------------------------------------------------------------------------- #
def test_waypoint_trajectory_risk_disabled_is_unaffected_by_flag_default():
    """directional_risk.waypoint_trajectory_risk_enabled defaults False (see
    _node()'s base fixture) -- waypoint_yield must keep using the current-
    pose-only per-sector lookup, exactly like before this feature existed."""
    node = _node(
        action_mode="waypoint_yield",
        human_states={"h0": {"x": 1.0, "y": 0.0, "yaw": math.pi, "v": 0.3}},
    )
    assert node._waypoint_trajectory_risk_enabled is False
    baseline = Environment._compute_directional_risk(node, 0.0, target_v=0.0, target_cmd_steering=0.0)
    moving = Environment._compute_directional_risk(node, 0.0, target_v=1.5, target_cmd_steering=0.3)
    assert baseline == moving


def test_waypoint_trajectory_risk_enabled_swept_path_catches_crossing_human():
    """A robot driving STRAIGHT AHEAD and a pedestrian crossing perpendicular
    whose paths nearly collide at the swept-path's ENDPOINT (t=1.0s, matching
    test_speed_steering_swept_path_catches_crossing_human's setup exactly).

    The DISABLED (per-sector) target -- compute_directional_risk_map -- has
    NO notion of the robot's own motion at all: it only ever looks at where
    each human is PROJECTED to be at the single horizon time, from the
    robot's CURRENT (stationary) pose, then bins that into a sector. It
    therefore reports a much LOWER risk here (the human's horizon-projected
    position is 2.0 m from the robot's CURRENT position) than the true
    swept-path answer (an actual near-collision, since the robot itself
    moves to meet the human at that same point/time). ENABLED must catch the
    true near-collision the endpoint-only, motion-blind lookup misses.
    """
    cfg = _directional_risk_cfg(horizons_sec=[1.0])
    human_states = {"h0": {"x": 2.0, "y": -2.0, "yaw": math.pi / 2, "v": 2.0}}
    base_kwargs = dict(
        action_mode="waypoint_yield",
        vehicle_wheelbase_m=1.0,
        _directional_risk_cfg=cfg,
        latest_actual_signed_speed=2.0,
        latest_center_steering=0.0,
        human_states=human_states,
    )

    node_off = _node(_waypoint_trajectory_risk_enabled=False, **base_kwargs)
    risk_off, min_dist_off = Environment._compute_directional_risk(
        node_off, 0.0, target_v=2.0, target_cmd_steering=0.0)
    # Sector lookup: human projects to (2, 0) at t=1.0 (2.0 m from the
    # robot's CURRENT, unmoved position) -> risk = 1 - 2.0/3.0.
    assert risk_off == pytest.approx(1.0 / 3.0, abs=1e-3)

    node_on = _node(_waypoint_trajectory_risk_enabled=True, **base_kwargs)
    risk_on, min_dist_on = Environment._compute_directional_risk(
        node_on, 0.0, target_v=2.0, target_cmd_steering=0.0)
    assert risk_on == pytest.approx(1.0)
    assert min_dist_on == pytest.approx(0.0)
    assert risk_on > risk_off  # swept path catches what the endpoint-only lookup misses


def test_waypoint_trajectory_risk_enabled_turn_away_scores_lower_than_straight():
    """In a genuinely risky situation (human directly ahead), a swerve-away
    command must score LOWER risk than driving straight at the same target
    speed -- the swept path actually curves away from the human, unlike the
    old sector-lookup target (current-pose-only, blind to the robot's own
    commanded motion)."""
    cfg = _directional_risk_cfg(horizons_sec=[1.0])
    node = _node(
        action_mode="waypoint_yield",
        _waypoint_trajectory_risk_enabled=True,
        vehicle_wheelbase_m=1.0,
        _directional_risk_cfg=cfg,
        latest_actual_signed_speed=2.0,
        latest_center_steering=0.0,
        human_states={"h0": {"x": 1.5, "y": 0.0, "yaw": 0.0, "v": 0.0}},
    )
    risk_straight, min_dist_straight = Environment._compute_directional_risk(
        node, 0.0, target_v=2.0, target_cmd_steering=0.0)
    risk_turn, min_dist_turn = Environment._compute_directional_risk(
        node, 0.3, target_v=2.0, target_cmd_steering=0.3)
    assert risk_straight > 0.5
    assert risk_turn < risk_straight
    assert min_dist_turn > min_dist_straight


def test_waypoint_trajectory_risk_enabled_yield_has_real_braking_distance():
    """Regression (waypoint_yield analogue of
    test_speed_steering_braking_from_speed_reaches_true_stopping_point): a
    YIELD command (target_v=0.0, the yield_creep_speed_mps default) does NOT
    teleport the robot to an instant stop -- it must still cover the real
    braking distance at brake_decel_mps2 before the swept path stops."""
    cfg = _directional_risk_cfg(horizons_sec=[1.0])
    stopping_distance = 2.0 ** 2 / (2 * 6.0)  # exact analytic braking distance
    node = _node(
        action_mode="waypoint_yield",
        _waypoint_trajectory_risk_enabled=True,
        vehicle_wheelbase_m=1.0,
        _directional_risk_cfg=cfg,
        latest_actual_signed_speed=2.0,
        latest_center_steering=0.0,
        human_states={"h0": {"x": stopping_distance, "y": 0.0, "yaw": 0.0, "v": 0.0}},
    )
    # target_v=0.0 mirrors hybrid_action_to_command's YIELD branch (speed
    # capped at controller_yield_creep_mps, default 0.0 -> full stop target).
    risk_dir, min_dist_dir = Environment._compute_directional_risk(
        node, 0.0, target_v=0.0, target_cmd_steering=0.0)
    assert min_dist_dir == pytest.approx(0.0, abs=1e-6)
    assert risk_dir == pytest.approx(1.0, abs=1e-6)


def test_waypoint_trajectory_risk_enabled_yield_scores_lower_than_move():
    """In a risky situation, YIELD (target_v=0.0, real braking distance ~
    0.333 m at this fixture's dynamics) must score LOWER risk than MOVE
    (target_v=2.0, continuing at cruise speed straight through the human)."""
    cfg = _directional_risk_cfg(horizons_sec=[1.0])
    node = _node(
        action_mode="waypoint_yield",
        _waypoint_trajectory_risk_enabled=True,
        vehicle_wheelbase_m=1.0,
        _directional_risk_cfg=cfg,
        latest_actual_signed_speed=2.0,
        latest_center_steering=0.0,
        # 1.9 m ahead: well past the ~0.333 m YIELD braking distance, but
        # reached by t=0.95s < horizon (1.0s) at a steady 2 m/s MOVE.
        human_states={"h0": {"x": 1.9, "y": 0.0, "yaw": 0.0, "v": 0.0}},
    )
    risk_yield, min_dist_yield = Environment._compute_directional_risk(
        node, 0.0, target_v=0.0, target_cmd_steering=0.0)
    risk_move, min_dist_move = Environment._compute_directional_risk(
        node, 0.0, target_v=2.0, target_cmd_steering=0.0)
    # MOVE passes right by the human (near-collision, some sampling slack
    # since the closest swept-path sample need not land exactly on the
    # human); YIELD brakes to a stop well short of it.
    assert risk_move > 0.9
    assert min_dist_move < 0.05
    assert risk_yield < risk_move
    assert min_dist_yield > min_dist_move


def test_waypoint_trajectory_risk_enabled_stopped_robot_catches_intermediate_time_close_pass():
    """waypoint_yield analogue of
    test_speed_steering_stopped_robot_catches_intermediate_time_close_pass: a
    robot already at rest AND commanding YIELD (target_v=0.0) must still
    catch a pedestrian who is only close to it at some time STRICTLY BETWEEN
    now and the horizon."""
    cfg = _directional_risk_cfg(horizons_sec=[1.0])
    vx, vy = 3.0, -5.0  # crosses (x=-1.8, y=3.0) -> (x=0, y=0) at t=0.6s
    v = math.hypot(vx, vy)
    yaw = math.atan2(vy, vx)
    node = _node(
        action_mode="waypoint_yield",
        _waypoint_trajectory_risk_enabled=True,
        vehicle_wheelbase_m=1.0,
        _directional_risk_cfg=cfg,
        _dr_rollout_path_samples=5,
        latest_actual_signed_speed=0.0,
        latest_center_steering=0.0,
        human_states={"h0": {"x": -1.8, "y": 3.0, "yaw": yaw, "v": v}},
    )
    risk_dir, min_dist_dir = Environment._compute_directional_risk(
        node, 0.0, target_v=0.0, target_cmd_steering=0.0)
    assert risk_dir == pytest.approx(1.0)
    assert min_dist_dir == pytest.approx(0.0)


def test_waypoint_trajectory_risk_enabled_robot_human_time_alignment():
    """Robot and human positions must be compared at MATCHING sample times,
    not robot@t=0 vs human@t=horizon: the human here reaches the robot's
    CURRENT (t=0) position (0, 0) only at t=1.0s, by which point the ALSO-
    MOVING robot (2 m/s straight ahead, same direction/speed) has itself
    moved away from (0, 0) -- so the two stay a constant 2.0 m apart at every
    matching sample time. A time-MISMATCHED check (robot@t=0=(0,0) vs.
    human@t=horizon=(0,0)) would wrongly report a dead-on collision
    (risk=1.0, min_dist=0.0); the correct matching-time answer must not."""
    cfg = _directional_risk_cfg(horizons_sec=[1.0])
    node = _node(
        action_mode="waypoint_yield",
        _waypoint_trajectory_risk_enabled=True,
        vehicle_wheelbase_m=1.0,
        _directional_risk_cfg=cfg,
        latest_actual_signed_speed=2.0,
        latest_center_steering=0.0,
        # Walks +x at the SAME speed as the robot, starting 2m behind the
        # robot's t=0 position -> constant 2.0 m separation at every matching
        # t (NOT 0, which a t=0-vs-t=horizon mismatch would wrongly produce).
        human_states={"h0": {"x": -2.0, "y": 0.0, "yaw": 0.0, "v": 2.0}},
    )
    risk_dir, min_dist_dir = Environment._compute_directional_risk(
        node, 0.0, target_v=2.0, target_cmd_steering=0.0)
    # Dc=3.0 (cfg default) -> risk = 1 - 2.0/3.0, min_dist = 2.0/3.0.
    assert risk_dir == pytest.approx(1.0 / 3.0, abs=1e-3)
    assert min_dist_dir == pytest.approx(2.0 / 3.0, abs=1e-3)
    assert risk_dir < 0.9  # far from the wrong mismatched-time answer (1.0)


# --------------------------------------------------------------------------- #
#  _append_aux_labels: action-risk sentinel prepend
# --------------------------------------------------------------------------- #
def test_append_aux_labels_no_prepend_when_disabled():
    node = _node(action_risk_head_env_enabled=False)
    state = [1.0, 2.0, 3.0]
    out = Environment._append_aux_labels(node, state, action_risk_target=(0.5, 0.5))
    assert out == state  # unchanged: byte-identical when the env flag is off


def test_append_aux_labels_no_prepend_when_target_none():
    node = _node(action_risk_head_env_enabled=True)
    state = [1.0, 2.0, 3.0]
    out = Environment._append_aux_labels(node, state, action_risk_target=None)
    assert out == state  # unchanged: reset() calls with target=None


def test_append_aux_labels_prepends_sentinel_block_when_enabled():
    node = _node(action_risk_head_env_enabled=True)
    state = [1.0, 2.0, 3.0]
    out = Environment._append_aux_labels(node, state, action_risk_target=(0.42, 0.87))
    assert out[:3] == state
    assert out[3] == aux_labels.ACTION_RISK_WIRE_SENTINEL
    assert abs(out[4] - 0.42) < 1e-6
    assert abs(out[5] - 0.87) < 1e-6
    assert len(out) == 6  # no aux tail (aux disabled in this fixture)


def test_action_risk_block_composes_with_aux_tail():
    """Both PHASE2's action-risk block AND the aux label tail can coexist: the
    action-risk block comes first, the aux tail (self-describing) last, and
    EnvInterface's strip order (action-risk THEN aux) recovers both cleanly."""
    aux_cfg = _directional_risk_cfg(horizons_sec=[0.5, 1.0, 1.5])
    node = _node(
        action_risk_head_env_enabled=True,
        _aux_pred_enabled=True,
        _aux_label_cfg=aux_cfg,
        latest_actual_signed_speed=0.0,
        get_logger=lambda: types.SimpleNamespace(warn=lambda *a, **k: None),
    )
    state = [1.0, 2.0, 3.0]
    out = Environment._append_aux_labels(node, state, action_risk_target=(0.1, 0.9))
    assert out[:3] == state
    assert out[3] == aux_labels.ACTION_RISK_WIRE_SENTINEL

    remainder = out[3:]
    target, after_action_risk = aux_labels.strip_action_risk_wire(remainder)
    assert target is not None
    assert abs(target[0] - 0.1) < 1e-6
    assert abs(target[1] - 0.9) < 1e-6
    meta, label = aux_labels.parse_aux_wire(after_action_risk)
    assert meta["num_sectors"] == aux_cfg.num_sectors
    assert len(label) == aux_cfg.label_dim
