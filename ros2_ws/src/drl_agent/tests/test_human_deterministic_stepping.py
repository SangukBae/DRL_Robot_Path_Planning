"""ROS-free unit tests for deterministic (sim-step-synchronized) human motion.

Stage-1 finding: the human-motion timer (``_human_timer_callback``) fires on a
wall-clock ``rclpy`` timer (confirmed live: ``/gym_node`` reports
``use_sim_time=False``), decoupled from Gazebo's simulated clock and from RL
step boundaries. Two compounding reproducibility issues:

1. The NUMBER of ticks that fire during one RL step's ``propagate_state``
   window is wall-clock-speed dependent (CPU load, GIL contention with the
   sleep-based physics propagation, etc.) -- not fixed at 2 per 0.1s step
   even though ``human_update_rate=20Hz`` nominally implies exactly that.
   Each tick draws from ``self._human_np_rng`` (stop/retarget sampling), so a
   different tick count means a different number of RNG draws consumed
   in-episode -- the exact human trajectory for the SAME seed becomes
   dependent on wall-clock scheduling speed.
2. Even when a tick DOES fire, it uses a FIXED ``dt = 1/human_update_rate``
   per tick regardless of how much real time actually elapsed since the last
   fire, so timer jitter/drops are never corrected -- motion silently drifts
   out of sync with elapsed wall time under scheduling pressure.

This is DISTINCT from the already-mitigated issue in ``test_human_rng.py``:
that suite locks in that CROSS-episode spawn config is reseeded independent of
the previous episode's (variable) motion-tick count. It does NOT test
WITHIN-episode trajectory reproducibility, which is what's wall-clock-broken
here.

The fix: ``compute_human_tick_plan`` splits a given physics-propagation
duration into an EXACT, integer number of fixed-dt ticks (independent of any
wall-clock timing), and ``_advance_humans_one_tick`` is the single shared
per-tick body used by BOTH the legacy wall-clock timer callback AND the new
deterministic per-RL-step driver -- so enabling determinism changes only WHO
calls the tick logic and how many times, never the tick logic itself.
"""

import sys
import types

import pytest


def _install_ros_stubs():
    class _Meta(type):
        _n = [0]

        def __getattr__(cls, name):
            _Meta._n[0] += 1
            return _Meta._n[0]

    def _dummy(name):
        return _Meta(name, (), {})

    for name in (
        "squaternion", "rclpy", "rclpy.parameter",
        "geometry_msgs", "geometry_msgs.msg",
        "nav_msgs", "nav_msgs.msg",
        "sensor_msgs", "sensor_msgs.msg",
        "visualization_msgs", "visualization_msgs.msg",
        "drl_agent_interfaces", "drl_agent_interfaces.msg",
        "ros_gz_interfaces", "ros_gz_interfaces.msg", "ros_gz_interfaces.srv",
    ):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__getattr__ = _dummy
            sys.modules[name] = mod


_install_ros_stubs()

import drl_agent.env.humans.human_motion_manager as hmm  # noqa: E402


# ---------------------------------------------------------------------------
# compute_human_tick_plan: pure function, no node/lock/RNG involved.
# ---------------------------------------------------------------------------

def test_matches_current_phase2_both_config_2_ticks_per_step():
    # time_delta=0.1, human_update_rate=20.0 -> exactly 2 ticks of 0.05s each.
    n_ticks, sub_dt = hmm.compute_human_tick_plan(0.1, 20.0)
    assert n_ticks == 2
    assert sub_dt == pytest.approx(0.05)


def test_reset_double_length_propagate_scales_ticks_proportionally():
    # reset_callback calls propagate_state(2 * time_delta) -- must yield double
    # the ticks at the SAME sub_dt, not double the sub_dt.
    n_ticks, sub_dt = hmm.compute_human_tick_plan(0.2, 20.0)
    assert n_ticks == 4
    assert sub_dt == pytest.approx(0.05)


def test_rejects_non_integer_multiple_fail_fast():
    # time_delta=0.1 / human_update_rate=15 -> 1.5 ticks, not an integer.
    with pytest.raises(ValueError, match="not an integer multiple"):
        hmm.compute_human_tick_plan(0.1, 15.0)


def test_rejects_duration_shorter_than_one_tick():
    with pytest.raises(ValueError, match="shorter than one"):
        hmm.compute_human_tick_plan(0.01, 20.0)


def test_rejects_non_positive_rate():
    with pytest.raises(ValueError, match="human_update_rate"):
        hmm.compute_human_tick_plan(0.1, 0.0)


def test_tolerates_float_rounding_noise():
    # 0.1 is not exactly representable in binary float; the raw ratio won't be
    # an exact integer even for a config that's conceptually exact.
    n_ticks, sub_dt = hmm.compute_human_tick_plan(0.1, 20.0)
    assert n_ticks == 2
    n_ticks3, sub_dt3 = hmm.compute_human_tick_plan(0.3, 10.0)
    assert n_ticks3 == 3
    assert sub_dt3 == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# _advance_humans_one_tick: shared per-tick body (fast-path + lock + RNG use).
# ---------------------------------------------------------------------------

class _RecordingRng:
    """Counts .rand() draws without needing real numpy randomness."""

    def __init__(self):
        self.draws = 0

    def rand(self):
        self.draws += 1
        return 1.0  # never fires the stop-probability branch


class _FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeParam:
    def __init__(self, value):
        self.value = value


class _Node(hmm.HumanMotionMixin):
    def __init__(self, humans=None, enabled=True):
        self.human_states = humans if humans is not None else {}
        self._human_updates_enabled = enabled
        self._human_lock = _FakeLock()
        self._human_np_rng = _RecordingRng()
        self.human_retarget_prob_per_sec = 0.0
        self.human_stop_prob_per_sec = 0.0
        self.human_social_avoid_enabled = False
        self.human_goal_driven_enabled = False
        self.human_goal_reach_threshold = 0.3
        self.human_max_accel = 1.0
        self.human_max_yaw_accel = 1.0
        self.goal_obstacle_lower = -10.0
        self.goal_obstacle_upper = 10.0
        self.obstacle_wall_margin = 1.0
        self.spawned_obstacle_records = {}
        self.published_batches = []
        # Normal-motion-branch params (kept inert: retarget/timeout disabled,
        # target far enough away that dist_to_target never triggers a retarget
        # within the handful of ticks these tests run).
        self.human_max_segment_sec = 0.0
        self.human_min_retarget_interval = 1.0
        self.human_heading_jitter_on_retarget_only = True
        self.human_heading_jitter = 0.0
        self.human_desired_yaw_rate_limit = 3.0
        self.human_k_yaw = 1.0
        self.human_max_yaw_rate = 1.0
        self.human_social_avoid_radius = 0.0
        self.human_social_avoid_strength = 0.0
        self.human_social_avoid_max_heading_offset = 0.0
        self.human_robot_avoid_radius = 0.0
        self.human_robot_avoid_strength = 0.0
        self.human_robot_avoid_max_heading_offset = 0.0
        self.gt_x = 0.0
        self.gt_y = 0.0

    def get_parameter(self, name):
        return _FakeParam(False)

    def _publish_model_poses(self, batch):
        self.published_batches.append(list(batch))

    def _collect_human_part_poses(self, state, batch):
        batch.append((state.get("visual_torso", "h"), 0, 0, 0, 0, 0, 0, 1))

    def _resolve_human_wall_collision(self, x, y, nx, ny, r):
        return nx, ny, False

    def _resolve_human_static_obstacle_collision(self, x, y, nx, ny, r):
        return nx, ny, False


def _make_human_state(**overrides):
    state = dict(
        x=0.0, y=0.0, yaw=0.0, v=0.5, w=0.0,
        target_x=5.0, target_y=0.0, radius=0.3,
        gait_phase=0.0, gait_freq_hz=1.5, leg_swing_amp_rad=0.3,
        leg_length=0.9, hip_z=0.5, leg_y_offset=0.1, torso_z=0.9,
        speed=0.5, mode_stop_prob=0.0, visual_torso="h1_v_torso",
        proxy_torso="h1_p_torso",
    )
    state.update(overrides)
    return state


def test_no_humans_is_a_true_no_op_zero_rng_draws_zero_publishes():
    node = _Node(humans={}, enabled=True)
    node._advance_humans_one_tick(0.05)
    assert node._human_np_rng.draws == 0
    assert node.published_batches == []


def test_disabled_is_a_true_no_op_even_with_humans_present():
    node = _Node(humans={"h1": _make_human_state()}, enabled=False)
    node._advance_humans_one_tick(0.05)
    assert node._human_np_rng.draws == 0
    assert node.published_batches == []


_DRAWS_PER_HUMAN_PER_TICK = 2  # stop-probability check + retarget-probability check


def test_active_humans_draw_rng_and_publish_once_per_tick():
    node = _Node(humans={"h1": _make_human_state()}, enabled=True)
    node._advance_humans_one_tick(0.05)
    assert node._human_np_rng.draws == _DRAWS_PER_HUMAN_PER_TICK
    assert len(node.published_batches) == 1


def test_wall_clock_timer_callback_delegates_to_shared_tick_body():
    # The legacy timer path must funnel through the SAME shared tick body --
    # not a separate, divergent copy of the update logic.
    node = _Node(humans={"h1": _make_human_state()}, enabled=True)
    node.human_update_rate = 20.0
    node._human_timer_callback()
    assert node._human_np_rng.draws == _DRAWS_PER_HUMAN_PER_TICK
    assert len(node.published_batches) == 1


def test_tick_count_is_independent_of_how_many_times_a_wall_clock_delay_is_simulated():
    # The core reproducibility guarantee: N calls to _advance_humans_one_tick
    # ALWAYS consumes exactly N * (draws per tick) RNG draws, regardless of any
    # wall-clock timing around those calls -- there is no wall-clock dependent
    # branch left in the shared tick body itself.
    for n_calls in (1, 2, 5):
        node = _Node(humans={"h1": _make_human_state()}, enabled=True)
        for _ in range(n_calls):
            node._advance_humans_one_tick(0.05)
        assert node._human_np_rng.draws == n_calls * _DRAWS_PER_HUMAN_PER_TICK
