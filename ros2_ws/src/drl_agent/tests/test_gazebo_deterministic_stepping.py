"""ROS-free unit tests for Stage-2 deterministic Gazebo physics stepping.

Empirically verified (live Docker + Gazebo) via ros_gz's WorldControl service:
sending ``multi_step=N`` together with ``pause=True`` in ONE ControlWorld
request steps physics by EXACTLY N iterations of the world's configured
``max_step_size`` (confirmed: N=1/50/100 -> sim-time deltas of exactly
0.001/0.050/0.100s) and leaves the world paused afterward (confirmed via a
follow-up /clock stay-paused check) -- this replaces the legacy sleep-based
``unpause -> time.sleep(duration) -> pause`` (2 world-control calls) with ONE
call, halving that path's _await_future-polling overhead. The service call
itself returns near-instantly (<1ms), but physics + sensor computation is still
wall-clock bound. The shipped environment path therefore sleeps for the
requested duration and then waits for scan/odom freshness. A stricter embedded
/clock completion wait was tried and deliberately excluded because it hung real
/step calls, even though isolated probes confirmed multi_step's exact sim-time
advance.

``compute_physics_step_count`` is the pure (duration, physics_step_size) ->
n_steps helper, mirroring human_motion_manager.compute_human_tick_plan's
fail-fast-on-non-integer-multiple contract.
"""

import pytest

import drl_agent.env.simulation.gazebo_service_wait as gsw


def test_matches_confirmed_world_physics_config_100_steps_per_rl_step():
    # time_delta=0.1s, max_step_size=0.001s (drl_arena.world, confirmed) -> 100.
    assert gsw.compute_physics_step_count(0.1, 0.001) == 100


def test_reset_double_length_propagate_scales_step_count_proportionally():
    assert gsw.compute_physics_step_count(0.2, 0.001) == 200


def test_matches_live_probe_values_1_and_50_steps():
    # Directly mirrors the live multi_step=1 / multi_step=50 probe results
    # (sim_delta measured as exactly 0.001s / 0.050s).
    assert gsw.compute_physics_step_count(0.001, 0.001) == 1
    assert gsw.compute_physics_step_count(0.05, 0.001) == 50


def test_rejects_non_integer_multiple_fail_fast():
    with pytest.raises(ValueError, match="not an integer multiple"):
        gsw.compute_physics_step_count(0.1, 0.003)  # 33.33 steps


def test_rejects_duration_shorter_than_one_step():
    with pytest.raises(ValueError, match="shorter than one"):
        gsw.compute_physics_step_count(0.0001, 0.001)


def test_rejects_non_positive_step_size():
    with pytest.raises(ValueError, match="physics_step_size"):
        gsw.compute_physics_step_count(0.1, 0.0)


def test_tolerates_float_rounding_noise():
    # 0.1 isn't exactly representable in binary float.
    assert gsw.compute_physics_step_count(0.1, 0.001) == 100
    assert gsw.compute_physics_step_count(0.3, 0.001) == 300
