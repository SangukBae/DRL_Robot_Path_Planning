"""Regression test for the PHASE1B fixed-eval-suite branch of
ResetPipelineMixin._reset_callback_impl (env/simulation/reset_pipeline.py).

reset_pipeline.py is ROS-free (no rclpy/squaternion-at-import dependency in
the fixed-eval-suite branch itself), so this drives the REAL unbound method
on a minimal fake node (types.SimpleNamespace + stubs), following the same
"call the real method with a fake self" pattern as test_directional_risk_env.py.

Regression coverage: when fixed_eval_suite_enabled=true and the env is in
curriculum_eval_mode, _reset_callback_impl derives a suite-local episode seed
via drl_agent.common.seed_utils (derive_resume_seed + seed_basic_rngs) BEFORE
falling through to the normal per-episode human-RNG reseed. This branch was
previously never exercised by the test suite, so a missing `seed_utils`
import in reset_pipeline.py (NameError at these two call sites) went
undetected — see memory / commit history for the fix.
"""

import threading
import types

import pytest

from drl_agent.env.simulation.reset_pipeline import ResetPipelineMixin


class _StopHere(Exception):
    """Sentinel raised right after the code under test, so the real method
    body runs far enough to prove the fixed-eval-suite seeding lines execute
    without error, without needing to mock the rest of this ~200-line method
    (Gazebo control, start/goal sampling, ...)."""


def _make_fake_node(*, suite_enabled, eval_mode, suite_seed=777, suite_idx=3):
    node = types.SimpleNamespace()
    node._human_lock = threading.Lock()
    node.human_states = {"stale": object()}
    node._episode_count = 5

    params = {
        "fixed_eval_suite_enabled": suite_enabled,
        "curriculum_eval_mode": eval_mode,
        "fixed_eval_suite_reset_token": 0,
        "fixed_eval_suite_base_seed": suite_seed,
    }

    def get_parameter(name):
        return types.SimpleNamespace(value=params[name])

    node.get_parameter = get_parameter
    # Force the "toggle changed" branch so _fixed_suite_episode_index resets
    # to 0 before being read/incremented below (mirrors a fresh eval run).
    node._fixed_suite_eval_mode_prev = not eval_mode
    node._fixed_suite_last_reset_token = -1
    node._fixed_suite_episode_index = suite_idx
    node.pool_build_seed = 1234

    def _stop(*_a, **_kw):
        raise _StopHere()

    node._seed_human_rngs = _stop
    return node


def test_fixed_eval_suite_branch_derives_seed_without_nameerror():
    """The bug this guards against: seed_utils.derive_resume_seed /
    seed_utils.seed_basic_rngs raised NameError because reset_pipeline.py
    never imported drl_agent.common.seed_utils, even though the module was
    used at these two call sites."""
    node = _make_fake_node(suite_enabled=True, eval_mode=True)

    with pytest.raises(_StopHere):
        ResetPipelineMixin._reset_callback_impl(node, None, None)

    # The suite-local index must have been consumed (read then incremented)
    # by the branch under test, not the ever-growing _episode_count.
    assert node._fixed_suite_episode_index == 1
    assert node._fixed_suite_last_episode_index == 0


def test_fixed_eval_suite_disabled_skips_seed_utils_branch():
    """Sanity check: with the suite OFF (the default), the normal
    _episode_count-based path runs instead — same _StopHere sentinel, but
    _fixed_suite_last_episode_index must stay None (never entered the
    suite-seeding branch at all)."""
    node = _make_fake_node(suite_enabled=False, eval_mode=False)
    node._fixed_suite_last_episode_index = "untouched"

    with pytest.raises(_StopHere):
        ResetPipelineMixin._reset_callback_impl(node, None, None)

    assert node._fixed_suite_last_episode_index is None
