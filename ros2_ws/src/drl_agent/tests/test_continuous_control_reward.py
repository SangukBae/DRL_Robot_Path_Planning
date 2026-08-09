"""ROS-free unit tests for PHASE3's speed_steering-only continuous-control
reward shaping (reward_calculator.compute_reward's continuous_control_
reward_* params) plus its config/CSV wiring.

Covers:
  * disabled (default) -> reward byte-identical to before this feature.
  * action_mode gate: even with continuous_control_reward_enabled=True, any
    mode other than "speed_steering" (waypoint / waypoint_yield) is a no-op
    -- this is enforced INSIDE compute_reward, not just by the caller.
  * heading-error-delta reward: signed decrease/increase/no-change, incl. the
    +-pi wrap boundary; 0 on the episode's first step (prev_theta_err=None).
  * continuous-control penalty: steering-magnitude symmetry, steering/speed
    change penalties (0 when unchanged / on first step, positive on change),
    and that constant nonzero steering keeps costing the magnitude term
    every step (it does not decay/disappear).
  * the canonical environment_curriculum.yaml and phase2/both carry the
    block OFF (or absent).
  * the env-side debug CSV schema: _rotate_debug_csv's header contains the
    4 new columns in the documented position, and the (separately, inline)
    per-step row-write references the same 4 reward_terms keys in the same
    relative order with no header/row length drift.
"""

import math
import os
import re
import sys
import types

import pytest
import yaml

import drl_agent.env.rewards.reward_calculator as rc
from drl_agent.common import compat
from drl_agent.config import ProfileLoader


# --------------------------------------------------------------------------- #
#  compute_reward: disabled / non-speed_steering -> byte-identical baseline
# --------------------------------------------------------------------------- #
def _base_kwargs(**over):
    kw = dict(
        target=False, collision=False,
        v=0.5, w=0.0,
        prev_goal_dist=5.0, curr_goal_dist=4.8,   # delta_d = 0.2 (positive progress)
        theta_err=0.1,
        return_terms=True,
    )
    kw.update(over)
    return kw


def _ccr_kwargs(**over):
    kw = dict(
        continuous_control_reward_enabled=True,
        action_mode="speed_steering",
        prev_theta_err=0.3,
        cmd_steering=0.1,
        prev_cmd_steering=0.05,
        prev_cmd_speed=0.4,
        steering_limit_rad=0.3767,  # ~21.58 deg
        heading_delta_weight=0.2,
        steering_magnitude_weight=0.005,
        steering_change_weight=0.01,
        speed_change_weight=0.005,
    )
    kw.update(over)
    return kw


def test_disabled_matches_baseline():
    reward_off, terms_off = rc.compute_reward(**_base_kwargs())
    reward_on_but_disabled, terms_on_but_disabled = rc.compute_reward(
        **_base_kwargs(**_ccr_kwargs(continuous_control_reward_enabled=False)))
    assert reward_off == reward_on_but_disabled
    for k in ("reward_heading_delta", "penalty_steering_magnitude",
              "penalty_steering_change", "penalty_speed_change"):
        assert terms_off[k] == 0.0
        assert terms_on_but_disabled[k] == 0.0


@pytest.mark.parametrize("mode", ["waypoint", "waypoint_yield", None])
def test_non_speed_steering_action_mode_is_noop(mode):
    """Enforced INSIDE compute_reward: even if a caller mistakenly leaves
    continuous_control_reward_enabled=True while action_mode != 'speed_steering'
    (waypoint/waypoint_yield, i.e. phase2/both), every new term stays exactly 0
    and the reward is byte-identical to the fully-disabled baseline."""
    reward_off, _ = rc.compute_reward(**_base_kwargs())
    reward_mode, terms_mode = rc.compute_reward(
        **_base_kwargs(**_ccr_kwargs(action_mode=mode)))
    assert reward_mode == reward_off
    for k in ("reward_heading_delta", "penalty_steering_magnitude",
              "penalty_steering_change", "penalty_speed_change"):
        assert terms_mode[k] == 0.0


def test_terminal_rewards_short_circuit_continuous_control_shaping():
    reward, terms = rc.compute_reward(**_base_kwargs(
        target=True, **_ccr_kwargs()))
    assert reward == 20.0
    for k in ("reward_heading_delta", "penalty_steering_magnitude",
              "penalty_steering_change", "penalty_speed_change"):
        assert terms[k] == 0.0


# --------------------------------------------------------------------------- #
#  (a) heading-error-delta reward: POTENTIAL-BASED shaping
#      potential(theta) = |wrap(theta)| / pi in [0, 1]
#      reward_heading_delta = weight * (prev_potential - curr_potential)
# --------------------------------------------------------------------------- #
def _potential(theta):
    return abs(math.atan2(math.sin(theta), math.cos(theta))) / math.pi


def test_heading_error_decrease_is_positive():
    # |prev|=0.3 -> |curr|=0.1: error shrank -> positive reward.
    _, terms = rc.compute_reward(**_base_kwargs(
        theta_err=0.1, **_ccr_kwargs(prev_theta_err=0.3)))
    assert terms["reward_heading_delta"] > 0.0
    assert terms["reward_heading_delta"] == pytest.approx(
        0.2 * (_potential(0.3) - _potential(0.1)))


def test_heading_error_increase_is_negative():
    # |prev|=0.1 -> |curr|=0.3: error grew -> negative reward (signed, not clamped to >=0).
    _, terms = rc.compute_reward(**_base_kwargs(
        theta_err=0.3, **_ccr_kwargs(prev_theta_err=0.1)))
    assert terms["reward_heading_delta"] < 0.0
    assert terms["reward_heading_delta"] == pytest.approx(
        0.2 * (_potential(0.1) - _potential(0.3)))


def test_heading_error_unchanged_is_zero():
    _, terms = rc.compute_reward(**_base_kwargs(
        theta_err=0.2, **_ccr_kwargs(prev_theta_err=0.2)))
    assert terms["reward_heading_delta"] == 0.0


def test_heading_error_sign_flip_same_magnitude_is_zero():
    # |theta_err| is the same on both sides of 0 -> |curr|==|prev| -> delta 0,
    # even though the SIGNED heading error flipped sign (oscillation guard).
    _, terms = rc.compute_reward(**_base_kwargs(
        theta_err=-0.2, **_ccr_kwargs(prev_theta_err=0.2)))
    assert terms["reward_heading_delta"] == 0.0


def test_heading_delta_naturally_bounded_by_weight_no_clip_param():
    """Potential-based shaping needs no explicit clip: even a huge (non-
    physical) heading swing stays within [-weight, +weight] because
    potential(theta) in [0, 1] by construction."""
    _, terms = rc.compute_reward(**_base_kwargs(
        theta_err=0.0, **_ccr_kwargs(prev_theta_err=3.0, heading_delta_weight=0.2)))
    assert abs(terms["reward_heading_delta"]) <= 0.2 + 1e-12
    assert terms["reward_heading_delta"] == pytest.approx(0.2 * _potential(3.0))


def test_heading_delta_wrap_boundary_no_spurious_large_delta():
    """theta_err near +pi and prev near -pi are numerically ~2*pi apart but
    represent almost the SAME heading error once wrapped -- must not produce
    a huge spurious delta (would happen with naive abs(curr-prev)/pi)."""
    _, terms = rc.compute_reward(**_base_kwargs(
        theta_err=math.pi - 0.01,
        **_ccr_kwargs(prev_theta_err=-math.pi + 0.01)))
    # |wrap(pi-0.01)| == |wrap(-pi+0.01)| == pi-0.01 -> potentials equal -> delta ~ 0.
    assert abs(terms["reward_heading_delta"]) < 1e-9


def test_heading_delta_round_trip_nets_exactly_zero():
    """Regression for the reviewed oscillation-exploit bug: a clipped RAW
    delta (clip(|prev|-|curr|, -c, +c)) can pay out net-positive reward over
    a path that returns to the SAME heading error, whenever one worsening
    step's raw delta exceeds the clip while the recovery steps are paid in
    full (e.g. 0 -> 1.0 -> 0.75 -> 0.50 -> 0.25 -> 0 with clip=0.25 nets
    +0.15). This test sums the raw per-step terms with NO discounting
    (plain arithmetic sum, matching what compute_reward's docstring calls
    the "undiscounted step-reward sum") -- that undiscounted sum telescopes
    to weight * (potential_start - potential_end) over ANY path, so a true
    round trip (same start/end heading error) MUST net to (near) exactly 0,
    regardless of the intermediate path or how large any single step's swing
    is. This does NOT claim the same holds for TQC's actual gamma<1
    discounted return -- under discounting the same round trip is position-
    dependent and is not guaranteed to net to 0 (see the caller-facing
    caveat in compute_reward's continuous_control_reward_enabled docstring)."""
    path = [0.0, 1.0, 0.75, 0.50, 0.25, 0.0]
    total = 0.0
    prev = None
    for theta in path:
        _, terms = rc.compute_reward(**_base_kwargs(
            theta_err=theta, **_ccr_kwargs(prev_theta_err=prev, heading_delta_weight=0.2)))
        total += terms["reward_heading_delta"]
        prev = theta
    assert total == pytest.approx(0.0, abs=1e-9)


def test_heading_delta_zero_on_first_step_prev_none():
    _, terms = rc.compute_reward(**_base_kwargs(
        theta_err=0.5, **_ccr_kwargs(prev_theta_err=None)))
    assert terms["reward_heading_delta"] == 0.0


# --------------------------------------------------------------------------- #
#  (b) continuous-control penalty: steering magnitude / change, speed change
# --------------------------------------------------------------------------- #
def test_steering_magnitude_penalty_symmetric_in_sign():
    _, terms_pos = rc.compute_reward(**_base_kwargs(
        **_ccr_kwargs(cmd_steering=0.2, prev_cmd_steering=0.2)))
    _, terms_neg = rc.compute_reward(**_base_kwargs(
        **_ccr_kwargs(cmd_steering=-0.2, prev_cmd_steering=-0.2)))
    assert terms_pos["penalty_steering_magnitude"] == pytest.approx(
        terms_neg["penalty_steering_magnitude"])
    assert terms_pos["penalty_steering_magnitude"] == pytest.approx(
        0.005 * 0.2 / 0.3767)


def test_steering_magnitude_penalty_persists_for_constant_nonzero_steering():
    """A policy holding a constant nonzero steering (the observed circling
    local optimum) must keep paying the magnitude penalty every step -- it
    must not decay to 0 just because steering isn't CHANGING."""
    _, terms = rc.compute_reward(**_base_kwargs(
        **_ccr_kwargs(cmd_steering=0.15, prev_cmd_steering=0.15)))
    assert terms["penalty_steering_magnitude"] > 0.0
    assert terms["penalty_steering_change"] == 0.0  # unchanged -> no change penalty


def test_no_change_penalty_when_command_unchanged():
    _, terms = rc.compute_reward(**_base_kwargs(
        **_ccr_kwargs(cmd_steering=0.1, prev_cmd_steering=0.1,
                      prev_cmd_speed=0.5, v=0.5)))
    assert terms["penalty_steering_change"] == 0.0
    assert terms["penalty_speed_change"] == 0.0


def test_change_penalty_positive_when_command_changes():
    _, terms = rc.compute_reward(**_base_kwargs(
        v=0.8, **_ccr_kwargs(cmd_steering=0.2, prev_cmd_steering=-0.1,
                              prev_cmd_speed=0.3)))
    assert terms["penalty_steering_change"] > 0.0
    assert terms["penalty_steering_change"] == pytest.approx(0.01 * 0.3 / 0.3767)
    assert terms["penalty_speed_change"] > 0.0
    assert terms["penalty_speed_change"] == pytest.approx(0.005 * 0.5 / 1.5)  # v_max default


def test_change_penalties_zero_on_first_step_prev_none():
    _, terms = rc.compute_reward(**_base_kwargs(
        v=0.8, **_ccr_kwargs(cmd_steering=0.3, prev_cmd_steering=None,
                              prev_cmd_speed=None)))
    assert terms["penalty_steering_change"] == 0.0
    assert terms["penalty_speed_change"] == 0.0
    # Magnitude is NOT gated on prev state -- only the change terms are.
    assert terms["penalty_steering_magnitude"] > 0.0


def test_cmd_steering_none_leaves_all_control_penalties_zero():
    _, terms = rc.compute_reward(**_base_kwargs(
        **_ccr_kwargs(cmd_steering=None, prev_cmd_steering=0.1)))
    assert terms["penalty_steering_magnitude"] == 0.0
    assert terms["penalty_steering_change"] == 0.0


# --------------------------------------------------------------------------- #
#  Fail-fast guards: defense-in-depth for direct compute_reward() callers
#  that bypass ConfigValidator._check_continuous_control_reward (which
#  already blocks these on the real profile-launch path -- see
#  test_continuous_control_reward.py's validator section below). A silent
#  near-zero-divisor fallback here would blow the corresponding penalty up
#  instead of erroring, so compute_reward() raises ValueError itself too.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_limit", [None, 0.0, -0.1, float("nan"), "bad", object()])
def test_missing_or_invalid_steering_limit_raises_when_cmd_steering_given(bad_limit):
    """Covers non-numeric values too (e.g. a stray string/object slipping in
    from a misconfigured direct caller) -- must raise the SAME clean
    ValueError, not a bare TypeError from an un-guarded numeric comparison."""
    with pytest.raises(ValueError, match="steering_limit_rad"):
        rc.compute_reward(**_base_kwargs(
            **_ccr_kwargs(cmd_steering=0.1, steering_limit_rad=bad_limit)))


def test_missing_steering_limit_does_not_raise_when_cmd_steering_is_none():
    # The guard only fires for the term actually being computed -- with no
    # cmd_steering, steering_limit_rad is simply unused.
    reward, terms = rc.compute_reward(**_base_kwargs(
        **_ccr_kwargs(cmd_steering=None, steering_limit_rad=None)))
    assert terms["penalty_steering_magnitude"] == 0.0


@pytest.mark.parametrize("bad_cruise,bad_vmax", [
    (0.0, 0.0), (-1.0, -1.0), (None, 0.0), (float("nan"), float("nan")),
    ("bad", 0.0),
])
def test_missing_or_invalid_cruise_normalizer_raises_when_prev_cmd_speed_given(bad_cruise, bad_vmax):
    """Covers non-numeric cruise_speed_mps/v_max (e.g. cruise_speed_mps="bad")
    -- the regression this was written for: a raw `x and x > 0.0` comparison
    against a string raises TypeError before ever reaching the intended
    ValueError. Must raise the SAME clean ValueError in every case."""
    with pytest.raises(ValueError, match="cruise_speed_mps|v_max"):
        rc.compute_reward(**_base_kwargs(
            v_max=bad_vmax, **_ccr_kwargs(prev_cmd_speed=0.3, cmd_steering=None,
                                           cruise_speed_mps=bad_cruise)))


def test_cruise_normalizer_falls_back_to_valid_v_max_without_raising():
    # cruise_speed_mps unset/0 but v_max is a valid positive fallback -> no error.
    _, terms = rc.compute_reward(**_base_kwargs(
        v=0.8, v_max=1.5, **_ccr_kwargs(prev_cmd_speed=0.3, cmd_steering=None,
                                         cruise_speed_mps=None)))
    assert terms["penalty_speed_change"] > 0.0


def test_missing_cruise_normalizer_does_not_raise_when_prev_cmd_speed_is_none():
    reward, terms = rc.compute_reward(**_base_kwargs(
        v_max=0.0, **_ccr_kwargs(prev_cmd_speed=None, cmd_steering=None, cruise_speed_mps=0.0)))
    assert terms["penalty_speed_change"] == 0.0


# --------------------------------------------------------------------------- #
#  Full aggregate reward includes the new terms with correct signs
# --------------------------------------------------------------------------- #
def test_aggregate_reward_adds_heading_delta_and_subtracts_penalties():
    reward, terms = rc.compute_reward(**_base_kwargs(**_ccr_kwargs()))
    reward_baseline, _ = rc.compute_reward(**_base_kwargs())
    expected_shift = (
        terms["reward_heading_delta"]
        - terms["penalty_steering_magnitude"]
        - terms["penalty_steering_change"]
        - terms["penalty_speed_change"]
    )
    assert reward == pytest.approx(reward_baseline + expected_shift)


# --------------------------------------------------------------------------- #
#  Profile / canonical config wiring
# --------------------------------------------------------------------------- #
def _load_env_block(path):
    with open(path, "r") as f:
        return (yaml.safe_load(f) or {}).get("environment", {})


def _canonical_env_yaml_path():
    root = compat.package_source_root()
    if not root:
        return None
    path = os.path.join(root, "config", "environment_curriculum.yaml")
    return path if os.path.isfile(path) else None


def _experiments_present():
    return bool(ProfileLoader().profiles_root())


pytestmark_profiles = pytest.mark.skipif(
    not _experiments_present(), reason="drl_experiments profiles root not found")


def test_canonical_config_has_continuous_control_reward_off():
    path = _canonical_env_yaml_path()
    if path is None:
        pytest.skip("drl_agent source root / canonical config not found")
    env = _load_env_block(path)
    ccr = env.get("continuous_control_reward", {})
    assert ccr.get("enabled") is False


@pytestmark_profiles
def test_phase2_both_config_unaffected():
    """phase2/both must not carry (or need) the new block, and its action_mode
    stays the pre-existing waypoint_yield contract."""
    spec = ProfileLoader().load("phase2/both")
    env = _load_env_block(spec.config_paths["environment"])
    assert "continuous_control_reward" not in env
    assert env.get("action_mode", "waypoint_yield") != "speed_steering"


# --------------------------------------------------------------------------- #
#  ConfigValidator: fail-fast checks for continuous_control_reward
# --------------------------------------------------------------------------- #
def _validator_check(docs, overrides=None):
    """Drive ConfigValidator._check_continuous_control_reward in isolation
    (mirrors the class's own _check_risk_map_reward/_check_action_risk_head_
    consistency unit-test style -- no ProfileSpec/files needed)."""
    from drl_agent.config.validation import ConfigValidator

    validator = ConfigValidator.__new__(ConfigValidator)
    validator.spec = types.SimpleNamespace(overrides=dict(overrides or {}))
    from drl_agent.config.schema import ValidationReport
    rep = ValidationReport()
    validator._check_continuous_control_reward(rep, docs)
    return rep


def _env_docs(**ccr_over):
    ccr = dict(enabled=True, heading_delta_weight=0.2,
               steering_magnitude_weight=0.005, steering_change_weight=0.01,
               speed_change_weight=0.005)
    ccr.update(ccr_over)
    return {"environment": {"environment": {
        "action_mode": "speed_steering", "action_dim": 2,
        "vehicle_steering_limit_deg": 21.58,
        "controller_cruise_speed_mps": 2.0,
        "continuous_control_reward": ccr,
    }}}


def test_validator_disabled_block_skips_all_checks():
    docs = _env_docs(enabled=False)
    docs["environment"]["environment"]["vehicle_steering_limit_deg"] = 0.0
    docs["environment"]["environment"]["controller_cruise_speed_mps"] = -1.0
    rep = _validator_check(docs)
    assert rep.errors == []
    assert rep.warnings == []
    assert rep.info["continuous_control_reward.enabled"] is False


def test_validator_passes_for_valid_phase3_style_config():
    rep = _validator_check(_env_docs())
    assert rep.errors == []
    assert rep.warnings == []


@pytest.mark.parametrize("bad_key,bad_value", [
    ("heading_delta_weight", -0.1),
    ("steering_magnitude_weight", float("nan")),
    ("steering_change_weight", float("-inf")),
    ("speed_change_weight", "not-a-number"),
])
def test_validator_rejects_invalid_weight(bad_key, bad_value):
    rep = _validator_check(_env_docs(**{bad_key: bad_value}))
    assert any(bad_key in e for e in rep.errors), rep.errors


@pytest.mark.parametrize("bad_value", [0.0, -5.0])
def test_validator_rejects_non_positive_steering_limit(bad_value):
    docs = _env_docs()
    docs["environment"]["environment"]["vehicle_steering_limit_deg"] = bad_value
    rep = _validator_check(docs)
    assert any("vehicle_steering_limit_deg" in e for e in rep.errors), rep.errors


@pytest.mark.parametrize("bad_value", [0.0, -1.0])
def test_validator_rejects_non_positive_cruise_speed(bad_value):
    docs = _env_docs()
    docs["environment"]["environment"]["controller_cruise_speed_mps"] = bad_value
    rep = _validator_check(docs)
    assert any("controller_cruise_speed_mps" in e for e in rep.errors), rep.errors


def test_validator_warns_when_enabled_but_action_mode_not_speed_steering():
    docs = _env_docs()
    docs["environment"]["environment"]["action_mode"] = "waypoint_yield"
    docs["environment"]["environment"]["action_dim"] = 3
    rep = _validator_check(docs)
    assert rep.errors == []
    assert any("action_mode" in w for w in rep.warnings), rep.warnings


# --------------------------------------------------------------------------- #
#  Env-side debug CSV schema (_rotate_debug_csv header + the inline per-step
#  row-write, source-verified since the row-write is not a standalone method)
# --------------------------------------------------------------------------- #
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


class _StubLogger:
    def info(self, msg):
        pass


def test_debug_csv_header_contains_new_columns_in_order(tmp_path):
    node = types.SimpleNamespace(
        _env_log_dir=str(tmp_path),
        get_logger=lambda: _StubLogger(),
    )
    Environment._rotate_debug_csv(node)
    with open(node._env_step_csv, "r", newline="") as f:
        header = next(iter(__import__("csv").reader(f)))
    assert header.count("reward_heading_delta") == 1
    assert header.count("penalty_steering_magnitude") == 1
    assert header.count("penalty_steering_change") == 1
    assert header.count("penalty_speed_change") == 1
    i_risk_bonus = header.index("reward_risk_map_bonus")
    i_terminal = header.index("reward_terminal")
    assert header[i_risk_bonus + 1:i_terminal] == [
        "reward_heading_delta", "penalty_steering_magnitude",
        "penalty_steering_change", "penalty_speed_change",
    ]


def test_debug_csv_row_write_references_new_keys_in_matching_order_and_count():
    """The per-step row-write is inline in _step_callback_impl (not a separate
    method), so this is a source-level structural check: the writerow(...)
    list must reference the 4 new reward_terms keys in the SAME relative
    order as the header, and the total column count must still match."""
    src = open(env_mod.__file__, "r").read()

    header_match = re.search(r"header = \[(.*?)\]\n", src, re.S)
    assert header_match, "header list literal not found"
    header_items = re.findall(r'"([a-zA-Z0-9_]+)"', header_match.group(1))

    row_match = re.search(r"csv\.writer\(_f\)\.writerow\(\[(.*?)\]\)\n", src, re.S)
    assert row_match, "per-step row writerow(...) literal not found"
    row_body = row_match.group(1)
    row_keys_in_order = re.findall(r'reward_terms\["([a-zA-Z0-9_]+)"\]', row_body)

    new_cols = ["reward_heading_delta", "penalty_steering_magnitude",
                "penalty_steering_change", "penalty_speed_change"]
    for col in new_cols:
        assert col in header_items
        assert col in row_keys_in_order

    # In the row-write, the 4 new dict-key reads must be CONTIGUOUS, in the
    # same order as declared, sandwiched between risk_map_bonus and terminal
    # -- mirroring their position in the header (verified separately by
    # test_debug_csv_header_contains_new_columns_in_order).
    i_bonus = row_keys_in_order.index("risk_map_bonus")
    i_terminal = row_keys_in_order.index("terminal")
    assert row_keys_in_order[i_bonus + 1:i_terminal] == new_cols

    # Overall column-count parity (approx count of writerow(...) top-level
    # items via round(/int(bool(/literal-episode-step, mirroring the existing
    # 46-vs-50 style check done by hand during development).
    n_round = row_body.count("round(")
    n_intbool = row_body.count("int(bool(")
    n_literal = row_body.count("self._episode_count") + row_body.count("self.current_episode_step")
    assert n_round + n_intbool + n_literal == len(header_items)


# --------------------------------------------------------------------------- #
#  reset_callback actually clears the per-episode carry state (real code
#  path, not just pure-function inputs -- Environment._reset_continuous_
#  control_reward_state is called from BOTH __init__ and reset_callback; it
#  is exercised here directly since driving the full reset_callback needs a
#  live ROS/Gazebo node, but this is the real unbound method, not a re-
#  implementation of its logic).
# --------------------------------------------------------------------------- #
def test_reset_continuous_control_reward_state_clears_all_three_fields():
    node = types.SimpleNamespace(
        _prev_theta_err=0.42,
        _prev_ccr_cmd_steering=0.1,
        _prev_ccr_cmd_speed=0.5,
    )
    Environment._reset_continuous_control_reward_state(node)
    assert node._prev_theta_err is None
    assert node._prev_ccr_cmd_steering is None
    assert node._prev_ccr_cmd_speed is None


def test_reset_callback_impl_source_calls_the_real_reset_method():
    """Structural guard: _reset_callback_impl (the actual per-episode reset
    logic behind the reset_callback service wrapper) must actually CALL
    _reset_continuous_control_reward_state() (not reimplement the three
    assignments inline, which could silently drift from the tested method)."""
    src = open(env_mod.__file__, "r").read()
    reset_start = src.index("def _reset_callback_impl(")
    next_def = src.index("\n    def ", reset_start + 1)
    reset_body = src[reset_start:next_def]
    assert "self._reset_continuous_control_reward_state()" in reset_body
