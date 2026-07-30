"""PHASE3 profile tests: phase3/speed_steering_risk_balanced.

Covers:
  * the profile loads and validates (ConfigValidator, no ROS/Gazebo needed);
  * its environment_curriculum.yaml carries the correct action contract
    (action_mode=speed_steering, action_dim=2) and the state_dim implied by
    observation_time_context stacking is unchanged from phase2/both (327-D);
  * resume is rejected when no checkpoint exists at all, or when one exists
    but its recorded action contract doesn't match (see
    test_config_validation.py for the accept/reject-by-contract cases);
  * the risk-balanced-replay / weighted-loss config keys this profile
    activates are actually present with sane values;
  * yield_reward is fully disabled (no stage-5-style contract change, no
    per-stage overrides) and reset_buffer_on_promote_to is empty;
  * phase2/both (and the rest of the phase2 matrix) still loads and
    validates completely unchanged by this addition.
"""

import os

import pytest
import yaml

from drl_agent.common import compat
from drl_agent.config import ProfileLoader
from drl_agent.config.validation import ConfigValidator

try:
    from drl_agent.env.observation.obs_time_context import ObsTimeContext
    _HAVE_OTC = True
except Exception:  # pragma: no cover
    _HAVE_OTC = False


def _experiments_present():
    return bool(ProfileLoader().profiles_root())


pytestmark = pytest.mark.skipif(not _experiments_present(),
                                reason="drl_experiments profiles root not found")

PHASE3_NAME = "phase3/speed_steering_risk_balanced"


def _load_env_yaml(spec):
    with open(spec.config_paths["environment"], "r") as f:
        return (yaml.safe_load(f) or {}).get("environment", {})


def _load_hp_yaml(spec):
    with open(spec.config_paths["hparams"], "r") as f:
        return (yaml.safe_load(f) or {}).get("hyperparameters", {})


# --------------------------------------------------------------------------- #
#  Profile loads + validates
# --------------------------------------------------------------------------- #
def test_phase3_profile_loads():
    spec = ProfileLoader().load(PHASE3_NAME)
    assert spec.algorithm == "tqc"
    assert spec.trainer == "curriculum"
    assert spec.output_prefix == "tqc_phase3_speed_steering_risk_balanced"
    for key, path in spec.config_paths.items():
        assert os.path.isfile(path), f"{key} missing at {path}"


def test_phase3_profile_validates_fresh_run():
    spec = ProfileLoader().load(PHASE3_NAME)
    rep = ConfigValidator(spec).validate(resume=False)
    assert not rep.errors, rep.errors
    assert rep.info["environment.action_mode"] == "speed_steering"


def test_phase3_profile_rejects_resume_when_no_checkpoint_exists(tmp_path):
    """No checkpoint on disk -> resume is still a hard error (the generic
    resume-state check), even though a same-contract speed_steering resume is
    no longer unconditionally rejected -- see test_config_validation.py for
    the action-contract accept/reject-by-mismatch cases."""
    spec = ProfileLoader().load(PHASE3_NAME)
    root = str(tmp_path / "emptyroot")
    os.makedirs(root)
    rep = ConfigValidator(spec).validate(resume=True, seed=0, package_root=root)
    assert rep.errors
    assert any("no checkpoint" in e for e in rep.errors)


# --------------------------------------------------------------------------- #
#  Rollout dynamics must not silently drift from hunter_se_cmd_prefilter.yaml
# --------------------------------------------------------------------------- #
def _hunter_se_prefilter_yaml_path():
    """Locate hunter_se_cmd_prefilter.yaml via TWO independent strategies so
    the config-drift check below stays meaningful in an installed-only
    environment too, not just a source-tree checkout:
      1. sibling-package path off drl_agent's own SOURCE root (source-tree
         layout: .../ros2_ws/src/{drl_agent,hunter_se_gazebo}/...).
      2. ament_index's INSTALLED share directory for hunter_se_gazebo (its
         CMakeLists installs config/ verbatim) -- works after `colcon build`
         even when the source tree itself isn't present, e.g. a Docker image
         that only ships install/.

    Returns (path_or_None, src_root_found). ``src_root_found`` lets the
    caller distinguish "drl_agent isn't even part of a discoverable source
    tree" (a genuinely isolated/partial checkout -- skip is fine) from "the
    sibling src/ tree IS there but hunter_se_gazebo/config is somehow missing
    from it, and no installed copy exists either" (in THIS repo the two
    packages always live in the same monorepo checkout, so that combination
    means something is actually broken/moved -- fail, don't silently skip).
    """
    root = compat.package_source_root()
    src_root_found = bool(root)
    if root:
        path = os.path.join(os.path.dirname(root), "hunter_se_gazebo", "config",
                             "hunter_se_cmd_prefilter.yaml")
        if os.path.isfile(path):
            return path, src_root_found
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("hunter_se_gazebo")
        path = os.path.join(share, "config", "hunter_se_cmd_prefilter.yaml")
        if os.path.isfile(path):
            return path, src_root_found
    except Exception:
        pass
    return None, src_root_found


def test_phase3_rollout_dynamics_match_prefilter_config():
    """Regression for the reviewed config-drift risk: the two YAML files
    (phase3's environment_curriculum.yaml and hunter_se_cmd_prefilter.yaml)
    are genuinely separate config sources -- a code comment asking humans to
    keep them in sync does NOT prevent one from silently changing while the
    other doesn't. pure_pursuit.ackermann_swept_path's rollout is a forward
    PREDICTION of what the prefilter will actually do, so directional_risk.
    rollout_{accel_limit_mps2,brake_decel_mps2,steering_rate_deg_s} must stay
    numerically identical to the prefilter's own {accel_limit_mps2,
    brake_decel_mps2,steering_rate_deg_s} -- if this test ever fails, either
    the rollout is silently modeling the wrong vehicle, or this test's
    expectations need updating alongside the intentional config change.
    """
    prefilter_path, src_root_found = _hunter_se_prefilter_yaml_path()
    if prefilter_path is None:
        if src_root_found:
            pytest.fail(
                "drl_agent's sibling src/ tree was found but "
                "hunter_se_gazebo/config/hunter_se_cmd_prefilter.yaml is missing "
                "from it, and no installed hunter_se_gazebo share directory was "
                "found either -- in this repo the two packages always live in "
                "the same checkout, so this looks like a genuinely broken/moved "
                "prefilter config, not an isolated drl_agent-only checkout."
            )
        pytest.skip("hunter_se_gazebo not found (source tree or installed share) "
                     "-- looks like an isolated drl_agent-only checkout, not the "
                     "full repo")
    with open(prefilter_path, "r") as f:
        prefilter = (yaml.safe_load(f) or {}).get(
            "hunter_se_cmd_prefilter", {}).get("ros__parameters", {})

    spec = ProfileLoader().load(PHASE3_NAME)
    env = _load_env_yaml(spec)
    dr = env.get("directional_risk", {})

    assert dr.get("rollout_accel_limit_mps2") == pytest.approx(
        prefilter["accel_limit_mps2"])
    assert dr.get("rollout_brake_decel_mps2") == pytest.approx(
        prefilter["brake_decel_mps2"])
    assert dr.get("rollout_steering_rate_deg_s") == pytest.approx(
        prefilter["steering_rate_deg_s"])


# --------------------------------------------------------------------------- #
#  Action contract: action_mode / action_dim / implied state_dim
# --------------------------------------------------------------------------- #
def test_phase3_action_contract_is_speed_steering_dim2():
    spec = ProfileLoader().load(PHASE3_NAME)
    env = _load_env_yaml(spec)
    assert env.get("action_mode") == "speed_steering"
    assert env.get("action_dim") == 2
    assert len(env.get("actions_low", [])) == 2
    assert len(env.get("actions_high", [])) == 2


@pytest.mark.skipif(not _HAVE_OTC, reason="obs_time_context not importable")
def test_phase3_state_dim_matches_phase2_both_stacking():
    """action_mode is independent of observation_time_context stacking -- the
    stacked state_dim (327 = 80*4 + 7) must be UNCHANGED from phase2/both."""
    spec3 = ProfileLoader().load(PHASE3_NAME)
    spec2 = ProfileLoader().load("phase2/both")
    env3 = _load_env_yaml(spec3)
    env2 = _load_env_yaml(spec2)

    def stacked_dim(env):
        otc = dict(env.get("observation_time_context", {}) or {})
        ctx = ObsTimeContext(
            int(env["environment_state_dim"]), int(env["agent_state_dim"]),
            enabled=bool(otc.get("enabled", False)),
            obs_frame_stack=int(otc.get("obs_frame_stack", 1)),
            stack_agent_state=bool(otc.get("stack_agent_state", False)),
        )
        return ctx.stacked_state_dim()

    assert stacked_dim(env3) == stacked_dim(env2) == 327


# --------------------------------------------------------------------------- #
#  Yield reward fully neutralized (no stage-5-style contract change)
# --------------------------------------------------------------------------- #
def test_phase3_yield_reward_disabled_and_no_per_stage_overrides():
    spec = ProfileLoader().load(PHASE3_NAME)
    env = _load_env_yaml(spec)
    yr = env.get("yield_reward", {})
    assert yr.get("enabled") is False
    assert yr.get("action_enabled") is False
    with open(spec.config_paths["environment"], "r") as f:
        full = yaml.safe_load(f) or {}
    stages = full.get("curriculum", {}).get("stages", [])
    assert len(stages) == 10
    assert all("yield_reward" not in s for s in stages), \
        "speed_steering action semantics must be identical across every stage"


def test_phase3_no_buffer_reset_on_any_stage_promotion():
    spec = ProfileLoader().load(PHASE3_NAME)
    with open(spec.config_paths["curriculum"], "r") as f:
        cur = (yaml.safe_load(f) or {}).get("curriculum_settings", {})
    assert cur.get("reset_buffer_on_promote_to") == []


# --------------------------------------------------------------------------- #
#  Risk-balanced replay + weighted losses are actually turned ON
# --------------------------------------------------------------------------- #
def test_phase3_risk_balanced_replay_enabled_with_sane_ratios():
    spec = ProfileLoader().load(PHASE3_NAME)
    hp = _load_hp_yaml(spec)
    rb = hp.get("replay_buffer", {})
    assert rb.get("risk_meta", {}).get("enabled") is True
    rbs = rb.get("risk_balanced_sampling", {})
    assert rbs.get("enabled") is True
    ratios = (rbs.get("ratio_uniform"), rbs.get("ratio_human_risk"), rbs.get("ratio_collision"))
    assert ratios == (0.5, 0.25, 0.25)


def test_phase3_weighted_losses_enabled_within_conservative_range():
    spec = ProfileLoader().load(PHASE3_NAME)
    hp = _load_hp_yaml(spec)
    arh = hp.get("action_risk_head", {})
    assert 5.0 <= arh.get("pos_weight", 0.0) <= 10.0
    assert arh.get("pos_weight_cap", 0.0) <= 10.0

    aux = hp.get("aux_prediction", {})
    assert 5.0 <= aux.get("risk_map_positive_weight", 0.0) <= 10.0
    assert aux.get("risk_map_positive_weight_cap", 0.0) <= 10.0
    assert 5.0 <= aux.get("hazard_pos_weight", 0.0) <= 10.0
    assert aux.get("hazard_pos_weight_cap", 0.0) <= 10.0


# --------------------------------------------------------------------------- #
#  phase2 matrix (incl. phase2/both) stays completely unaffected
# --------------------------------------------------------------------------- #
_PHASE2 = ("phase2/baseline", "phase2/reward_shaping_only",
           "phase2/action_risk_head_only", "phase2/both",
           "phase2/obs_norm_optim_split")


@pytest.mark.parametrize("name", _PHASE2)
def test_phase2_profiles_still_validate_unchanged(name):
    spec = ProfileLoader().load(name)
    rep = ConfigValidator(spec).validate(resume=False)
    assert not rep.errors, rep.errors
    # None of the phase2 configs set action_mode -> default inference from
    # action_dim must still resolve to the pre-existing waypoint_yield mode.
    assert rep.info["environment.action_mode"] == "waypoint_yield"
