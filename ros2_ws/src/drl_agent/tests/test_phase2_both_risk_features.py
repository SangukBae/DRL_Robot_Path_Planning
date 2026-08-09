"""phase2/both_trajrisk_rbs profile tests: risk-balanced replay + weighted
loss, and the waypoint/waypoint_yield trajectory-risk target.

Covers:
  * both features are ON in phase2/both_trajrisk_rbs, with sane/conservative
    values;
  * canonical (config/) defaults and the rest of the phase2 matrix
    (baseline / reward_shaping_only / action_risk_head_only /
    obs_norm_optim_split, both_legacy) stay OFF -- these two features are
    opt-in ONLY for phase2/both_trajrisk_rbs, not a canonical behaviour change;
  * phase2/both_trajrisk_rbs's action_dim/state_dim/actions_low/actions_high
    (the Pure Pursuit control contract) are UNCHANGED by either feature;
  * phase2/both_trajrisk_rbs's directional_risk.rollout_* mirrors
    hunter_se_cmd_prefilter.yaml exactly (config-drift regression);
  * a fresh (non-resume) validate of phase2/both_trajrisk_rbs is clean.

Resume-hard-reject-on-feature-contract-mismatch is covered generically in
test_config_validation.py (not profile-specific) -- not touched by this file.
"""

import os

import pytest
import yaml

from drl_agent.common import compat
from drl_agent.config import ProfileLoader
from drl_agent.config.validation import ConfigValidator

PHASE2_BOTH_TRAJRISK_RBS = "phase2/both_trajrisk_rbs"
PHASE2_BOTH_LEGACY = "phase2/both_legacy"
_OTHER_PHASE2 = ("phase2/baseline", "phase2/reward_shaping_only",
                  "phase2/action_risk_head_only", "phase2/obs_norm_optim_split",
                  PHASE2_BOTH_LEGACY)


def _experiments_present():
    return bool(ProfileLoader().profiles_root())


pytestmark = pytest.mark.skipif(not _experiments_present(),
                                 reason="drl_experiments profiles root not found")


def _load_env_yaml(spec):
    with open(spec.config_paths["environment"], "r") as f:
        return (yaml.safe_load(f) or {}).get("environment", {})


def _load_hp_yaml(spec):
    with open(spec.config_paths["hparams"], "r") as f:
        return (yaml.safe_load(f) or {}).get("hyperparameters", {})


# --------------------------------------------------------------------------- #
#  Profile loads + validates
# --------------------------------------------------------------------------- #
def test_phase2_both_profile_validates_fresh_run():
    spec = ProfileLoader().load(PHASE2_BOTH_TRAJRISK_RBS)
    rep = ConfigValidator(spec).validate(resume=False)
    assert not rep.errors, rep.errors
    assert rep.info["environment.action_mode"] == "waypoint_yield"
    assert rep.info["directional_risk.waypoint_trajectory_risk_enabled"] is True
    assert rep.info["replay_buffer.risk_balanced_sampling.enabled"] is True


# --------------------------------------------------------------------------- #
#  Feature 1: risk-balanced replay + weighted loss -- ON, sane values
# --------------------------------------------------------------------------- #
def test_phase2_both_risk_balanced_replay_enabled_with_sane_ratios():
    spec = ProfileLoader().load(PHASE2_BOTH_TRAJRISK_RBS)
    hp = _load_hp_yaml(spec)
    rb = hp.get("replay_buffer", {})
    assert rb.get("risk_meta", {}).get("enabled") is True
    rbs = rb.get("risk_balanced_sampling", {})
    assert rbs.get("enabled") is True
    ratios = (rbs.get("ratio_uniform"), rbs.get("ratio_human_risk"), rbs.get("ratio_collision"))
    assert ratios == (0.5, 0.25, 0.25)


def test_phase2_both_weighted_losses_enabled_within_conservative_range():
    spec = ProfileLoader().load(PHASE2_BOTH_TRAJRISK_RBS)
    hp = _load_hp_yaml(spec)
    arh = hp.get("action_risk_head", {})
    assert 3.0 <= arh.get("pos_weight", 0.0) <= 6.0
    assert arh.get("pos_weight_cap", 0.0) <= 10.0
    assert arh.get("loss_type") == "smooth_l1"

    aux = hp.get("aux_prediction", {})
    assert 3.0 <= aux.get("risk_map_positive_weight", 0.0) <= 6.0
    assert aux.get("risk_map_positive_weight_cap", 0.0) <= 10.0
    assert aux.get("risk_map_loss_type") == "smooth_l1"


# --------------------------------------------------------------------------- #
#  Feature 2: waypoint trajectory risk -- ON, rollout dynamics match prefilter
# --------------------------------------------------------------------------- #
def _hunter_se_prefilter_yaml_path():
    """Kept self-contained per this repo's test-file convention rather than
    cross-importing another test module."""
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


def test_phase2_both_rollout_dynamics_match_prefilter_config():
    """Config-drift regression for phase2/both_trajrisk_rbs's
    waypoint_trajectory_risk_enabled rollout."""
    prefilter_path, src_root_found = _hunter_se_prefilter_yaml_path()
    if prefilter_path is None:
        if src_root_found:
            pytest.fail(
                "drl_agent's sibling src/ tree was found but "
                "hunter_se_gazebo/config/hunter_se_cmd_prefilter.yaml is missing "
                "from it, and no installed hunter_se_gazebo share directory was "
                "found either."
            )
        pytest.skip("hunter_se_gazebo not found (source tree or installed share)")
    with open(prefilter_path, "r") as f:
        prefilter = (yaml.safe_load(f) or {}).get(
            "hunter_se_cmd_prefilter", {}).get("ros__parameters", {})

    spec = ProfileLoader().load(PHASE2_BOTH_TRAJRISK_RBS)
    env = _load_env_yaml(spec)
    dr = env.get("directional_risk", {})

    assert dr.get("waypoint_trajectory_risk_enabled") is True
    assert dr.get("rollout_accel_limit_mps2") == pytest.approx(prefilter["accel_limit_mps2"])
    assert dr.get("rollout_brake_decel_mps2") == pytest.approx(prefilter["brake_decel_mps2"])
    assert dr.get("rollout_steering_rate_deg_s") == pytest.approx(prefilter["steering_rate_deg_s"])


# --------------------------------------------------------------------------- #
#  Pure Pursuit / action / state contract unchanged
# --------------------------------------------------------------------------- #
def test_phase2_both_action_contract_unchanged():
    """Neither feature touches the waypoint_yield action/control contract:
    action_dim stays 3 (r, theta, yield), actions_low/high stay the original
    hybrid-action bounds, and no action_mode override is introduced (still
    inferred as waypoint_yield from action_dim>=3, same as before either
    feature existed)."""
    spec = ProfileLoader().load(PHASE2_BOTH_TRAJRISK_RBS)
    env = _load_env_yaml(spec)
    assert "action_mode" not in env
    assert env.get("action_dim") == 3
    assert env.get("actions_low") == [0.0, -0.524, -1.0]
    assert env.get("actions_high") == [2.0, 0.524, 1.0]


# --------------------------------------------------------------------------- #
#  Everything else stays OFF: canonical + the rest of the phase2 matrix
# --------------------------------------------------------------------------- #
def test_canonical_defaults_are_off():
    from drl_agent.common import compat as _compat
    root = _compat.package_source_root()
    if not root:
        pytest.skip("drl_agent source root not found")
    env_path = os.path.join(root, "config", "environment_curriculum.yaml")
    hp_path = os.path.join(root, "config", "hyperparameters_tqc.yaml")
    with open(env_path, "r") as f:
        env = (yaml.safe_load(f) or {}).get("environment", {})
    with open(hp_path, "r") as f:
        hp = (yaml.safe_load(f) or {}).get("hyperparameters", {})

    assert env.get("directional_risk", {}).get("waypoint_trajectory_risk_enabled") is False
    rb = hp.get("replay_buffer", {})
    assert rb.get("risk_meta", {}).get("enabled") is False
    assert rb.get("risk_balanced_sampling", {}).get("enabled") is False


@pytest.mark.parametrize("name", _OTHER_PHASE2)
def test_other_phase2_profiles_do_not_enable_either_feature(name):
    spec = ProfileLoader().load(name)
    rep = ConfigValidator(spec).validate(resume=False)
    assert not rep.errors, rep.errors
    assert rep.info["directional_risk.waypoint_trajectory_risk_enabled"] is False
    assert rep.info["replay_buffer.risk_balanced_sampling.enabled"] is False


def test_phase2_both_legacy_keeps_old_both_feature_contract():
    spec = ProfileLoader().load(PHASE2_BOTH_LEGACY)
    rep = ConfigValidator(spec).validate(resume=False)
    assert not rep.errors, rep.errors
    assert spec.output_prefix == "tqc_phase2_both_legacy"
    assert rep.info["risk_map_reward.enabled(env)"] is True
    assert rep.info["action_risk_head.env_enabled"] is True
    assert rep.info["action_risk_head.agent_enabled"] is True
    assert rep.info["directional_risk.waypoint_trajectory_risk_enabled"] is False
    assert rep.info["replay_buffer.risk_balanced_sampling.enabled"] is False
