"""phase2/both_trajrisk_rbs_cf_st profile tests: the CF stability/
generalization improvements (actor-penalty warm-up/ramp, executed-action
target, weighted_mean horizon aggregation) layered on top of
phase2/both_trajrisk_rbs, plus the ST (spatiotemporal_lidar) toggle already
carried by that profile name.

Covers:
  * a fresh (non-resume) validate of phase2/both_trajrisk_rbs_cf_st is clean,
    with counterfactual_multi_horizon_risk actually ON;
  * the new CF keys (actor_penalty_warmup/ramp_updates, weighted_mean +
    horizon_weights, executed_action_loss_weight) are present and sane;
  * phase2/both_trajrisk_rbs (the baseline this profile forks from) carries
    the SAME new keys but stays enabled=false -- so toggling `enabled` alone
    is a clean A/B, and --validate-only catches a malformed horizon_weights
    even while the feature is off.
"""

import pytest
import yaml

from drl_agent.config import ProfileLoader
from drl_agent.config.validation import ConfigValidator

PHASE2_CF_ST = "phase2/both_trajrisk_rbs_cf_st"
PHASE2_BASELINE = "phase2/both_trajrisk_rbs"


def _experiments_present():
    return bool(ProfileLoader().profiles_root())


pytestmark = pytest.mark.skipif(not _experiments_present(),
                                 reason="drl_experiments profiles root not found")


def _load_hp_yaml(spec):
    with open(spec.config_paths["hparams"], "r") as f:
        return (yaml.safe_load(f) or {}).get("hyperparameters", {})


def test_cf_st_profile_validates_fresh_run():
    spec = ProfileLoader().load(PHASE2_CF_ST)
    rep = ConfigValidator(spec).validate(resume=False)
    assert not rep.errors, rep.errors
    assert rep.info["counterfactual_multi_horizon_risk.env_enabled"] is True
    assert rep.info["counterfactual_multi_horizon_risk.agent_enabled"] is True


def test_cf_st_profile_new_keys_present_and_sane():
    spec = ProfileLoader().load(PHASE2_CF_ST)
    hp = _load_hp_yaml(spec)
    cf = hp.get("counterfactual_multi_horizon_risk", {})
    assert cf.get("enabled") is True
    assert cf.get("actor_penalty_warmup_updates") == 5000
    assert cf.get("actor_penalty_ramp_updates") == 10000
    assert cf.get("actor_risk_aggregation") == "weighted_mean"
    assert cf.get("horizon_weights") == [0.4, 0.3, 0.2, 0.1]
    assert len(cf.get("horizon_weights")) == len(cf.get("horizons_sec"))
    assert cf.get("executed_action_loss_weight") == 1.0
    # Unchanged carry-over settings.
    assert cf.get("horizons_sec") == [0.5, 1.0, 1.5, 2.0]
    assert len(cf.get("candidate_actions")) == 7
    assert cf.get("hidden_dim") == 128
    assert cf.get("use_temporal_context") is True
    assert cf.get("enable_from_stage") == 3


def test_baseline_profile_carries_synced_cf_keys_but_stays_off():
    spec = ProfileLoader().load(PHASE2_BASELINE)
    rep = ConfigValidator(spec).validate(resume=False)
    assert not rep.errors, rep.errors
    assert rep.info["counterfactual_multi_horizon_risk.env_enabled"] is False
    assert rep.info["counterfactual_multi_horizon_risk.agent_enabled"] is False

    hp = _load_hp_yaml(spec)
    cf = hp.get("counterfactual_multi_horizon_risk", {})
    assert cf.get("enabled") is False
    assert cf.get("actor_penalty_warmup_updates") == 5000
    assert cf.get("actor_penalty_ramp_updates") == 10000
    assert cf.get("actor_risk_aggregation") == "weighted_mean"
    assert cf.get("horizon_weights") == [0.4, 0.3, 0.2, 0.1]
    assert cf.get("executed_action_loss_weight") == 1.0
