"""phase2/tqc_vanilla profile tests: "pure TQC" curriculum profile -- actor +
TQC quantile critic only, every training-time model extension OFF on both
the env side and the agent side.

Covers:
  * the profile is discoverable by ProfileLoader and validates cleanly
    (ConfigValidator, no ROS/Gazebo needed);
  * the action contract is unchanged from phase2/baseline/both
    (action_mode=waypoint_yield, action_dim=3, same actions_low/high);
  * every model extension is OFF on both sides: aux_prediction,
    observation_time_context, temporal_actor_context, risk_map_reward,
    action_risk_head (env + agent), critic_risk_input,
    continuous_control_reward, directional_risk.waypoint_trajectory_risk_
    enabled, replay_buffer.risk_meta / risk_balanced_sampling,
    observation_normalization, optimizer_groups, prioritized;
  * the requested promotion-gate values (eval_eps=40, eval_freq=12000,
    consecutive_eval_passes=1, min_stage_steps=30000,
    min_stage_episodes=20) are in effect;
  * output_prefix / base_file_name are tqc_phase2_vanilla on both sides;
  * this profile does not perturb phase2/baseline, phase2/both, or any
    other existing phase2 profile;
  * the profile's ACTUAL hyperparameters_tqc.yaml constructs a real
    drl_agent.rl.algorithms.tqc.agent.Agent without raising -- a config-only
    (YAML flag / ConfigValidator) check is not enough: a prior revision of
    this profile had aux_prediction.enabled=false but left the same block's
    action_conditioned_aux=true, which Agent.__init__ fail-fasts on
    (RuntimeError: "aux_prediction.action_conditioned_aux=true requires
    aux_prediction.enabled=true") -- ConfigValidator never inspects that
    combination, so only an actual Agent construction catches it.
"""

import os

import pytest
import yaml

from drl_agent.config import ProfileLoader
from drl_agent.config.validation import ConfigValidator

try:
    import numpy as np

    from drl_agent.rl.algorithms.tqc.agent import Agent
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

TQC_VANILLA = "phase2/tqc_vanilla"
_OTHER_PROFILES = (
    "phase2/baseline",
    "phase2/both",
    "phase2/both_legacy",
    "phase2/both_trajrisk_rbs",
    "phase2/reward_shaping_only",
    "phase2/action_risk_head_only",
    "phase2/obs_norm_optim_split",
)


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


def _load_train_yaml(spec):
    with open(spec.config_paths["train"], "r") as f:
        return (yaml.safe_load(f) or {}).get("train_settings", {})


def _load_curriculum_yaml(spec):
    with open(spec.config_paths["curriculum"], "r") as f:
        return (yaml.safe_load(f) or {}).get("curriculum_settings", {})


# --------------------------------------------------------------------------- #
#  Discovery + validation
# --------------------------------------------------------------------------- #
def test_tqc_vanilla_is_discoverable():
    assert TQC_VANILLA in ProfileLoader().available_profiles()


def test_tqc_vanilla_profile_validates_fresh_run():
    spec = ProfileLoader().load(TQC_VANILLA)
    rep = ConfigValidator(spec).validate(resume=False)
    assert not rep.errors, rep.errors
    assert rep.info["environment.action_mode"] == "waypoint_yield"
    assert rep.info["risk_map_reward.enabled(env)"] is False
    assert rep.info["action_risk_head.env_enabled"] is False
    assert rep.info["action_risk_head.agent_enabled"] is False
    assert rep.info["continuous_control_reward.enabled"] is False
    assert rep.info["directional_risk.waypoint_trajectory_risk_enabled"] is False
    assert rep.info["replay_buffer.risk_balanced_sampling.enabled"] is False


def test_tqc_vanilla_profile_manifest_fields():
    spec = ProfileLoader().load(TQC_VANILLA)
    assert spec.algorithm == "tqc"
    assert spec.trainer == "curriculum"
    assert spec.output_prefix == "tqc_phase2_vanilla"
    assert spec.overrides == {
        "risk_map_reward_enabled": False,
        "action_risk_head_enabled": False,
    }


# --------------------------------------------------------------------------- #
#  Action contract unchanged (waypoint_yield, action_dim=3)
# --------------------------------------------------------------------------- #
def test_tqc_vanilla_action_contract_matches_phase2_baseline():
    spec = ProfileLoader().load(TQC_VANILLA)
    env = _load_env_yaml(spec)
    assert "action_mode" not in env  # inferred as waypoint_yield from action_dim>=3
    assert env.get("action_dim") == 3
    assert env.get("actions_low") == [0.0, -0.524, -1.0]
    assert env.get("actions_high") == [2.0, 0.524, 1.0]


# --------------------------------------------------------------------------- #
#  Every model extension OFF, both sides
# --------------------------------------------------------------------------- #
def test_tqc_vanilla_env_side_extensions_all_off():
    spec = ProfileLoader().load(TQC_VANILLA)
    env = _load_env_yaml(spec)

    assert env.get("aux_prediction", {}).get("enabled") is False
    assert env.get("observation_time_context", {}).get("enabled") is False
    assert env.get("risk_map_reward", {}).get("enabled") is False
    assert env.get("action_risk_head", {}).get("enabled") is False
    assert env.get("continuous_control_reward", {}).get("enabled") is False
    assert env.get("directional_risk", {}).get(
        "waypoint_trajectory_risk_enabled") is False


def test_tqc_vanilla_agent_side_extensions_all_off():
    spec = ProfileLoader().load(TQC_VANILLA)
    hp = _load_hp_yaml(spec)

    aux = hp.get("aux_prediction", {})
    assert aux.get("enabled") is False
    # Regression: a sub-feature flag left true inside a disabled aux_prediction
    # block fails Agent.__init__ fail-fast (see module docstring) even though
    # aux_prediction.enabled itself is false -- ConfigValidator never checks
    # this, only Agent construction does (see test_tqc_vanilla_agent_constructs_
    # without_error below).
    assert aux.get("action_conditioned_aux") is False
    assert aux.get("action_condition_attention") is False
    assert aux.get("ttc_head_enabled") is False
    assert aux.get("hazard_sector_head_enabled") is False
    assert hp.get("temporal_actor_context", {}).get("enabled") is False
    assert hp.get("action_risk_head", {}).get("enabled") is False
    assert hp.get("critic_risk_input", {}).get("enabled") is False
    rb = hp.get("replay_buffer", {})
    assert rb.get("risk_meta", {}).get("enabled") is False
    assert rb.get("risk_balanced_sampling", {}).get("enabled") is False
    assert hp.get("observation_normalization", {}).get("enabled") is False
    assert hp.get("optimizer_groups", {}).get("enabled") is False
    assert hp.get("prioritized") is False


# --------------------------------------------------------------------------- #
#  Real Agent construction from the profile's ACTUAL hyperparameters_tqc.yaml
#  -- catches config-internal contradictions ConfigValidator does not check
#  (e.g. aux_prediction.enabled=false but action_conditioned_aux=true, which
#  Agent.__init__ raises RuntimeError on).
# --------------------------------------------------------------------------- #
STATE_DIM = 87   # waypoint_yield, observation_time_context OFF: 80 (obs) + 7 (agent)
ACTION_DIM = 3   # waypoint_yield: [r, theta, yield]
ENV_OBS_DIM = 80
ENV_AGENT_DIM = 7


@pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")
def test_tqc_vanilla_agent_constructs_without_error(tmp_path):
    spec = ProfileLoader().load(TQC_VANILLA)
    hp = _load_hp_yaml(spec)

    # Must not raise -- this is the exact reproduction of the reviewed HIGH
    # finding (aux_prediction.enabled=false + action_conditioned_aux=true).
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0, hp, log_dir=str(tmp_path),
                  env_obs_dim=ENV_OBS_DIM, env_agent_dim=ENV_AGENT_DIM)

    assert agent.aux_enabled is False
    assert agent.aux_action_conditioned is False
    assert agent.aux_head is None
    assert agent.temporal_actor_enabled is False
    assert agent.action_risk_enabled is False
    assert agent.critic_risk_input_enabled is False
    assert agent.obs_norm_cfg.enabled is False
    assert agent.risk_balanced_enabled is False
    assert agent.store_risk_meta is False

    # aux_prediction + temporal_actor_context both off -> encoder is the
    # parameter-free identity passthrough (out_dim == state_dim).
    assert agent.encoder.out_dim == STATE_DIM

    a = agent.select_action(
        np.zeros(STATE_DIM, dtype="float32"), use_exploration=False)
    assert a.shape == (ACTION_DIM,)


# --------------------------------------------------------------------------- #
#  Promotion-gate values
# --------------------------------------------------------------------------- #
def test_tqc_vanilla_promotion_gate_values():
    spec = ProfileLoader().load(TQC_VANILLA)
    train = _load_train_yaml(spec)
    curr = _load_curriculum_yaml(spec)

    assert train.get("eval_eps") == 40
    assert train.get("eval_freq") == 12000
    assert curr.get("consecutive_eval_passes") == 1
    assert curr.get("min_stage_steps") == 30000
    assert curr.get("min_stage_episodes") == 20


def test_tqc_vanilla_output_prefix_matches_base_file_name():
    spec = ProfileLoader().load(TQC_VANILLA)
    train = _load_train_yaml(spec)
    assert spec.output_prefix == "tqc_phase2_vanilla"
    assert train.get("base_file_name") == "tqc_phase2_vanilla"


# --------------------------------------------------------------------------- #
#  10-stage curriculum / map / human / reward config preserved from baseline
# --------------------------------------------------------------------------- #
def test_tqc_vanilla_curriculum_stages_match_baseline():
    vanilla_spec = ProfileLoader().load(TQC_VANILLA)
    baseline_spec = ProfileLoader().load("phase2/baseline")
    vanilla_env = _load_env_yaml(vanilla_spec)
    baseline_env = _load_env_yaml(baseline_spec)

    assert vanilla_env.get("curriculum") == baseline_env.get("curriculum")
    assert vanilla_env.get("human_risk_penalty") == baseline_env.get("human_risk_penalty")
    assert vanilla_env.get("yield_reward") == baseline_env.get("yield_reward")
    assert vanilla_env.get("stall") == baseline_env.get("stall")
    assert vanilla_env.get("anti_freeze") == baseline_env.get("anti_freeze")


# --------------------------------------------------------------------------- #
#  Other profiles untouched
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", _OTHER_PROFILES)
def test_other_profiles_still_validate_unaffected(name):
    spec = ProfileLoader().load(name)
    rep = ConfigValidator(spec).validate(resume=False)
    assert not rep.errors, rep.errors


def test_phase2_baseline_unchanged_by_tqc_vanilla():
    spec = ProfileLoader().load("phase2/baseline")
    env = _load_env_yaml(spec)
    hp = _load_hp_yaml(spec)
    assert env.get("aux_prediction", {}).get("enabled") is True
    assert env.get("observation_time_context", {}).get("enabled") is True
    assert hp.get("aux_prediction", {}).get("enabled") is True
    assert hp.get("temporal_actor_context", {}).get("enabled") is True
    train = _load_train_yaml(spec)
    assert train.get("base_file_name") == "tqc_phase2_baseline"


def test_phase2_both_unchanged_by_tqc_vanilla():
    spec = ProfileLoader().load("phase2/both")
    env = _load_env_yaml(spec)
    assert env.get("risk_map_reward", {}).get("enabled") is True
    assert env.get("action_risk_head", {}).get("enabled") is True
    assert env.get("directional_risk", {}).get(
        "waypoint_trajectory_risk_enabled") is True
