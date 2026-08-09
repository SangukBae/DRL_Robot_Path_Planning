"""Unit tests for drl_agent.config.validation (ConfigValidator).

Covers every check the validator promises: missing files, env/agent
action_risk_head mismatch, profile-override vs yaml disagreement,
risk_map_reward source-of-truth recording, output_prefix vs base_file_name,
and resume-state reporting (found / not found) against a synthetic run tree.
"""

import json
import os

import pytest
import yaml

from drl_agent.config.schema import ProfileSpec
from drl_agent.config.validation import ConfigValidator


def _write_yaml(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(doc, f)


def _make_configs(tmp_path, *, env_arh=False, agent_arh=False, rmr=False,
                  base_file_name="tqc_x"):
    d = str(tmp_path / "prof")
    paths = {
        "environment": os.path.join(d, "environment_curriculum.yaml"),
        "train": os.path.join(d, "train_tqc_config.yaml"),
        "hparams": os.path.join(d, "hyperparameters_tqc.yaml"),
        "curriculum": os.path.join(d, "train_tqc_curriculum_config.yaml"),
    }
    _write_yaml(paths["environment"], {"environment": {
        "risk_map_reward": {"enabled": rmr},
        "action_risk_head": {"enabled": env_arh},
    }})
    _write_yaml(paths["train"], {"train_settings": {
        "seed": 0, "base_file_name": base_file_name, "load_model": False,
    }})
    _write_yaml(paths["hparams"], {"hyperparameters": {
        "action_risk_head": {"enabled": agent_arh},
    }})
    _write_yaml(paths["curriculum"], {"curriculum_settings": {"stages": []}})
    return d, paths


def _spec(d, paths, *, output_prefix="tqc_x", overrides=None, trainer="curriculum"):
    return ProfileSpec(name="t/x", algorithm="tqc", trainer=trainer,
                       profile_dir=d, config_paths=paths,
                       output_prefix=output_prefix, overrides=overrides or {})


def test_all_consistent_passes(tmp_path):
    d, paths = _make_configs(tmp_path, env_arh=True, agent_arh=True, rmr=True)
    rep = ConfigValidator(_spec(d, paths, overrides={
        "risk_map_reward_enabled": True, "action_risk_head_enabled": True,
    })).validate()
    assert rep.ok, rep.summary()
    assert rep.info["risk_map_reward.enabled(env)"] is True
    assert rep.info["action_risk_head.env_enabled"] is True


def test_missing_config_file_is_error(tmp_path):
    d, paths = _make_configs(tmp_path)
    os.remove(paths["hparams"])
    rep = ConfigValidator(_spec(d, paths)).validate()
    assert not rep.ok
    assert any("hparams" in e and "does not exist" in e for e in rep.errors)


def test_missing_curriculum_key_for_curriculum_trainer(tmp_path):
    d, paths = _make_configs(tmp_path)
    del paths["curriculum"]
    rep = ConfigValidator(_spec(d, paths)).validate()
    assert any("curriculum" in e for e in rep.errors)


def test_env_agent_action_risk_head_mismatch_is_error(tmp_path):
    d, paths = _make_configs(tmp_path, env_arh=True, agent_arh=False)
    rep = ConfigValidator(_spec(d, paths)).validate()
    assert not rep.ok
    assert any("action_risk_head.enabled mismatch" in e for e in rep.errors)


def test_profile_override_disagreeing_with_yaml_is_error(tmp_path):
    d, paths = _make_configs(tmp_path, env_arh=False, agent_arh=False)
    rep = ConfigValidator(_spec(d, paths, overrides={
        "action_risk_head_enabled": True,
    })).validate()
    assert any("declares action_risk_head_enabled=True" in e for e in rep.errors)

    d, paths = _make_configs(tmp_path / "b", rmr=False)
    rep = ConfigValidator(_spec(d, paths, overrides={
        "risk_map_reward_enabled": True,
    })).validate()
    assert any("declares risk_map_reward_enabled=True" in e for e in rep.errors)


def test_output_prefix_mismatch_is_error(tmp_path):
    d, paths = _make_configs(tmp_path, base_file_name="something_else")
    rep = ConfigValidator(_spec(d, paths, output_prefix="tqc_x")).validate()
    assert any("output_prefix" in e for e in rep.errors)


def test_risk_map_reward_block_in_hparams_warns(tmp_path):
    d, paths = _make_configs(tmp_path)
    with open(paths["hparams"], "w") as f:
        yaml.safe_dump({"hyperparameters": {
            "action_risk_head": {"enabled": False},
            "risk_map_reward": {"enabled": True},
        }}, f)
    rep = ConfigValidator(_spec(d, paths)).validate()
    assert rep.ok
    assert any("risk_map_reward" in w and "IGNORED" in w for w in rep.warnings)


# --------------------------------------------------------------------------- #
#  Resume-state checks                                                         #
# --------------------------------------------------------------------------- #

def _fake_run(tmp_path, base="tqc_x", seed=0, run_id="20260101_000000_tqc_x_seed0",
              *, with_replay_buffer=True, with_curriculum_state=True):
    root = str(tmp_path / "pkgroot")
    run = os.path.join(root, "runtime", "experiments", run_id)
    models = os.path.join(run, "models")
    logs = os.path.join(run, "logs")
    os.makedirs(models)
    os.makedirs(logs)
    prefix = f"{base}_seed_{seed}_20260101"
    suffixes = ["_actor.pth", "_critic.pth"]
    if with_replay_buffer:
        suffixes.append("_replay_buffer.npz")
    for suffix in suffixes:
        open(os.path.join(models, prefix + suffix), "w").close()
    if with_curriculum_state:
        open(os.path.join(logs, "curriculum_state.json"), "w").close()
    return root, run, prefix


def test_resume_reports_existing_checkpoint(tmp_path):
    d, paths = _make_configs(tmp_path)
    root, run, prefix = _fake_run(tmp_path)
    rep = ConfigValidator(_spec(d, paths)).validate(
        resume=True, seed=0, package_root=root)
    assert rep.ok, rep.summary()
    assert rep.info["resume.resumable"] is True
    assert rep.info["resume.layout"] == "new"
    assert rep.info["resume.checkpoint_prefix"] == prefix
    assert rep.info["resume.run_dir"] == run


def test_resume_without_checkpoint_is_error(tmp_path):
    d, paths = _make_configs(tmp_path)
    root = str(tmp_path / "emptyroot")
    os.makedirs(root)
    rep = ConfigValidator(_spec(d, paths)).validate(
        resume=True, seed=3, package_root=root)
    assert not rep.ok
    assert any("resume requested but no checkpoint" in e for e in rep.errors)
    assert rep.info["resume.resumable"] is False


# --------------------------------------------------------------------------- #
#  action_mode contract check (speed_steering resume gate)                     #
# --------------------------------------------------------------------------- #

def _make_speed_steering_configs(tmp_path, **kw):
    d, paths = _make_configs(tmp_path, **kw)
    with open(paths["environment"], "r") as f:
        doc = yaml.safe_load(f)
    doc["environment"]["action_mode"] = "speed_steering"
    doc["environment"]["action_dim"] = 2
    _write_yaml(paths["environment"], doc)
    return d, paths


def _write_action_contract(run, action_mode, action_dim):
    configs_dir = os.path.join(run, "configs")
    os.makedirs(configs_dir, exist_ok=True)
    with open(os.path.join(configs_dir, "profile_manifest.json"), "w") as f:
        json.dump({"action_mode": action_mode, "action_dim": action_dim}, f)


def test_speed_steering_resume_allowed_when_checkpoint_contract_matches(tmp_path):
    d, paths = _make_speed_steering_configs(tmp_path)
    root, run, _prefix = _fake_run(tmp_path)
    _write_action_contract(run, "speed_steering", 2)
    rep = ConfigValidator(_spec(d, paths)).validate(
        resume=True, seed=0, package_root=root)
    assert rep.ok, rep.summary()
    assert rep.info["environment.action_mode"] == "speed_steering"


def test_speed_steering_resume_rejected_when_checkpoint_contract_differs(tmp_path):
    d, paths = _make_speed_steering_configs(tmp_path)
    root, run, _prefix = _fake_run(tmp_path)
    _write_action_contract(run, "waypoint_yield", 3)
    rep = ConfigValidator(_spec(d, paths)).validate(
        resume=True, seed=0, package_root=root)
    assert not rep.ok
    assert any("incompatible action contracts" in e for e in rep.errors)


def test_speed_steering_resume_rejected_when_manifest_missing(tmp_path):
    d, paths = _make_speed_steering_configs(tmp_path)
    root, _run, _prefix = _fake_run(tmp_path)  # no configs/profile_manifest.json
    rep = ConfigValidator(_spec(d, paths)).validate(
        resume=True, seed=0, package_root=root)
    assert not rep.ok
    assert any("no action-contract record" in e for e in rep.errors)


def test_waypoint_yield_resume_unaffected_by_action_mode_check(tmp_path):
    """A non-speed_steering profile never even consults the manifest -- the
    pre-existing curriculum resume checks (replay buffer / curriculum state)
    are the only gate."""
    d, paths = _make_configs(tmp_path)  # no action_dim set -> default action_mode "waypoint"
    root, _run, _prefix = _fake_run(tmp_path)  # no action-contract manifest
    rep = ConfigValidator(_spec(d, paths)).validate(
        resume=True, seed=0, package_root=root)
    assert rep.ok, rep.summary()


def test_curriculum_resume_missing_replay_buffer_is_hard_error(tmp_path):
    d, paths = _make_configs(tmp_path)
    _fake_run(tmp_path, with_replay_buffer=False, with_curriculum_state=True)
    root = str(tmp_path / "pkgroot")
    rep = ConfigValidator(_spec(d, paths, trainer="curriculum")).validate(
        resume=True, seed=0, package_root=root)
    assert not rep.ok
    assert any("no replay buffer found" in e and "EMPTY replay buffer" in e
               for e in rep.errors)
    assert rep.info["resume.has_replay_buffer"] is False
    # resumable is still True (a checkpoint DOES exist) -- it's the STRICTER
    # curriculum-specific check that fails.
    assert rep.info["resume.resumable"] is True


def test_curriculum_resume_missing_curriculum_state_is_hard_error(tmp_path):
    d, paths = _make_configs(tmp_path)
    _fake_run(tmp_path, with_replay_buffer=True, with_curriculum_state=False)
    root = str(tmp_path / "pkgroot")
    rep = ConfigValidator(_spec(d, paths, trainer="curriculum")).validate(
        resume=True, seed=0, package_root=root)
    assert not rep.ok
    assert any("no curriculum_state.json found" in e and "RESTART" in e
               for e in rep.errors)
    assert rep.info["resume.has_curriculum_state"] is False


def test_curriculum_resume_missing_both_reports_both_errors(tmp_path):
    d, paths = _make_configs(tmp_path)
    _fake_run(tmp_path, with_replay_buffer=False, with_curriculum_state=False)
    root = str(tmp_path / "pkgroot")
    rep = ConfigValidator(_spec(d, paths, trainer="curriculum")).validate(
        resume=True, seed=0, package_root=root)
    assert len(rep.errors) == 2
    assert any("replay buffer" in e for e in rep.errors)
    assert any("curriculum_state.json" in e for e in rep.errors)


def test_base_trainer_resume_missing_replay_buffer_only_warns(tmp_path):
    d, paths = _make_configs(tmp_path)
    _fake_run(tmp_path, with_replay_buffer=False, with_curriculum_state=False)
    root = str(tmp_path / "pkgroot")
    rep = ConfigValidator(_spec(d, paths, trainer="base")).validate(
        resume=True, seed=0, package_root=root)
    assert rep.ok, rep.summary()  # warning only -- does not block launch
    assert any("MODEL-ONLY resume" in w for w in rep.warnings)
    # curriculum_state absence is irrelevant to a base trainer -- no error/warn.
    assert not any("curriculum_state" in w for w in rep.warnings)


# --------------------------------------------------------------------------- #
#  RISK_BALANCE / TRAJ_RISK feature-contract check (resume gate)               #
# --------------------------------------------------------------------------- #

def _make_risk_feature_configs(tmp_path, *, traj_risk=True, risk_balanced=True, **kw):
    d, paths = _make_configs(tmp_path, **kw)
    with open(paths["environment"], "r") as f:
        doc = yaml.safe_load(f)
    doc["environment"]["directional_risk"] = {
        "waypoint_trajectory_risk_enabled": traj_risk,
    }
    _write_yaml(paths["environment"], doc)
    with open(paths["hparams"], "r") as f:
        hdoc = yaml.safe_load(f)
    hdoc["hyperparameters"]["replay_buffer"] = {
        "risk_balanced_sampling": {"enabled": risk_balanced},
    }
    _write_yaml(paths["hparams"], hdoc)
    return d, paths


def _write_risk_feature_contract(run, *, traj_risk, risk_balanced,
                                  action_mode="waypoint_yield", action_dim=3):
    configs_dir = os.path.join(run, "configs")
    os.makedirs(configs_dir, exist_ok=True)
    with open(os.path.join(configs_dir, "profile_manifest.json"), "w") as f:
        json.dump({
            "action_mode": action_mode, "action_dim": action_dim,
            "waypoint_trajectory_risk_enabled": traj_risk,
            "risk_balanced_sampling_enabled": risk_balanced,
        }, f)


def test_risk_feature_contract_recorded_on_fresh_validate(tmp_path):
    d, paths = _make_risk_feature_configs(tmp_path, traj_risk=True, risk_balanced=True)
    rep = ConfigValidator(_spec(d, paths)).validate(resume=False)
    assert rep.ok, rep.summary()
    assert rep.info["directional_risk.waypoint_trajectory_risk_enabled"] is True
    assert rep.info["replay_buffer.risk_balanced_sampling.enabled"] is True


def test_risk_feature_resume_allowed_when_contract_matches(tmp_path):
    d, paths = _make_risk_feature_configs(tmp_path, traj_risk=True, risk_balanced=True)
    root, run, _prefix = _fake_run(tmp_path)
    _write_risk_feature_contract(run, traj_risk=True, risk_balanced=True)
    rep = ConfigValidator(_spec(d, paths)).validate(
        resume=True, seed=0, package_root=root)
    assert rep.ok, rep.summary()


def test_risk_feature_resume_rejected_when_contract_differs(tmp_path):
    d, paths = _make_risk_feature_configs(tmp_path, traj_risk=True, risk_balanced=True)
    root, run, _prefix = _fake_run(tmp_path)
    _write_risk_feature_contract(run, traj_risk=False, risk_balanced=False)
    rep = ConfigValidator(_spec(d, paths)).validate(
        resume=True, seed=0, package_root=root)
    assert not rep.ok
    assert any("risk feature contract mismatch" in e for e in rep.errors)


def test_risk_feature_resume_rejected_when_only_one_flag_differs(tmp_path):
    d, paths = _make_risk_feature_configs(tmp_path, traj_risk=True, risk_balanced=False)
    root, run, _prefix = _fake_run(tmp_path)
    _write_risk_feature_contract(run, traj_risk=True, risk_balanced=True)
    rep = ConfigValidator(_spec(d, paths)).validate(
        resume=True, seed=0, package_root=root)
    assert not rep.ok
    assert any("risk feature contract mismatch" in e for e in rep.errors)


def test_risk_feature_resume_rejected_when_manifest_missing(tmp_path):
    d, paths = _make_risk_feature_configs(tmp_path, traj_risk=True, risk_balanced=False)
    root, _run, _prefix = _fake_run(tmp_path)  # no configs/profile_manifest.json
    rep = ConfigValidator(_spec(d, paths)).validate(
        resume=True, seed=0, package_root=root)
    assert not rep.ok
    assert any("no profile_manifest.json feature contract" in e for e in rep.errors)


def test_risk_feature_resume_unaffected_when_both_features_off(tmp_path):
    """When the profile itself never turns either feature on AND no manifest
    contract exists, resume is unaffected by this check (mirrors
    test_waypoint_yield_resume_unaffected_by_action_mode_check)."""
    d, paths = _make_configs(tmp_path)  # no directional_risk / replay_buffer keys at all
    root, _run, _prefix = _fake_run(tmp_path)  # no profile_manifest.json
    rep = ConfigValidator(_spec(d, paths)).validate(
        resume=True, seed=0, package_root=root)
    assert rep.ok, rep.summary()


def test_risk_feature_resume_rejected_when_current_off_but_checkpoint_had_features_on(tmp_path):
    """Regression for the reviewed MEDIUM finding: the check previously
    short-circuited BEFORE reading the manifest whenever the CURRENT config
    had both features off -- silently allowing a resume into a checkpoint
    that was trained WITH risk-balanced replay / trajectory-risk targets ON.
    A manifest that DOES carry the feature contract must now be compared
    unconditionally, regardless of the current config's own on/off state."""
    d, paths = _make_configs(tmp_path)  # current config: both features OFF (absent keys)
    root, run, _prefix = _fake_run(tmp_path)
    _write_risk_feature_contract(run, traj_risk=True, risk_balanced=True)
    rep = ConfigValidator(_spec(d, paths)).validate(
        resume=True, seed=0, package_root=root)
    assert not rep.ok
    assert any("risk feature contract mismatch" in e for e in rep.errors)


def test_risk_feature_resume_allowed_when_current_off_and_checkpoint_also_off(tmp_path):
    """Symmetric non-mismatch case: manifest DOES carry the contract (both
    keys present, both False) and current config is also both off -> no
    mismatch, allowed."""
    d, paths = _make_configs(tmp_path)  # current config: both features OFF
    root, run, _prefix = _fake_run(tmp_path)
    _write_risk_feature_contract(run, traj_risk=False, risk_balanced=False)
    rep = ConfigValidator(_spec(d, paths)).validate(
        resume=True, seed=0, package_root=root)
    assert rep.ok, rep.summary()


# --------------------------------------------------------------------------- #
#  LOW fix: rollout dynamics / balance-ratio fail-fast validation             #
# --------------------------------------------------------------------------- #

def _make_traj_risk_configs(tmp_path, *, rollout_overrides=None, ratio_overrides=None, **kw):
    d, paths = _make_configs(tmp_path, **kw)
    with open(paths["environment"], "r") as f:
        doc = yaml.safe_load(f)
    dr = {"waypoint_trajectory_risk_enabled": True}
    dr.update(rollout_overrides or {})
    doc["environment"]["directional_risk"] = dr
    _write_yaml(paths["environment"], doc)
    if ratio_overrides is not None:
        with open(paths["hparams"], "r") as f:
            hdoc = yaml.safe_load(f)
        rbs = {"enabled": True}
        rbs.update(ratio_overrides)
        hdoc["hyperparameters"]["replay_buffer"] = {"risk_balanced_sampling": rbs}
        _write_yaml(paths["hparams"], hdoc)
    return d, paths


def test_rollout_path_samples_zero_is_error(tmp_path):
    d, paths = _make_traj_risk_configs(tmp_path, rollout_overrides={"rollout_path_samples": 0})
    rep = ConfigValidator(_spec(d, paths)).validate()
    assert not rep.ok
    assert any("rollout_path_samples=0" in e and "EMPTY path" in e for e in rep.errors)


def test_rollout_path_samples_negative_is_error(tmp_path):
    d, paths = _make_traj_risk_configs(tmp_path, rollout_overrides={"rollout_path_samples": -3})
    rep = ConfigValidator(_spec(d, paths)).validate()
    assert not rep.ok
    assert any("rollout_path_samples=-3" in e for e in rep.errors)


def test_rollout_non_positive_dynamics_are_errors(tmp_path):
    d, paths = _make_traj_risk_configs(tmp_path, rollout_overrides={
        "horizon_sec": 0.0,
        "rollout_accel_limit_mps2": -1.0,
        "rollout_brake_decel_mps2": float("nan"),
        "rollout_steering_rate_deg_s": 0.0,
    })
    rep = ConfigValidator(_spec(d, paths)).validate()
    assert not rep.ok
    joined = " | ".join(rep.errors)
    assert "horizon_sec=0.0" in joined
    assert "rollout_accel_limit_mps2=-1.0" in joined
    assert "rollout_brake_decel_mps2=nan" in joined
    assert "rollout_steering_rate_deg_s=0.0" in joined


def test_rollout_config_valid_values_pass(tmp_path):
    d, paths = _make_traj_risk_configs(tmp_path, rollout_overrides={
        "rollout_path_samples": 15, "horizon_sec": 1.0,
        "rollout_accel_limit_mps2": 6.0, "rollout_brake_decel_mps2": 6.0,
        "rollout_steering_rate_deg_s": 200.0,
    })
    rep = ConfigValidator(_spec(d, paths)).validate()
    assert rep.ok, rep.summary()


def test_rollout_check_inert_when_neither_speed_steering_nor_traj_risk(tmp_path):
    """A garbage rollout_path_samples is harmless (and unchecked) when the
    rollout is never reachable: action_mode is plain waypoint_yield and
    waypoint_trajectory_risk_enabled is false/absent."""
    d, paths = _make_configs(tmp_path)
    with open(paths["environment"], "r") as f:
        doc = yaml.safe_load(f)
    doc["environment"]["directional_risk"] = {"rollout_path_samples": 0}
    _write_yaml(paths["environment"], doc)
    rep = ConfigValidator(_spec(d, paths)).validate()
    assert rep.ok, rep.summary()


def test_rollout_check_applies_to_speed_steering_too(tmp_path):
    d, paths = _make_speed_steering_configs(tmp_path)
    with open(paths["environment"], "r") as f:
        doc = yaml.safe_load(f)
    doc["environment"]["directional_risk"] = {"rollout_path_samples": 0}
    _write_yaml(paths["environment"], doc)
    rep = ConfigValidator(_spec(d, paths)).validate()
    assert not rep.ok
    assert any("rollout_path_samples=0" in e for e in rep.errors)


def test_risk_balanced_ratio_negative_is_error(tmp_path):
    d, paths = _make_traj_risk_configs(tmp_path, ratio_overrides={"ratio_human_risk": -0.1})
    rep = ConfigValidator(_spec(d, paths)).validate()
    assert not rep.ok
    assert any("ratio_human_risk=-0.1" in e for e in rep.errors)


def test_risk_balanced_ratio_non_numeric_is_error(tmp_path):
    d, paths = _make_traj_risk_configs(tmp_path, ratio_overrides={"ratio_collision": "oops"})
    rep = ConfigValidator(_spec(d, paths)).validate()
    assert not rep.ok
    assert any("ratio_collision='oops'" in e for e in rep.errors)


def test_risk_balanced_ratio_check_inert_when_sampling_disabled(tmp_path):
    d, paths = _make_configs(tmp_path)
    with open(paths["hparams"], "r") as f:
        hdoc = yaml.safe_load(f)
    hdoc["hyperparameters"]["replay_buffer"] = {
        "risk_balanced_sampling": {"enabled": False, "ratio_human_risk": -5.0},
    }
    _write_yaml(paths["hparams"], hdoc)
    rep = ConfigValidator(_spec(d, paths)).validate()
    assert rep.ok, rep.summary()


# --------------------------------------------------------------------------- #
#  Follow-up fixes: strict integer rollout_path_samples + all-zero ratio sum  #
# --------------------------------------------------------------------------- #

def test_rollout_path_samples_fractional_float_is_error(tmp_path):
    """int(15.5) == 15 would otherwise silently truncate a typo'd fractional
    sample count instead of rejecting it."""
    d, paths = _make_traj_risk_configs(tmp_path, rollout_overrides={"rollout_path_samples": 15.5})
    rep = ConfigValidator(_spec(d, paths)).validate()
    assert not rep.ok
    assert any("rollout_path_samples=15.5" in e and "fractional" in e for e in rep.errors)


def test_rollout_path_samples_whole_number_float_passes(tmp_path):
    """A whole-number float (e.g. YAML's `15.0`) is a legitimate integer
    value, not a typo -- must NOT be rejected."""
    d, paths = _make_traj_risk_configs(tmp_path, rollout_overrides={"rollout_path_samples": 15.0})
    rep = ConfigValidator(_spec(d, paths)).validate()
    assert rep.ok, rep.summary()


def test_rollout_path_samples_bool_is_error(tmp_path):
    """bool is a subclass of int in Python -- int(True) == 1 would otherwise
    silently accept `rollout_path_samples: true` as a valid sample count."""
    d, paths = _make_traj_risk_configs(tmp_path, rollout_overrides={"rollout_path_samples": True})
    rep = ConfigValidator(_spec(d, paths)).validate()
    assert not rep.ok
    assert any("rollout_path_samples=True" in e for e in rep.errors)


def test_risk_balanced_all_zero_ratios_is_error(tmp_path):
    d, paths = _make_traj_risk_configs(tmp_path, ratio_overrides={
        "ratio_uniform": 0.0, "ratio_human_risk": 0.0, "ratio_collision": 0.0,
    })
    rep = ConfigValidator(_spec(d, paths)).validate()
    assert not rep.ok
    assert any("must sum to > 0" in e for e in rep.errors)


def test_risk_balanced_partial_zero_ratios_still_pass(tmp_path):
    """Only the ALL-zero case is an error -- a legitimate config that zeroes
    out just one pool (e.g. uniform=0, all weight on the two event pools)
    must still validate."""
    d, paths = _make_traj_risk_configs(tmp_path, ratio_overrides={
        "ratio_uniform": 0.0, "ratio_human_risk": 0.5, "ratio_collision": 0.5,
    })
    rep = ConfigValidator(_spec(d, paths)).validate()
    assert rep.ok, rep.summary()


def test_risk_balanced_default_ratios_when_all_absent_pass(tmp_path):
    """enabled=true with no ratio_* keys at all -> LAP's own defaults
    (0.5/0.25/0.25) apply and sum to 1.0 -- must not be flagged."""
    d, paths = _make_configs(tmp_path)
    with open(paths["environment"], "r") as f:
        doc = yaml.safe_load(f)
    doc["environment"]["directional_risk"] = {"waypoint_trajectory_risk_enabled": True}
    _write_yaml(paths["environment"], doc)
    with open(paths["hparams"], "r") as f:
        hdoc = yaml.safe_load(f)
    hdoc["hyperparameters"]["replay_buffer"] = {"risk_balanced_sampling": {"enabled": True}}
    _write_yaml(paths["hparams"], hdoc)
    rep = ConfigValidator(_spec(d, paths)).validate()
    assert rep.ok, rep.summary()
