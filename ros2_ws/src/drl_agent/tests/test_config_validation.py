"""Unit tests for drl_agent.config.validation (ConfigValidator).

Covers every check the validator promises: missing files, env/agent
action_risk_head mismatch, profile-override vs yaml disagreement,
risk_map_reward source-of-truth recording, output_prefix vs base_file_name,
and resume-state reporting (found / not found) against a synthetic run tree.
"""

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
