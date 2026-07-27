"""Unit tests for drl_agent.training.run_manager (RunManager) and the
checkpoint discovery in drl_agent.rl.checkpointing.manager.

RunManager must delegate to run_layout (identical run_id format / resume
precedence) and additionally record config hashes + a profile manifest.
"""

import json
import os

from drl_agent.rl.checkpointing.manager import (CheckpointManager,
                                                latest_checkpoint_prefix)
from drl_agent.training.run_manager import RunManager, config_hashes


def test_config_hashes_stable_and_missing_tolerant(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("k: 1\n")
    h1 = config_hashes({"a": str(p), "missing": str(tmp_path / "nope.yaml")})
    h2 = config_hashes({"a": str(p)})
    assert h1["a"].startswith("sha256:") and h1["a"] == h2["a"]
    assert h1["missing"] == ""


def test_fresh_run_layout_and_manifest(tmp_path):
    root = str(tmp_path)
    mgr = RunManager(root, "tqc_x", 0, load_model=False)
    layout = mgr.resolve()
    assert not layout["is_legacy"] and layout["is_fresh"]
    assert layout["run_dir"].startswith(
        os.path.join(root, "runtime", "experiments"))
    mgr.create_dirs()
    assert os.path.isdir(layout["models_dir"])

    cfg = tmp_path / "train.yaml"
    cfg.write_text("train_settings: {}\n")
    path = mgr.write_profile_manifest(
        profile_name="phase2/both",
        config_paths={"train": str(cfg)},
        overrides={"risk_map_reward_enabled": True},
        extra={"note": "unit"},
    )
    data = json.loads(open(path).read())
    assert data["profile"] == "phase2/both"
    assert data["base_file_name"] == "tqc_x"
    assert data["config_hashes"]["train"].startswith("sha256:")
    assert data["overrides"] == {"risk_map_reward_enabled": True}
    assert data["note"] == "unit"


def test_resume_reuses_existing_run_dir(tmp_path):
    root = str(tmp_path)
    # Seed an existing new-structure run with a checkpoint.
    run = os.path.join(root, "runtime", "experiments",
                       "20260101_000000_tqc_x_seed0")
    models = os.path.join(run, "models")
    os.makedirs(models)
    open(os.path.join(models, "tqc_x_seed_0_20260101_actor.pth"), "w").close()

    layout = RunManager(root, "tqc_x", 0, load_model=True).resolve()
    assert layout["run_dir"] == run
    assert not layout["is_fresh"]


def test_legacy_layout_gets_no_manifest(tmp_path):
    root = str(tmp_path)
    legacy_models = os.path.join(root, "runtime", "tqc", "seed_0", "pytorch_models")
    os.makedirs(legacy_models)
    open(os.path.join(legacy_models, "tqc_x_seed_0_20250101_actor.pth"), "w").close()

    mgr = RunManager(root, "tqc_x", 0, load_model=True)
    layout = mgr.resolve()
    assert layout["is_legacy"]
    assert mgr.write_profile_manifest(profile_name="p") == ""  # never touch legacy dirs


def test_latest_checkpoint_prefix_ignores_mid_episode_snapshots(tmp_path):
    d = str(tmp_path)
    open(os.path.join(d, "tqc_x_seed_0_20260101_checkpoint_actor.pth"), "w").close()
    assert latest_checkpoint_prefix(d, "tqc_x", 0) == ""
    open(os.path.join(d, "tqc_x_seed_0_20260102_actor.pth"), "w").close()
    assert latest_checkpoint_prefix(d, "tqc_x", 0) == "tqc_x_seed_0_20260102"


def test_checkpoint_manager_prefers_new_layout_over_legacy(tmp_path):
    root = str(tmp_path)
    legacy_models = os.path.join(root, "runtime", "tqc", "seed_0", "pytorch_models")
    os.makedirs(legacy_models)
    open(os.path.join(legacy_models, "tqc_x_seed_0_20250101_actor.pth"), "w").close()
    new_models = os.path.join(root, "runtime", "experiments",
                              "20260101_000000_tqc_x_seed0", "models")
    os.makedirs(new_models)
    open(os.path.join(new_models, "tqc_x_seed_0_20260101_actor.pth"), "w").close()

    state = CheckpointManager(package_root=root).describe_resume_state("tqc_x", 0)
    assert state.resumable and state.layout_kind == "new"

    # Remove the new-layout run -> falls back to legacy.
    os.remove(os.path.join(new_models, "tqc_x_seed_0_20260101_actor.pth"))
    state = CheckpointManager(package_root=root).describe_resume_state("tqc_x", 0)
    assert state.resumable and state.layout_kind == "legacy"
