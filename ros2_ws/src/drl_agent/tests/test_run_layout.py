"""ROS-free unit tests for utils/run_layout.py (RUN_LAYOUT:
runtime/experiments/<run_id>/{configs,logs,models,analysis} per-run structure).

Covers:
  * run_id format: <YYYYMMDD_HHMMSS>_<base_file_name>_seed<seed>
  * tagged_filename: empty tag -> plain name, non-empty tag -> legacy "_<tag>"
  * resolve_run_layout: fresh run (load_model=False) always mints a new dir;
    resume (load_model=True) reuses an existing new-structure dir, else falls
    back to an existing legacy checkpoint (WITHOUT migrating it), else mints
    a new dir when nothing exists to resume.
  * create_run_dirs / copy_run_configs behave correctly for both layouts.
"""

import os

import drl_agent.training.run_layout as rl


def _touch_checkpoint(run_dir, subdir, base_file_name, seed, ts="20260701_000000"):
    """Create a fake, resumable (non-mid-episode) actor checkpoint file so a
    new/legacy run dir is recognised as having something to resume from."""
    models_dir = os.path.join(run_dir, subdir)
    os.makedirs(models_dir, exist_ok=True)
    open(os.path.join(
        models_dir, f"{base_file_name}_seed_{seed}_{ts}_actor.pth"), "w").close()


# --------------------------------------------------------------------------- #
#  make_run_id / tagged_filename                                                #
# --------------------------------------------------------------------------- #

def test_make_run_id_format():
    run_id = rl.make_run_id("tqc_phase2_both", 0, ts="20260726_101200")
    assert run_id == "20260726_101200_tqc_phase2_both_seed0"


def test_make_run_id_defaults_to_now(monkeypatch):
    # ts=None -> uses datetime.now(); just check the shape, not the exact value.
    run_id = rl.make_run_id("tqc", 3)
    assert run_id.endswith("_tqc_seed3")
    ts_part = run_id.split("_tqc_seed3")[0]
    assert len(ts_part) == len("20260726_101200")


def test_tagged_filename_empty_tag_is_plain():
    assert rl.tagged_filename("episode_rewards", ".csv", "") == "episode_rewards.csv"


def test_tagged_filename_nonempty_tag_is_legacy_suffixed():
    assert (rl.tagged_filename("episode_rewards", ".csv", "20260726_101200")
            == "episode_rewards_20260726_101200.csv")


# --------------------------------------------------------------------------- #
#  find_latest_new_run_dir / legacy_has_checkpoint                              #
# --------------------------------------------------------------------------- #

def test_find_latest_new_run_dir_none_when_absent(tmp_path):
    assert rl.find_latest_new_run_dir(str(tmp_path), "tqc_both", 0) is None


def test_find_latest_new_run_dir_picks_most_recent(tmp_path):
    root = rl.experiments_root(str(tmp_path))
    older = os.path.join(root, "20260101_000000_tqc_both_seed0")
    newer = os.path.join(root, "20260726_101200_tqc_both_seed0")
    other_seed = os.path.join(root, "20260726_101200_tqc_both_seed1")
    _touch_checkpoint(older, "models", "tqc_both", 0)
    _touch_checkpoint(newer, "models", "tqc_both", 0)
    _touch_checkpoint(other_seed, "models", "tqc_both", 1)
    # mtimes deliberately REVERSED vs. run_id order: proves the pick is by the
    # run_id's own timestamp prefix, not directory mtime (2026-07 review fix).
    os.utime(older, (2000, 2000))
    os.utime(newer, (1000, 1000))

    found = rl.find_latest_new_run_dir(str(tmp_path), "tqc_both", 0)
    assert found == newer


def test_find_latest_new_run_dir_skips_checkpointless_dirs(tmp_path):
    # A dir with no checkpoint (failed/smoke-test/empty run) must never be
    # picked as a resume target -- 2026-07 review: this used to block the
    # legacy-checkpoint fallback below it (see
    # test_resolve_resume_falls_back_to_legacy_ignoring_empty_new_dir).
    root = rl.experiments_root(str(tmp_path))
    empty_dir = os.path.join(root, "20260726_101200_tqc_both_seed0")
    os.makedirs(empty_dir)  # models/ never created -> nothing to resume

    assert rl.find_latest_new_run_dir(str(tmp_path), "tqc_both", 0) is None


def test_find_latest_new_run_dir_skips_dir_with_only_mid_episode_checkpoint(tmp_path):
    root = rl.experiments_root(str(tmp_path))
    d = os.path.join(root, "20260726_101200_tqc_both_seed0")
    models_dir = os.path.join(d, "models")
    os.makedirs(models_dir)
    open(os.path.join(models_dir, "tqc_both_seed_0_20260701_checkpoint_actor.pth"), "w").close()

    assert rl.find_latest_new_run_dir(str(tmp_path), "tqc_both", 0) is None


def test_legacy_has_checkpoint_false_when_absent(tmp_path):
    assert rl.legacy_has_checkpoint(str(tmp_path), "tqc_agent", 0) is False


def test_legacy_has_checkpoint_true_when_actor_file_present(tmp_path):
    models_dir = os.path.join(rl.legacy_run_dir(str(tmp_path), 0), "pytorch_models")
    os.makedirs(models_dir)
    open(os.path.join(models_dir, "tqc_agent_seed_0_20260701_actor.pth"), "w").close()
    assert rl.legacy_has_checkpoint(str(tmp_path), "tqc_agent", 0) is True


def test_legacy_has_checkpoint_ignores_mid_episode_checkpoint_file(tmp_path):
    # A "_checkpoint_actor.pth" is a mid-episode safety save, not a resumable
    # end-of-episode checkpoint -- mirrors train_tqc_base._find_latest_prefix.
    models_dir = os.path.join(rl.legacy_run_dir(str(tmp_path), 0), "pytorch_models")
    os.makedirs(models_dir)
    open(os.path.join(models_dir, "tqc_agent_seed_0_20260701_checkpoint_actor.pth"), "w").close()
    assert rl.legacy_has_checkpoint(str(tmp_path), "tqc_agent", 0) is False


# --------------------------------------------------------------------------- #
#  resolve_run_layout                                                           #
# --------------------------------------------------------------------------- #

def test_resolve_fresh_run_always_mints_new_dir(tmp_path):
    layout = rl.resolve_run_layout(
        str(tmp_path), "tqc_both", 0, load_model=False, ts="20260726_101200")
    assert layout["is_legacy"] is False
    assert layout["is_fresh"] is True
    assert layout["run_id"] == "20260726_101200_tqc_both_seed0"
    assert layout["run_dir"] == os.path.join(
        rl.experiments_root(str(tmp_path)), "20260726_101200_tqc_both_seed0")
    assert layout["configs_dir"].endswith("configs")
    assert layout["models_dir"].endswith("models")


def test_resolve_fresh_run_ignores_existing_legacy_checkpoint(tmp_path):
    # load_model=False -> new structure regardless of what legacy history exists.
    models_dir = os.path.join(rl.legacy_run_dir(str(tmp_path), 0), "pytorch_models")
    os.makedirs(models_dir)
    open(os.path.join(models_dir, "tqc_both_seed_0_20260701_actor.pth"), "w").close()

    layout = rl.resolve_run_layout(str(tmp_path), "tqc_both", 0, load_model=False)
    assert layout["is_legacy"] is False


def test_resolve_resume_reuses_existing_new_run_dir(tmp_path):
    root = rl.experiments_root(str(tmp_path))
    existing = os.path.join(root, "20260101_000000_tqc_both_seed0")
    _touch_checkpoint(existing, "models", "tqc_both", 0)

    layout = rl.resolve_run_layout(str(tmp_path), "tqc_both", 0, load_model=True)
    assert layout["is_legacy"] is False
    assert layout["is_fresh"] is False
    assert layout["run_dir"] == existing
    assert layout["run_id"] == "20260101_000000_tqc_both_seed0"


def test_resolve_resume_falls_back_to_legacy_when_no_new_dir(tmp_path):
    models_dir = os.path.join(rl.legacy_run_dir(str(tmp_path), 0), "pytorch_models")
    os.makedirs(models_dir)
    open(os.path.join(models_dir, "tqc_both_seed_0_20260701_actor.pth"), "w").close()

    layout = rl.resolve_run_layout(str(tmp_path), "tqc_both", 0, load_model=True)
    assert layout["is_legacy"] is True
    assert layout["run_dir"] == rl.legacy_run_dir(str(tmp_path), 0)
    assert layout["configs_dir"] is None
    assert layout["analysis_dir"] is None
    assert layout["models_dir"].endswith("pytorch_models")


def test_resolve_resume_prefers_new_dir_over_legacy_when_both_exist(tmp_path):
    # A run that has already migrated forward (new dir exists) must not fall
    # back to the stale legacy checkpoint.
    root = rl.experiments_root(str(tmp_path))
    existing_new = os.path.join(root, "20260101_000000_tqc_both_seed0")
    _touch_checkpoint(existing_new, "models", "tqc_both", 0)
    _touch_checkpoint(rl.legacy_run_dir(str(tmp_path), 0), "pytorch_models", "tqc_both", 0)

    layout = rl.resolve_run_layout(str(tmp_path), "tqc_both", 0, load_model=True)
    assert layout["is_legacy"] is False
    assert layout["run_dir"] == existing_new


def test_resolve_resume_falls_back_to_legacy_ignoring_empty_new_dir(tmp_path):
    # 2026-07 review: an empty/checkpointless new-structure dir (failed run,
    # smoke test, ...) must NOT block the legacy-checkpoint fallback.
    root = rl.experiments_root(str(tmp_path))
    empty_new_dir = os.path.join(root, "20260726_101200_tqc_both_seed0")
    os.makedirs(empty_new_dir)  # no models/ -- nothing to resume from here
    _touch_checkpoint(rl.legacy_run_dir(str(tmp_path), 0), "pytorch_models", "tqc_both", 0)

    layout = rl.resolve_run_layout(str(tmp_path), "tqc_both", 0, load_model=True)
    assert layout["is_legacy"] is True
    assert layout["run_dir"] == rl.legacy_run_dir(str(tmp_path), 0)


def test_resolve_resume_with_nothing_to_resume_mints_new_dir(tmp_path):
    layout = rl.resolve_run_layout(
        str(tmp_path), "tqc_both", 0, load_model=True, ts="20260726_101200")
    assert layout["is_legacy"] is False
    assert layout["is_fresh"] is True
    assert layout["run_id"] == "20260726_101200_tqc_both_seed0"


# --------------------------------------------------------------------------- #
#  create_run_dirs / copy_run_configs                                           #
# --------------------------------------------------------------------------- #

def test_create_run_dirs_new_layout_creates_all_four_subdirs(tmp_path):
    layout = rl.resolve_run_layout(
        str(tmp_path), "tqc_both", 0, load_model=False, ts="20260726_101200")
    rl.create_run_dirs(layout)
    for key in ("run_dir", "configs_dir", "logs_dir", "models_dir", "analysis_dir"):
        assert os.path.isdir(layout[key]), f"{key} not created"


def test_create_run_dirs_legacy_layout_creates_original_four_dirs_only(tmp_path):
    legacy_models = os.path.join(rl.legacy_run_dir(str(tmp_path), 0), "pytorch_models")
    os.makedirs(legacy_models)
    open(os.path.join(legacy_models, "tqc_both_seed_0_20260701_actor.pth"), "w").close()
    layout = rl.resolve_run_layout(str(tmp_path), "tqc_both", 0, load_model=True)

    rl.create_run_dirs(layout)
    assert os.path.isdir(layout["logs_dir"])
    assert os.path.isdir(layout["models_dir"])
    assert os.path.isdir(layout["final_models_dir"])
    assert os.path.isdir(layout["results_dir"])
    # No new-structure dirs were invented inside the existing legacy run dir.
    assert not os.path.isdir(os.path.join(layout["run_dir"], "configs"))
    assert not os.path.isdir(os.path.join(layout["run_dir"], "analysis"))


def test_copy_run_configs_copies_existing_files(tmp_path):
    src_dir = tmp_path / "src_configs"
    src_dir.mkdir()
    train_cfg = src_dir / "train_tqc_config.yaml"
    train_cfg.write_text("train_settings: {}\n")
    dest_dir = tmp_path / "configs"
    dest_dir.mkdir()

    result = rl.copy_run_configs(str(dest_dir), {
        "train": str(train_cfg),
        "hyperparameters": "",
        "curriculum": str(tmp_path / "does_not_exist.yaml"),
    })
    assert result["train"] == str(dest_dir / "train_tqc_config.yaml")
    assert os.path.isfile(result["train"])
    assert result["hyperparameters"] == ""
    assert result["curriculum"] == ""


def test_copy_run_configs_preserves_content(tmp_path):
    src = tmp_path / "hyperparameters_tqc.yaml"
    src.write_text("hyperparameters:\n  discount: 0.99\n")
    dest_dir = tmp_path / "configs"
    dest_dir.mkdir()

    result = rl.copy_run_configs(str(dest_dir), {"hyperparameters": str(src)})
    with open(result["hyperparameters"]) as f:
        assert f.read() == "hyperparameters:\n  discount: 0.99\n"


# --------------------------------------------------------------------------- #
#  write_csv_header_if_new (2026-07 review: resume must not truncate logs)      #
# --------------------------------------------------------------------------- #

def test_write_csv_header_if_new_writes_header_for_fresh_file(tmp_path):
    path = str(tmp_path / "episode_rewards.csv")
    wrote = rl.write_csv_header_if_new(path, ["a", "b"])
    assert wrote is True
    with open(path, newline="") as f:
        assert f.read() == "a,b\r\n"


def test_write_csv_header_if_new_does_not_touch_existing_file(tmp_path):
    path = str(tmp_path / "episode_rewards.csv")
    with open(path, "w", newline="") as f:
        f.write("a,b\r\n1,2\r\n3,4\r\n")

    wrote = rl.write_csv_header_if_new(path, ["a", "b"])
    assert wrote is False
    with open(path, newline="") as f:
        # The pre-existing rows (simulating a resumed run's prior history)
        # must survive byte-for-byte -- this is the exact truncation bug the
        # 2026-07 review caught.
        assert f.read() == "a,b\r\n1,2\r\n3,4\r\n"
