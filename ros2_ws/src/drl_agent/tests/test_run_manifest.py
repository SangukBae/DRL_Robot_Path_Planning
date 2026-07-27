"""ROS-free unit tests for utils/aux_ablation_logging.py run-manifest tracking.

These lock in the reproducibility-provenance fix: the old manifests recorded an
empty git_commit (git was invoked from the non-repo install copy) and were
overwritten each run, so a same-seed comparison could not be traced back to the
exact code + config that produced it.
"""

import json
import os
import tempfile

import drl_agent.training.aux_ablation_logging as al
import drl_agent.evaluation.analysis.aux_ablation_summary as asum


class _FakeAgent:
    aux_cfg = None
    aux_enabled = False


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def test_file_sha1_hashes_and_is_safe_on_missing():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("hello: world\n")
        path = f.name
    try:
        h1 = al.file_sha1(path)
        assert h1 and len(h1) == 40       # sha1 hexdigest
        assert al.file_sha1(path) == h1   # deterministic
    finally:
        os.unlink(path)
    assert al.file_sha1("/no/such/file") == ""
    assert al.file_sha1("") == ""


def test_git_info_finds_repo_from_source_tree():
    # This module lives inside the repo, so walking up from its real path must
    # find a git work tree (the exact failure the empty-commit manifests had).
    info = al.git_info(os.path.dirname(os.path.realpath(al.__file__)))
    assert info["repo_dir"]                       # a repo was found
    assert info["commit"] and info["commit_short"]
    assert isinstance(info["dirty"], bool)


def test_find_git_repo_walks_up_to_a_dotgit():
    # Inside the repo: walking up from this test file finds a dir that actually
    # contains `.git` (the walk-up that the empty-commit manifests never did).
    here = os.path.dirname(os.path.realpath(__file__))
    repo = al._find_git_repo(here)
    assert repo and os.path.exists(os.path.join(repo, ".git"))
    # A path with no `.git` ancestor returns None (root has none). We assert via
    # a constructed root-only walk so the result is independent of where the
    # sandbox places temp dirs (some CI roots carry a stray `.git`).
    assert al._find_git_repo("/") is None or os.path.exists("/.git")


def test_git_info_recovers_when_repo_dir_is_a_non_repo_install_copy():
    # Regression: past manifests recorded an EMPTY git_commit because git was
    # invoked from the install copy (.../install/.../share/...), which is NOT a
    # git work tree. git_info must fall back (module realpath / cwd) and still
    # resolve the commit instead of silently returning "".
    source_repo = al.git_info(os.path.dirname(os.path.realpath(al.__file__)))
    if not source_repo["commit"]:
        import pytest
        pytest.skip("tests are not running inside a git work tree")
    with tempfile.TemporaryDirectory() as non_repo_install_copy:
        info = al.git_info(non_repo_install_copy)   # mimics the install path
        assert info["commit"] == source_repo["commit"]      # NOT empty
        assert info["commit_short"] == source_repo["commit_short"]


def test_write_run_manifest_records_commit_for_non_repo_repo_dir():
    # End-to-end: even when the trainer passes a non-repo install dir as
    # repo_dir, the written manifest must carry a non-empty git_commit (the field
    # that was blank in every old run_manifest.json).
    source_repo = al.git_info(os.path.dirname(os.path.realpath(al.__file__)))
    if not source_repo["commit"]:
        import pytest
        pytest.skip("tests are not running inside a git work tree")
    with tempfile.TemporaryDirectory() as d:
        install_copy = os.path.join(d, "install", "share", "drl_agent")
        os.makedirs(install_copy)
        al.write_run_manifest(d, seed=0, agent=_FakeAgent(), run_tag="R",
                              repo_dir=install_copy)
        m = json.load(open(os.path.join(d, "run_manifest_R.json")))
        assert m["git_commit"]            # non-empty
        assert m["git_commit_short"]


def test_write_run_manifest_writes_per_run_copy_and_hashes_configs():
    with tempfile.TemporaryDirectory() as d:
        cfg = os.path.join(d, "train.yaml")
        with open(cfg, "w") as f:
            f.write("train_settings: {seed: 0}\n")

        al.write_run_manifest(
            d, seed=0, agent=_FakeAgent(), run_tag="20991231_235959",
            train_config_file=cfg, curriculum_config_file=cfg,
            hyperparameters_file=cfg, environment_config_file=cfg,
            seed_source="config(0)", determinism={"requested": True},
        )

        files = set(os.listdir(d))
        # both the latest pointer AND the per-run history copy must exist
        assert "run_manifest.json" in files
        assert "run_manifest_20991231_235959.json" in files

        m = json.load(open(os.path.join(d, "run_manifest_20991231_235959.json")))
        assert m["run_tag"] == "20991231_235959"
        assert m["seed_source"] == "config(0)"
        # human_rng key always present (empty dict when not supplied)
        assert "human_rng" in m
        # every config a run reads is pinned by content hash, not just path
        assert len(m["train_config_sha1"]) == 40
        assert len(m["curriculum_config_sha1"]) == 40
        assert len(m["hyperparameters_sha1"]) == 40
        assert m["determinism"] == {"requested": True}
        assert "versions" in m and "python" in m["versions"]


def test_write_run_manifest_does_not_overwrite_distinct_run_tags():
    with tempfile.TemporaryDirectory() as d:
        al.write_run_manifest(d, seed=0, agent=_FakeAgent(), run_tag="RUN_A")
        al.write_run_manifest(d, seed=1, agent=_FakeAgent(), run_tag="RUN_B")
        # Distinct run tags keep distinct history files (the old single-file
        # manifest lost RUN_A here).
        a = json.load(open(os.path.join(d, "run_manifest_RUN_A.json")))
        b = json.load(open(os.path.join(d, "run_manifest_RUN_B.json")))
        assert a["seed"] == 0 and b["seed"] == 1


# ── downstream consumer: eval-summary ↔ manifest pairing ────────────────────

def test_manifest_records_human_rng_policy():
    # Priority 3/4: the manifest must record the pedestrian-RNG reproducibility
    # policy (sub-stream isolation + Option-B resume contract).
    with tempfile.TemporaryDirectory() as d:
        human_rng = {
            "enabled": True,
            "policy": "substream_isolated_from_global; per_episode_reseed; exact_resume_disabled",
            "base_seed": 0,
            "base_seed_source": "env /seed service = run seed (self.seed)",
            "resume_guarantee": "deterministic_per_checkpoint; NOT bit-exact continuation",
        }
        al.write_run_manifest(d, seed=0, agent=_FakeAgent(), run_tag="HR",
                              human_rng=human_rng)
        m = json.load(open(os.path.join(d, "run_manifest_HR.json")))
        assert m["human_rng"]["enabled"] is True
        assert "per_episode_reseed" in m["human_rng"]["policy"]
        assert "exact_resume_disabled" in m["human_rng"]["policy"]
        assert m["human_rng"]["base_seed"] == 0
        assert "deterministic_per_checkpoint" in m["human_rng"]["resume_guarantee"]


def test_run_tag_of_parses_eval_summary_name():
    assert asum._run_tag_of("/x/eval_summary_20260615_002749.csv") == "20260615_002749"
    assert asum._run_tag_of("/x/something_else.csv") == ""


def test_load_manifest_pairs_each_csv_with_its_own_run():
    # Two runs share one logs/ dir. The summary must pair RUN_A's CSV with
    # RUN_A's manifest, NOT the latest (RUN_B) shared run_manifest.json.
    with tempfile.TemporaryDirectory() as d:
        al.write_run_manifest(d, seed=0, agent=_FakeAgent(), run_tag="RUN_A")
        al.write_run_manifest(d, seed=1, agent=_FakeAgent(), run_tag="RUN_B")
        # shared run_manifest.json now reflects the latest run (RUN_B).
        assert json.load(open(os.path.join(d, "run_manifest.json")))["seed"] == 1

        man_a = asum._load_manifest(os.path.join(d, "eval_summary_RUN_A.csv"))
        man_b = asum._load_manifest(os.path.join(d, "eval_summary_RUN_B.csv"))
        assert man_a["seed"] == 0          # paired with its OWN run, not latest
        assert man_b["seed"] == 1


def test_load_manifest_refuses_mismatched_shared_manifest():
    # A CSV whose per-run manifest is missing must NOT be paired with a shared
    # manifest that belongs to a different run (silent mislabeling).
    with tempfile.TemporaryDirectory() as d:
        al.write_run_manifest(d, seed=1, agent=_FakeAgent(), run_tag="RUN_B")
        man = asum._load_manifest(os.path.join(d, "eval_summary_RUN_A.csv"))
        assert man is None


def test_load_manifest_falls_back_for_legacy_untagged_manifest():
    # Back-compat: a legacy run_manifest.json without a run_tag is still paired
    # with a CSV when no per-run manifest exists.
    with tempfile.TemporaryDirectory() as d:
        _write(os.path.join(d, "run_manifest.json"), json.dumps({"seed": 7}))
        man = asum._load_manifest(os.path.join(d, "eval_summary_RUN_A.csv"))
        assert man is not None and man["seed"] == 7
