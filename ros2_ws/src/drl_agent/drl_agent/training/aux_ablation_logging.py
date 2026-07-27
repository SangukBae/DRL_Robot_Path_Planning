# AUX_ABLATION: logging helpers for the auxiliary-prediction ablation study.
#
# Goal: let "aux_prediction.enabled=false" and "=true" runs be compared at paper
# quality (final performance, sample efficiency, aux-loss curve, generalization)
# WITHOUT touching the learning loop.  Everything here is pure I/O metadata; it
# reuses the existing CSV/JSON columns and only stamps run identity (seed,
# aux_enabled, aux_version) + a dedicated eval-summary CSV and a run manifest.
#
# All run-identity logging funnels through this module so the trainers stay a
# one-line integration each, and so the schema lives in ONE place.

import csv
import hashlib
import json
import os
import subprocess

import drl_agent.training.run_layout as run_layout

# Common run-identity columns appended to existing per-episode CSVs.
META_COLUMN_NAMES = ["seed", "aux_enabled", "aux_version"]


def agent_aux_meta(agent) -> dict:
    """Describe an agent's auxiliary-prediction config (null-safe).

    Works for both aux-enabled and aux-disabled agents, and even for agents that
    predate the aux feature (every field falls back to a safe default).
    """
    cfg = getattr(agent, "aux_cfg", None)
    enabled = bool(getattr(agent, "aux_enabled", False))
    return {
        "aux_enabled": int(enabled),
        "aux_version": int(getattr(cfg, "version", 0)) if cfg is not None else 0,
        "num_sectors": int(getattr(cfg, "num_sectors", 0)) if cfg is not None else 0,
        "horizons_sec": list(getattr(cfg, "horizons_sec", [])) if cfg is not None else [],
        "loss_weight": float(getattr(cfg, "loss_weight", 0.0)) if cfg is not None else 0.0,
        "min_distance_loss_weight": (
            float(getattr(cfg, "min_distance_loss_weight", 0.0)) if cfg is not None else 0.0
        ),
        "use_distributional_aux": (
            bool(getattr(cfg, "use_distributional_aux", False)) if cfg is not None else False
        ),
        "temporal_enabled": (
            bool(getattr(cfg, "temporal_enabled", False)) if cfg is not None else False
        ),
    }


def meta_columns(seed, agent) -> list:
    """Return the 3 common CSV meta values: [seed, aux_enabled, aux_version]."""
    m = agent_aux_meta(agent)
    return [int(seed) if seed is not None else -1, m["aux_enabled"], m["aux_version"]]


def _run_git(args, cwd):
    """Run a git command in `cwd`, returning stripped stdout (raises on failure)."""
    out = subprocess.check_output(
        ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL
    )
    return out.decode("utf-8", "ignore").strip()


def _find_git_repo(start):
    """Walk up from `start` to the first directory that is a git work tree.

    Returns the repo dir, or None. This is the fix for the empty ``git_commit``
    in past manifests: the trainer's ``__file__`` resolves to the INSTALL copy
    (``.../install/drl_agent/share/...``) which is NOT under ``.git``, so a git
    call rooted there silently failed. Walking up (and from realpath, which a
    symlink-install resolves back into the source tree) finds the real repo.
    """
    try:
        d = os.path.abspath(start)
    except Exception:
        return None
    while True:
        dotgit = os.path.join(d, ".git")
        if os.path.isdir(dotgit) or os.path.isfile(dotgit):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def git_info(repo_dir=None) -> dict:
    """Best-effort git identity for the run (never raises).

    Records full + short commit, branch, and the DIRTY-worktree flag — so a run
    made from uncommitted edits is never mistaken for the committed code. Tries,
    in order: an explicit ``repo_dir``, this module's real source path, and CWD.
    """
    info = {
        "commit": "", "commit_short": "", "branch": "",
        "dirty": None, "describe": "", "repo_dir": "",
    }
    candidates = []
    if repo_dir:
        candidates.append(repo_dir)
    candidates.append(os.path.dirname(os.path.realpath(__file__)))
    candidates.append(os.getcwd())
    for c in candidates:
        repo = _find_git_repo(c)
        if not repo:
            continue
        # A `.git` may exist but not be a USABLE repo (e.g. an empty repo with no
        # commits, or a stray `.git` above a temp dir). If HEAD can't be resolved,
        # keep trying the remaining candidates instead of giving up with an empty
        # commit — this is exactly the install-copy case the empty-commit
        # manifests hit (repo_dir was a non-repo install path).
        try:
            commit = _run_git(["rev-parse", "HEAD"], repo)
        except Exception:
            continue
        if not commit:
            continue
        info["repo_dir"] = repo
        info["commit"] = commit
        try:
            info["commit_short"] = _run_git(["rev-parse", "--short", "HEAD"], repo)
        except Exception:
            info["commit_short"] = commit[:7]
        try:
            info["branch"] = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo)
        except Exception:
            pass
        try:
            info["dirty"] = bool(_run_git(["status", "--porcelain"], repo).strip())
        except Exception:
            pass
        try:
            info["describe"] = _run_git(
                ["describe", "--always", "--dirty", "--tags"], repo)
        except Exception:
            info["describe"] = info["commit_short"]
        return info
    return info


def git_commit(repo_dir=None) -> str:
    """Best-effort short git hash; returns '' on any failure (never raises)."""
    return git_info(repo_dir).get("commit_short", "")


def file_sha1(path) -> str:
    """SHA-1 of a file's bytes; '' on any failure. Lets the manifest pin the
    EXACT config content that produced a run, not just its path (two runs can
    share a path while the file changed underneath them)."""
    if not path:
        return ""
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def runtime_versions() -> dict:
    """Best-effort interpreter / library / GPU versions for the manifest."""
    import platform
    out = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import numpy as _np
        out["numpy"] = _np.__version__
    except Exception:
        pass
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            try:
                out["cudnn"] = torch.backends.cudnn.version()
            except Exception:
                pass
        out["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        out["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    except Exception:
        pass
    return out


# Manifest fields that define a run's aux configuration.  Aggregation groups
# runs by exactly these (the env-side geometry is guaranteed equal to the agent
# values by the wire fail-fast, so the agent view is the running config).
MANIFEST_CONFIG_KEYS = [
    "aux_enabled", "aux_version", "num_sectors", "horizons_sec",
    "loss_weight", "min_distance_loss_weight", "use_distributional_aux",
    "temporal_enabled",
]


def write_run_manifest(log_dir, *, seed, agent, train_config_file="",
                       environment_config_file="", environment_config_sha1="",
                       env_aux=None, aux_eval=None, file_name="", repo_dir=None,
                       run_tag="", curriculum_config_file="",
                       hyperparameters_file="", seed_source="",
                       determinism=None, human_rng=None, extra=None,
                       # RUN_LAYOUT: runtime/experiments/<run_id>/ identity +
                       # config provenance (original path -> copied-into-run-dir
                       # path). All default "" / None so a caller that predates
                       # this (e.g. a non-curriculum trainer) still gets a valid
                       # manifest with these fields blank.
                       run_id="", run_dir="", logs_dir="", models_dir="",
                       configs_dir="", analysis_dir="",
                       copied_train_config_file="",
                       copied_curriculum_config_file="",
                       copied_hyperparameters_file="",
                       copied_environment_config_file=""):
    """Write the per-run config/identity manifest at run start (config-tracking).

    Lets per-seed aggregation avoid mixing configs AND lets us answer "what code
    + config produced this run?" after the fact — the gap that made the
    20260613/14 vs 20260615 comparison unprovable (the old manifests recorded an
    empty git_commit and were overwritten each run).

    Two files are written:
      * ``run_manifest.json``                  — latest run (back-compat readers)
      * ``run_manifest_<run_tag>.json``        — PER-RUN copy (history preserved,
        legacy layout only -- a RUN_LAYOUT run_tag is empty so this second copy
        is skipped; the run folder itself is already unique)

    ``environment_config_file`` / ``environment_config_sha1`` / ``env_aux`` should
    come from the running env node (via ROS parameters), not a re-discovered
    file, so the manifest records the TRUE env config that produced the run.

    RUN_LAYOUT fields (``run_id``/``run_dir``/``logs_dir``/``models_dir``/
    ``configs_dir``/``analysis_dir``, and the ``copied_*_config_file`` paths)
    record the runtime/experiments/<run_id>/ structure this run used, alongside
    the ``*_config_file`` (original source path) fields already above — so a
    reader can see BOTH where a config was read from AND where its frozen copy
    for this run lives.
    """
    m = agent_aux_meta(agent)
    g = git_info(repo_dir)
    manifest = {
        "run_tag": str(run_tag or ""),
        # RUN_LAYOUT identity (blank for a legacy-layout run / an older caller).
        "run_id": str(run_id or ""),
        "run_dir": str(run_dir or ""),
        "logs_dir": str(logs_dir or ""),
        "models_dir": str(models_dir or ""),
        "configs_dir": str(configs_dir or ""),
        "analysis_dir": str(analysis_dir or ""),
        "seed": int(seed) if seed is not None else None,
        "seed_source": str(seed_source or ""),
        "aux_enabled": m["aux_enabled"],
        "aux_version": m["aux_version"],
        "num_sectors": m["num_sectors"],
        "horizons_sec": m["horizons_sec"],
        "loss_weight": m["loss_weight"],
        "min_distance_loss_weight": m["min_distance_loss_weight"],
        "use_distributional_aux": m["use_distributional_aux"],
        "temporal_enabled": m["temporal_enabled"],
        # ── Config provenance: path + content hash for EVERY config a run reads,
        #    so a changed-but-same-path file can never silently alter a rerun. ──
        "train_config_file": str(train_config_file or ""),
        "train_config_sha1": file_sha1(train_config_file),
        "curriculum_config_file": str(curriculum_config_file or ""),
        "curriculum_config_sha1": file_sha1(curriculum_config_file),
        "hyperparameters_file": str(hyperparameters_file or ""),
        "hyperparameters_sha1": file_sha1(hyperparameters_file),
        "environment_config_file": str(environment_config_file or ""),
        # env sha1 comes from the running env node (authoritative); fall back to
        # hashing the file ourselves only if the node did not report one.
        "environment_config_sha1": (
            str(environment_config_sha1 or "") or file_sha1(environment_config_file)
        ),
        # RUN_LAYOUT: explicit original-vs-copied path pairs (item 6 of the
        # runtime/experiments/<run_id>/ spec). "original_*" duplicates the
        # "*_config_file" values above under the requested names; "copied_*"
        # is where that file was frozen inside THIS run's configs/ dir ("" for
        # a legacy-layout run, which has no configs/ dir -- see run_layout.py).
        "original_train_config_file": str(train_config_file or ""),
        "copied_train_config_file": str(copied_train_config_file or ""),
        "original_curriculum_config_file": str(curriculum_config_file or ""),
        "copied_curriculum_config_file": str(copied_curriculum_config_file or ""),
        "original_hyperparameters_file": str(hyperparameters_file or ""),
        "copied_hyperparameters_file": str(copied_hyperparameters_file or ""),
        "original_environment_config_file": str(environment_config_file or ""),
        "copied_environment_config_file": str(copied_environment_config_file or ""),
        # What the env node reports it is ACTUALLY running (label geometry).
        "env_aux": dict(env_aux or {}),
        # Formal aux-evaluation config (thresholds / reference speeds) so paper
        # numbers are reproducible without re-reading the trainer defaults.
        "aux_eval": dict(aux_eval or {}),
        # Reproducibility knobs actually applied this run (cudnn / det-algos).
        "determinism": dict(determinism or {}),
        # Pedestrian RNG reproducibility policy (sub-stream isolation + resume
        # contract). Values come from the running env node when available.
        "human_rng": dict(human_rng or {}),
        "file_name": str(file_name or ""),
        # ── Code identity ──
        "git_commit": g.get("commit", ""),
        "git_commit_short": g.get("commit_short", ""),
        "git_branch": g.get("branch", ""),
        "git_dirty": g.get("dirty", None),
        "git_describe": g.get("describe", ""),
        "git_repo_dir": g.get("repo_dir", ""),
        # ── Runtime identity ──
        "versions": runtime_versions(),
    }
    if extra:
        manifest["extra"] = dict(extra)
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        if run_tag:
            per_run = os.path.join(log_dir, f"run_manifest_{run_tag}.json")
            with open(per_run, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
    except Exception:
        pass
    return os.path.join(log_dir, "run_manifest.json")


# Paper-comparison eval summary: one row per periodic evaluation, so a single
# run's learning curve AND the aux on/off comparison are both readable here.
# Columns are APPEND-ONLY: the trailing main-policy (stl/psc/h_coll_rate/
# lidar_clearance_rate) and aux-eval (aux_*) columns were added later; older
# readers of the leading columns are unaffected. `psc` / `h_coll_rate` are blank
# when the ENV emits no human labels; `aux_*` are blank when the agent aux head
# is disabled.
EVAL_SUMMARY_HEADER = [
    "seed", "aux_enabled", "aux_version",
    "eval_global_t", "curriculum_stage", "eval_eps",
    "success_rate", "collision_rate", "timeout_rate",
    "mean_reward", "mean_final_goal_dist",
    "spl", "cte", "jerk",
    # main-policy extras. `psc` / `h_coll_rate` are LABEL-derived (blank when the
    # env emits no human labels); `lidar_clearance_rate` is the state-stream
    # clearance proxy (always present, not human PSC).
    "stl", "psc", "h_coll_rate", "lidar_clearance_rate",
    # formal aux-eval metrics (blank when the agent aux head is disabled)
    "aux_risk_rmse", "aux_min_dist_mae_m", "aux_peak_sector_acc", "aux_near_event_f1",
]


class EvalSummaryCSV:
    """Append-only eval summary CSV (logs/eval_summary_<run>.csv, or plain
    logs/eval_summary.csv for a RUN_LAYOUT new-structure run -- see
    run_layout.tagged_filename)."""

    def __init__(self, log_dir, run_tag, seed, agent):
        self.path = os.path.join(
            log_dir, run_layout.tagged_filename("eval_summary", ".csv", run_tag))
        self._seed = seed
        self._agent = agent
        run_layout.write_csv_header_if_new(self.path, EVAL_SUMMARY_HEADER)

    def append(self, *, eval_global_t, curriculum_stage, eval_eps, metrics):
        """metrics: the merged eval dict (base rates + aggregated paper metrics).

        Reads success/collision/timeout_rate, mean_reward, mean_goal_dist and the
        aggregated spl / mean_cross_track_error_m / mean_action_jerk.
        """
        seed, aux_en, aux_v = meta_columns(self._seed, self._agent)

        def g(key, default=0.0):
            try:
                return round(float(metrics.get(key, default)), 4)
            except Exception:
                return default

        def gblank(key):
            # Blank (not 0.0) when a metric is absent — e.g. aux_* for aux-off
            # runs — so a missing value is never mistaken for a real zero.
            if key not in metrics or metrics.get(key) is None:
                return ""
            try:
                v = float(metrics[key])
                return "" if v != v else round(v, 6)   # NaN -> blank
            except Exception:
                return ""

        row = [
            seed, aux_en, aux_v,
            int(eval_global_t), int(curriculum_stage), int(eval_eps),
            g("success_rate"), g("collision_rate"), g("timeout_rate"),
            g("mean_reward"), g("mean_goal_dist"),
            g("spl"), g("mean_cross_track_error_m"), g("mean_action_jerk"),
            g("stl"), gblank("psc"), gblank("h_coll_rate"), g("lidar_clearance_rate"),
            gblank("aux_risk_rmse"), gblank("aux_min_dist_mae_m"),
            gblank("aux_peak_sector_acc"), gblank("aux_near_event_f1"),
        ]
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)
