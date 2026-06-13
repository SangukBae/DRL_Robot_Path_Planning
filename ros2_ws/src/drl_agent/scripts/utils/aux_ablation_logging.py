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
import json
import os
import subprocess

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


def git_commit(repo_dir=None) -> str:
    """Best-effort short git hash; returns '' on any failure (never raises)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir, stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8", "ignore").strip()
    except Exception:
        return ""


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
                       env_aux=None, aux_eval=None, file_name="", repo_dir=None):
    """Write logs/run_manifest.json once at run start (config-tracking).

    Lets per-seed aggregation later avoid mixing configs.  Returns the path.

    ``environment_config_file`` / ``environment_config_sha1`` / ``env_aux`` should
    come from the running env node (via ROS parameters), not a re-discovered
    file, so the manifest records the TRUE env config that produced the run.
    """
    m = agent_aux_meta(agent)
    manifest = {
        "seed": int(seed) if seed is not None else None,
        "aux_enabled": m["aux_enabled"],
        "aux_version": m["aux_version"],
        "num_sectors": m["num_sectors"],
        "horizons_sec": m["horizons_sec"],
        "loss_weight": m["loss_weight"],
        "min_distance_loss_weight": m["min_distance_loss_weight"],
        "use_distributional_aux": m["use_distributional_aux"],
        "temporal_enabled": m["temporal_enabled"],
        "train_config_file": str(train_config_file or ""),
        "environment_config_file": str(environment_config_file or ""),
        "environment_config_sha1": str(environment_config_sha1 or ""),
        # What the env node reports it is ACTUALLY running (label geometry).
        "env_aux": dict(env_aux or {}),
        # Formal aux-evaluation config (thresholds / reference speeds) so paper
        # numbers are reproducible without re-reading the trainer defaults.
        "aux_eval": dict(aux_eval or {}),
        "file_name": str(file_name or ""),
        "git_commit": git_commit(repo_dir),
    }
    path = os.path.join(log_dir, "run_manifest.json")
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception:
        pass
    return path


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
    """Append-only eval summary CSV (logs/eval_summary_<run>.csv)."""

    def __init__(self, log_dir, run_tag, seed, agent):
        self.path = os.path.join(log_dir, f"eval_summary_{run_tag}.csv")
        self._seed = seed
        self._agent = agent
        if not os.path.isfile(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(EVAL_SUMMARY_HEADER)

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
