#!/usr/bin/env python3
"""``dynamic_avoidance_metrics_<run_tag>.csv`` — single-file dynamic-obstacle
(pedestrian) avoidance diagnostic log for curriculum training.

Purpose
-------
Dynamic-avoidance-relevant signals used to be scattered across five CSVs
(``episode_metrics`` / ``episode_driving`` / ``curriculum_episode_rewards`` /
``curriculum_eval_per_map`` / ``eval_metrics``). This log **consolidates** the
subset that matters for diagnosing dynamic-obstacle avoidance and **adds**
privileged human-interaction / clutter / yield / near-event diagnostics, so the
analysis of "how well did the policy avoid pedestrians" needs ONLY this file.

The existing CSVs are left untouched (no format break); the columns below are
re-exposed here. Consolidation map (source column -> this file):

  * curriculum_episode_rewards: episode, global_t, curriculum_stage, map_type,
      total_reward, steps, goal_reached->success, collision, timeout,
      final_goal_dist_m, mean_gazebo_rtf, seed, aux_enabled, aux_version
  * episode_metrics (paper): near_collision_count, mean_speed_mps,
      lidar_clearance_rate
  * episode_driving: mean_cmd_v_mps, mean_cmd_steering_rad, low_obs_speed_frac
  * curriculum_eval_per_map / eval_metrics: their human-safety aggregates
      (h_coll_rate / psc) are eval-time means of the SAME per-episode human
      proximity this file records per episode via min_human_distance_m /
      time_below_human_clearance_ratio / near_human_event.

New privileged columns come from the env's DynamicAvoidanceEpisodeDiag
(dynamic_avoidance_telemetry.py), reachable even for an aux-OFF baseline.

NaN convention
--------------
Undefined numeric metrics are written as ``float('nan')`` (rendered ``""``-free
as ``nan`` by csv, read back as NaN by pandas); undefined strings as ``""``.
"""

import os
import csv
import math

import numpy as np

import drl_agent.training.run_layout as run_layout


DYN_AVOID_HEADER = [
    # ── identifiers ───────────────────────────────────────────────────────
    "episode", "global_t", "curriculum_stage", "map_type",
    "seed", "aux_enabled", "aux_version",
    # ── episode outcome (from curriculum_episode_rewards) ─────────────────
    "success", "collision", "timeout", "total_reward", "steps",
    # ── safety / driving (from episode_metrics + episode_driving) ─────────
    "near_collision_count", "mean_speed_mps", "lidar_clearance_rate",
    "final_goal_dist_m", "mean_gazebo_rtf",
    "mean_cmd_v_mps", "mean_cmd_steering_rad", "low_obs_speed_frac",
    # ── human interaction / clutter / near-event (privileged env diag) ────
    "collision_object_type",           # none | static | human | unknown
    "min_human_distance_m",
    "time_below_human_clearance_ratio",
    "human_ttc_min",
    "near_human_event",
    "near_human_event_success",
    "has_human_interaction",
    "has_static_clutter_pressure",
    "static_clutter_steps",
    "human_observed_steps",
    "nearest_human_mode",
    "human_modes_present",
    # ── yield diagnostics (env's actual per-stage-gated decision) ─────────
    "yield_available",
    "yield_used",
    "yield_trigger_count",
    "yield_steps",
    "yield_in_risk_steps",       # yield effectiveness numerator (yield & risk)
    "yield_no_risk_steps",       # yield false-positives (yield & no risk)
    "risk_steps",                # yield effectiveness denominator (risk steps)
]

_NAN = float("nan")


def _is_missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def driving_mean(values):
    """Buffer mean matching the ``episode_driving`` columns in train_tqc_base
    (plain ``np.mean``; NaN only for an empty buffer).

    The motion buffers are all-finite or all-NaN for a whole run (the motion-log
    contract is fixed at startup), so NaN filtering is unnecessary to reproduce
    the source column byte-for-byte — and filtering would silently DIVERGE from
    episode_driving, which is exactly what the consolidated log must avoid."""
    return float(np.mean(values)) if values else _NAN


def low_obs_speed_frac(obs_speed_buf, threshold_mps):
    """``low_obs_speed_frac`` computed IDENTICALLY to episode_driving
    (train_tqc_base): fraction of observed-speed samples whose *magnitude*
    (abs of the SIGNED speed) is below ``threshold_mps``.

    NaN samples count as not-low (``abs(nan) < thr`` is False), the denominator is
    ALL steps, and the result is NaN only for an empty buffer. Using the
    signed-speed magnitude (not the raw value) matters when the robot reverses, so
    the consolidated file never disagrees with episode_driving for an episode."""
    if not obs_speed_buf:
        return _NAN
    thr = max(float(threshold_mps or 0.0), 1e-9)
    return float(np.mean([
        1.0 if abs(float(sv)) < thr else 0.0 for sv in obs_speed_buf
    ]))


class DynamicAvoidanceCSV:
    """Writer for the per-episode dynamic-avoidance diagnostic CSV.

    Keeping the schema here guarantees header == row and keeps the trainer
    integration to a single :meth:`write_episode` call.
    """

    def __init__(self, log_dir: str, run_tag: str):
        # RUN_LAYOUT: empty run_tag (new-structure run) -> plain filename.
        self.path = os.path.join(
            log_dir,
            run_layout.tagged_filename("dynamic_avoidance_metrics", ".csv", run_tag),
        )
        # RUN_LAYOUT: only write the header for a genuinely NEW file -- a
        # resume into an existing new-structure run dir (plain filename) must
        # APPEND to its history, not truncate it.
        run_layout.write_csv_header_if_new(self.path, DYN_AVOID_HEADER)

    def write_episode(self, *, episode, global_t, stage, map_type,
                      seed, aux_enabled, aux_version,
                      success, collision, timeout, total_reward, steps,
                      ep_metrics, final_goal_dist_m, mean_gazebo_rtf,
                      mean_cmd_v_mps, mean_cmd_steering_rad, low_obs_speed_frac,
                      diag):
        """Assemble one row from the trainer's per-episode aggregates + the env's
        privileged dynamic-avoidance diagnostic dict (``diag``; ``{}`` when the
        env does not provide it)."""
        d = diag or {}

        def g(key, default=_NAN):
            v = d.get(key, default)
            return default if v is None else v

        # collision object type: env provides none/static/human; when the env did
        # not report a diag at all, we cannot classify -> unknown (only if there
        # actually was a collision), else none.
        coll_type = d.get("collision_object_type")
        if coll_type is None:
            coll_type = "unknown" if bool(collision) else "none"

        # near_human_event may be 0/1 or NaN (no pedestrian ever observed).
        nhe = g("near_human_event")
        if _is_missing(nhe):
            nhe_out = _NAN
            nhe_success = _NAN
        else:
            nhe_out = int(nhe)
            nhe_success = int(bool(nhe) and bool(success))

        row = [
            episode, global_t, stage, (map_type or ""),
            int(seed) if seed is not None else -1,
            int(aux_enabled), int(aux_version),
            int(bool(success)), int(bool(collision)), int(bool(timeout)),
            round(float(total_reward), 4), int(steps),
            # safety / driving
            ep_metrics.get("near_collision_count", 0),
            ep_metrics.get("mean_speed_mps", _NAN),
            ep_metrics.get("lidar_clearance_rate", _NAN),
            _round(final_goal_dist_m, 4),
            _round(mean_gazebo_rtf, 4),
            _round(mean_cmd_v_mps, 4),
            _round(mean_cmd_steering_rad, 4),
            _round(low_obs_speed_frac, 4),
            # human interaction / clutter / near-event
            str(coll_type),
            _round(g("min_human_distance_m"), 4),
            _round(g("time_below_human_clearance_ratio"), 4),
            _round(g("human_ttc_min"), 4),
            nhe_out,
            nhe_success,
            _int_or_nan(g("has_human_interaction")),
            _int_or_nan(g("has_static_clutter_pressure")),
            _int_or_nan(g("static_clutter_steps")),
            _int_or_nan(g("human_observed_steps")),
            str(g("nearest_human_mode", "")),
            str(g("human_modes_present", "")),
            # yield diagnostics
            _int_or_nan(g("yield_available")),
            _int_or_nan(g("yield_used")),
            _int_or_nan(g("yield_trigger_count")),
            _int_or_nan(g("yield_steps")),
            _int_or_nan(g("yield_in_risk_steps")),
            _int_or_nan(g("yield_no_risk_steps")),
            _int_or_nan(g("risk_steps")),
        ]
        assert len(row) == len(DYN_AVOID_HEADER), (
            f"dynamic-avoidance row/header length mismatch: "
            f"{len(row)} != {len(DYN_AVOID_HEADER)}")
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)


def _round(v, ndigits):
    """Round a float, passing NaN/None through as NaN (never crashes)."""
    if _is_missing(v):
        return _NAN
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return _NAN


def _int_or_nan(v):
    """Coerce to int, passing NaN/None through as NaN."""
    if _is_missing(v):
        return _NAN
    try:
        return int(v)
    except (TypeError, ValueError):
        return _NAN
