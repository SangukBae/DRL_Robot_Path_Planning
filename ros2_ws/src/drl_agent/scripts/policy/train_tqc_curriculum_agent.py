#!/usr/bin/env python3
"""Curriculum-learning subclass of TrainTQCBase.

Extends TrainTQCBase (train_tqc_base.py) with:
  - Loading curriculum_settings from train_tqc_curriculum_config.yaml
  - evaluate_and_print() returns success / collision / timeout rates (dict)
  - Automatic stage advancement via /gym_node/set_parameters
  - curriculum_stage column in per-episode CSV log
  - curriculum_state.json checkpoint for resume / inspection

Usage:
  ros2 run drl_agent train_tqc_curriculum_agent.py

The environment must be running environment_curriculum.py (not environment.py)
so that the curriculum_stage / curriculum_num_stages parameters exist on /gym_node.
"""

import os
import sys
import csv
import math
import time
import json
import pickle
import random
from datetime import datetime

import numpy as np
import torch
import rclpy
from rcl_interfaces.srv import GetParameters, SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

# Allow direct script execution (not only via ros2 run)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "environment")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
)

from train_tqc_base import TrainTQCBase
from file_manager import load_yaml
from episode_metrics import EpisodeMetrics, PaperMetricsCSV
# AUX_PRED: expected wire-format version for the env<->agent label contract.
from aux_prediction_labels import AUX_WIRE_VERSION
# AUX_ABLATION: run-identity / ablation logging helpers.
import aux_ablation_logging as aux_log
# AUX_PRED: formal aux-evaluation metric helpers (pure numpy).
from aux_eval_metrics import AuxEvalAccumulator, split_label


class _LabelProximity:
    """Per-episode human-proximity tracker from privileged aux labels.

    H-Coll and the TRUE (human) PSC are derived from the env's privileged
    human-distance labels, NOT from the agent's aux head.  So they are available
    for an aux-OFF agent baseline too, as long as the ENV emits labels
    (aux_prediction.enabled on the env side).  When the env emits no labels both
    metrics are reported BLANK (None) — never a misleading 0.
    """

    def __init__(self, personal_space_m: float, h_coll_radius_m: float):
        self.ps = float(personal_space_m)
        self.hcr = float(h_coll_radius_m)
        self.reset()

    def reset(self):
        self.min_m = float("inf")
        self.steps = 0
        self.intrusions = 0

    def add_dist(self, dist_m):
        """Fold one step's nearest-human distance [m] (ignored if not finite)."""
        if dist_m is None or not math.isfinite(dist_m):
            return
        self.steps += 1
        if dist_m < self.min_m:
            self.min_m = dist_m
        if dist_m < self.ps:
            self.intrusions += 1

    @property
    def available(self) -> bool:
        return self.steps > 0

    def psc(self):
        """Fraction of label-available steps that respected personal space.
        None when no label was seen (env labels off)."""
        return (1.0 - self.intrusions / self.steps) if self.steps > 0 else None

    def h_coll(self, collision: bool):
        """1 if a collision episode ended with a human inside h_coll_radius.
        None when no label was seen (env labels off)."""
        if self.steps == 0:
            return None
        return int(bool(collision) and self.min_m < self.hcr)


class TrainTQCCurriculum(TrainTQCBase):
    """TQC trainer with automatic curriculum stage advancement.

    Inherits all setup and model I/O from TrainTQCBase.
    Adds:
      1. eval metrics (success / collision / timeout rates)
      2. stage-pass checking (consecutive evals threshold)
      3. ROS2 set_parameters call to push new stage to EnvironmentCurriculum
      4. curriculum-aware CSV log
    """

    # AUX_PRED: this is the trainer that fully supports auxiliary prediction.
    AUX_SUPPORTED = True

    def _init_csv_loggers(self):
        """Create CSV logs for curriculum training (step-level log disabled)."""
        self._csv_run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._step_csv    = None  # step-level debug log disabled for curriculum
        self._reward_csv  = os.path.join(
            self.log_dir, f"episode_rewards_{self._csv_run_tag}.csv"
        )
        self._driving_csv = os.path.join(
            self.log_dir, f"episode_driving_{self._csv_run_tag}.csv"
        )

        # AUX_ABLATION: append [seed, aux_enabled, aux_version] (header == row;
        # the base _write_episode_logs writes these meta values for both files).
        _meta = aux_log.META_COLUMN_NAMES
        reward_header = [
            "episode", "global_t", "steps", "total_reward", "mean_reward",
            "goal_reached", "collision", "timeout", "eval_cut", "final_goal_dist_m",
        ] + _meta
        driving_header = [
            "episode", "global_t", "steps", "mean_v_norm", "mean_abs_w_norm",
            "initial_goal_dist_m", "final_goal_dist_m", "goal_dist_reduction_m",
            "min_lidar_m", "mean_min_lidar_m", "goal_reached", "eval_cut",
            "mean_gazebo_rtf",
        ] + _meta
        for path, header in [
            (self._reward_csv,  reward_header),
            (self._driving_csv, driving_header),
        ]:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

        self.get_logger().info(f"Episode rewards CSV: {self._reward_csv}")
        self.get_logger().info(f"Episode driving CSV: {self._driving_csv}")
        self.get_logger().info("Policy step CSV: disabled")

    def __init__(self):
        super().__init__()   # loads train_tqc_config.yaml, builds agent, etc.

        # Load curriculum advancement rules
        cur_cfg_path = self._find_config_file("train_tqc_curriculum_config.yaml")
        if cur_cfg_path:
            cur = load_yaml(cur_cfg_path).get("curriculum_settings", {})
        else:
            self.get_logger().warn(
                "[Curriculum] train_tqc_curriculum_config.yaml not found — using defaults."
            )
            cur = {}

        self.cur_enabled         = bool(cur.get("enabled", True))
        self.cur_min_stage_steps = int(cur.get("min_stage_steps", 10000))
        self.cur_min_stage_eps   = int(cur.get("min_stage_episodes", 20))
        self.cur_pass_sr         = list(cur.get("pass_eval_success_rate",
                                                 [0.90, 0.85, 0.75, 0.70]))
        self.cur_pass_cr         = list(cur.get("pass_eval_collision_rate",
                                                 [0.05, 0.10, 0.15, 0.20]))
        self.cur_consec_passes   = int(cur.get("consecutive_eval_passes", 2))

        # AUX_PRED: per-step auxiliary label tracking.  EnvInterface.reset()/
        # step() slice the env-appended future-risk label off and expose it as
        # self.last_aux_label; the loop snapshots it here so each stored label
        # stays aligned with its encoder-input state.
        self._aux_enabled = bool(getattr(self.rl_agent, "aux_enabled", False))
        self._aux_label_cur = None    # label paired with the CURRENT state s_t
        self._aux_label_next = None   # label paired with the next state s_{t+1}

        # Runtime state
        self._curriculum_stage       = 0
        # Structured map curriculum: map_type of the episode currently running
        # (read from the env's read-only current_map_type parameter after reset).
        self._cur_episode_map_type   = ""
        self._stage_start_step       = 0
        self._stage_start_ep         = 0
        self._consecutive_pass_count = 0
        self._total_episodes         = 0
        self._resume_global_t        = 0
        self._resume_loaded          = False
        self._last_global_t          = 0
        # Partial episode state (saved on interrupt, restored on resume)
        self._resume_epoch           = 1
        self._partial_ep_timesteps   = 0
        self._partial_ep_reward      = 0.0
        self._stage_restart_from_weights = False
        self._stage_restart_source_prefix = ""
        self._stage_restart_weights_dir = ""

        # ROS2 clients for the gym_node (EnvironmentCurriculum)
        # Node is named "gym_node" — matches Environment.__init__("gym_node")
        self._param_set_client = self.create_client(
            SetParameters, "/gym_node/set_parameters"
        )
        self._param_get_client = self.create_client(
            GetParameters, "/gym_node/get_parameters"
        )

        # Extra CSV that includes the curriculum_stage column
        self._curriculum_reward_csv = os.path.join(
            self.log_dir,
            f"curriculum_episode_rewards_{self._csv_run_tag}.csv",
        )
        with open(self._curriculum_reward_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "episode", "global_t", "steps",
                "total_reward", "mean_reward",
                "goal_reached", "collision", "timeout", "eval_cut",
                "final_goal_dist_m", "curriculum_stage", "mean_gazebo_rtf",
            ] + aux_log.META_COLUMN_NAMES   # AUX_ABLATION: seed/aux_enabled/aux_version
              + ["map_type"])               # structured map curriculum
        self.get_logger().info(
            f"[Curriculum] Episode log (with stage): {self._curriculum_reward_csv}"
        )

        # Structured map curriculum: per-map evaluation breakdown (one row per
        # map_type per evaluation) so the paper can report corridor / intersection
        # / clutter / lobby performance separately, not just the mixed average.
        self._curriculum_eval_per_map_csv = os.path.join(
            self.log_dir, f"curriculum_eval_per_map_{self._csv_run_tag}.csv",
        )
        with open(self._curriculum_eval_per_map_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "global_t", "curriculum_stage", "map_type", "eval_eps",
                "success_rate", "collision_rate", "timeout_rate",
                "mean_reward", "mean_goal_dist",
                # append-only extras (label-derived human metrics + formal aux
                # metrics; blank when env labels / agent aux head are off)
                "h_coll_rate", "psc",
                "aux_risk_rmse", "aux_min_dist_mae_m",
                "aux_peak_sector_acc", "aux_near_event_f1",
            ])
        self.get_logger().info(
            f"[Curriculum] Per-map eval log: {self._curriculum_eval_per_map_csv}"
        )

        # Paper metrics (SPL, path length, CTE, jerk, STL, clearance, ...) → CSVs.
        self.declare_parameter("near_collision_dist_m", 0.5)
        self.declare_parameter("metric_time_delta", 0.1)
        # STL reference speed + LiDAR-clearance radius (state-stream proxy knobs).
        self.declare_parameter("stl_ref_speed_mps", 1.0)
        self.declare_parameter("lidar_clearance_radius_m", 0.5)
        _ncd = self.get_parameter("near_collision_dist_m").get_parameter_value().double_value
        _mdt = self.get_parameter("metric_time_delta").get_parameter_value().double_value
        _stl_ref = self.get_parameter("stl_ref_speed_mps").get_parameter_value().double_value
        _lcr = self.get_parameter("lidar_clearance_radius_m").get_parameter_value().double_value
        self._em = EpisodeMetrics(
            self.environment_dim,
            time_delta=_mdt if _mdt > 0 else 0.1,
            near_collision_dist_m=_ncd if _ncd > 0 else 0.5,
            stl_ref_speed_mps=_stl_ref if _stl_ref > 0 else 1.0,
            lidar_clearance_radius_m=_lcr if _lcr > 0 else 0.5,
        )
        self._paper = PaperMetricsCSV(self.log_dir, self._csv_run_tag)
        self.get_logger().info(
            f"[Metrics] Paper CSVs: {self._paper.episode_path} | {self._paper.eval_path}"
        )

        # AUX_ABLATION: per-eval summary CSV (paper aux on/off comparison) + a
        # one-shot run manifest so per-seed aggregation never mixes configs.
        self._aux_eval_summary = aux_log.EvalSummaryCSV(
            self.log_dir, self._csv_run_tag, self.seed, self.rl_agent)
        # Ask the running env node what it ACTUALLY loaded (path + hash + aux
        # geometry).  Fall back to best-effort discovery only if unavailable.
        _envp = self._fetch_env_aux_params()

        # ── Label-derived human metrics (H-Coll + true PSC) ──────────────────
        # These use the ENV's privileged human-distance LABELS, not the agent aux
        # head, so they work for an aux-OFF agent baseline whenever the env emits
        # labels (aux_prediction.enabled on the env side). Label geometry (H, K,
        # D_c) is taken from the running ENV — the single source of truth.
        self.declare_parameter("h_coll_radius_m", 0.5)
        self.declare_parameter("psc_personal_space_m", 0.5)
        _hcr = self.get_parameter("h_coll_radius_m").get_parameter_value().double_value
        _psm = self.get_parameter("psc_personal_space_m").get_parameter_value().double_value
        self._h_coll_radius_m = _hcr if _hcr > 0 else 0.5
        self._psc_personal_space_m = _psm if _psm > 0 else 0.5
        self._label_enabled = bool(_envp.get("aux_enabled"))
        self._label_horizons = list(_envp.get("aux_horizons_sec") or [])
        self._label_H = len(self._label_horizons)
        self._label_K = int(_envp.get("aux_num_sectors") or 0)
        self._label_Dc = float(_envp.get("aux_risk_distance_scale") or 3.0)
        # H-Coll / PSC available iff the env actually appends labels with valid geometry.
        self._h_coll_available = bool(self._label_enabled and self._label_H > 0 and self._label_K > 0)
        # Per-episode human-proximity tracker (used by training loop console).
        self._ep_prox = _LabelProximity(self._psc_personal_space_m, self._h_coll_radius_m)

        # ── Formal aux-evaluation setup (paper aux metrics) ──────────────────
        # Computed in evaluate_and_print() only when the agent has an aux HEAD.
        self.declare_parameter("aux_near_event_threshold_m", 0.5)
        _net = self.get_parameter("aux_near_event_threshold_m").get_parameter_value().double_value
        self._aux_near_event_threshold_m = _net if _net > 0 else 0.5
        _aux_cfg = getattr(self.rl_agent, "aux_cfg", None)
        self._aux_eval_on = bool(getattr(self.rl_agent, "aux_eval_enabled", False))
        # Aux-head geometry (equals env label geometry by the wire fail-fast).
        self._aux_eval_H = int(getattr(_aux_cfg, "num_horizons", 0)) if _aux_cfg else 0
        self._aux_eval_K = int(getattr(_aux_cfg, "num_sectors", 0)) if _aux_cfg else 0
        self._aux_eval_Dc = self._label_Dc
        self._aux_eval_action_conditioned = bool(
            getattr(self.rl_agent, "aux_action_conditioned", False))
        self._aux_eval_ac_steps = int(getattr(_aux_cfg, "action_conditioned_steps", 4)) if _aux_cfg else 4
        self._aux_eval_cfg = {
            "aux_eval_on": self._aux_eval_on,
            "h_coll_available": self._h_coll_available,
            "near_event_threshold_m": self._aux_near_event_threshold_m,
            "h_coll_radius_m": self._h_coll_radius_m,
            "psc_personal_space_m": self._psc_personal_space_m,
            "stl_ref_speed_mps": _stl_ref if _stl_ref > 0 else 1.0,
            "lidar_clearance_radius_m": _lcr if _lcr > 0 else 0.5,
            "risk_distance_scale": self._label_Dc,
            "action_conditioned": self._aux_eval_action_conditioned,
        }
        self.get_logger().info(
            f"[AUX_EVAL] aux_head_metrics={self._aux_eval_on} | "
            f"label_human_metrics(H-Coll/PSC)={self._h_coll_available} | "
            f"H={self._label_H} K={self._label_K} D_c={self._label_Dc:.2f}m | "
            f"near_event<{self._aux_near_event_threshold_m:.2f}m "
            f"action_cond={self._aux_eval_action_conditioned}"
        )
        _env_cfg = _envp.get("loaded_config_path", "")
        if not _env_cfg:
            try:
                _env_cfg = self._find_config_file("environment_curriculum.yaml") or ""
            except Exception:
                _env_cfg = ""
        aux_log.write_run_manifest(
            self.log_dir, seed=self.seed, agent=self.rl_agent,
            train_config_file=getattr(self, "_train_cfg_path", ""),
            environment_config_file=_env_cfg,
            environment_config_sha1=_envp.get("loaded_config_sha1", ""),
            env_aux={
                "aux_enabled": _envp.get("aux_enabled"),
                "num_sectors": _envp.get("aux_num_sectors"),
                "horizons_sec": _envp.get("aux_horizons_sec"),
                "risk_distance_scale": _envp.get("aux_risk_distance_scale"),
            },
            aux_eval=self._aux_eval_cfg,   # formal aux-eval thresholds / ref speeds
            file_name=getattr(self, "file_name", ""),
            repo_dir=os.path.dirname(os.path.abspath(__file__)),
        )
        self.get_logger().info(
            f"[AUX_ABLATION] eval summary: {self._aux_eval_summary.path}"
        )

        self.get_logger().info(
            f"[Curriculum] Trainer ready — "
            f"enabled={self.cur_enabled} "
            f"min_steps={self.cur_min_stage_steps} "
            f"min_eps={self.cur_min_stage_eps} "
            f"consec={self.cur_consec_passes}"
        )
        self.declare_parameter("resume_weight_prefix", "")
        self.declare_parameter("resume_stage", -1)
        self.declare_parameter("resume_weights_dir", "")
        resume_weight_prefix = (
            self.get_parameter("resume_weight_prefix")
            .get_parameter_value().string_value.strip()
        )
        resume_stage = int(
            self.get_parameter("resume_stage").get_parameter_value().integer_value
        )
        resume_weights_dir = (
            self.get_parameter("resume_weights_dir")
            .get_parameter_value().string_value.strip()
        )

        if resume_weight_prefix and resume_stage < 0:
            raise ValueError(
                "resume_stage must be >= 0 when resume_weight_prefix is provided."
            )
        if resume_stage >= 0 and not resume_weight_prefix:
            raise ValueError(
                "resume_weight_prefix must be provided when resume_stage is set."
            )

        if resume_weight_prefix:
            self._start_from_specific_weights(
                resume_weight_prefix, resume_stage, resume_weights_dir
            )
        elif self.load_model:
            self._load_curriculum_state()

    # ------------------------------------------------------------------ #
    #  Stage control helpers                                                #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------
    # AUX_PRED: auxiliary-label capture
    # ------------------------------------------------------------------
    # The label is sliced off centrally in EnvInterface.reset()/step() (so every
    # client strips it consistently) and exposed via self.last_aux_label.  Here
    # we only snapshot it into the (cur / next) bookkeeping used to align each
    # stored label with its encoder-input state, plus a one-time config sanity
    # check at startup.
    def _check_aux_label_contract(self):
        """AUX_PRED: fail-fast on any agent/env aux mismatch (STRUCTURAL).

        The auxiliary label geometry lives in two configs (agent-side
        hyperparameters_tqc.yaml and env-side environment_curriculum.yaml).
        Rather than clipping or trusting a total-length match (different
        num_sectors / horizons_sec can yield the same length), the env sends a
        geometry header with the label; here we compare that header field-by-
        field against the agent's aux config and raise immediately on ANY
        inconsistency (missing label, wrong num_sectors, wrong number of
        horizons, different horizon values, or mismatched label length).
        """
        if not self._aux_enabled:
            return
        exp = getattr(self.rl_agent, "aux_cfg", None)
        lab = self.last_aux_label
        meta = self.last_aux_meta

        if lab is None:
            raise RuntimeError(
                "[AUX_PRED] agent aux_prediction.enabled=true but the environment "
                "appended no future-risk label. Set aux_prediction.enabled=true in "
                "environment_curriculum.yaml (and rebuild) so the env emits labels."
            )
        if meta is None:
            raise RuntimeError(
                "[AUX_PRED] env label is missing its geometry header (malformed or "
                "version-incompatible wire format). Rebuild so env and agent share "
                "the same aux_prediction_labels module."
            )
        if meta.get("version") != AUX_WIRE_VERSION:
            raise RuntimeError(
                f"[AUX_PRED] wire-format version mismatch: env="
                f"{meta.get('version')} but agent expects {AUX_WIRE_VERSION}. "
                "Rebuild so env and agent share the same aux_prediction_labels "
                "module (the wire layout changed incompatibly)."
            )
        if exp is None:
            return

        hint = ("Make num_sectors / horizons_sec IDENTICAL in "
                "hyperparameters_tqc.yaml and environment_curriculum.yaml, then rebuild.")
        if meta["num_sectors"] != exp.num_sectors:
            raise RuntimeError(
                f"[AUX_PRED] num_sectors mismatch: env={meta['num_sectors']} "
                f"agent={exp.num_sectors}. {hint}"
            )
        if meta["num_horizons"] != exp.num_horizons:
            raise RuntimeError(
                f"[AUX_PRED] horizon count mismatch: env={meta['num_horizons']} "
                f"agent={exp.num_horizons}. {hint}"
            )
        env_h = list(meta["horizons_sec"])
        agent_h = list(exp.horizons_sec)
        if any(abs(float(a) - float(b)) > 1e-3 for a, b in zip(env_h, agent_h)):
            raise RuntimeError(
                f"[AUX_PRED] horizon values mismatch: env={env_h} agent={agent_h}. {hint}"
            )
        if lab.shape[0] != exp.label_dim:
            raise RuntimeError(
                f"[AUX_PRED] label length mismatch: env sent {lab.shape[0]} but the "
                f"agent expects {exp.label_dim}. {hint}"
            )

    def _set_curriculum_stage(self, stage: int) -> bool:
        """Push curriculum_stage to /gym_node via set_parameters service."""
        if not self._param_set_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(
                "[Curriculum] /gym_node/set_parameters not available — "
                "is environment_curriculum.py running?"
            )
            return False
        req = SetParameters.Request()
        req.parameters = [
            Parameter(
                name="curriculum_stage",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_INTEGER,
                    integer_value=int(stage),
                ),
            )
        ]
        future = self._param_set_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is None:
            self.get_logger().warn("[Curriculum] /gym_node/set_parameters timed out.")
            return False
        ok = all(r.successful for r in future.result().results)
        if ok:
            self._curriculum_stage = stage
            self.get_logger().info(
                f"[Curriculum] Environment stage set to {stage}."
            )
        else:
            self.get_logger().warn(
                f"[Curriculum] set_parameters for stage={stage} rejected by gym_node."
            )
        return ok

    def _fetch_eval_mode(self):
        """Read the env's ACTUAL curriculum_eval_mode bool via get_parameters.
        Returns True/False, or None when unavailable (service down / param not
        declared on an older env)."""
        try:
            if not self._param_get_client.wait_for_service(timeout_sec=1.0):
                return None
            req = GetParameters.Request()
            req.names = ["curriculum_eval_mode"]
            future = self._param_get_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
            res = future.result()
            if res is None or not res.values:
                return None
            pv = res.values[0]
            if pv.type == ParameterType.PARAMETER_BOOL:
                return bool(pv.bool_value)
        except Exception:
            return None
        return None

    def _set_eval_mode(self, on: bool) -> bool:
        """Set the env's curriculum_eval_mode so evaluation episodes use
        eval_map_types (round-robin) instead of the training map distribution,
        while the env stays in train mode (obstacles/humans still activate).

        Returns True only after CONFIRMING via get_parameters that the env's
        actual value equals `on` — not merely that the SetParameters request was
        accepted. This makes the caller's eval_map_applied flag reflect ground
        truth even when a set times out but applies, or the env was already in the
        target state from a prior (failed-looking) toggle.
        """
        try:
            if self._param_set_client.wait_for_service(timeout_sec=3.0):
                req = SetParameters.Request()
                req.parameters = [
                    Parameter(
                        name="curriculum_eval_mode",
                        value=ParameterValue(
                            type=ParameterType.PARAMETER_BOOL,
                            bool_value=bool(on),
                        ),
                    )
                ]
                future = self._param_set_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            else:
                self.get_logger().warn(
                    "[Curriculum] /gym_node/set_parameters unavailable for "
                    "curriculum_eval_mode toggle — will confirm actual value below."
                )
        except Exception as e:
            self.get_logger().warn(f"[Curriculum] could not set eval_mode={on}: {e}")
        # Confirm the actually-applied value (ground truth), regardless of the
        # request-level result above.
        actual = self._fetch_eval_mode()
        return actual is not None and actual == bool(on)

    def _save_curriculum_state(self, global_t: int):
        """Write curriculum_state.json + RNG state files for full off-policy resume."""
        path = os.path.join(self.log_dir, "curriculum_state.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "stage":                  self._curriculum_stage,
                    "stage_start_step":       self._stage_start_step,
                    "stage_start_episode":    self._stage_start_ep,
                    "consecutive_pass_count": self._consecutive_pass_count,
                    "global_t":               global_t,
                    "total_episodes":         self._total_episodes,
                    "epoch":                  self._resume_epoch,
                    "ep_timesteps":           self._partial_ep_timesteps,
                    "ep_total_reward":        self._partial_ep_reward,
                },
                f,
                indent=2,
            )
        # RNG states — binary, saved alongside the JSON
        try:
            with open(os.path.join(self.log_dir, "rng_state.pkl"), "wb") as f:
                pickle.dump(
                    {"numpy": np.random.get_state(), "python": random.getstate()},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            torch.save(
                torch.get_rng_state(),
                os.path.join(self.log_dir, "rng_torch.pt"),
            )
            if torch.cuda.is_available():
                torch.save(
                    torch.cuda.get_rng_state(),
                    os.path.join(self.log_dir, "rng_cuda.pt"),
                )
        except Exception as _e:
            self.get_logger().warn(f"[Curriculum] RNG state save failed: {_e}")

    def _load_curriculum_state(self) -> bool:
        """Restore saved curriculum progress when resuming a run."""
        path = os.path.join(self.log_dir, "curriculum_state.json")
        if not os.path.isfile(path):
            self.get_logger().info(
                "[Curriculum] No curriculum_state.json found; resume will restart "
                "from stage 0 even though model weights were loaded."
            )
            return False
        try:
            with open(path, "r") as f:
                state = json.load(f)
            self._curriculum_stage = int(state.get("stage", 0))
            self._stage_start_step = int(state.get("stage_start_step", 0))
            self._stage_start_ep = int(state.get("stage_start_episode", 0))
            self._consecutive_pass_count = int(
                state.get("consecutive_pass_count", 0)
            )
            self._resume_global_t      = int(state.get("global_t", 0))
            self._total_episodes       = int(state.get("total_episodes", 0))
            self._resume_epoch         = int(state.get("epoch", 1))
            self._partial_ep_timesteps = int(state.get("ep_timesteps", 0))
            self._partial_ep_reward    = float(state.get("ep_total_reward", 0.0))
            self._last_global_t = self._resume_global_t
            self._resume_loaded = True
            self.get_logger().info(
                f"[Curriculum] Restored state from {path} | "
                f"stage={self._curriculum_stage} "
                f"global_t={self._resume_global_t} "
                f"episodes={self._total_episodes} "
                f"pass_streak={self._consecutive_pass_count}"
            )
            # Restore RNG states for reproducible off-policy resume
            try:
                pkl = os.path.join(self.log_dir, "rng_state.pkl")
                if os.path.isfile(pkl):
                    with open(pkl, "rb") as f:
                        rng = pickle.load(f)
                    np.random.set_state(rng["numpy"])
                    random.setstate(rng["python"])
                pt = os.path.join(self.log_dir, "rng_torch.pt")
                if os.path.isfile(pt):
                    torch.set_rng_state(torch.load(pt))
                cuda_pt = os.path.join(self.log_dir, "rng_cuda.pt")
                if torch.cuda.is_available() and os.path.isfile(cuda_pt):
                    torch.cuda.set_rng_state(torch.load(cuda_pt))
                self.get_logger().info("[Curriculum] RNG states restored.")
            except Exception as _e:
                self.get_logger().warn(
                    f"[Curriculum] RNG state restore failed: {_e}"
                )
            return True
        except Exception as e:
            self.get_logger().warn(
                f"[Curriculum] Failed to load curriculum_state.json: {e}. "
                "Falling back to fresh curriculum progression."
            )
            return False

    def _start_from_specific_weights(
        self, weight_prefix: str, stage: int, weights_dir: str
    ):
        """Load a specific model prefix and restart curriculum from a user stage."""
        weights_dir = os.path.expanduser(weights_dir) if weights_dir else self.pytorch_models_dir
        actor_path = os.path.join(weights_dir, f"{weight_prefix}_actor.pth")
        if not os.path.isfile(actor_path):
            raise FileNotFoundError(
                f"Specified model prefix not found: {actor_path}"
            )

        self.rl_agent.load(
            weights_dir,
            weight_prefix,
            load_optimizer_state=False,
            load_replay_buffer=False,
        )

        restart_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_name = f"{weight_prefix}_stage{int(stage)}_restart_{restart_tag}"
        self._curriculum_stage = int(stage)
        self._stage_start_step = self.timesteps_before_training
        self._stage_start_ep = 0
        self._consecutive_pass_count = 0
        self._total_episodes = 0
        self._resume_global_t = self.timesteps_before_training
        self._resume_loaded = False
        self._resume_epoch = 1
        self._partial_ep_timesteps = 0
        self._partial_ep_reward = 0.0
        self._last_global_t = self._resume_global_t
        self._stage_restart_from_weights = True
        self._stage_restart_source_prefix = weight_prefix
        self._stage_restart_weights_dir = weights_dir

        self.get_logger().info(
            f"[Curriculum] Loaded explicit model prefix '{weight_prefix}' "
            f"from {weights_dir} and will restart from stage {stage}."
        )

    def _fetch_num_stages(self) -> int:
        """Query curriculum_num_stages from the running gym_node.

        EnvironmentCurriculum declares this parameter at startup with the exact
        count from the config file the environment was actually launched with,
        so trainer and environment always agree on stage count.
        Falls back to 5 if the parameter or service is unavailable.
        """
        if not self._param_get_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(
                "[Curriculum] /gym_node/get_parameters unavailable — "
                "defaulting to 5 stages. Is environment_curriculum.py running?"
            )
            return 5
        req = GetParameters.Request()
        req.names = ["curriculum_num_stages"]
        future = self._param_get_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is None or not future.result().values:
            self.get_logger().warn(
                "[Curriculum] curriculum_num_stages not found on gym_node — "
                "defaulting to 5."
            )
            return 5
        n = int(future.result().values[0].integer_value)
        if n < 1:
            return 5
        self.get_logger().info(f"[Curriculum] gym_node reports {n} stages.")
        return n

    def _fetch_env_aux_params(self) -> dict:
        """AUX_ABLATION: read the env node's actual loaded-config path / hash and
        running aux geometry from gym_node, so run_manifest.json records the TRUE
        env settings instead of a re-discovered file.  Returns {} on any failure.
        """
        out = {}
        try:
            if not self._param_get_client.wait_for_service(timeout_sec=3.0):
                return out
            names = ["loaded_config_path", "loaded_config_sha1", "aux_enabled",
                     "aux_num_sectors", "aux_horizons_sec", "aux_risk_distance_scale"]
            req = GetParameters.Request()
            req.names = names
            future = self._param_get_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            res = future.result()
            if res is None or not res.values:
                return out
            for name, pv in zip(names, res.values):
                t = pv.type
                if t == ParameterType.PARAMETER_STRING:
                    out[name] = pv.string_value
                elif t == ParameterType.PARAMETER_BOOL:
                    out[name] = bool(pv.bool_value)
                elif t == ParameterType.PARAMETER_INTEGER:
                    out[name] = int(pv.integer_value)
                elif t == ParameterType.PARAMETER_DOUBLE:
                    out[name] = float(pv.double_value)
                elif t == ParameterType.PARAMETER_DOUBLE_ARRAY:
                    out[name] = [float(x) for x in pv.double_array_value]
                # PARAMETER_NOT_SET -> param absent on this env (older build)
        except Exception as e:
            self.get_logger().warn(f"[AUX_ABLATION] could not read env aux params: {e}")
        return out

    def _fetch_current_map_type(self) -> str:
        """Read the env's read-only `current_map_type` parameter (structured map
        curriculum). Returns "" when unavailable (e.g. layout disabled / older
        env). Cheap: one get_parameters round-trip, called once per episode."""
        try:
            if not self._param_get_client.wait_for_service(timeout_sec=1.0):
                return ""
            req = GetParameters.Request()
            req.names = ["current_map_type"]
            future = self._param_get_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
            res = future.result()
            if res is None or not res.values:
                return ""
            pv = res.values[0]
            if pv.type == ParameterType.PARAMETER_STRING:
                return pv.string_value
        except Exception:
            return ""
        return ""

    # ------------------------------------------------------------------ #
    #  AUX_PRED: formal-eval helpers                                        #
    # ------------------------------------------------------------------ #
    def _human_min_dist_m_from_label(self, label):
        """Privileged nearest-human distance [m] from an env label: the closest
        future approach over all horizons (min over the min_dist_norm block * D_c).

        Depends ONLY on the env emitting labels (self._h_coll_available), NOT on
        the agent aux head — so H-Coll / true PSC work for an aux-OFF agent
        baseline too. Returns None when no usable label is available."""
        if not self._h_coll_available or label is None or self._label_H <= 0:
            return None
        arr = np.asarray(label, dtype=np.float64).ravel()
        risk_dim = self._label_H * self._label_K
        if arr.shape[0] < risk_dim + self._label_H:
            return None
        md_norm = arr[risk_dim:risk_dim + self._label_H]
        return float(np.min(md_norm)) * self._label_Dc

    def _build_future_actions(self, actions_list):
        """Boundary-safe future-action tensor for action-conditioned aux eval.

        actions_list[i] = a_i (action taken from s_i), all within ONE episode.
        For step i returns [a_i, .., a_{i+K-1}] zero-padded past the episode end,
        and valid_len[i] = min(K, T - i) (>= 1) — NEVER reading across the
        boundary, identical to the training-time alignment.
        """
        K = max(1, int(self._aux_eval_ac_steps))
        T = len(actions_list)
        adim = len(actions_list[0]) if T > 0 else 0
        fut = np.zeros((T, K, adim), dtype=np.float32)
        vlen = np.ones((T,), dtype=np.int64)
        for i in range(T):
            n = min(K, T - i)
            vlen[i] = max(1, n)
            for j in range(n):
                fut[i, j] = np.asarray(actions_list[i + j], dtype=np.float32)
        return fut, vlen

    def _aux_eval_episode(self, acc, states_list, labels_list, actions_list, map_type):
        """Run the aux head over one finished eval episode and add the batch to
        the accumulator (single-step OR action-conditioned, boundary-safe)."""
        if not self._aux_eval_on or not states_list:
            return
        states = np.asarray(states_list, dtype=np.float32)
        labels = np.asarray(labels_list, dtype=np.float64)
        if self._aux_eval_action_conditioned:
            fut, vlen = self._build_future_actions(actions_list)
            preds = self.rl_agent.aux_predict_eval(states, fut, vlen)
        else:
            preds = self.rl_agent.aux_predict_eval(states)
        if preds is None:
            return
        risk_gt, md_gt = split_label(labels, self._aux_eval_H, self._aux_eval_K)
        risk_pred = preds["risk_map"]
        md_pred = preds.get("min_dist")   # None when the head has no min-dist out
        acc.add_batch(
            risk_pred.reshape(len(states_list), -1),
            risk_gt.reshape(len(states_list), -1),
            md_pred,
            md_gt,
            map_type=map_type or "na",
        )

    def _check_stage_advance(self, global_t: int, metrics: dict, num_stages: int) -> bool:
        """Return True when this eval pass should count toward stage promotion."""
        if not self.cur_enabled:
            return False
        if self._curriculum_stage >= num_stages - 1:
            return False   # already at the final stage
        # No promotion during warmup
        if global_t <= self.timesteps_before_training:
            return False
        # Minimum time / episode count in the current stage
        if global_t - self._stage_start_step < self.cur_min_stage_steps:
            return False
        if self._total_episodes - self._stage_start_ep < self.cur_min_stage_eps:
            return False

        stage_idx   = min(self._curriculum_stage, len(self.cur_pass_sr) - 1)
        required_sr = self.cur_pass_sr[stage_idx]
        required_cr = self.cur_pass_cr[stage_idx]
        return (
            metrics.get("success_rate",   0.0) >= required_sr
            and metrics.get("collision_rate", 1.0) <= required_cr
        )

    # ------------------------------------------------------------------ #
    #  Override: evaluate_and_print → returns dict of metrics              #
    # ------------------------------------------------------------------ #

    def evaluate_and_print(self, evals, epoch, start_time):
        """Run eval_eps episodes and return a metrics dict (not just mean reward)."""
        self.get_logger().info("=" * 55)
        self.get_logger().info(
            f"[Curriculum] Evaluating — Epoch {epoch} | Stage {self._curriculum_stage}"
        )
        self.get_logger().info(f"Elapsed: {time.time() - start_time:.1f}s")
        self.get_logger().info("=" * 55)

        ENV_DIM = self.environment_dim
        rewards, final_dists = [], []
        success_count = collision_count = timeout_count = 0
        per_ep_metrics = []
        # Structured map curriculum: per-map_type breakdown of this evaluation.
        per_map = {}   # map_type -> {n, success, collision, timeout, reward, goal_dist, h_coll}
        # Formal aux-evaluation accumulator (global + per map_type). Only used
        # when the agent has an aux head; otherwise it stays empty (aux off path).
        aux_acc = None
        if self._aux_eval_on:
            aux_acc = AuxEvalAccumulator(
                num_horizons=self._aux_eval_H, num_sectors=self._aux_eval_K,
                risk_distance_scale=self._aux_eval_Dc,
                near_event_threshold_m=self._aux_near_event_threshold_m,
            )
        # Switch the env to the eval_map_types round-robin for these episodes
        # (obstacles still activate; env stays in train mode). If the toggle fails
        # we do NOT silently evaluate on the training mix — warn loudly and flag
        # the fallback so the per-map numbers are not mistaken for eval_map_types.
        eval_mode_on = self._set_eval_mode(True)
        if not eval_mode_on:
            self.get_logger().warn(
                "[Curriculum] ⚠ FAILED to enable curriculum_eval_mode "
                "(set_parameters rejected/timed out). This evaluation FALLS BACK to "
                "the TRAINING map distribution — the per-map breakdown below is NOT "
                "an eval_map_types evaluation. Check that environment_curriculum.py "
                "is running and exposes curriculum_eval_mode."
            )
        # try/finally so an exception mid-eval (reset/step/logging) can never leave
        # the env stuck in eval mode — otherwise later TRAINING resets would keep
        # cycling eval_map_types.
        try:
            for _ in range(self.eval_eps):
                state    = self.reset()
                map_type = self._fetch_current_map_type()
                done     = False
                ep_steps = 0
                ep_rew   = 0.0
                self._em.reset(state)
                # Per-step buffers for the formal aux eval (state_t, label_t paired
                # with s_t, and a_t taken from s_t). Boundary-safe: one episode.
                ev_states, ev_labels, ev_actions = [], [], []
                # Label-based human proximity (H-Coll / true PSC). Gated on the
                # ENV emitting labels — independent of the agent aux head.
                prox = _LabelProximity(self._psc_personal_space_m, self._h_coll_radius_m)

                while not done and ep_steps < self.max_episode_steps:
                    # Capture s_t + its paired aux label BEFORE stepping (aux head
                    # eval needs the encoder input s_t and its label).
                    if self._aux_eval_on and self.last_aux_label is not None:
                        ev_states.append(np.asarray(state, dtype=np.float32).ravel())
                        ev_labels.append(np.asarray(self.last_aux_label, dtype=np.float64).ravel())
                    action = self.rl_agent.select_action(
                        state, use_checkpoint=False, use_exploration=False
                    )
                    if self._aux_eval_on and len(ev_states) == len(ev_actions) + 1:
                        ev_actions.append(np.asarray(action, dtype=np.float32).ravel())
                    state, reward, done, info = self.step(action)
                    self._em.update(state, action)
                    # Fold the post-step label (paired with s_{t+1}) so the FINAL
                    # collision step's proximity is counted (it would be missed if
                    # we only read labels before stepping).
                    prox.add_dist(self._human_min_dist_m_from_label(self.last_aux_label))
                    ep_rew   += reward
                    ep_steps += 1

                s = np.asarray(state, dtype=np.float32).ravel()
                final_dist = float(s[ENV_DIM])
                final_dists.append(final_dist)
                rewards.append(ep_rew)
                per_ep_metrics.append(self._em.compute(bool(done and info)))

                ep_success   = bool(done and info)
                ep_collision = bool(done and not info)
                ep_timeout   = bool(not done)
                if ep_success:
                    success_count   += 1
                elif ep_collision:
                    collision_count += 1
                else:
                    timeout_count   += 1
                # Label-derived human metrics (None when the env emits no labels).
                ep_h_coll = prox.h_coll(ep_collision)
                ep_psc    = prox.psc()

                # Formal aux metrics for this episode (boundary-safe).
                if aux_acc is not None and ev_states:
                    # Align lengths defensively (last action may be missing if the
                    # episode ended on the captured state).
                    L = min(len(ev_states), len(ev_labels), len(ev_actions)) \
                        if self._aux_eval_action_conditioned else min(len(ev_states), len(ev_labels))
                    if L > 0:
                        self._aux_eval_episode(
                            aux_acc, ev_states[:L], ev_labels[:L],
                            ev_actions[:L] if self._aux_eval_action_conditioned else None,
                            map_type,
                        )

                m = per_map.setdefault(
                    map_type or "na",
                    {"n": 0, "success": 0, "collision": 0, "timeout": 0,
                     "reward": 0.0, "goal_dist": 0.0,
                     "h_coll": 0, "h_coll_n": 0, "psc_sum": 0.0, "psc_n": 0},
                )
                m["n"]         += 1
                m["success"]   += int(ep_success)
                m["collision"] += int(ep_collision)
                m["timeout"]   += int(ep_timeout)
                m["reward"]    += ep_rew
                m["goal_dist"] += final_dist
                # Only count episodes where labels were available (avail) so the
                # rates are over the eligible denominator, not all episodes.
                if ep_h_coll is not None:
                    m["h_coll"]   += int(ep_h_coll)
                    m["h_coll_n"] += 1
                if ep_psc is not None:
                    m["psc_sum"]  += float(ep_psc)
                    m["psc_n"]    += 1
        finally:
            # Always return the env to the training map distribution so the next
            # training reset is NOT stuck on the eval round-robin — even if the
            # eval loop raised. Warn if the restore itself fails.
            if not self._set_eval_mode(False):
                self.get_logger().warn(
                    "[Curriculum] ⚠ FAILED to clear curriculum_eval_mode — subsequent "
                    "TRAINING resets may keep cycling eval_map_types until it clears."
                )

        n = self.eval_eps
        metrics = {
            "mean_reward":    float(np.mean(rewards)),
            "std_reward":     float(np.std(rewards)),
            "success_rate":   success_count   / n,
            "collision_rate": collision_count / n,
            "timeout_rate":   timeout_count   / n,
            "mean_goal_dist": float(np.mean(final_dists)),
            # False → eval ran on the training map mix (eval_map_types NOT applied).
            "eval_map_applied": bool(eval_mode_on),
        }
        # Aggregate paper metrics (SPL, CTE, jerk, STL, lidar_clearance, ...).
        _agg = PaperMetricsCSV.aggregate(per_ep_metrics)
        metrics.update(_agg)
        # Label-derived human metrics, aggregated over the EPISODES THAT HAD
        # LABELS (env-side). None (→ blank) when no eval episode had labels, so an
        # aux-off-env run never reports a misleading 0. An aux-OFF agent with the
        # env labels ON still gets real H-Coll / PSC.
        _hc_sum = sum(d["h_coll"] for d in per_map.values())
        _hc_n   = sum(d["h_coll_n"] for d in per_map.values())
        _psc_sum = sum(d["psc_sum"] for d in per_map.values())
        _psc_n   = sum(d["psc_n"] for d in per_map.values())
        metrics["h_coll_rate"] = (float(_hc_sum) / _hc_n) if _hc_n > 0 else None
        # TRUE (human personal-space) PSC overrides the state-stream proxy name in
        # the summary; the LiDAR clearance proxy is kept separately as
        # lidar_clearance_rate (from _agg).
        metrics["psc"] = (float(_psc_sum) / _psc_n) if _psc_n > 0 else None
        # Formal aux-eval metrics (global). Merged into metrics so the eval
        # summary CSV + console can read them; absent/blank when aux is off.
        aux_eval_metrics = aux_acc.finalize() if aux_acc is not None else {}
        if aux_eval_metrics:
            metrics.update(aux_eval_metrics)
        aux_eval_per_map = aux_acc.finalize_per_map() if aux_acc is not None else {}
        self._paper.write_eval(
            epoch=epoch, global_t=self._last_global_t,
            stage=self._curriculum_stage, eval_eps=n,
            base=metrics, metrics_mean=_agg,
        )

        # AUX_ABLATION: one eval-summary row per evaluation -> aux on/off and
        # learning-curve comparison at the same timesteps.
        try:
            self._aux_eval_summary.append(
                eval_global_t=self._last_global_t,
                curriculum_stage=self._curriculum_stage,
                eval_eps=n, metrics=metrics,
            )
        except Exception as _e:
            self.get_logger().warn(f"[AUX_ABLATION] eval-summary append failed: {_e}")

        _psc_v = metrics.get("psc")
        _hc_v = metrics.get("h_coll_rate")
        _psc_s = f"{_psc_v:.3f}" if _psc_v is not None else "n/a"
        _hc_s = f"{_hc_v*100:.1f}%" if _hc_v is not None else "n/a"
        self.get_logger().info(
            f"Eval {n} eps | "
            f"Reward {metrics['mean_reward']:.3f}±{metrics['std_reward']:.3f} | "
            f"Success {metrics['success_rate']*100:.1f}% | "
            f"Collision {metrics['collision_rate']*100:.1f}% | "
            f"Timeout {metrics['timeout_rate']*100:.1f}% | "
            f"GoalDist {metrics['mean_goal_dist']:.3f}m | "
            f"SPL {metrics['spl']:.3f} | STL {metrics.get('stl', 0.0):.3f} | "
            f"PSC {_psc_s} | H-Coll {_hc_s} | "
            f"Clearance {metrics.get('lidar_clearance_rate', 0.0):.3f} | "
            f"CTE {metrics['mean_cross_track_error_m']:.3f}m"
        )
        # AUX_PRED: formal aux-evaluation line — printed ONLY when aux is on.
        if self._aux_eval_on and aux_eval_metrics:
            _pk = aux_eval_metrics.get("aux_peak_sector_acc")
            _pk_s = f"{_pk:.3f}" if (_pk is not None and _pk == _pk) else "n/a"
            self.get_logger().info(
                "Eval(aux) | "
                f"AuxLossEval(RiskRMSE) {aux_eval_metrics['aux_risk_rmse']:.4f} | "
                f"MinDistMAE(m) {aux_eval_metrics['aux_min_dist_mae_m']:.4f} | "
                f"PeakAcc {_pk_s} | "
                f"EventF1 {aux_eval_metrics['aux_near_event_f1']:.3f} "
                f"(thr<{self._aux_near_event_threshold_m:.2f}m, N={aux_eval_metrics.get('aux_eval_samples', 0)})"
            )

        # Structured map curriculum: write + log the per-map_type breakdown so a
        # mixed-map stage average never hides one map collapsing.
        metrics["per_map"] = {}
        if not eval_mode_on:
            self.get_logger().warn(
                "[Curriculum] per-map breakdown below is a FALLBACK (training map "
                "distribution), not eval_map_types — see warning above."
            )
        try:
            with open(self._curriculum_eval_per_map_csv, "a", newline="") as f:
                w = csv.writer(f)
                def _b(am, key):
                    # blank when a metric is absent / None / NaN (aux off / no labels)
                    v = am.get(key) if isinstance(am, dict) else am
                    if v is None:
                        return ""
                    try:
                        fv = float(v)
                        return "" if fv != fv else round(fv, 6)
                    except Exception:
                        return ""
                for mt in sorted(per_map.keys()):
                    d = per_map[mt]
                    nn = max(d["n"], 1)
                    sr, cr, tr = d["success"] / nn, d["collision"] / nn, d["timeout"] / nn
                    mr, gd = d["reward"] / nn, d["goal_dist"] / nn
                    # Label-derived rates over label-available episodes only (None
                    # → blank when this map had no labels).
                    hc = (d["h_coll"] / d["h_coll_n"]) if d["h_coll_n"] > 0 else None
                    psc = (d["psc_sum"] / d["psc_n"]) if d["psc_n"] > 0 else None
                    am = aux_eval_per_map.get(mt, {})
                    metrics["per_map"][mt] = {
                        "eval_eps": d["n"], "success_rate": sr,
                        "collision_rate": cr, "timeout_rate": tr,
                        "mean_reward": mr, "mean_goal_dist": gd,
                        "h_coll_rate": hc, "psc": psc, **am,
                    }
                    w.writerow([
                        epoch, self._last_global_t, self._curriculum_stage, mt, d["n"],
                        round(sr, 4), round(cr, 4), round(tr, 4),
                        round(mr, 4), round(gd, 4),
                        _b(None, hc), _b(None, psc),
                        _b(am, "aux_risk_rmse"), _b(am, "aux_min_dist_mae_m"),
                        _b(am, "aux_peak_sector_acc"), _b(am, "aux_near_event_f1"),
                    ])
                    _hc_ms = f"{hc*100:.1f}%" if hc is not None else "n/a"
                    self.get_logger().info(
                        f"  [map={mt:<12}] n={d['n']:<3} "
                        f"Success {sr*100:5.1f}% | Collision {cr*100:5.1f}% | "
                        f"Timeout {tr*100:5.1f}% | GoalDist {gd:.3f}m | H-Coll {_hc_ms}"
                    )
        except Exception as _e:
            self.get_logger().warn(f"[Curriculum] per-map eval log failed: {_e}")

        evals.append(metrics["mean_reward"])
        np.save(f"{self.results_dir}/{self.file_name}", evals)
        return metrics

    # ------------------------------------------------------------------ #
    #  Override: train_online — adds stage advancement around eval          #
    # ------------------------------------------------------------------ #

    def train_online(self):
        """Training loop identical to TrainTQC.train_online() plus curriculum."""
        start_time = time.time()

        # Restore eval history and epoch counter so the curve is continuous.
        evals_path = f"{self.results_dir}/{self.file_name}.npy"
        if self._resume_loaded and os.path.isfile(evals_path):
            evals = list(np.load(evals_path))
            self.get_logger().info(
                f"[Curriculum] Loaded {len(evals)} past eval points from {evals_path}."
            )
        else:
            evals = []
        # Derive epoch from actual eval history so it stays in sync with
        # evals.npy even if curriculum_state.json was written before the
        # epoch counter was incremented (race window after crash).
        epoch = len(evals) + 1

        next_eval_t             = self.eval_freq if self.eval_freq > 0 else None
        training_enabled_logged = False

        # Query the actual stage count from the running environment node.
        # This guarantees trainer and environment share the same stage count
        # regardless of which config file the environment was launched with.
        num_stages = self._fetch_num_stages()
        self._curriculum_stage = max(0, min(self._curriculum_stage, num_stages - 1))
        if self._stage_restart_from_weights:
            self.get_logger().info(
                f"[Curriculum] Restarting from explicit weights "
                f"'{self._stage_restart_source_prefix}' at stage "
                f"{self._curriculum_stage}."
            )
            if not self._set_curriculum_stage(self._curriculum_stage):
                raise RuntimeError(
                    "[Curriculum] Cannot push requested restart stage to gym_node. "
                    "Make sure environment_curriculum.py is running and "
                    "/gym_node/set_parameters is reachable."
                )
            self._stage_start_step = self._resume_global_t
            self._stage_start_ep = self._total_episodes
        elif self._resume_loaded:
            self.get_logger().info(
                f"[Curriculum] Resuming curriculum from stage "
                f"{self._curriculum_stage} at global step {self._resume_global_t}."
            )
            if not self._set_curriculum_stage(self._curriculum_stage):
                raise RuntimeError(
                    "[Curriculum] Cannot restore saved curriculum stage on gym_node. "
                    "Make sure environment_curriculum.py is running and "
                    "/gym_node/set_parameters is reachable."
                )
        else:
            # Always force stage 0 before warmup begins.
            # This is critical for fresh starts: the environment node may still
            # hold a non-zero curriculum_stage from a previous session.
            self.get_logger().info(
                f"[Curriculum] Enforcing stage 0 (empty) for warmup "
                f"({self.timesteps_before_training} steps)."
            )
            if not self._set_curriculum_stage(0):
                raise RuntimeError(
                    "[Curriculum] Cannot push stage 0 to gym_node before warmup. "
                    "Make sure environment_curriculum.py is running and "
                    "/gym_node/set_parameters is reachable."
                )
            self._stage_start_step = 0
            self._stage_start_ep   = 0

        self.get_logger().info(
            f"[Curriculum] Training starts — {num_stages} stages total."
        )

        ENV_DIM = self.environment_dim
        state           = self.reset()
        # Structured map curriculum: record which map_type this episode runs on.
        self._cur_episode_map_type = self._fetch_current_map_type()
        # AUX_PRED: snapshot the label paired with s_0 and sanity-check config.
        self._aux_label_cur = self.last_aux_label
        self._check_aux_label_contract()
        # Always start a fresh episode on resume: carrying over ep_timesteps /
        # ep_total_reward from a different env rollout would corrupt timeout
        # logic and episode-level logs.  _partial_ep_* are saved to JSON for
        # crash-location debugging only and are NOT applied here.
        ep_total_reward = 0.0
        ep_timesteps    = 0
        ep_num          = self._total_episodes + 1
        ep_finished     = False
        _ep_v_buf:          list = []
        _ep_w_buf:          list = []
        _ep_min_lidar_buf:  list = []
        _ep_gazebo_rtf_buf: list = []
        _state0 = np.asarray(state, dtype=np.float32).ravel()
        _ep_initial_goal_dist = float(_state0[ENV_DIM])
        if next_eval_t is not None and self._resume_global_t > 0:
            next_eval_t = ((self._resume_global_t // self.eval_freq) + 1) * self.eval_freq

        for t in range(self._resume_global_t + 1, self.max_timesteps + 1):
            self._last_global_t = t
            if ep_timesteps == 0:
                self._em.reset(state)   # new episode → reset paper-metric tracker
                self._ep_prox.reset()   # label-based H-Coll / PSC tracker
            train_ready = t >= self.timesteps_before_training
            use_policy  = t >  self.timesteps_before_training
            if train_ready and not training_enabled_logged:
                self.get_logger().info(
                    f"[Curriculum] Warmup done at step {t} — "
                    f"gradient updates + policy actions enabled."
                )
                training_enabled_logged = True

            if use_policy:
                action = self.rl_agent.select_action(state)
            else:
                action = self.sample_action_space()

            next_state, reward, ep_finished, info = self.step(action)
            # AUX_PRED: label paired with the next state s_{t+1} (sliced off
            # centrally in EnvInterface.step and exposed as last_aux_label).
            self._aux_label_next = self.last_aux_label

            # Timeout penalty (same as base class)
            if ep_timesteps == self.max_episode_steps - 1 and not ep_finished:
                reward -= 20.0

            done = float(ep_finished) if ep_timesteps < self.max_episode_steps else 0.0
            # AUX_PRED: store the auxiliary label paired with `state` (s_t, the
            # encoder input), then advance the label bookkeeping with the state.
            self.rl_agent.replay_buffer.add(
                state, action, next_state, reward, done,
                aux_target=self._aux_label_cur,
            )
            self._aux_label_cur = self._aux_label_next

            state            = next_state
            self._em.update(state, action)
            # Label-based H-Coll / PSC: fold the nearest-human distance from the
            # aux label paired with the NEW state (so the collision step counts).
            # No-op when the env emits no labels (returns None).
            self._ep_prox.add_dist(self._human_min_dist_m_from_label(self._aux_label_next))
            ep_total_reward += reward
            ep_timesteps    += 1
            # Mirror to instance vars so Ctrl+C saves correct partial state
            self._partial_ep_timesteps = ep_timesteps
            self._partial_ep_reward    = ep_total_reward

            _s_after = np.asarray(state, dtype=np.float32).ravel()
            _ep_v_buf.append(float(action[0]))
            _ep_w_buf.append(float(action[1]))
            _ep_min_lidar_buf.append(float(np.min(_s_after[:ENV_DIM])))
            if np.isfinite(self._latest_gazebo_rtf):
                _ep_gazebo_rtf_buf.append(float(self._latest_gazebo_rtf))

            if train_ready and not self.use_checkpoints:
                self.rl_agent.train()

            eval_due       = bool(next_eval_t is not None and t >= next_eval_t)
            episode_limit  = ep_timesteps >= self.max_episode_steps
            force_eval_cut = eval_due and not ep_finished and not episode_limit

            if ep_finished or episode_limit or force_eval_cut:
                # AUX_PRED: mark the just-stored transition as an episode boundary
                # (goal / collision / timeout / eval-cut) so the action-conditioned
                # future-action lookup never crosses into the next episode.  No-op
                # unless the buffer tracks boundaries.
                self.rl_agent.replay_buffer.mark_last_traj_end()

                # Base-class episode log (reward_csv, driving_csv)
                result = self._write_episode_logs(
                    ep_num=ep_num, global_t=t,
                    ep_timesteps=ep_timesteps, ep_total_reward=ep_total_reward,
                    state=state, info=info,
                    episode_done=ep_finished, episode_limit=episode_limit,
                    ep_v_buf=_ep_v_buf, ep_w_buf=_ep_w_buf,
                    ep_min_lidar_buf=_ep_min_lidar_buf,
                    ep_gazebo_rtf_buf=_ep_gazebo_rtf_buf,
                    ep_initial_goal_dist=_ep_initial_goal_dist,
                    eval_cut=force_eval_cut,
                )

                # Curriculum episode log (adds stage column)
                final_dist   = float(np.asarray(state, dtype=np.float32).ravel()[ENV_DIM])
                goal_reached = bool(ep_finished and info) and not force_eval_cut
                collision    = bool(ep_finished and not goal_reached) and not force_eval_cut
                timeout      = bool(episode_limit and not ep_finished) and not force_eval_cut
                # Compute the per-episode paper metrics ONCE (reused by the CSV +
                # the policy-performance console summary below).
                ep_metrics = self._em.compute(goal_reached)
                # Label-derived human metrics (None when the env emits no labels —
                # then the console shows n/a, never a misleading 0).
                ep_h_coll = self._ep_prox.h_coll(collision)
                ep_psc    = self._ep_prox.psc()
                if not force_eval_cut:   # skip partial (eval-interrupted) episodes
                    self._paper.write_episode(
                        episode=ep_num, global_t=t, stage=self._curriculum_stage,
                        success=goal_reached, collision=collision, timeout=timeout,
                        total_reward=ep_total_reward, steps=ep_timesteps,
                        metrics=ep_metrics,
                    )
                with open(self._curriculum_reward_csv, "a", newline="") as _f:
                    csv.writer(_f).writerow([
                        ep_num, t, ep_timesteps,
                        round(ep_total_reward, 4),
                        round(ep_total_reward / max(ep_timesteps, 1), 4),
                        int(goal_reached), int(collision), int(timeout),
                        int(force_eval_cut),
                        round(final_dist, 4),
                        self._curriculum_stage,
                        round(float(np.mean(_ep_gazebo_rtf_buf)), 4)
                        if _ep_gazebo_rtf_buf else float("nan"),
                    ] + self._aux_log_meta_cols()    # AUX_ABLATION
                      + [self._cur_episode_map_type])  # structured map curriculum

                self._total_episodes = ep_num
                # Episode is done — next save should reflect a fresh episode start
                self._partial_ep_timesteps = 0
                self._partial_ep_reward    = 0.0
                self._save_curriculum_state(t)
                # One-line episode summary with policy-performance metrics. Same
                # format for aux on/off. PSC / H-Coll are label-derived (n/a when
                # the env emits no human labels — never a misleading 0).
                _psc_s   = f"{ep_psc:.2f}" if ep_psc is not None else "n/a"
                _hcoll_s = f"{ep_h_coll}" if ep_h_coll is not None else "n/a"
                self.get_logger().info(
                    f"T:{t} | Ep:{ep_num} | Steps:{ep_timesteps} | "
                    f"Reward:{ep_total_reward:.3f} | {result} | "
                    f"Stage:{self._curriculum_stage} | "
                    f"SPL:{ep_metrics['spl']:.2f} | STL:{ep_metrics['stl']:.2f} | "
                    f"PSC:{_psc_s} | H-Coll:{_hcoll_s}"
                )

                if self.use_checkpoints and train_ready:
                    self.rl_agent.train_and_checkpoint(ep_timesteps, ep_total_reward)

                if eval_due:
                    self.save_models(self.pytorch_models_dir, self.file_name)
                    metrics = self.evaluate_and_print(evals, epoch, start_time)
                    epoch  += 1
                    self._resume_epoch = epoch   # persisted by next _save_curriculum_state
                    while next_eval_t is not None and next_eval_t <= t:
                        next_eval_t += self.eval_freq

                    # ── Stage advancement logic ─────────────────────────
                    if self._check_stage_advance(t, metrics, num_stages):
                        self._consecutive_pass_count += 1
                        self.get_logger().info(
                            f"[Curriculum] Pass {self._consecutive_pass_count}/"
                            f"{self.cur_consec_passes} for stage "
                            f"{self._curriculum_stage} "
                            f"(sr={metrics['success_rate']*100:.1f}% "
                            f"cr={metrics['collision_rate']*100:.1f}%)"
                        )
                        if self._consecutive_pass_count >= self.cur_consec_passes:
                            new_stage = self._curriculum_stage + 1
                            self.get_logger().info(
                                f"[Curriculum] ★ Promoting to stage {new_stage}! "
                                f"(sr={metrics['success_rate']*100:.1f}% ≥ "
                                f"{self.cur_pass_sr[min(self._curriculum_stage, len(self.cur_pass_sr)-1)]*100:.0f}% | "
                                f"cr={metrics['collision_rate']*100:.1f}% ≤ "
                                f"{self.cur_pass_cr[min(self._curriculum_stage, len(self.cur_pass_cr)-1)]*100:.0f}%)"
                            )
                            self._set_curriculum_stage(new_stage)
                            self._stage_start_step       = t
                            self._stage_start_ep         = ep_num
                            self._consecutive_pass_count = 0
                            self._save_curriculum_state(t)
                    else:
                        if self._consecutive_pass_count > 0:
                            self.get_logger().info(
                                f"[Curriculum] Pass streak reset "
                                f"(sr={metrics['success_rate']*100:.1f}% "
                                f"cr={metrics['collision_rate']*100:.1f}%)"
                            )
                        self._consecutive_pass_count = 0

                # Reset episode
                state           = self.reset()
                # Structured map curriculum: record the new episode's map_type.
                self._cur_episode_map_type = self._fetch_current_map_type()
                # AUX_PRED: refresh the label paired with the new episode's s_0.
                self._aux_label_cur = self.last_aux_label
                ep_total_reward = 0.0
                ep_timesteps    = 0
                ep_num         += 1
                ep_finished     = False
                _ep_v_buf.clear()
                _ep_w_buf.clear()
                _ep_min_lidar_buf.clear()
                _ep_gazebo_rtf_buf.clear()
                _ep_initial_goal_dist = float(
                    np.asarray(state, dtype=np.float32).ravel()[ENV_DIM]
                )

        self.get_logger().info("[Curriculum] Training complete!")
        self.save_models(self.final_models_dir, self.file_name)
        self._save_curriculum_state(self.max_timesteps)
        self.done_training = True


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TrainTQCCurriculum()
        node.train_online()
        while rclpy.ok() and not node.done_training:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("\n[Curriculum] Training interrupted by user.")
        if node is not None:
            try:
                node._save_curriculum_state(getattr(node, "_last_global_t", 0))
            except Exception:
                pass
    except Exception as e:
        print(f"[Curriculum] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
