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
# rcl_interfaces parameter plumbing now lives in gym_parameter_client.py.

# Allow direct script execution (not only via ros2 run)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "environment")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
)

from train_tqc_base import TrainTQCBase
from environment_interface import EnvServiceError
from gym_parameter_client import GymParameterClient
import curriculum_stage_logic
import curriculum_state_io
from curriculum_eval_runner import CurriculumEvalMixin
from curriculum_aux_eval import CurriculumAuxEvalMixin
from file_manager import load_yaml
import seed_utils
from episode_metrics import EpisodeMetrics, PaperMetricsCSV
# AUX_PRED: expected wire-format version for the env<->agent label contract.
from aux_prediction_labels import AUX_WIRE_VERSION
# AUX_ABLATION: run-identity / ablation logging helpers.
import aux_ablation_logging as aux_log
# AUX_PRED: formal aux-evaluation metric helpers (pure numpy).
from aux_eval_metrics import AuxEvalAccumulator, split_label


# Per-episode human-proximity tracker (H-Coll / true PSC) lives in the pure,
# testable curriculum_metrics module.
from curriculum_metrics import _LabelProximity


class TrainTQCCurriculum(
    CurriculumEvalMixin,
    CurriculumAuxEvalMixin,
    TrainTQCBase,
):
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
        # Remember resolved path so the run manifest can hash the curriculum config.
        self._curriculum_cfg_path = cur_cfg_path or ""
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
        # Quality gates (in ADDITION to success/collision). Conservative: an
        # absent/empty list → non-blocking default (SPL 0.0, clearance 0.0), so
        # older configs promote exactly as before. The shipped config sets a mild
        # SPL floor so "arrived but wandered" episodes don't pass on reward alone.
        self.cur_pass_spl        = list(cur.get("pass_eval_spl", []))
        self.cur_pass_clear      = list(cur.get("pass_eval_lidar_clearance_rate", []))
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

        # ROS2 parameter interaction with the gym_node (EnvironmentCurriculum) is
        # encapsulated in GymParameterClient. Node is named "gym_node" — matches
        # Environment.__init__("gym_node").
        self._gym_params = GymParameterClient(self)
        # Back-compat aliases (kept in case other code paths reference the raw
        # clients); all parameter traffic now goes through self._gym_params.
        self._param_set_client = self._gym_params.set_client
        self._param_get_client = self._gym_params.get_client

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
            run_tag=self._csv_run_tag,
            seed_source=getattr(self, "_seed_source", ""),
            train_config_file=getattr(self, "_train_cfg_path", ""),
            curriculum_config_file=getattr(self, "_curriculum_cfg_path", ""),
            hyperparameters_file=getattr(self, "_hparams_path", ""),
            environment_config_file=_env_cfg,
            environment_config_sha1=_envp.get("loaded_config_sha1", ""),
            env_aux={
                "aux_enabled": _envp.get("aux_enabled"),
                "num_sectors": _envp.get("aux_num_sectors"),
                "horizons_sec": _envp.get("aux_horizons_sec"),
                "risk_distance_scale": _envp.get("aux_risk_distance_scale"),
            },
            aux_eval=self._aux_eval_cfg,   # formal aux-eval thresholds / ref speeds
            determinism=getattr(self, "_determinism_info", None),
            # Pedestrian RNG policy: prefer the env node's reported values; the
            # base seed is the run seed the trainer pushed via /seed. resume_*
            # documents the Option-B contract (deterministic per checkpoint).
            human_rng={
                "enabled": _envp.get("human_rng_enabled", True),
                "policy": _envp.get(
                    "human_rng_policy",
                    "substream_isolated_from_global; "
                    "per_episode_reseed=SeedSequence(base_seed,episode_count); "
                    "resume=deterministic_per_checkpoint; exact_resume_disabled"),
                "base_seed": _envp.get("human_rng_base_seed", int(self.seed)),
                "base_seed_source": "env /seed service = run seed (self.seed)",
                "episode_derivation":
                    "RandomState+Random seeded from SeedSequence([base_seed, episode_count]) each reset",
                "resume_guarantee":
                    "deterministic_per_checkpoint (env re-seeded from "
                    "derive_resume_seed(base_seed, resume_global_t)); "
                    "NOT bit-exact continuation",
            },
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
            f"consec={self.cur_consec_passes} | "
            f"gates: sr={self.cur_pass_sr} cr={self.cur_pass_cr} "
            f"spl={self.cur_pass_spl or 'off'} "
            f"clearance={self.cur_pass_clear or 'off'}"
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
            if self._load_curriculum_state():
                self._reseed_env_for_resume()

    def _reseed_env_for_resume(self):
        """Define the ENVIRONMENT-RNG resume scope explicitly (the env runs in a
        SEPARATE process; its numpy/random state is NOT snapshotted across the
        process boundary, unlike the trainer-side RNGs which ARE restored exactly
        from rng_state.pkl / rng_torch.pt / rng_cuda.pt).

        On resume the env is re-seeded DETERMINISTICALLY from (base_seed,
        resume_global_t). Consequence (documented contract):
          • Reproducible: the same checkpoint always resumes the same env
            obstacle/human sampling stream.
          • NOT bit-exact: it does not byte-continue the pre-interrupt env stream,
            and it deliberately decorrelates from the already-trained early-stage
            layouts (so a resumed run does not replay the identical first-N
            episodes). This is a RESUMABLE guarantee, not exact continuation.
        Raising the bar to exact continuation would require an env-side RNG
        get/set-state service — intentionally out of scope for this change."""
        resume_env_seed = seed_utils.derive_resume_seed(
            self.seed, self._resume_global_t)
        # set_env_seed -> env.seed_callback re-bases BOTH the global env RNGs AND
        # the dedicated HUMAN sub-stream on resume_env_seed (the human stream is
        # then reseeded per episode as (resume_env_seed, episode_count)). So the
        # human RNG follows the SAME Option-B contract: deterministic per
        # checkpoint, not a bit-exact continuation.
        self.set_env_seed(resume_env_seed)
        self.get_logger().info(
            f"[Curriculum] Resume: env RNG (global + human sub-stream) re-seeded "
            f"deterministically to {resume_env_seed} (= base_seed {self.seed} + "
            f"global_t {self._resume_global_t}). Env sampling is reproducible PER "
            f"CHECKPOINT but does NOT bit-continue the pre-interrupt stream; "
            f"trainer-side numpy/random/torch RNGs are restored exactly."
        )

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
    def _set_curriculum_stage(self, stage: int) -> bool:
        """Delegate to GymParameterClient; cache the new stage on success."""
        ok = self._gym_params.set_curriculum_stage(stage)
        if ok:
            self._curriculum_stage = stage
        return ok

    def _fetch_eval_mode(self):
        """Delegate to GymParameterClient.get_eval_mode."""
        return self._gym_params.get_eval_mode()

    def _set_eval_mode(self, on: bool) -> bool:
        """Delegate to GymParameterClient.set_eval_mode."""
        return self._gym_params.set_eval_mode(on)


    def _save_curriculum_state(self, global_t: int):
        """Delegate to curriculum_state_io.save_curriculum_state (format unchanged)."""
        curriculum_state_io.save_curriculum_state(self, global_t)

    def _load_curriculum_state(self) -> bool:
        """Delegate to curriculum_state_io.load_curriculum_state (format unchanged)."""
        return curriculum_state_io.load_curriculum_state(self)

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
        """Delegate to GymParameterClient.get_num_stages."""
        return self._gym_params.get_num_stages()

    def _fetch_env_aux_params(self) -> dict:
        """Delegate to GymParameterClient.get_env_aux_params."""
        return self._gym_params.get_env_aux_params()

    def _fetch_current_map_type(self) -> str:
        """Delegate to GymParameterClient.get_current_map_type."""
        return self._gym_params.get_current_map_type()


    # ------------------------------------------------------------------ #
    #  AUX_PRED: formal-eval helpers                                        #
    # ------------------------------------------------------------------ #
    def _check_stage_advance(self, global_t: int, metrics: dict, num_stages: int) -> bool:
        """Return True when this eval pass should count toward stage promotion.

        The pure decision (gate thresholds vs. eval metrics) lives in
        ``curriculum_stage_logic.should_advance_stage``; this wrapper supplies the
        trainer's current state + config and logs any gate failures."""
        advance, reasons = curriculum_stage_logic.should_advance_stage(
            enabled=self.cur_enabled,
            curriculum_stage=self._curriculum_stage,
            num_stages=num_stages,
            global_t=global_t,
            timesteps_before_training=self.timesteps_before_training,
            stage_start_step=self._stage_start_step,
            min_stage_steps=self.cur_min_stage_steps,
            total_episodes=self._total_episodes,
            stage_start_ep=self._stage_start_ep,
            min_stage_eps=self.cur_min_stage_eps,
            pass_sr=self.cur_pass_sr,
            pass_cr=self.cur_pass_cr,
            pass_spl=self.cur_pass_spl,
            pass_clear=self.cur_pass_clear,
            metrics=metrics,
        )
        if reasons:
            self.get_logger().info(
                f"[Curriculum] Stage {self._curriculum_stage} promotion gate NOT met: "
                + ", ".join(reasons)
            )
        return advance

    # ------------------------------------------------------------------ #
    #  Override: evaluate_and_print → returns dict of metrics              #
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
                            f"cr={metrics['collision_rate']*100:.1f}% "
                            f"spl={metrics.get('spl', 0.0):.3f})"
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
    except EnvServiceError as e:
        # Environment service died (Gazebo/env node hung or crashed). Fail-fast:
        # save a checkpoint + curriculum state so the run is resumable, then stop
        # cleanly instead of hanging or losing progress.
        print(f"\n[Curriculum] Environment service failure: {e}")
        if node is not None:
            try:
                node.save_models(node.pytorch_models_dir, node.file_name)
                node._save_curriculum_state(getattr(node, "_last_global_t", 0))
                print("[Curriculum] Saved checkpoint after service failure — "
                      "run is resumable.")
            except Exception as se:
                print(f"[Curriculum] Checkpoint save after service failure failed: {se}")
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
