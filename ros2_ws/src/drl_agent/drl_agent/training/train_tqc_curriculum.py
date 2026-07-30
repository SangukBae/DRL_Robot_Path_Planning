#!/usr/bin/env python3
"""Curriculum-learning subclass of TrainTQCBase.

Extends TrainTQCBase (train_tqc_base.py) with:
  - Loading curriculum_settings from train_tqc_curriculum_config.yaml
  - evaluate_and_print() returns success / collision / timeout rates (dict)
  - Automatic stage advancement via /gym_node/set_parameters
  - curriculum_stage column in per-episode CSV log
  - curriculum_state.json checkpoint for resume / inspection

Usage:
  ros2 run drl_agent train_tqc_curriculum.py

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


from drl_agent.training.train_tqc_base import TrainTQCBase
from drl_agent.env.environment_interface import EnvServiceError
from drl_agent.training.gym_parameter_client import GymParameterClient
import drl_agent.training.curriculum.stage_logic as curriculum_stage_logic
import drl_agent.training.curriculum.state_io as curriculum_state_io
from drl_agent.training.curriculum.eval_runner import CurriculumEvalMixin
from drl_agent.training.curriculum.aux_eval import CurriculumAuxEvalMixin
from drl_agent.common.file_manager import load_yaml
import drl_agent.common.seed_utils as seed_utils
from drl_agent.training.episode_metrics import EpisodeMetrics, PaperMetricsCSV
# DYN_AVOID: consolidated per-episode dynamic-obstacle avoidance diagnostic CSV.
# driving_mean / low_obs_speed_frac live in the pure module so they reproduce the
# episode_driving definitions exactly AND stay unit-testable off-ROS.
from drl_agent.training.dynamic_avoidance_log import (
    DynamicAvoidanceCSV, driving_mean as _driving_mean,
    low_obs_speed_frac as _low_speed_frac,
)
import drl_agent.common.pure_pursuit as pure_pursuit
# AUX_PRED: expected wire-format version for the env<->agent label contract.
from drl_agent.env.observation.aux_prediction_labels import AUX_WIRE_VERSION
# AUX_ABLATION: run-identity / ablation logging helpers.
import drl_agent.training.aux_ablation_logging as aux_log
# RUN_LAYOUT: per-run runtime/experiments/<run_id>/ directory structure.
import drl_agent.training.run_layout as run_layout
# AUX_PRED: formal aux-evaluation metric helpers (pure numpy).
from drl_agent.training.aux_eval_metrics import AuxEvalAccumulator, split_label


# Per-episode human-proximity tracker (H-Coll / true PSC) lives in the pure,
# testable curriculum_metrics module.
from drl_agent.training.curriculum.metrics import _LabelProximity


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

    # ------------------------------------------------------------------ #
    #  RUN_LAYOUT: runtime/experiments/<run_id>/{configs,logs,models,      #
    #  analysis} per-run directory structure (overrides TrainTQCBase's     #
    #  flat runtime/tqc/seed_<seed>/ layout for THIS trainer only --       #
    #  other TrainTQCBase subclasses, e.g. generalization_eval.py, are     #
    #  unaffected).                                                        #
    # ------------------------------------------------------------------ #

    def _setup_directories(self):
        """Resolve this run's directory layout and create every subdir.

        Precedence:
          1. Explicit override (``run_dir`` ROS param or ``DRL_AGENT_RUN_DIR``
             env var) -- used AS-IS as the run dir (unchanged from the prior
             behaviour), but still gets the new configs/logs/models/analysis
             substructure created inside it.
          2. Otherwise: ``run_layout.resolve_run_layout`` -- a fresh
             ``runtime/experiments/<run_id>/`` for a new run, or a reused
             existing new-structure / legacy-structure dir when resuming
             (``load_model=true``) into one. See run_layout.py's module
             docstring for the full precedence and the "never migrate a
             legacy run" guarantee.
        """
        self.declare_parameter("run_dir", "")
        run_dir_param = (
            self.get_parameter("run_dir").get_parameter_value().string_value.strip()
        )
        env_run_dir = os.environ.get("DRL_AGENT_RUN_DIR", "").strip()

        if run_dir_param or env_run_dir:
            base_run_dir = os.path.expanduser(run_dir_param or env_run_dir)
            layout = run_layout.new_layout(base_run_dir, is_fresh=True)
        else:
            package_root = self._resolve_drl_agent_source_root()
            layout = run_layout.resolve_run_layout(
                package_root, self.base_file_name, self.seed,
                load_model=self.load_model,
            )

        self._run_layout          = layout
        self.run_dir               = layout["run_dir"]
        self.run_id                = layout["run_id"]
        self._using_legacy_layout  = bool(layout["is_legacy"])
        self.log_dir                = layout["logs_dir"]
        self.pytorch_models_dir     = layout["models_dir"]
        if self._using_legacy_layout:
            # Legacy resume: replicate the OLD 4-separate-dir structure
            # exactly (no configs/analysis dirs -- nothing new-structure-only
            # is ever written into an existing seed_<seed> folder).
            self.final_models_dir  = layout["final_models_dir"]
            self.results_dir       = layout["results_dir"]
            self.configs_dir       = None
            self.analysis_dir      = None
        else:
            # New structure: a single models/ dir doubles as "final" (each
            # run already has its own folder, so the old checkpoint-vs-final
            # directory split no longer carries information), and logs/
            # doubles as the small eval-history bookkeeping dir (evals.npy).
            self.final_models_dir  = layout["models_dir"]
            self.results_dir       = layout["logs_dir"]
            self.configs_dir       = layout["configs_dir"]
            self.analysis_dir      = layout["analysis_dir"]

        run_layout.create_run_dirs(layout)
        self.get_logger().info(
            f"[RunLayout] run_dir={self.run_dir} "
            f"(legacy={self._using_legacy_layout}, run_id={self.run_id})"
        )

    def _init_csv_loggers(self):
        """Create CSV logs for curriculum training (step-level log disabled).

        RUN_LAYOUT: ``self._csv_run_tag`` is the legacy "_<timestamp>" filename
        suffix. New-structure runs leave it empty (the run folder itself
        already carries the timestamp, so a plain "episode_rewards.csv" is
        enough and stays readable without a glob); a legacy-structure run
        (resumed into an existing seed_<seed> dir) keeps generating a fresh
        timestamp exactly as before, so multiple restarts keep accumulating
        distinct per-run CSVs there like they always have.
        """
        self._csv_run_tag = (
            "" if not self._using_legacy_layout
            else datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self._step_csv    = None  # step-level debug log disabled for curriculum
        self._reward_csv  = os.path.join(
            self.log_dir, run_layout.tagged_filename("episode_rewards", ".csv", self._csv_run_tag)
        )
        self._driving_csv = os.path.join(
            self.log_dir, run_layout.tagged_filename("episode_driving", ".csv", self._csv_run_tag)
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
            "mean_cmd_v_mps", "mean_cmd_steering_rad",
            "mean_obs_speed_mps", "mean_abs_obs_yaw_rate_rads",
            "mean_obs_steering_rad", "mean_abs_cmd_obs_speed_err_mps",
            "low_obs_speed_frac",
        ] + _meta
        # RUN_LAYOUT: only write the header for a genuinely NEW file -- a
        # resume into an existing new-structure run dir (plain filename, same
        # path every process start) must APPEND to its history, not truncate
        # it (_write_episode_logs in train_tqc_base.py already opens "a").
        for path, header in [
            (self._reward_csv,  reward_header),
            (self._driving_csv, driving_header),
        ]:
            run_layout.write_csv_header_if_new(path, header)

        self.get_logger().info(f"Episode rewards CSV: {self._reward_csv}")
        self.get_logger().info(f"Episode driving CSV: {self._driving_csv}")
        self.get_logger().info("Policy step CSV: disabled")

    def _init_motion_logging_contract(self):
        """Load the environment action/control contract for episode telemetry."""
        self._motion_log_enabled = False
        self._motion_low_speed_threshold_mps = 0.12
        env_cfg_path = self._find_config_file(
            "environment_curriculum.yaml", self._train_config_file_param)
        if not env_cfg_path:
            self.get_logger().warn(
                "[Curriculum] environment_curriculum.yaml not found; "
                "episode cmd/speed telemetry disabled."
            )
            return

        _full_cfg = load_yaml(env_cfg_path)
        env_cfg = _full_cfg.get("environment", {})
        # Curriculum stages live at TOP LEVEL (curriculum.stages), sibling to
        # `environment` — mirror environment_curriculum._load_curriculum_config.
        cur_cfg = _full_cfg.get("curriculum", {}) or {}
        try:
            self._motion_actions_low = np.asarray(env_cfg["actions_low"], dtype=np.float32)
            self._motion_actions_high = np.asarray(env_cfg["actions_high"], dtype=np.float32)
            self._motion_wheelbase_m = float(env_cfg["vehicle_wheelbase_m"])
            self._motion_steer_limit_rad = math.radians(
                float(env_cfg["vehicle_steering_limit_deg"])
            )
            self._motion_cruise_speed_mps = float(env_cfg["controller_cruise_speed_mps"])
            self._motion_min_speed_mps = float(env_cfg["controller_min_speed_mps"])
            self._motion_speed_steer_factor = float(env_cfg["controller_speed_steer_factor"])
            self._motion_low_speed_distance_m = float(
                env_cfg.get("controller_low_speed_distance_m", 0.0)
            )
            # 3D hybrid-action contract (for telemetry consistency with the env).
            self._motion_action_dim = int(env_cfg.get("action_dim", 2))
            # action_mode: same inference rule as environment.py (default
            # derived from action_dim -> identical dispatch for every existing
            # config that doesn't set this key).
            self._motion_action_mode = str(env_cfg.get(
                "action_mode",
                "waypoint_yield" if self._motion_action_dim >= 3 else "waypoint",
            )).strip().lower()
            self._motion_lookahead_min_m = float(
                env_cfg.get("controller_lookahead_min_m", 0.8)
            )
            self._motion_v_move_min_mps = float(
                env_cfg.get("controller_v_move_min_mps", 0.35)
            )
            self._motion_yield_creep_mps = float(
                env_cfg.get("controller_yield_creep_mps", 0.0)
            )
            _yr = dict(env_cfg.get("yield_reward", {}) or {})
            self._motion_yield_action_enabled = bool(_yr.get("action_enabled", True))
            self._motion_yield_action_threshold = float(_yr.get("action_threshold", 0.3))
            # Per-stage effective yield gate: mirror the env's stage override
            # (environment_curriculum._apply_curriculum_stage) so telemetry reflects
            # the ACTUAL contract per stage (early stages seal yield). Index i = the
            # effective action_enabled for stage i; a stage that omits the key
            # inherits the base default.
            self._motion_yield_action_enabled_by_stage = [
                bool(dict(s.get("yield_reward", {}) or {}).get(
                    "action_enabled", self._motion_yield_action_enabled))
                for s in (cur_cfg.get("stages", []) or [])
            ]
            _af = dict(env_cfg.get("anti_freeze", {}) or {})
            self._motion_low_speed_threshold_mps = float(
                _af.get("speed_threshold_mps", self._motion_low_speed_threshold_mps)
            )
            self._motion_log_enabled = True
        except Exception as e:
            self.get_logger().warn(
                f"[Curriculum] Failed to load motion-log contract from "
                f"{env_cfg_path}: {e}"
            )

    def _augment_profile_manifest_with_action_contract(self):
        """RESUME-SAFETY: append action_mode/action_dim to the run's already-
        written configs/profile_manifest.json (see train_tqc_base.py's
        _write_profile_manifest_if_requested, which runs earlier in
        super().__init__() -- before _init_motion_logging_contract has parsed
        these values, hence this separate augmentation pass instead of
        writing them inline there).

        This is what lets a LATER resume attempt verify the checkpoint's
        action contract before loading it (see config/validation.py's
        _check_action_mode) instead of unconditionally rejecting every
        speed_steering resume. No-op (never raises) when no manifest was
        written this run (non-profile launch, or the manifest write itself
        already failed) -- the resume gate then falls back to rejecting
        outright, which is the pre-existing, safe behaviour.
        """
        target_dir = getattr(self, "configs_dir", None) or self.log_dir
        path = os.path.join(target_dir, "profile_manifest.json")
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r") as f:
                manifest = json.load(f)
            manifest["action_mode"] = str(getattr(self, "_motion_action_mode", "waypoint"))
            manifest["action_dim"] = int(getattr(self, "_motion_action_dim", 2))
            with open(path, "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            self.get_logger().warn(
                f"[Profile] manifest action-contract augmentation failed (ignored): {e}"
            )

    def _motion_telemetry_sample(self, state, action):
        """Return per-step command + observed-motion telemetry for episode CSVs."""
        cmd_v = float("nan")
        cmd_steer = float("nan")
        if self._motion_log_enabled:
            _mode = getattr(self, "_motion_action_mode", "waypoint")
            if _mode == "speed_steering":
                cmd_v, cmd_steer = pure_pursuit.speed_steering_action_to_command(
                    action, self._motion_cruise_speed_mps, self._motion_steer_limit_rad,
                )
            elif getattr(self, "_motion_action_dim", 2) >= 3:
                # 3D hybrid: mirror the env's command exactly (incl. yield floors).
                # Use the CURRENT stage's effective yield gate so telemetry matches
                # the env's per-stage seal (early stages have yield disabled).
                _by_stage = getattr(self, "_motion_yield_action_enabled_by_stage", []) or []
                _stage = int(getattr(self, "_curriculum_stage", 0))
                _yield_enabled = (
                    _by_stage[_stage] if 0 <= _stage < len(_by_stage)
                    else self._motion_yield_action_enabled
                )
                cmd_v, cmd_steer, _theta, _ctl = pure_pursuit.hybrid_action_to_command(
                    action, self._motion_actions_low, self._motion_actions_high,
                    self._motion_wheelbase_m, self._motion_steer_limit_rad,
                    self._motion_cruise_speed_mps, self._motion_speed_steer_factor,
                    yield_enabled=_yield_enabled,
                    yield_threshold=self._motion_yield_action_threshold,
                    lookahead_min_m=self._motion_lookahead_min_m,
                    v_move_min_mps=self._motion_v_move_min_mps,
                    yield_creep_speed_mps=self._motion_yield_creep_mps,
                )
            else:
                _, _, x_wp, y_wp = pure_pursuit.action_to_waypoint(
                    action, self._motion_actions_low, self._motion_actions_high
                )
                cmd_v, cmd_steer = pure_pursuit.waypoint_to_command(
                    x_wp,
                    y_wp,
                    self._motion_wheelbase_m,
                    self._motion_steer_limit_rad,
                    self._motion_cruise_speed_mps,
                    self._motion_min_speed_mps,
                    self._motion_speed_steer_factor,
                    low_speed_distance_m=self._motion_low_speed_distance_m,
                )

        arr = np.asarray(state, dtype=np.float32).ravel()
        obs_speed = float("nan")
        obs_yaw_rate = float("nan")
        obs_steer = float("nan")
        if arr.size >= self.environment_dim + 7:
            base = int(self.environment_dim)
            obs_speed = float(arr[base + 4])
            obs_yaw_rate = float(arr[base + 5])
            obs_steer = float(arr[base + 6])

        return cmd_v, cmd_steer, obs_speed, obs_yaw_rate, obs_steer

    def _compute_risk_meta(self, action_risk_target):
        """RISK_BALANCE: per-transition metadata for the replay buffer's
        (optional) risk_meta field -- [stage, human_event, risk_positive,
        collision_or_near], see buffer.py's RISK_META_COLUMNS. Best-effort from
        signals already available at the add() call site; never raises (a
        best-effort miss just leaves that flag 0, which only ever makes
        risk-balanced sampling MORE conservative -- it falls back toward the
        uniform pool, never fabricates a false-positive event).

        human_event    : True iff a usable nearest-human distance label exists
                          this step (mirrors _human_min_dist_m_from_label; None
                          when no human labels / no human present).
        risk_positive  : action_risk_target's risk_dir (if the Action-Risk Head
                          env-side target is enabled) exceeds
                          risk_meta_positive_threshold; else (head disabled)
                          falls back to "a human is within
                          risk_meta_human_risk_distance_m".
        collision_or_near: full-360 environment_state min distance (the SAME
                          quantity check_collision() uses) at s_{t+1} is below
                          near_collision_dist_m -- covers both a true collision
                          step and a close near-miss. Sourced from
                          self.last_collision / self.last_min_obstacle_dist_m
                          (EnvInterface.step()'s side-channel cache of the
                          env's response.collision / response.min_obstacle_
                          dist_m -- the GROUND-TRUTH full-360 verdict; NOT
                          derived from next_state[:environment_dim], which is
                          the front-180 obs_state RL-input slice, a different
                          array that merely happens to share environment_dim's
                          length).
        """
        stage = float(int(getattr(self, "_curriculum_stage", 0)))
        human_min_dist = self._human_min_dist_m_from_label(self._aux_label_next)
        # RISK_BALANCE fix: _human_min_dist_m_from_label returns the SAFE
        # SENTINEL distance (== self._label_Dc) whenever no human is within
        # D_c -- including the human-free case entirely -- not None. Treating
        # "a label exists" as "a human is present" put every human-free
        # transition (e.g. stage 0-2, no pedestrians at all) into the human
        # pool, degrading the 25% human-risk sampling quota toward uniform.
        # A human only genuinely "present" (within sensing range) when its
        # distance is STRICTLY below the sentinel.
        human_event = human_min_dist is not None and human_min_dist < self._label_Dc

        risk_positive = False
        if action_risk_target is not None:
            _ar = np.asarray(action_risk_target, dtype=np.float32).ravel()
            if _ar.size >= 1:
                risk_positive = bool(_ar[0] > self._risk_meta_positive_threshold)
        elif human_min_dist is not None:
            risk_positive = bool(human_min_dist < self._risk_meta_human_risk_distance_m)

        collision_or_near = bool(self.last_collision) or bool(
            float(self.last_min_obstacle_dist_m) < self._risk_meta_near_collision_dist_m)

        return (stage, float(human_event), float(risk_positive), float(collision_or_near))

    def __init__(self):
        super().__init__()   # loads train_tqc_config.yaml, builds agent, etc.

        # Load curriculum advancement rules
        cur_cfg_path = self._find_config_file(
            "train_tqc_curriculum_config.yaml", self._train_config_file_param)
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
        # Human-safety HARD gates + per-map fail-fast gates (all default-empty →
        # disabled, so older curriculum configs promote exactly as before). These
        # only block when configured AND the underlying metric is available.
        self.cur_pass_psc         = list(cur.get("pass_eval_psc", []))
        self.cur_pass_h_coll      = list(cur.get("pass_eval_h_coll_rate", []))
        self.cur_pass_per_map_sr  = list(cur.get("pass_eval_per_map_success_rate", []))
        self.cur_pass_per_map_cr  = list(cur.get("pass_eval_per_map_collision_rate", []))
        self.cur_consec_passes   = int(cur.get("consecutive_eval_passes", 2))
        # Replay-buffer reset on promotion INTO a stage whose normalized action /
        # control contract DIFFERS from the previous stage's (old transitions would
        # otherwise poison the critic; clearing + re-warming avoids that). The
        # current curriculum uses ONE stop-capable contract from Stage 0 onward, so
        # no stage changes semantics and this list is EMPTY (never reset). The
        # mechanism is kept generic in case a contract-changing stage is re-added.
        # rewarmup_steps random-action steps refill the buffer with on-contract data
        # before gradient updates resume.
        self.cur_reset_buffer_stages = set(
            int(s) for s in (cur.get("reset_buffer_on_promote_to", []) or [])
        )
        self.cur_rewarmup_steps = int(
            cur.get("rewarmup_steps", self.timesteps_before_training)
        )

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
        # Absolute global step until which post-promotion re-warmup runs (random
        # actions, no gradient updates). 0 → not re-warming. Persisted to JSON.
        self._rewarmup_until_t       = 0
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
        self._init_motion_logging_contract()
        self._augment_profile_manifest_with_action_contract()

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
            run_layout.tagged_filename("curriculum_episode_rewards", ".csv", self._csv_run_tag),
        )
        # RUN_LAYOUT: header only for a genuinely NEW file -- a resume into an
        # existing new-structure run dir (plain filename) must APPEND to its
        # history (line ~1185 already opens "a"), not truncate it.
        run_layout.write_csv_header_if_new(self._curriculum_reward_csv, [
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
            self.log_dir,
            run_layout.tagged_filename("curriculum_eval_per_map", ".csv", self._csv_run_tag),
        )
        # RUN_LAYOUT: header only for a genuinely NEW file (see the comment
        # above the curriculum_reward_csv header write for why).
        run_layout.write_csv_header_if_new(self._curriculum_eval_per_map_csv, [
            "epoch", "global_t", "curriculum_stage", "map_type", "eval_eps",
            "success_rate", "collision_rate", "timeout_rate",
            "mean_reward", "mean_goal_dist",
            # append-only extras (label-derived human metrics + formal aux
            # metrics; blank when env labels / agent aux head are off)
            "h_coll_rate", "psc",
            "aux_risk_rmse", "aux_min_dist_mae_m",
            "aux_peak_sector_acc", "aux_near_event_f1",
            # append-only: per-map paper metrics (same PaperMetricsCSV.aggregate()
            # the global eval_summary row uses), for corridor/intersection/lobby/
            # clutter breakdowns of clearance compliance + path efficiency.
            "lidar_clearance_rate", "spl",
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
        # RISK_BALANCE: reuse this SAME near-collision distance (already an
        # existing config knob for the paper-metric H-Coll tracker) as the
        # "collision_or_near" replay-metadata threshold, so both concepts stay
        # numerically consistent instead of introducing a second knob.
        self._risk_meta_near_collision_dist_m = _ncd if _ncd > 0 else 0.5
        # RISK_BALANCE: thresholds for the "risk_positive" metadata flag (see
        # _compute_risk_meta). Read from the agent's replay_buffer.risk_meta
        # config block; harmless defaults when the block/agent attribute is
        # absent (this metadata is only ever CONSUMED when store_risk_meta is
        # on, so an unused default here changes nothing for phase2/both).
        _rm_meta_cfg = dict(
            self.rl_agent.hyperparameters.get("replay_buffer", {}).get("risk_meta", {}) or {})
        self._risk_meta_positive_threshold = float(
            _rm_meta_cfg.get("risk_positive_threshold", 0.5))
        self._risk_meta_human_risk_distance_m = float(
            _rm_meta_cfg.get("human_risk_distance_m", 2.0))
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

        # DYN_AVOID: single-file dynamic-obstacle avoidance diagnostic log. It
        # consolidates the dynamic-avoidance-relevant columns scattered across the
        # other CSVs (see dynamic_avoidance_log.py) and adds the env's privileged
        # human-interaction / clutter / yield / near-event diagnostics, so this one
        # file is enough to analyse pedestrian-avoidance behaviour.
        self._dyn_avoid_csv = DynamicAvoidanceCSV(self.log_dir, self._csv_run_tag)
        self.get_logger().info(
            f"[DYN_AVOID] Dynamic-avoidance CSV: {self._dyn_avoid_csv.path}"
        )
        # Count episodes whose diagnostics could not be read (env param missing /
        # service or parse failure). Used to rate-limit a WARN so a broken
        # telemetry path is visible at run time and NOT silently logged as NaN.
        self._dyn_diag_unavailable = 0

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
        # AUX_PRED (v2): temporal-context eval flags (faithful backward history).
        self._aux_eval_temporal = bool(
            getattr(self.rl_agent, "aux_temporal_enabled", False))
        self._aux_eval_hist_len = int(getattr(_aux_cfg, "history_len", 4)) if _aux_cfg else 4
        # Observation time-context (frame stacking) is detectable here as a
        # state_dim wider than one [obs+agent] frame (the env reports the stacked
        # state_dim, per-frame environment_dim/agent_dim unchanged). When BOTH the
        # actor-visible obs stacking AND the aux-only temporal branch are on, the
        # aux temporal GRU now summarises a window of ALREADY-stacked states — a
        # history-of-histories whose marginal value is small and largely redundant
        # with the actor's own stacked input. Kept available for ablation, but
        # warn so the redundancy is explicit (consider disabling temporal_enabled
        # once obs stacking is on). See aux_prediction_temporal.py header.
        self._obs_time_context_on = (
            int(self.state_dim) > int(self.environment_dim) + int(self.agent_dim))
        if self._obs_time_context_on and self._aux_eval_temporal:
            self.get_logger().warn(
                "[AUX_PRED] observation time-context (frame stacking) AND the "
                "aux-only temporal branch are BOTH enabled — the temporal GRU now "
                "operates on already-stacked states (redundant). Consider "
                "aux_prediction.temporal_enabled=false now that the actor sees "
                "time context directly.")
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
            "temporal": self._aux_eval_temporal,
        }
        self.get_logger().info(
            f"[AUX_EVAL] aux_head_metrics={self._aux_eval_on} | "
            f"label_human_metrics(H-Coll/PSC)={self._h_coll_available} | "
            f"H={self._label_H} K={self._label_K} D_c={self._label_Dc:.2f}m | "
            f"near_event<{self._aux_near_event_threshold_m:.2f}m "
            f"action_cond={self._aux_eval_action_conditioned} "
            f"temporal={self._aux_eval_temporal}"
        )
        _env_cfg = _envp.get("loaded_config_path", "")
        if not _env_cfg:
            try:
                _env_cfg = self._find_config_file(
                    "environment_curriculum.yaml", self._train_config_file_param) or ""
            except Exception:
                _env_cfg = ""

        # RUN_LAYOUT: freeze the 4 configs THIS run actually used into
        # configs/ (new-layout runs only -- a legacy-layout resume has no
        # configs_dir, see run_layout.legacy_layout, and is left untouched).
        _copied_cfg = {"train": "", "curriculum": "", "hyperparameters": "", "environment": ""}
        if self.configs_dir:
            _copied_cfg = run_layout.copy_run_configs(self.configs_dir, {
                "train": getattr(self, "_train_cfg_path", ""),
                "curriculum": getattr(self, "_curriculum_cfg_path", ""),
                "hyperparameters": getattr(self, "_hparams_path", ""),
                "environment": _env_cfg,
            })
            self.get_logger().info(f"[RunLayout] configs copied into {self.configs_dir}")

        aux_log.write_run_manifest(
            self.log_dir, seed=self.seed, agent=self.rl_agent,
            run_tag=self._csv_run_tag,
            seed_source=getattr(self, "_seed_source", ""),
            train_config_file=getattr(self, "_train_cfg_path", ""),
            curriculum_config_file=getattr(self, "_curriculum_cfg_path", ""),
            hyperparameters_file=getattr(self, "_hparams_path", ""),
            environment_config_file=_env_cfg,
            environment_config_sha1=_envp.get("loaded_config_sha1", ""),
            # RUN_LAYOUT: run identity + frozen-config paths (blank/legacy-safe
            # when this run is using the legacy runtime/tqc/seed_<seed> layout).
            run_id=getattr(self, "run_id", "") or "",
            run_dir=getattr(self, "run_dir", "") or "",
            logs_dir=self.log_dir,
            models_dir=getattr(self, "pytorch_models_dir", "") or "",
            configs_dir=self.configs_dir or "",
            analysis_dir=self.analysis_dir or "",
            copied_train_config_file=_copied_cfg.get("train", ""),
            copied_curriculum_config_file=_copied_cfg.get("curriculum", ""),
            copied_hyperparameters_file=_copied_cfg.get("hyperparameters", ""),
            copied_environment_config_file=_copied_cfg.get("environment", ""),
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
        """Delegate to GymParameterClient; cache the new stage on success.

        Also notify the agent so the TEMPORAL_ACTOR feature strength and the aux
        loss weight can ramp in with the curriculum (no tensor-shape change)."""
        ok = self._gym_params.set_curriculum_stage(stage)
        if ok:
            self._curriculum_stage = stage
            if hasattr(self.rl_agent, "set_curriculum_stage"):
                self.rl_agent.set_curriculum_stage(stage)
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
        # A3/A4 (residual critic / FiLM aux) are FRESH-RUN-ONLY: loading a prefix
        # into them would fresh-init the mismatched critic/aux modules -> hybrid
        # run. Same guard as load_model in TrainTQCBase.
        _fr = self._experimental_arch_reason()
        if _fr:
            raise RuntimeError(
                f"resume_weight_prefix is not allowed with FRESH-RUN-ONLY "
                f"architecture(s): {_fr}. Loading weights into a residual critic / "
                f"FiLM aux head would fresh-init the mismatched modules and produce "
                f"a hybrid run. Start a fresh run instead."
            )
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

    def _fetch_dynamic_diag(self) -> dict:
        """DYN_AVOID: read the env's privileged per-episode dynamic-avoidance
        diagnostics for the just-finished episode (delegates to
        GymParameterClient.get_dynamic_diag).

        The env ALWAYS publishes a full diagnostic dict (all keys present) even
        when no pedestrian was active — so an empty {} here means the telemetry
        path is broken (parameter absent on an older env, or a service/parse
        failure), NOT "no humans". We warn (rate-limited) in that case so the
        resulting unknown/NaN rows are attributable to a dead telemetry path and
        not misread as genuinely human-free episodes."""
        diag = self._gym_params.get_dynamic_diag()
        if not diag:
            self._dyn_diag_unavailable += 1
            n = self._dyn_diag_unavailable
            # Warn on the 1st, 2nd, 5th, then every 50th unavailable episode.
            if n <= 2 or n == 5 or n % 50 == 0:
                self.get_logger().warn(
                    f"[DYN_AVOID] episode_dynamic_diag unavailable (empty) — "
                    f"dynamic-avoidance telemetry is NOT being collected; "
                    f"dynamic_avoidance_metrics rows will be unknown/NaN "
                    f"(occurrence #{n}). Check that the running env node exposes "
                    f"the 'episode_dynamic_diag' parameter (rebuild if it predates "
                    f"DYN_AVOID)."
                )
        return diag


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
            pass_psc=self.cur_pass_psc,
            pass_h_coll=self.cur_pass_h_coll,
            pass_per_map_sr=self.cur_pass_per_map_sr,
            pass_per_map_cr=self.cur_pass_per_map_cr,
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

        # If a crash happened mid re-warmup, the reloaded buffer still holds the
        # pre-reset (off-contract) transitions — clear it again so the resumed
        # re-warmup refills with on-contract data.
        if self._resume_loaded and self._rewarmup_until_t > self._resume_global_t:
            self.rl_agent.replay_buffer.reset()
            self.get_logger().info(
                f"[Curriculum] Resumed during post-promotion re-warmup "
                f"(until step {self._rewarmup_until_t}); replay buffer re-cleared."
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
        _ep_cmd_v_buf:      list = []
        _ep_cmd_steer_buf:  list = []
        _ep_obs_speed_buf:  list = []
        _ep_obs_yaw_buf:    list = []
        _ep_obs_steer_buf:  list = []
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
            # Post-promotion re-warmup: after a buffer reset at a contract-changing
            # stage boundary, take random actions and skip gradient updates until
            # the buffer refills with on-contract data.
            if self._rewarmup_until_t:
                if t <= self._rewarmup_until_t:
                    train_ready = False
                    use_policy  = False
                else:
                    self.get_logger().info(
                        f"[Curriculum] Post-promotion re-warmup complete at step {t} "
                        f"— training resumes (buffer size="
                        f"{self.rl_agent.replay_buffer.size})."
                    )
                    self._rewarmup_until_t = 0
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
            # PHASE2: the Action-Risk Head's target is ALREADY action-conditioned
            # (computed by the env for the a_t just executed, pre-step) -- unlike
            # last_aux_label it needs no cur/next carry-over, just read straight
            # off EnvInterface for this same step() call (None when disabled).
            action_risk_target = self.last_action_risk_target
            # PHASE2 fail-fast: when this agent expects a supervision target
            # (hyperparameters_tqc.yaml action_risk_head.enabled=true) but the
            # env never emitted one this step, the buffer would otherwise
            # silently zero-pad it (buffer.py's add()) and the head would train
            # toward an all-zero "always safe" target with no visible error.
            # This ALWAYS means the env-side mirror is off (environment(_
            # curriculum).yaml action_risk_head.enabled=false) or the wire got
            # dropped -- environment.py computes this target unconditionally
            # (never returns None) whenever its own flag is on, and a genuine
            # env-side exception aborts the whole /step call instead of
            # returning a state, so a None here is never a transient hiccup.
            if self.rl_agent.action_risk_enabled and action_risk_target is None:
                raise RuntimeError(
                    "[PHASE2] hyperparameters_tqc.yaml action_risk_head.enabled="
                    "true but the env produced NO action-risk supervision target "
                    "this step (last_action_risk_target is None). The two "
                    "action_risk_head.enabled switches (env config AND agent "
                    "config) must BOTH be true -- check "
                    "environment_curriculum.yaml's action_risk_head.enabled. "
                    "Refusing to silently train the head toward an all-zero "
                    "target."
                )

            # Timeout penalty (same as base class)
            if ep_timesteps == self.max_episode_steps - 1 and not ep_finished:
                reward -= 20.0

            done = float(ep_finished) if ep_timesteps < self.max_episode_steps else 0.0
            # RISK_BALANCE: only bother computing metadata when the buffer
            # actually stores it (store_risk_meta off -> add() ignores the
            # kwarg anyway, but skip the (cheap but non-zero) computation too).
            risk_meta = None
            if getattr(self.rl_agent.replay_buffer, "risk_meta", None) is not None:
                risk_meta = self._compute_risk_meta(action_risk_target)
            # AUX_PRED: store the auxiliary label paired with `state` (s_t, the
            # encoder input), then advance the label bookkeeping with the state.
            self.rl_agent.replay_buffer.add(
                state, action, next_state, reward, done,
                aux_target=self._aux_label_cur,
                action_risk_target=action_risk_target,
                risk_meta=risk_meta,
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
            _cmd_v, _cmd_steer, _obs_speed, _obs_yaw, _obs_steer = (
                self._motion_telemetry_sample(state, action)
            )
            _ep_cmd_v_buf.append(float(_cmd_v))
            _ep_cmd_steer_buf.append(float(_cmd_steer))
            _ep_obs_speed_buf.append(float(_obs_speed))
            _ep_obs_yaw_buf.append(float(_obs_yaw))
            _ep_obs_steer_buf.append(float(_obs_steer))
            if np.isfinite(self._latest_gazebo_rtf):
                _ep_gazebo_rtf_buf.append(float(self._latest_gazebo_rtf))

            if train_ready and not self.use_checkpoints:
                # A1 (UTD ratio): run updates_per_env_step gradient updates per
                # env step (default 1 == baseline). The checkpoint path
                # (train_and_checkpoint below) is intentionally left unchanged —
                # it already batches updates by timesteps_since_update.
                for _ in range(self.updates_per_env_step):
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
                    ep_cmd_v_buf=_ep_cmd_v_buf,
                    ep_cmd_steer_buf=_ep_cmd_steer_buf,
                    ep_obs_speed_buf=_ep_obs_speed_buf,
                    ep_obs_yaw_rate_buf=_ep_obs_yaw_buf,
                    ep_obs_steer_buf=_ep_obs_steer_buf,
                    low_obs_speed_threshold_mps=self._motion_low_speed_threshold_mps,
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
                _mean_cmd_v = (float(np.mean(_ep_cmd_v_buf))
                               if _ep_cmd_v_buf else float("nan"))
                _mean_obs_v = (float(np.mean(_ep_obs_speed_buf))
                               if _ep_obs_speed_buf else float("nan"))
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

                # DYN_AVOID: one consolidated dynamic-avoidance row per completed
                # episode (skip eval-cut partials, matching the paper CSV). Reads
                # the env's privileged diagnostics BEFORE the reset below clears
                # them, and reuses ep_metrics / the motion buffers already computed.
                if not force_eval_cut:
                    _dyn = self._fetch_dynamic_diag()
                    _seed_m, _aux_en_m, _aux_ver_m = self._aux_log_meta_cols()
                    # Re-expose the episode_driving columns with the SAME
                    # definitions as train_tqc_base (plain buffer mean; low-speed
                    # fraction over abs(signed speed)) so the consolidated file
                    # matches the source file for every episode.
                    _mean_cmd_v_d = _driving_mean(_ep_cmd_v_buf)
                    _mean_cmd_steer = _driving_mean(_ep_cmd_steer_buf)
                    _mean_rtf = _driving_mean(_ep_gazebo_rtf_buf)
                    _low_obs_frac = _low_speed_frac(
                        _ep_obs_speed_buf, self._motion_low_speed_threshold_mps)
                    self._dyn_avoid_csv.write_episode(
                        episode=ep_num, global_t=t, stage=self._curriculum_stage,
                        map_type=self._cur_episode_map_type,
                        seed=_seed_m, aux_enabled=_aux_en_m, aux_version=_aux_ver_m,
                        success=goal_reached, collision=collision, timeout=timeout,
                        total_reward=ep_total_reward, steps=ep_timesteps,
                        ep_metrics=ep_metrics, final_goal_dist_m=final_dist,
                        mean_gazebo_rtf=_mean_rtf,
                        mean_cmd_v_mps=_mean_cmd_v_d,
                        mean_cmd_steering_rad=_mean_cmd_steer,
                        low_obs_speed_frac=_low_obs_frac,
                        diag=_dyn,
                    )

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
                    f"CmdV:{_mean_cmd_v:.2f} | ObsV:{_mean_obs_v:.2f} | "
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
                            # Fail-fast: only proceed once the env has actually
                            # switched stages. Otherwise the trainer would reset
                            # its buffer / re-warmup while the env stays on the old
                            # contract — a silent desync. (No-op unless a
                            # contract-changing stage is configured; currently none.)
                            if not self._set_curriculum_stage(new_stage):
                                raise RuntimeError(
                                    f"[Curriculum] Failed to push stage {new_stage} to "
                                    f"gym_node (/gym_node/set_parameters); aborting "
                                    f"before buffer reset / re-warmup to avoid a "
                                    f"trainer/environment stage desync."
                                )
                            self._stage_start_step       = t
                            self._stage_start_ep         = ep_num
                            self._consecutive_pass_count = 0
                            # Contract-changing boundary: clear the buffer so
                            # off-contract data does not poison the new stage's
                            # critic. Reset is gated ONLY by the stage list, which is
                            # EMPTY in the current single-contract curriculum, so this
                            # branch does not run; rewarmup_steps controls just the
                            # re-warmup length (0 → reset only, resume immediately).
                            if new_stage in self.cur_reset_buffer_stages:
                                self.rl_agent.replay_buffer.reset()
                                if self.cur_rewarmup_steps > 0:
                                    self._rewarmup_until_t = t + self.cur_rewarmup_steps
                                    self.get_logger().info(
                                        f"[Curriculum] Stage {new_stage} changes the "
                                        f"action/control contract — replay buffer "
                                        f"cleared; re-warmup for {self.cur_rewarmup_steps} "
                                        f"steps (until step {self._rewarmup_until_t})."
                                    )
                                else:
                                    self.get_logger().info(
                                        f"[Curriculum] Stage {new_stage} changes the "
                                        f"action/control contract — replay buffer "
                                        f"cleared; training resumes immediately "
                                        f"(rewarmup_steps=0)."
                                    )
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
                _ep_cmd_v_buf.clear()
                _ep_cmd_steer_buf.clear()
                _ep_obs_speed_buf.clear()
                _ep_obs_yaw_buf.clear()
                _ep_obs_steer_buf.clear()
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
        # STAGE 6: flush any buffered TensorBoard/JSON metrics on EVERY exit
        # path (normal completion, KeyboardInterrupt, EnvServiceError, any
        # other exception) -- json_flush_interval > 1 batches physical JSON
        # writes, so without this a run that stops between flush intervals
        # would silently lose its most recent buffered records.
        if node is not None and getattr(node, "rl_agent", None) is not None:
            try:
                node.rl_agent.flush_logs()
            except Exception as fe:
                print(f"[Curriculum] flush_logs() failed during shutdown: {fe}")
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
