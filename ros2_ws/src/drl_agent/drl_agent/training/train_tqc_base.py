#!/usr/bin/env python3
"""Common base for all TQC training variants.

TrainTQCBase handles everything that is shared across TQC trainers:
  - train_tqc_config.yaml / hyperparameters_tqc.yaml loading
  - Output directory setup (pytorch_models, final_models, results, logs)
  - TQC agent construction and optional model loading
  - CSV loggers (episode_rewards, episode_driving, policy_step_debug)
  - Gazebo RTF estimation via /clock subscription
  - save_models() / _write_episode_logs()

Subclasses provide train_online() and evaluate_and_print() for their
specific training strategy (single-phase or curriculum).
"""

import os
import sys
import time
from datetime import datetime
import csv

import rclpy
import torch
import numpy as np

from rosgraph_msgs.msg import Clock


from drl_agent.rl.algorithms.tqc.agent import Agent
from drl_agent.env.environment_interface import EnvInterface
from drl_agent.common.file_manager import load_yaml
import drl_agent.common.seed_utils as seed_utils
import drl_agent.config.paths as config_paths
# AUX_ABLATION: run-identity / ablation logging helpers.
import drl_agent.training.aux_ablation_logging as aux_log


def _find_latest_prefix(models_dir: str, base: str, seed: int):
    """Return the file-name prefix of the most-recently modified checkpoint, or None."""
    import glob
    pat = os.path.join(models_dir, f"{base}_seed_{seed}_*_actor.pth")
    cands = [
        p for p in glob.glob(pat)
        if not os.path.basename(p).endswith("_checkpoint_actor.pth")
    ]
    if not cands:
        return None
    cands.sort(key=lambda p: os.stat(p).st_mtime, reverse=True)
    return os.path.basename(cands[0]).replace("_actor.pth", "")


class TrainTQCBase(EnvInterface):
    """Shared initialisation and I/O for all TQC training subclasses."""

    # AUX_PRED: auxiliary prediction is only wired into the curriculum trainer
    # (it needs the env-appended privileged labels and the cur/next label
    # bookkeeping).  Subclasses that genuinely support an aux-enabled agent set
    # this True; everything else hard-fails in __init__ instead of silently
    # training the encoder on missing / unaligned labels.
    AUX_SUPPORTED = False

    def __init__(self):
        super().__init__("train_tqc_node")

        # ----------------------------
        # Declare once: train_config_file
        # ----------------------------
        self.declare_parameter("train_config_file", "")
        user_param_path = (
            self.get_parameter("train_config_file")
            .get_parameter_value().string_value.strip()
        )

        # ----------------------------
        # Find and load training configuration file
        # ----------------------------
        train_cfg_path = self._find_config_file("train_tqc_config.yaml", user_param_path)
        if not train_cfg_path:
            self.get_logger().error("Could not find 'train_tqc_config.yaml'")
            raise FileNotFoundError("train_tqc_config.yaml not found")
        # AUX_ABLATION: remember resolved config path for the run manifest.
        self._train_cfg_path = train_cfg_path

        train_settings = load_yaml(train_cfg_path)["train_settings"]
        self.seed                       = train_settings["seed"]
        # Multi-seed sweep override (paper protocol): -p seed:=N or DRL_AGENT_SEED=N
        self.seed                       = self._resolve_seed_override(self.seed)
        self.max_episode_steps          = train_settings["max_episode_steps"]
        self.load_model                 = train_settings["load_model"]
        # PROFILE: optional CLI override for resume without editing the train
        # yaml (train_node.py translates -p resume:=true into this). Same
        # empty-string-means-unset convention as action_risk_head_enabled.
        self.declare_parameter("load_model", "")
        _lm_override_raw = (
            self.get_parameter("load_model").get_parameter_value().string_value
        )
        if _lm_override_raw.strip():
            self.load_model = config_paths.parse_bool_override(
                _lm_override_raw, bool(self.load_model)
            )
        self.max_timesteps              = train_settings["max_timesteps"]
        self.use_checkpoints            = train_settings["use_checkpoints"]
        self.eval_freq                  = train_settings["eval_freq"]
        self.timesteps_before_training  = train_settings["timesteps_before_training"]
        self.eval_eps                   = train_settings["eval_eps"]
        # A1 (UTD ratio): number of gradient updates per env step on the
        # non-checkpoint training path. Default 1 == baseline (1 env step -> 1
        # update); >1 runs N updates per env step so a slow ROS/Gazebo env can
        # still feed more gradient work to the GPU. Read here (shared by all TQC
        # trainers); only the curriculum loop actually consumes it. Absent key ->
        # 1, so existing configs are byte-for-byte unchanged.
        self.updates_per_env_step       = max(1, int(train_settings.get("updates_per_env_step", 1)))
        # Optional CLI override so an experiment can set the UTD ratio without a
        # separate train-config file: -p updates_per_env_step:=N (N>0 wins). 0 (the
        # default) means "no override -> use the config value above".
        self.declare_parameter("updates_per_env_step", 0)
        _upe_override = int(
            self.get_parameter("updates_per_env_step").get_parameter_value().integer_value
        )
        if _upe_override > 0:
            self.updates_per_env_step = _upe_override
        base_file_name                  = train_settings["base_file_name"]
        # RUN_LAYOUT: kept as an attribute (not just the local var above) so a
        # subclass's _setup_directories() override (see TrainTQCCurriculum) can
        # build a run_id from it without re-parsing train_tqc_config.yaml.
        self.base_file_name             = base_file_name

        current_date = datetime.now().strftime("%Y%m%d")
        self.file_name = f"{base_file_name}_seed_{self.seed}_{current_date}"

        # ----------------------------
        # Setup output directories
        # ----------------------------
        self._setup_directories()
        self._write_profile_manifest_if_requested()
        self._init_csv_loggers()

        # ----------------------------
        # Seeds — seed EVERY RNG the training process draws from so a fixed seed
        # is fully reproducible: the env services (numpy + random inside the env
        # node), plus this process's random / numpy / torch (CPU + all CUDA
        # devices). Previously `random` was left unseeded here, so any library
        # path using it broke reproducibility despite a fixed config seed.
        # ----------------------------
        self.set_env_seed(self.seed)
        seeded = seed_utils.seed_all(self.seed, seed_torch=True)
        self.get_logger().info(
            f"[Seed] Trainer RNGs seeded ({' + '.join(seeded)}) with {self.seed}; "
            f"env seed dispatched via /seed service."
        )
        # Record where the seed came from (config / ROS param / env var) so the
        # run manifest pins the EXACT seed provenance for multi-seed sweeps.
        self._seed_source = self._describe_seed_source(train_settings.get("seed"))

        # Reproducibility: make torch deterministic so the SAME seed + SAME code
        # produce the SAME training trajectory. Default ON; opt out with
        # train_settings.deterministic_torch=false (e.g. to recover the small
        # throughput that cuDNN auto-tuning buys). warn_only avoids a hard crash
        # if a rare op lacks a deterministic kernel.
        self._determinism_info = {"requested": False}
        if bool(train_settings.get("deterministic_torch", True)):
            self._determinism_info = seed_utils.enable_torch_determinism(
                warn_only=bool(train_settings.get("deterministic_warn_only", True))
            )
            self.get_logger().info(
                f"[Seed] Torch determinism applied: {self._determinism_info}"
            )
        else:
            self.get_logger().warn(
                "[Seed] deterministic_torch=false — cuDNN auto-tuning left ON; "
                "same-seed runs may not reproduce bit-for-bit."
            )

        # ----------------------------
        # Environment dimensions
        # ----------------------------
        state_dim, action_dim, max_action, environment_dim, agent_dim = self.get_dimensions()
        self.state_dim       = state_dim
        self.action_dim      = action_dim
        self.max_action      = max_action
        self.environment_dim = environment_dim
        self.agent_dim       = agent_dim

        # ----------------------------
        # Find and load hyperparameters
        # ----------------------------
        hparams_path = self._find_config_file("hyperparameters_tqc.yaml", user_param_path)
        if not hparams_path:
            self.get_logger().error("Could not find 'hyperparameters_tqc.yaml'")
            raise FileNotFoundError("hyperparameters_tqc.yaml not found")

        # AUX_ABLATION: remember resolved hyperparameter path for the manifest.
        self._hparams_path = hparams_path
        hyperparameters = load_yaml(hparams_path)["hyperparameters"]

        # PHASE2: optional CLI override for the agent-side action_risk_head.enabled,
        # mirroring the env-side risk_map_reward_enabled / action_risk_head_enabled
        # overrides in environment.py -- lets the 4-way PHASE2 experiment matrix
        # (neither / candidate1 / candidate2 / both) run via -p flags without
        # editing hyperparameters_tqc.yaml. "" (default) -> config value wins.
        self.declare_parameter("action_risk_head_enabled", "")
        _arh_override_raw = (
            self.get_parameter("action_risk_head_enabled")
            .get_parameter_value().string_value
        )
        if _arh_override_raw.strip():
            _arh_block = dict(hyperparameters.get("action_risk_head", {}) or {})
            _arh_block["enabled"] = config_paths.parse_bool_override(
                _arh_override_raw, bool(_arh_block.get("enabled", False))
            )
            hyperparameters["action_risk_head"] = _arh_block

        # ----------------------------
        # Create TQC agent
        # ----------------------------
        self.rl_agent = Agent(
            state_dim,
            action_dim,
            max_action,
            hyperparameters,
            log_dir=self.log_dir,
            # TEMPORAL_ACTOR: per-frame obs/agent dims so the agent can split the
            # transported stacked state into current(87) + scan history.
            env_obs_dim=environment_dim,
            env_agent_dim=agent_dim,
        )

        # AUX_ABLATION: let the agent stamp the seed on its JSON metric log.
        self.rl_agent.run_seed = self.seed

        # AUX_PRED: block aux on any non-supporting (e.g. non-curriculum) path.
        # type(self).AUX_SUPPORTED is resolved through the MRO, so this fires
        # even though the subclass __init__ runs super().__init__() first.
        if getattr(self.rl_agent, "aux_enabled", False) and not type(self).AUX_SUPPORTED:
            raise RuntimeError(
                "aux_prediction.enabled=true is only supported by "
                "train_tqc_curriculum_agent.py (and the aux-aware eval path). "
                "Set aux_prediction.enabled=false in hyperparameters_tqc.yaml for "
                f"this trainer ({type(self).__name__})."
            )

        self._skip_warmup_after_load = False
        self._last_sim_clock_sec  = None
        self._last_wall_clock_sec = None
        self._latest_gazebo_rtf   = float("nan")
        self._clock_sub = self.create_subscription(
            Clock, "/clock", self._on_clock, 50
        )

        # A3/A4 are FRESH-RUN-ONLY architectures (residual critic / FiLM aux head
        # change the critic/aux state_dict). The checkpoint loader silently
        # fresh-inits mismatched modules (see tqc_io.load), so a resume would
        # become a hybrid run — actor/encoder/replay carried over while critic/aux
        # restart. Refuse resume for these instead of relying on a docs warning.
        if self.load_model:
            _fr = self._experimental_arch_reason()
            if _fr:
                raise RuntimeError(
                    f"load_model=true is not allowed with FRESH-RUN-ONLY "
                    f"architecture(s): {_fr}. These change the critic/aux "
                    f"state_dict; the checkpoint loader would fresh-init the "
                    f"mismatched modules and silently produce a hybrid run. Start "
                    f"a fresh run (load_model=false), or use the baseline "
                    f"architecture (critic_residual=false, fusion_type=concat) to "
                    f"resume."
                )

        # ----------------------------
        # Checkpoint discovery
        # ----------------------------
        if self.load_model:
            latest = _find_latest_prefix(self.pytorch_models_dir, base_file_name, self.seed)
            if latest:
                self.file_name = latest
                self.get_logger().info(f"Resuming from checkpoint prefix: {self.file_name}")

        # ----------------------------
        # Optional: load existing model
        # ----------------------------
        if self.load_model:
            try:
                self.rl_agent.load(self.pytorch_models_dir, self.file_name)
                self._skip_warmup_after_load = True
                self.get_logger().info(
                    f"Loaded model from {self.pytorch_models_dir}/{self.file_name}"
                )
            except Exception as e:
                self.get_logger().warning(f"Could not load model: {e}")

        self.done_training = False
        self.log_training_setting_data()

    # ------------------------------------------------------------------ #
    #  Fresh-run-only architecture guard (A3/A4)                            #
    # ------------------------------------------------------------------ #

    def _experimental_arch_reason(self) -> str:
        """Return a human string naming any FRESH-RUN-ONLY architecture flag that
        is active on the constructed agent, or '' when none is.

        A3 (critic_residual) and A4 (aux fusion_type=film) change the critic / aux
        state_dict. Since the checkpoint loader fresh-inits mismatched modules
        instead of failing, any resume path (load_model / resume_weight_prefix)
        must be blocked for them so a run cannot silently become a hybrid."""
        reasons = []
        agent = getattr(self, "rl_agent", None)
        if agent is None:
            return ""
        if bool(getattr(agent, "critic_residual", False)):
            reasons.append("critic_residual=true (A3)")
        # A4 only matters when aux is ACTUALLY enabled: with
        # aux_prediction.enabled=false no aux head is built, so fusion_type=film
        # changes no state_dict and must not block an otherwise-valid resume.
        aux_cfg = getattr(agent, "aux_cfg", None)
        if bool(getattr(agent, "aux_enabled", False)) and \
                str(getattr(aux_cfg, "fusion_type", "concat")) == "film":
            reasons.append("aux_prediction.fusion_type=film (A4)")
        return ", ".join(reasons)

    # ------------------------------------------------------------------ #
    #  Config discovery                                                     #
    # ------------------------------------------------------------------ #

    def _describe_seed_source(self, config_seed) -> str:
        """Describe where the effective seed came from (manifest provenance).

        Mirrors the priority in ``_resolve_seed_override`` (ROS param > env var >
        config) without re-resolving — purely for logging the source label.
        """
        try:
            param = -1
            if self.has_parameter("seed"):
                param = self.get_parameter("seed").get_parameter_value().integer_value
            if param is not None and param >= 0:
                return f"ros_param seed:={param}"
            env_seed = os.environ.get("DRL_AGENT_SEED", "").strip()
            if env_seed.lstrip("-").isdigit() and int(env_seed) >= 0:
                return f"env DRL_AGENT_SEED={env_seed}"
            return f"config({config_seed})"
        except Exception:
            return "unknown"

    def _find_config_file(self, filename: str, user_param_path: str | None = None) -> str | None:
        """
        Robust config discovery.
        - If user_param_path is a file: return it.
        - If user_param_path is a directory: join with filename.
        - Else search: ament share -> env vars -> DRL_AGENT_SRC_PATH candidates -> repo-relative.
        """
        tried = []

        def _try_location_hint(hint: str):
            """A file-or-dir HINT (user param / env var): resolve via the shared
            config_paths helper, return the file if it exists, else record the
            attempt in `tried` (preserving the original error reporting)."""
            cand = config_paths.location_candidate(hint, filename)
            if cand:
                if os.path.isfile(cand):
                    return cand
                tried.append(cand)
            elif hint:
                tried.append(os.path.expanduser(hint))
            return None

        # 0) User-provided parameter (file or dir)
        hit = _try_location_hint(user_param_path)
        if hit:
            return hit

        share_dir = ""
        try:
            from ament_index_python.packages import get_package_share_directory
            share_dir = get_package_share_directory("drl_agent")
        except Exception:
            pass

        # 1) Environment variable: DRL_AGENT_TRAIN_CONFIG (file or dir)
        hit = _try_location_hint(os.environ.get("DRL_AGENT_TRAIN_CONFIG", "").strip())
        if hit:
            return hit

        # 2) Editable source tree (preferred over install/share so config edits
        # under ros2_ws/src/drl_agent take effect without requiring a rebuild).
        source_dirs = config_paths.source_package_candidates(
            "drl_agent",
            src_hint=os.environ.get("DRL_AGENT_SRC_PATH", "").strip(),
            installed_share_dir=share_dir,
            script_path=__file__,
        )
        for cand in config_paths.candidate_config_paths(filename, source_dirs):
            if os.path.isfile(cand):
                return cand
            tried.append(cand)

        # 3) Relative to this script (two common layouts)
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.normpath(os.path.join(here, "..", "config", filename)),
            os.path.normpath(os.path.join(here, "..", "..", "config", filename)),
        ]
        for cand in candidates:
            if os.path.isfile(cand):
                return cand
            tried.append(cand)

        # 4) Installed share directory (fallback only).
        if share_dir:
            cand = os.path.join(share_dir, "config", filename)
            if os.path.isfile(cand):
                return cand
            tried.append(cand)

        self.get_logger().error(
            "Could not find config '{}'. Tried:\n  {}".format(filename, "\n  ".join(tried))
        )
        return None

    # ------------------------------------------------------------------ #
    #  Directory management                                                 #
    # ------------------------------------------------------------------ #

    def _setup_directories(self):
        """Determine and create all output sub-directories."""
        self.declare_parameter("run_dir", "")
        run_dir_param = (
            self.get_parameter("run_dir").get_parameter_value().string_value.strip()
        )

        if run_dir_param:
            base_run_dir = os.path.expanduser(run_dir_param)
        elif os.environ.get("DRL_AGENT_RUN_DIR", "").strip():
            base_run_dir = os.path.expanduser(os.environ["DRL_AGENT_RUN_DIR"])
        else:
            package_root = self._resolve_drl_agent_source_root()
            # Per-seed isolation: each seed gets its own models/logs/curriculum_state
            base_run_dir = os.path.join(
                package_root, "runtime", "tqc", f"seed_{self.seed}"
            )

        self.run_dir             = base_run_dir
        self.pytorch_models_dir  = os.path.join(self.run_dir, "pytorch_models")
        self.final_models_dir    = os.path.join(self.run_dir, "final_models")
        self.results_dir         = os.path.join(self.run_dir, "results")
        self.log_dir             = os.path.join(self.run_dir, "logs")

        self.create_directories()

    def _write_profile_manifest_if_requested(self):
        """PROFILE: persist the launching profile's identity into the run dir.

        train_node.py serializes the profile manifest (name, config paths +
        sha256 hashes, overrides) into the DRL_AGENT_PROFILE_MANIFEST env var
        before exec'ing this trainer; here it lands as
        ``<configs|logs>/profile_manifest.json`` next to the run's other
        bookkeeping. Absent env var (every legacy invocation) -> no-op, and a
        failure never aborts the run (bookkeeping convention).
        """
        raw = os.environ.get("DRL_AGENT_PROFILE_MANIFEST", "").strip()
        if not raw:
            return
        try:
            import json
            manifest = json.loads(raw)
            manifest["run_dir"] = self.run_dir
            manifest["file_name"] = self.file_name
            target_dir = getattr(self, "configs_dir", None) or self.log_dir
            os.makedirs(target_dir, exist_ok=True)
            path = os.path.join(target_dir, "profile_manifest.json")
            with open(path, "w") as f:
                json.dump(manifest, f, indent=2)
            self.get_logger().info(f"[Profile] manifest written: {path}")
        except Exception as e:
            self.get_logger().warn(f"[Profile] manifest write failed (ignored): {e}")

    def _resolve_drl_agent_source_root(self) -> str:
        """Resolve the source-package root even when run from install/."""
        here = os.path.abspath(__file__)
        candidates = []

        src_env = os.environ.get("DRL_AGENT_SRC_PATH", "").strip()
        if src_env:
            src_env = os.path.expanduser(src_env)
            candidates.extend([
                os.path.join(src_env, "drl_agent"),
                os.path.join(src_env, "src", "drl_agent"),
                src_env,
            ])

        if "/install/" in here:
            ws_root = here.split("/install/")[0]
            candidates.append(os.path.join(ws_root, "src", "drl_agent"))

        cwd = os.path.abspath(os.getcwd())
        candidates.extend([
            os.path.join(cwd, "src", "drl_agent"),
            os.path.normpath(os.path.join(os.path.dirname(here), "..", "..")),
        ])

        for cand in candidates:
            if os.path.isdir(cand) and os.path.basename(cand) == "drl_agent":
                return os.path.normpath(cand)

        return os.path.normpath(os.path.join(os.path.dirname(here), "..", ".."))

    def create_directories(self):
        """Create all output directories."""
        for d in (
            self.run_dir,
            self.pytorch_models_dir,
            self.final_models_dir,
            self.results_dir,
            self.log_dir,
        ):
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  CSV logging                                                          #
    # ------------------------------------------------------------------ #

    def _aux_log_meta_cols(self):
        """AUX_ABLATION: [seed, aux_enabled, aux_version] for CSV stamping."""
        return aux_log.meta_columns(getattr(self, "seed", None),
                                    getattr(self, "rl_agent", None))

    def _init_csv_loggers(self):
        """Create a fresh set of per-run CSV log files."""
        self._csv_run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._reward_csv  = os.path.join(
            self.log_dir, f"episode_rewards_{self._csv_run_tag}.csv"
        )
        self._driving_csv = os.path.join(
            self.log_dir, f"episode_driving_{self._csv_run_tag}.csv"
        )
        self._step_csv = os.path.join(
            self.log_dir, f"policy_step_debug_{self._csv_run_tag}.csv"
        )

        # AUX_ABLATION: append [seed, aux_enabled, aux_version] to every per-run
        # CSV so runs can be grouped without a separate join (header == row).
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
        step_header = [
            "episode", "global_t", "episode_step", "action_source",
            "action_0_norm", "action_1_norm",
            "goal_dist_before_m", "goal_dist_after_m",
            "theta_before_rad", "theta_after_rad",
            "lidar_min_before_m", "lidar_min_after_m",
            "lidar_mean_before_m", "lidar_mean_after_m",
            "reward", "ep_finished", "target_flag",
        ]
        for path, header in [
            (self._reward_csv,  reward_header),
            (self._driving_csv, driving_header),
            (self._step_csv,    step_header),
        ]:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

        self.get_logger().info(f"Episode rewards CSV: {self._reward_csv}")
        self.get_logger().info(f"Episode driving CSV: {self._driving_csv}")
        self.get_logger().info(f"Policy step CSV: {self._step_csv}")

    def log_training_setting_data(self):
        """Log training configuration summary."""
        self.get_logger().info("=" * 50)
        self.get_logger().info("TQC Training Configuration")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"File name: {self.file_name}")
        self.get_logger().info(f"Seed: {self.seed}")
        self.get_logger().info(f"Run directory: {self.run_dir}")
        self.get_logger().info(f"PyTorch models: {self.pytorch_models_dir}")
        self.get_logger().info(f"Final models: {self.final_models_dir}")
        self.get_logger().info(f"Results: {self.results_dir}")
        self.get_logger().info(f"Logs: {self.log_dir}")
        self.get_logger().info(f"State dimension: {self.state_dim}")
        self.get_logger().info(f"Action dimension: {self.action_dim}")
        self.get_logger().info(f"Max action: {self.max_action}")
        self.get_logger().info(f"Max timesteps: {self.max_timesteps}")
        self.get_logger().info(f"Use checkpoints: {self.use_checkpoints}")
        self.get_logger().info("=" * 50)

    def _write_episode_logs(
        self,
        ep_num: int,
        global_t: int,
        ep_timesteps: int,
        ep_total_reward: float,
        state,
        info,
        episode_done: bool,
        episode_limit: bool,
        ep_v_buf: list,
        ep_w_buf: list,
        ep_min_lidar_buf: list,
        ep_gazebo_rtf_buf: list,
        ep_initial_goal_dist: float,
        eval_cut: bool = False,
        ep_cmd_v_buf: list | None = None,
        ep_cmd_steer_buf: list | None = None,
        ep_obs_speed_buf: list | None = None,
        ep_obs_yaw_rate_buf: list | None = None,
        ep_obs_steer_buf: list | None = None,
        low_obs_speed_threshold_mps: float | None = None,
    ):
        """Write episode summary logs and return the textual result label."""
        final_goal_dist = float(np.asarray(state, dtype=np.float32).ravel()[self.environment_dim])
        goal_reached = bool(episode_done and info) and not eval_cut
        collision    = bool(episode_done and (not goal_reached)) and not eval_cut
        timeout      = bool(episode_limit and (not episode_done)) and not eval_cut

        if goal_reached:
            result = "GOAL"
        elif collision:
            result = "COLLISION"
        elif eval_cut:
            result = "EVAL_CUT"
        else:
            result = "TIMEOUT"

        # AUX_ABLATION: run-identity columns appended to each row (header == row).
        _meta = self._aux_log_meta_cols()

        with open(self._reward_csv, "a", newline="") as _f:
            csv.writer(_f).writerow([
                ep_num, global_t, ep_timesteps,
                round(ep_total_reward, 4),
                round(ep_total_reward / max(ep_timesteps, 1), 4),
                int(goal_reached), int(collision), int(timeout), int(eval_cut),
                round(final_goal_dist, 4),
            ] + _meta)

        if ep_v_buf:
            driving_row = [
                ep_num, global_t, ep_timesteps,
                round(float(np.mean(ep_v_buf)), 4),
                round(float(np.mean(np.abs(ep_w_buf))), 4),
                round(ep_initial_goal_dist, 4),
                round(final_goal_dist, 4),
                round(ep_initial_goal_dist - final_goal_dist, 4),
                round(float(np.min(ep_min_lidar_buf)), 4),
                round(float(np.mean(ep_min_lidar_buf)), 4),
                int(goal_reached), int(eval_cut),
                round(float(np.mean(ep_gazebo_rtf_buf)), 4)
                if ep_gazebo_rtf_buf else float("nan"),
            ]
            if ep_cmd_v_buf is not None:
                driving_row.extend([
                    round(float(np.mean(ep_cmd_v_buf)), 4) if ep_cmd_v_buf else float("nan"),
                    round(float(np.mean(ep_cmd_steer_buf)), 4) if ep_cmd_steer_buf else float("nan"),
                    round(float(np.mean(ep_obs_speed_buf)), 4) if ep_obs_speed_buf else float("nan"),
                    round(float(np.mean(np.abs(ep_obs_yaw_rate_buf))), 4)
                    if ep_obs_yaw_rate_buf else float("nan"),
                    round(float(np.mean(ep_obs_steer_buf)), 4) if ep_obs_steer_buf else float("nan"),
                    round(float(np.mean([
                        abs(float(cv) - float(sv))
                        for cv, sv in zip(ep_cmd_v_buf, ep_obs_speed_buf)
                    ])), 4) if ep_cmd_v_buf and ep_obs_speed_buf else float("nan"),
                    round(float(np.mean([
                        1.0 if abs(float(sv)) < max(float(low_obs_speed_threshold_mps or 0.0), 1e-9) else 0.0
                        for sv in ep_obs_speed_buf
                    ])), 4) if ep_obs_speed_buf and low_obs_speed_threshold_mps is not None else float("nan"),
                ])
            with open(self._driving_csv, "a", newline="") as _f:
                csv.writer(_f).writerow(driving_row + _meta)

        return result

    # ------------------------------------------------------------------ #
    #  Model I/O                                                            #
    # ------------------------------------------------------------------ #

    def save_models(self, directory, file_name):
        """Save agent models to the given directory."""
        self.rl_agent.save(directory, file_name)
        self.get_logger().info(f"Models updated in {directory}")

    # ------------------------------------------------------------------ #
    #  ROS2 callbacks                                                       #
    # ------------------------------------------------------------------ #

    def _on_clock(self, msg: Clock):
        """Estimate Gazebo real-time factor from /clock."""
        sim_sec  = float(msg.clock.sec) + float(msg.clock.nanosec) * 1e-9
        wall_sec = time.monotonic()

        if self._last_sim_clock_sec is None or self._last_wall_clock_sec is None:
            self._last_sim_clock_sec  = sim_sec
            self._last_wall_clock_sec = wall_sec
            return

        delta_sim  = sim_sec  - self._last_sim_clock_sec
        delta_wall = wall_sec - self._last_wall_clock_sec
        self._last_sim_clock_sec  = sim_sec
        self._last_wall_clock_sec = wall_sec

        if delta_sim < 0.0 or delta_wall <= 1e-6:
            return

        self._latest_gazebo_rtf = delta_sim / delta_wall
