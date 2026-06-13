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

# Support direct script execution from the source tree as well as ros2 run
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "environment")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
)

from tqc_agent import Agent
from environment_interface import EnvInterface
from file_manager import load_yaml
# AUX_ABLATION: run-identity / ablation logging helpers.
import aux_ablation_logging as aux_log


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
        self.max_timesteps              = train_settings["max_timesteps"]
        self.use_checkpoints            = train_settings["use_checkpoints"]
        self.eval_freq                  = train_settings["eval_freq"]
        self.timesteps_before_training  = train_settings["timesteps_before_training"]
        self.eval_eps                   = train_settings["eval_eps"]
        base_file_name                  = train_settings["base_file_name"]

        current_date = datetime.now().strftime("%Y%m%d")
        self.file_name = f"{base_file_name}_seed_{self.seed}_{current_date}"

        # ----------------------------
        # Setup output directories
        # ----------------------------
        self._setup_directories()
        self._init_csv_loggers()

        # ----------------------------
        # Seeds
        # ----------------------------
        self.set_env_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

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

        # ----------------------------
        # Create TQC agent
        # ----------------------------
        self.rl_agent = Agent(
            state_dim,
            action_dim,
            max_action,
            hyperparameters,
            log_dir=self.log_dir,
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
    #  Config discovery                                                     #
    # ------------------------------------------------------------------ #

    def _find_config_file(self, filename: str, user_param_path: str | None = None) -> str | None:
        """
        Robust config discovery.
        - If user_param_path is a file: return it.
        - If user_param_path is a directory: join with filename.
        - Else search: ament share -> env vars -> DRL_AGENT_SRC_PATH candidates -> repo-relative.
        """
        tried = []

        # 0) User-provided parameter
        if user_param_path:
            p = os.path.expanduser(user_param_path)
            if os.path.isfile(p):
                base = os.path.dirname(p)
                cand = os.path.join(base, filename)
                if os.path.isfile(cand):
                    return cand
                tried.append(cand)
            elif os.path.isdir(p):
                cand = os.path.join(p, filename)
                if os.path.isfile(cand):
                    return cand
                tried.append(cand)
            else:
                tried.append(p)

        # 1) ament share directory
        try:
            from ament_index_python.packages import get_package_share_directory
            share_dir = get_package_share_directory("drl_agent")
            cand = os.path.join(share_dir, "config", filename)
            if os.path.isfile(cand):
                return cand
            tried.append(cand)
        except Exception:
            pass

        # 2) Environment variable: DRL_AGENT_TRAIN_CONFIG (file or dir)
        env_full = os.environ.get("DRL_AGENT_TRAIN_CONFIG", "").strip()
        if env_full:
            env_full = os.path.expanduser(env_full)
            if os.path.isfile(env_full):
                base = os.path.dirname(env_full)
                cand = os.path.join(base, filename)
                if os.path.isfile(cand):
                    return cand
                tried.append(cand)
            elif os.path.isdir(env_full):
                cand = os.path.join(env_full, filename)
                if os.path.isfile(cand):
                    return cand
                tried.append(cand)
            else:
                tried.append(env_full)

        # 3) DRL_AGENT_SRC_PATH candidates
        src = os.environ.get("DRL_AGENT_SRC_PATH", "").strip()
        if src:
            src = os.path.expanduser(src)
            candidates = [
                os.path.join(src, "drl_agent", "config"),
                os.path.join(src, "src", "drl_agent", "config"),
                os.path.join(src, "src", "drl_agent", "src", "drl_agent", "config"),
                os.path.join(src, "config"),
            ]
            for d in candidates:
                cand = os.path.join(d, filename)
                if os.path.isfile(cand):
                    return cand
                tried.append(cand)

        # 4) Relative to this script (two common layouts)
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.normpath(os.path.join(here, "..", "config", filename)),
            os.path.normpath(os.path.join(here, "..", "..", "config", filename)),
        ]
        for cand in candidates:
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
            with open(self._driving_csv, "a", newline="") as _f:
                csv.writer(_f).writerow([
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
                ] + _meta)

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
