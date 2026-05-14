#!/usr/bin/env python3

import os
import sys
import time
from datetime import datetime

import csv

import rclpy
import torch
import numpy as np

from sb3_td3_agent import SB3TD3Agent as Agent
from environment_interface import EnvInterface
from file_manager import DirectoryManager, load_yaml


class TrainSB3TD3(EnvInterface):
    def __init__(self):
        """Initialize SB3 TD3 training node"""
        super().__init__("train_sb3_td3_node")

        # ----------------------------
        # Declare once: train_config_file
        # ----------------------------
        self.declare_parameter("train_config_file", "")
        user_param_path = self.get_parameter("train_config_file").get_parameter_value().string_value.strip()

        # ----------------------------
        # Find and load training configuration file
        # ----------------------------
        train_cfg_path = self._find_config_file("train_sb3_td3_config.yaml", user_param_path)
        if not train_cfg_path:
            self.get_logger().error("Could not find 'train_sb3_td3_config.yaml'")
            raise FileNotFoundError("train_sb3_td3_config.yaml not found")

        train_settings = load_yaml(train_cfg_path)["train_settings"]
        self.seed = train_settings["seed"]
        self.max_episode_steps = train_settings["max_episode_steps"]
        self.load_model = train_settings["load_model"]
        self.max_timesteps = train_settings["max_timesteps"]
        self.use_checkpoints = False  # SB3 agents do not use checkpoint mode
        self.eval_freq = train_settings["eval_freq"]
        self.timesteps_before_training = train_settings["timesteps_before_training"]
        self.eval_eps = train_settings["eval_eps"]
        base_file_name = train_settings["base_file_name"]

        # Create file name with date stamp
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
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.environment_dim = environment_dim
        self.agent_dim = agent_dim

        # ----------------------------
        # Find and load hyperparameters
        # ----------------------------
        hparams_path = self._find_config_file("hyperparameters_sb3_td3.yaml", user_param_path)
        if not hparams_path:
            self.get_logger().error("Could not find 'hyperparameters_sb3_td3.yaml'")
            raise FileNotFoundError("hyperparameters_sb3_td3.yaml not found")

        # SB3 config files have a flat YAML structure (no ["hyperparameters"] key)
        hyperparameters = load_yaml(hparams_path)

        # ----------------------------
        # Create SB3 TD3 agent
        # ----------------------------
        self.rl_agent = Agent(
            state_dim,
            action_dim,
            max_action,
            hyperparameters,
            log_dir=self.log_dir
        )

        def _find_latest_prefix(models_dir, base, seed):
            import glob
            pat = os.path.join(models_dir, f"{base}_seed_{seed}_*_actor.pth")
            cands = glob.glob(pat)
            if not cands:
                return None
            cands.sort(key=lambda p: os.stat(p).st_mtime, reverse=True)
            return os.path.basename(cands[0]).replace("_actor.pth", "")

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
                self.get_logger().info(f"Loaded model from {self.pytorch_models_dir}/{self.file_name}")
            except Exception as e:
                self.get_logger().warning(f"Could not load model: {e}")

        self.done_training = False

        # Log training configuration
        self.log_training_setting_data()

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

        # Not found
        self.get_logger().error(
            "Could not find config '{}'. Tried:\n  {}".format(filename, "\n  ".join(tried))
        )
        return None

    def _setup_directories(self):
        """Setup output directories"""
        self.declare_parameter("run_dir", "")
        run_dir_param = self.get_parameter("run_dir").get_parameter_value().string_value.strip()

        if run_dir_param:
            base_run_dir = os.path.expanduser(run_dir_param)
        elif os.environ.get("DRL_AGENT_RUN_DIR", "").strip():
            base_run_dir = os.path.expanduser(os.environ["DRL_AGENT_RUN_DIR"])
        else:
            package_root = self._resolve_drl_agent_source_root()
            base_run_dir = os.path.join(package_root, "runtime", "sb3_td3_state_80_nstactics_5_obstacle_11")

        self.run_dir = base_run_dir

        self.pytorch_models_dir = os.path.join(self.run_dir, "pytorch_models")
        self.final_models_dir   = os.path.join(self.run_dir, "final_models")
        self.results_dir        = os.path.join(self.run_dir, "results")
        self.log_dir            = os.path.join(self.run_dir, "logs")

        self.create_directories()

    def _resolve_drl_agent_source_root(self) -> str:
        """Resolve the source-package root even when this script is run from install/."""
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
        """Create necessary directories safely"""
        for d in (self.run_dir, self.pytorch_models_dir, self.final_models_dir,
                  self.results_dir, self.log_dir):
            os.makedirs(d, exist_ok=True)

    def _init_csv_loggers(self):
        """Create a fresh set of per-run CSV log files."""
        self._csv_run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._reward_csv  = os.path.join(
            self.log_dir, f"episode_rewards_{self._csv_run_tag}.csv"
        )
        self._driving_csv = os.path.join(
            self.log_dir, f"episode_driving_{self._csv_run_tag}.csv"
        )
        self._step_csv    = os.path.join(
            self.log_dir, f"policy_step_debug_{self._csv_run_tag}.csv"
        )
        reward_header  = ["episode", "global_t", "steps", "total_reward", "mean_reward",
                          "goal_reached", "collision", "timeout", "final_goal_dist_m"]
        driving_header = ["episode", "global_t", "steps", "mean_v_norm", "mean_abs_w_norm",
                          "initial_goal_dist_m", "final_goal_dist_m", "goal_dist_reduction_m",
                          "min_lidar_m", "mean_min_lidar_m", "goal_reached"]
        step_header = ["episode", "global_t", "episode_step", "action_source",
                       "action_0_norm", "action_1_norm",
                       "goal_dist_before_m", "goal_dist_after_m",
                       "theta_before_rad", "theta_after_rad",
                       "lidar_min_before_m", "lidar_min_after_m",
                       "lidar_mean_before_m", "lidar_mean_after_m",
                       "reward", "ep_finished", "target_flag"]
        for path, header in [(self._reward_csv, reward_header),
                             (self._driving_csv, driving_header),
                             (self._step_csv, step_header)]:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

        self.get_logger().info(f"Episode rewards CSV: {self._reward_csv}")
        self.get_logger().info(f"Episode driving CSV: {self._driving_csv}")
        self.get_logger().info(f"Policy step CSV: {self._step_csv}")

    def log_training_setting_data(self):
        """Log training configuration"""
        self.get_logger().info("=" * 50)
        self.get_logger().info("SB3 TD3 Training Configuration")
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

    def save_models(self, directory, file_name):
        """Save agent models"""
        self.rl_agent.save(directory, file_name)
        self.get_logger().info(f"Models updated in {directory}")

    def train_online(self):
        """Main training loop"""
        start_time = time.time()
        evals = []
        epoch = 1
        timesteps_since_eval = 0
        allow_train = False

        ENV_DIM = self.environment_dim

        # Initialize episode
        state = self.reset()
        ep_total_reward = 0
        ep_timesteps = 0
        ep_num = 1
        ep_finished = False

        _state0 = np.asarray(state, dtype=np.float32).ravel()
        _ep_v_buf: list         = []
        _ep_w_buf: list         = []
        _ep_min_lidar_buf: list = []
        _ep_initial_goal_dist   = float(_state0[ENV_DIM])

        self.get_logger().info("Starting SB3 TD3 training...")

        # Main training loop
        for t in range(1, self.max_timesteps + 1):
            _state_before = np.asarray(state, dtype=np.float32).ravel()
            _lidar_before = _state_before[:ENV_DIM]
            _goal_before  = float(_state_before[ENV_DIM])
            _theta_before = float(_state_before[ENV_DIM + 1])

            # Select action
            if allow_train:
                action_source = "policy"
                action = self.rl_agent.select_action(state)
            else:
                action_source = "warmup"
                action = self.sample_action_space()

            # Environment step
            next_state, reward, ep_finished, info = self.step(action)

            # Timeout penalty
            if ep_timesteps == self.max_episode_steps - 1 and not ep_finished:
                reward -= 20.0

            # Store transition in replay buffer
            done = float(ep_finished) if ep_timesteps < self.max_episode_steps else 0.0
            self.rl_agent.replay_buffer.add(state, action, next_state, reward, done)

            # Update state
            state = next_state
            ep_total_reward += reward
            ep_timesteps += 1

            # Accumulate per-step driving data
            _s_np = np.asarray(state, dtype=np.float32).ravel()
            _lidar_after = _s_np[:ENV_DIM]
            _ep_v_buf.append(float(action[0]))
            _ep_w_buf.append(float(action[1]))
            _ep_min_lidar_buf.append(float(np.min(_lidar_after)))

            with open(self._step_csv, "a", newline="") as _f:
                csv.writer(_f).writerow([
                    ep_num, t, ep_timesteps, action_source,
                    round(float(action[0]), 6), round(float(action[1]), 6),
                    round(_goal_before, 6), round(float(_s_np[ENV_DIM]), 6),
                    round(_theta_before, 6), round(float(_s_np[ENV_DIM + 1]), 6),
                    round(float(np.min(_lidar_before)), 6), round(float(np.min(_lidar_after)), 6),
                    round(float(np.mean(_lidar_before)), 6), round(float(np.mean(_lidar_after)), 6),
                    round(float(reward), 6), int(bool(ep_finished)), int(bool(info)),
                ])

            # Train agent (off-policy, each step)
            if allow_train:
                self.rl_agent.train()

            # Episode finished
            if ep_finished or ep_timesteps >= self.max_episode_steps:
                _goal_reached = bool(info) and bool(ep_finished)
                _collision    = bool(ep_finished) and not _goal_reached
                _timeout      = not bool(ep_finished)
                if _goal_reached:
                    _result = "GOAL"
                elif _collision:
                    _result = "COLLISION"
                else:
                    _result = "TIMEOUT"

                self.get_logger().info(
                    f"Total T: {t} | Episode: {ep_num} | "
                    f"Episode T: {ep_timesteps} | Reward: {ep_total_reward:.3f} | "
                    f"Result: {_result}"
                )

                # CSV logging
                _final_goal_dist = float(np.asarray(state, dtype=np.float32).ravel()[ENV_DIM])
                with open(self._reward_csv, "a", newline="") as _f:
                    csv.writer(_f).writerow([
                        ep_num, t, ep_timesteps,
                        round(ep_total_reward, 4),
                        round(ep_total_reward / max(ep_timesteps, 1), 4),
                        int(_goal_reached), int(_collision), int(_timeout),
                        round(_final_goal_dist, 4),
                    ])
                if _ep_v_buf:
                    with open(self._driving_csv, "a", newline="") as _f:
                        csv.writer(_f).writerow([
                            ep_num, t, ep_timesteps,
                            round(float(np.mean(_ep_v_buf)), 4),
                            round(float(np.mean(np.abs(_ep_w_buf))), 4),
                            round(_ep_initial_goal_dist, 4),
                            round(_final_goal_dist, 4),
                            round(_ep_initial_goal_dist - _final_goal_dist, 4),
                            round(float(np.min(_ep_min_lidar_buf)), 4),
                            round(float(np.mean(_ep_min_lidar_buf)), 4),
                            int(_goal_reached),
                        ])

                # Evaluation
                if allow_train and timesteps_since_eval >= self.eval_freq:
                    timesteps_since_eval %= self.eval_freq
                    self.save_models(self.pytorch_models_dir, self.file_name)
                    self.evaluate_and_print(evals, epoch, start_time)
                    epoch += 1

                # Enable training after warmup
                if t >= self.timesteps_before_training:
                    allow_train = True

                # Reset episode
                state = self.reset()
                ep_total_reward = 0
                ep_timesteps = 0
                ep_num += 1
                ep_finished = False

                # Reset per-episode buffers
                _ep_v_buf.clear()
                _ep_w_buf.clear()
                _ep_min_lidar_buf.clear()
                _ep_initial_goal_dist = float(np.asarray(state, dtype=np.float32).ravel()[ENV_DIM])

            timesteps_since_eval += 1

        # Training complete
        self.get_logger().info("Training completed!")
        self.save_models(self.final_models_dir, self.file_name)
        self.done_training = True

    def evaluate_and_print(self, evals, epoch, start_time):
        """Evaluate agent performance"""
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Evaluating Epoch {epoch}")
        self.get_logger().info(f"Time elapsed: {time.time() - start_time:.2f}s")
        self.get_logger().info("=" * 50)

        total_reward = np.zeros(self.eval_eps)

        for ep in range(self.eval_eps):
            state = self.reset()
            done = False
            ep_timesteps = 0

            while not done and ep_timesteps < self.max_episode_steps:
                # Deterministic evaluation (no exploration noise)
                action = self.rl_agent.select_action(state, use_exploration=False)
                state, reward, done, _ = self.step(action)
                total_reward[ep] += reward
                ep_timesteps += 1

        mean_reward = np.mean(total_reward)
        std_reward  = np.std(total_reward)

        self.get_logger().info(
            f"Evaluation over {self.eval_eps} episodes: "
            f"{mean_reward:.3f} ± {std_reward:.3f}"
        )

        evals.append(mean_reward)
        np.save(f"{self.results_dir}/{self.file_name}", evals)

        return mean_reward


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)

    try:
        node = TrainSB3TD3()
        node.train_online()

        while rclpy.ok() and not node.done_training:
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
