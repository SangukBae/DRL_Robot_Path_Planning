#!/usr/bin/env python3

import os
import sys
import time
from datetime import datetime

import rclpy
import torch
import numpy as np

from tqc_agent import Agent
from environment_interface import EnvInterface
from file_manager import DirectoryManager, load_yaml


class TrainTQC(EnvInterface):
    def __init__(self):
        """Initialize TQC training node"""
        super().__init__("train_tqc_node")

        # ----------------------------
        # Declare once: train_config_file
        # ----------------------------
        self.declare_parameter("train_config_file", "")
        user_param_path = self.get_parameter("train_config_file").get_parameter_value().string_value.strip()

        # ----------------------------
        # Find and load training configuration file
        # ----------------------------
        train_cfg_path = self._find_config_file("train_tqc_config.yaml", user_param_path)
        if not train_cfg_path:
            self.get_logger().error("Could not find 'train_tqc_config.yaml'")
            raise FileNotFoundError("train_tqc_config.yaml not found")

        train_settings = load_yaml(train_cfg_path)["train_settings"]
        self.seed = train_settings["seed"]
        self.max_episode_steps = train_settings["max_episode_steps"]
        self.load_model = train_settings["load_model"]
        self.max_timesteps = train_settings["max_timesteps"]
        self.use_checkpoints = train_settings["use_checkpoints"]
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

        # ----------------------------
        # Seeds
        # ----------------------------
        self.set_env_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # ----------------------------
        # Environment dimensions
        # ----------------------------
        state_dim, action_dim, max_action = self.get_dimensions()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # ----------------------------
        # Find and load hyperparameters
        # ----------------------------
        hparams_path = self._find_config_file("hyperparameters_tqc.yaml", user_param_path)
        if not hparams_path:
            self.get_logger().error("Could not find 'hyperparameters_tqc.yaml'")
            raise FileNotFoundError("hyperparameters_tqc.yaml not found")

        hyperparameters = load_yaml(hparams_path)["hyperparameters"]

        # ----------------------------
        # Create TQC agent
        # ----------------------------
        self.rl_agent = Agent(
            state_dim,
            action_dim,
            max_action,
            hyperparameters,
            log_dir=self.log_dir
        )

        def _find_latest_prefix(models_dir, base, seed):
            import glob, os
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
        # Determine run directory
        self.declare_parameter("run_dir", "")
        run_dir_param = self.get_parameter("run_dir").get_parameter_value().string_value.strip()

        if run_dir_param:
            base_run_dir = os.path.expanduser(run_dir_param)
        elif os.environ.get("DRL_AGENT_RUN_DIR", "").strip():
            base_run_dir = os.path.expanduser(os.environ["DRL_AGENT_RUN_DIR"])
        else:
            base_run_dir = os.path.join(os.path.expanduser("~"), ".ros", "drl_agent", "tqc_state_80_nstactics_5_obstacle_11")

        self.run_dir = base_run_dir

        # Subdirectories
        self.pytorch_models_dir = os.path.join(self.run_dir, "pytorch_models")
        self.final_models_dir   = os.path.join(self.run_dir, "final_models")
        self.results_dir        = os.path.join(self.run_dir, "results")
        self.log_dir            = os.path.join(self.run_dir, "logs")

        # Create directories
        self.create_directories()

    def create_directories(self):
        """Create necessary directories safely"""
        for d in (self.run_dir, self.pytorch_models_dir, self.final_models_dir, self.results_dir, self.log_dir):
            os.makedirs(d, exist_ok=True)
    
    def log_training_setting_data(self):
        """Log training configuration"""
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
        
        # Initialize episode
        state = self.reset()
        ep_total_reward = 0
        ep_timesteps = 0
        ep_num = 1
        ep_finished = False

        ENV_DIM = int(self.state_dim - 4)  # lidar 360 + agent_state 4 라고 가정
        def split_obs(obs):
            obs_np = np.asarray(obs, dtype=np.float32).ravel()
            lidar = obs_np[:ENV_DIM]
            agent = obs_np[ENV_DIM:ENV_DIM+4]
            return lidar, agent
        
        self.get_logger().info("Starting TQC training...")
        
        # Main training loop
        for t in range(1, self.max_timesteps + 1):
                # (추가) 200스텝마다 입력 점검
            # if t % 200 == 0:
            #     lidar, agent = split_obs(state)
            #     self.get_logger().info(
            #         f"[OBS] lidar.shape={lidar.shape}, agent={agent.tolist()} | "
            #         f"lidar min/med/max={lidar.min():.2f}/{np.median(lidar):.2f}/{lidar.max():.2f}"
            #     )
            #     if not np.isfinite(lidar).all():
            #         self.get_logger().warn("[OBS] lidar contains NaN/Inf!")
            #     if (lidar < 0).any():
            #         self.get_logger().warn("[OBS] lidar has negatives!")

            # Select action
            if allow_train:
                action = self.rl_agent.select_action(state)
            else:
                # Warmup phase - random actions
                action = self.sample_action_space()
            
            # Environment step
            next_state, reward, ep_finished, info = self.step(action)
            
            # Store transition in replay buffer
            done = float(ep_finished) if ep_timesteps < self.max_episode_steps else 0.0
            self.rl_agent.replay_buffer.add(state, action, next_state, reward, done)
            
            # Update state
            state = next_state
            ep_total_reward += reward
            ep_timesteps += 1
            
            # Train agent (if not using checkpoints)
            if allow_train and not self.use_checkpoints:
                self.rl_agent.train()
            
            # Episode finished
            if ep_finished or ep_timesteps >= self.max_episode_steps:
                self.get_logger().info(
                    f"Total T: {t} | Episode: {ep_num} | "
                    f"Episode T: {ep_timesteps} | Reward: {ep_total_reward:.3f}"
                )
                
                # Checkpoint training if enabled
                if self.use_checkpoints and allow_train:
                    self.rl_agent.train_and_checkpoint(ep_timesteps, ep_total_reward)
                
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
                # Deterministic evaluation
                action = self.rl_agent.select_action(
                    state,
                    use_checkpoint=self.use_checkpoints,
                    use_exploration=False
                )
                state, reward, done, _ = self.step(action)
                total_reward[ep] += reward
                ep_timesteps += 1
        
        mean_reward = np.mean(total_reward)
        std_reward = np.std(total_reward)
        
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
        node = TrainTQC()
        node.train_online()
        
        # Keep node alive for ROS communication
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