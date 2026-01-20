#!/usr/bin/env python3

import os
import sys
import rclpy
import time
from datetime import date

import torch
import numpy as np
from sac_agent import Agent  # TD7からSACに変更
from environment_interface import EnvInterface
from file_manager import DirectoryManager, load_yaml


class TrainSAC(EnvInterface):  # クラス名をTrainSACに変更
    def __init__(self):
        super().__init__("train_sac_node")  # ノード名をSAC用に変更

        # ----------------------------
        # Load training config (robust)
        # ----------------------------
        self.declare_parameter("train_config_file", "")
        cfg_param = self.get_parameter("train_config_file").get_parameter_value().string_value.strip()

        train_cfg_name = "train_sac_config.yaml"
        hparams_name  = "hyperparameters_sac.yaml"  # SAC用のハイパーパラメータファイル

        cfg_dir = None
        candidates = []
        tried = []

        # 1) 사용자 파라미터(전체 파일 경로)
        if cfg_param:
            p = os.path.expanduser(cfg_param)
            if os.path.isfile(p):
                cfg_dir = os.path.dirname(p)
            else:
                tried.append(p)

        # 2) 설치된 share 경로
        if cfg_dir is None:
            try:
                from ament_index_python.packages import get_package_share_directory
                candidates.append(os.path.join(get_package_share_directory("drl_agent"), "config"))
            except Exception:
                pass

        # 3) 환경변수: DRL_AGENT_TRAIN_CONFIG (전체 파일 경로)
        if cfg_dir is None:
            env_full = os.environ.get("DRL_AGENT_TRAIN_CONFIG", "").strip()
            if env_full:
                env_full = os.path.expanduser(env_full)
                if os.path.isfile(env_full):
                    cfg_dir = os.path.dirname(env_full)
                else:
                    tried.append(env_full)

        # 4) 환경변수: DRL_AGENT_SRC_PATH 기반 후보들
        if cfg_dir is None:
            src = os.environ.get("DRL_AGENT_SRC_PATH", "").strip()
            if src:
                candidates += [
                    os.path.join(src, "drl_agent", "config"),
                    os.path.join(src, "src", "drl_agent", "config"),
                    os.path.join(src, "src", "drl_agent", "src", "drl_agent", "config"),
                    os.path.join(src, "config"),
                ]

        # 5) 소스 트리 상대 경로(개발 중 편의)
        if cfg_dir is None:
            here = os.path.dirname(os.path.abspath(__file__))
            candidates += [
                os.path.normpath(os.path.join(here, "..", "config")),       # .../scripts/config (혹시)
                os.path.normpath(os.path.join(here, "..", "..", "config")), # .../drl_agent/config
            ]

        # 후보들에서 train_config.yaml 탐색
        for d in candidates:
            p = os.path.join(d, train_cfg_name)
            if os.path.isfile(p):
                cfg_dir = d
                break
            tried.append(p)

        if cfg_dir is None:
            self.get_logger().error(
                "Could not find '{}'. Tried:\n  {}".format(train_cfg_name, "\n  ".join(tried))
            )
            sys.exit(-1)

        self.train_config_file_path = os.path.join(cfg_dir, train_cfg_name)
        self.hyperparameters_path   = os.path.join(cfg_dir, hparams_name)
        self.get_logger().info(f"Using train config: {self.train_config_file_path}")
        self.get_logger().info(f"Using hyperparameters: {self.hyperparameters_path}")

        # Load config file
        try:
            training_settings = load_yaml(self.train_config_file_path)["train_settings"]
        except Exception as e:
            self.get_logger().error(f"Unable to load config file: {e}")
            sys.exit(-1)

        # Extract training settings
        self.seed = training_settings["seed"]
        self.max_episode_steps = training_settings["max_episode_steps"]
        self.load_model = training_settings["load_model"]
        self.max_timesteps = training_settings["max_timesteps"]
        self.use_checkpoints = training_settings["use_checkpoints"]
        self.eval_freq = training_settings["eval_freq"]
        self.timesteps_before_training = training_settings["timesteps_before_training"]
        self.eval_eps = training_settings["eval_eps"]
        self.base_file_name = training_settings.get("base_file_name_sac", "sac_agent")  # SAC用のベース名
        self.file_name = (
            f"{self.base_file_name}_seed_{self.seed}_{date.today().strftime('%Y%m%d')}"
        )

        # -------------------------------------------------------
        # (수정) 출력/임시 디렉토리 설정: drl_agent_src_path 의존 제거
        # 우선순위: ROS 파라미터 run_dir → 환경변수 DRL_AGENT_RUN_DIR → 기본(~/.ros/drl_agent)
        # -------------------------------------------------------
        self.declare_parameter("run_dir", "")
        _run_param = self.get_parameter("run_dir").get_parameter_value().string_value.strip()

        if _run_param:
            base_run_dir = os.path.expanduser(_run_param)
        elif os.environ.get("DRL_AGENT_RUN_DIR"):
            base_run_dir = os.path.expanduser(os.environ["DRL_AGENT_RUN_DIR"])
        else:
            base_run_dir = os.path.join(os.path.expanduser("~"), ".ros", "drl_agent")

        os.makedirs(base_run_dir, exist_ok=True)

        # 기존 구조를 유지: base_run_dir/temp 아래에 모델/결과/로그 폴더 구성
        # SAC용 디렉토리를 분리하여 TD7와 구분
        temp_dir_path = os.path.join(base_run_dir, "temp_sac")
        self.pytorch_models_dir = os.path.join(temp_dir_path, "pytorch_models")
        self.final_models_dir   = os.path.join(temp_dir_path, "final_models")
        self.results_dir        = os.path.join(temp_dir_path, "results")
        self.log_dir            = os.path.join(temp_dir_path, "logs")

        for d in (temp_dir_path, self.pytorch_models_dir, self.final_models_dir, self.results_dir, self.log_dir):
            os.makedirs(d, exist_ok=True)
            
        # Set seed
        self.set_env_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Initialize the SAC agent
        try:
            hyperparameters = load_yaml(self.hyperparameters_path)["hyperparameters"]
        except Exception as e:
            self.get_logger().error(f"Unable to load hyperparameters file: {e}")
            sys.exit(-1)
            
        self.state_dim, self.action_dim, self.max_action = self.get_dimensions()
        self.rl_agent = Agent(
            self.state_dim,
            self.action_dim,
            self.max_action,
            hyperparameters,
            self.log_dir,
        )

        # Try to load the model
        if self.load_model:
            try:
                self.rl_agent.load(self.pytorch_models_dir, self.file_name)
                self.get_logger().info("SAC Model loaded")
            except Exception as e:
                self.get_logger().warning(f"Failed to load the SAC model: {e}")

        # Flag to indicate that the training is done
        self.done_training = False

        self.log_training_setting_data()

    def log_training_setting_data(self):
        """Log general info at the start of training"""
        self.border = "+" + "-" * 80 + "+"
        self.get_logger().info(self.border)
        self.get_logger().info(f"| SAC Agent Training")
        self.get_logger().info(f"| File name: {self.file_name}: Seed: {self.seed}")
        self.get_logger().info(self.border)
        self.get_logger().info("| Results will be saved in:")
        self.get_logger().info(f"|  {self.pytorch_models_dir}")
        self.get_logger().info(f"|  {self.final_models_dir}")
        self.get_logger().info(f"|  {self.results_dir}")
        self.get_logger().info(f"|  {self.log_dir}")
        self.get_logger().info(self.border)
        self.get_logger().info("| Environment")
        self.get_logger().info(self.border)
        self.get_logger().info(f"| State Dim: {self.state_dim}")
        self.get_logger().info(f"| Action Dim: {self.action_dim}")
        self.get_logger().info(f"| Max Action: {self.max_action}")
        self.get_logger().info(self.border)

    def create_directories(self):
        """Create directories for saving models"""
        directories = [
            self.pytorch_models_dir,
            self.final_models_dir,
            self.results_dir,
            self.log_dir,
        ]
        for dir in directories:
            dir_manager = DirectoryManager(dir)
            dir_manager.remove_if_present()
            dir_manager.create()

    def save_models(self, directory, file_name):
        """Save the models at the given step"""
        self.rl_agent.save(directory, file_name)
        self.get_logger().info("SAC Models updated")

    def train_online(self):
        """Train the SAC agent online"""
        # Initialize the variables
        start_time = time.time()
        evals = []
        epoch = 1
        timesteps_since_eval = 0
        allow_train = False

        state, ep_finished = self.reset(), False
        ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

        for t in range(1, self.max_timesteps + 1):

            if allow_train:
                # SACは探索的な行動選択をデフォルトで行う
                action = self.rl_agent.select_action(np.array(state), use_exploration=True)
            else:
                action = self.sample_action_space()

            # Act
            next_state, reward, ep_finished, _ = self.step(action)

            ep_total_reward += reward
            ep_timesteps += 1

            done = float(ep_finished) if ep_timesteps < self.max_episode_steps else 0
            self.rl_agent.replay_buffer.add(state, action, next_state, reward, done)

            state = next_state

            if allow_train and not self.use_checkpoints:
                self.rl_agent.train()

            if ep_finished or ep_timesteps == self.max_episode_steps:
                self.get_logger().info(
                    f"Total T: {t+1} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f}"
                )
                if allow_train and self.use_checkpoints:
                    self.rl_agent.train_and_checkpoint(ep_timesteps, ep_total_reward)

                if allow_train and timesteps_since_eval >= self.eval_freq:
                    timesteps_since_eval %= self.eval_freq
                    # Save the models
                    self.save_models(self.pytorch_models_dir, self.file_name)
                    self.evaluate_and_print(evals, epoch, start_time)
                    epoch += 1

                if t >= self.timesteps_before_training:
                    allow_train = True

                state, done = self.reset(), False
                ep_total_reward, ep_timesteps = 0, 0
                ep_num += 1

            timesteps_since_eval += 1
            
        # Final save
        self.save_models(self.final_models_dir, self.file_name)
        self.get_logger().info("Training completed! Final models saved.")
        
        # Indicate that the training is done
        self.done_training = True

    def evaluate_and_print(self, evals, epoch, start_time):
        """Evaluate the SAC agent and print the results"""

        self.get_logger().info(self.border)
        self.get_logger().info(f"| SAC Evaluation at epoch: {epoch}")
        self.get_logger().info(
            f"| Total time passed: {round((time.time()-start_time)/60.,2)} min(s)"
        )

        total_reward = np.zeros(self.eval_eps)
        for ep in range(self.eval_eps):
            state, done = self.reset(), False
            ep_timesteps = 0
            while not done and ep_timesteps < self.max_episode_steps:
                # 評価時は決定的な行動（平均値）を使用
                action = self.rl_agent.select_action(
                    np.array(state), self.use_checkpoints, use_exploration=False
                )
                # Act
                state, reward, done, _ = self.step(action)
                total_reward[ep] += reward
                ep_timesteps += 1

        avg_reward = total_reward.mean()
        std_reward = total_reward.std()
        
        self.get_logger().info(
            f"| Average reward over {self.eval_eps} episodes: {avg_reward:.3f} (+/- {std_reward:.3f})"
        )
        self.get_logger().info(self.border)
        evals.append(avg_reward)
        np.save(f"{self.results_dir}/{self.file_name}", evals)


def main(args=None):
    # Initialize the ROS2 communication
    rclpy.init(args=args)
    # Initialize the SAC training node
    train_sac_node = TrainSAC()
    # Start training
    train_sac_node.train_online()
    try:
        while rclpy.ok() and not train_sac_node.done_training:
            rclpy.spin_once(train_sac_node)
    except KeyboardInterrupt as e:
        train_sac_node.get_logger().warning(f"KeyboardInterrupt: {e}")
    finally:
        train_sac_node.get_logger().info("SAC training shutting down...")
        train_sac_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()