#!/usr/bin/env python3

import os
import sys
import rclpy
import time
from datetime import date

import torch
import numpy as np
from a3c_agent import Agent
from environment_interface import EnvInterface
from file_manager import DirectoryManager, load_yaml


class TrainA3C(EnvInterface):
    def __init__(self):
        # 노드명 TD7과 동일하게 맞춰 런치/스크립트 호환성 유지
        super().__init__("train_td7_node")

        # ----------------------------
        # Load training config (robust)
        # ----------------------------
        self.declare_parameter("train_config_file", "")
        cfg_param = self.get_parameter("train_config_file").get_parameter_value().string_value.strip()

        train_cfg_name = "train_a3c_config.yaml"
        hparams_name  = "hyperparameters_a3c.yaml"  # A3C용 하이퍼파라미터 파일

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
                os.path.normpath(os.path.join(here, "..", "config")),
                os.path.normpath(os.path.join(here, "..", "..", "config")),
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
        
        # Try A3C-specific hyperparameters file first, fallback to default
        self.hyperparameters_path = os.path.join(cfg_dir, hparams_name)
        if not os.path.isfile(self.hyperparameters_path):
            self.hyperparameters_path = os.path.join(cfg_dir, "hyperparameters.yaml")
            self.get_logger().info(f"A3C hyperparameters file not found, using default: {self.hyperparameters_path}")
        
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
        self.base_file_name = training_settings.get("base_file_name", "a3c_navigation")
        self.file_name = (
            f"{self.base_file_name}_seed_{self.seed}_{date.today().strftime('%Y%m%d')}"
        )

        # -------------------------------------------------------
        # Output directory setup (TD7와 동일 경로 구조로 통일)
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

        # TD7와 동일하게 temp 사용
        temp_dir_path = os.path.join(base_run_dir, "temp")
        self.pytorch_models_dir = os.path.join(temp_dir_path, "pytorch_models")
        self.final_models_dir   = os.path.join(temp_dir_path, "final_models")
        self.results_dir        = os.path.join(temp_dir_path, "results")
        self.log_dir            = os.path.join(temp_dir_path, "logs")

        for d in (temp_dir_path, self.pytorch_models_dir, self.final_models_dir, self.results_dir, self.log_dir):
            os.makedirs(d, exist_ok=True)

        # Set seed for reproducibility
        self.set_env_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Initialize the A3C agent
        try:
            hyperparameters = load_yaml(self.hyperparameters_path)
            # Support both formats
            if "hyperparameters" in hyperparameters:
                hyperparameters = hyperparameters["hyperparameters"]
        except Exception as e:
            self.get_logger().error(f"Unable to load hyperparameters file: {e}")
            sys.exit(-1)
            
        # Pass through TD7-style flags when missing
        if "use_checkpoints" not in hyperparameters:
            hyperparameters["use_checkpoints"] = self.use_checkpoints
            
        self.state_dim, self.action_dim, self.max_action = self.get_dimensions()
        self.rl_agent = Agent(
            self.state_dim,
            self.action_dim,
            self.max_action,
            hyperparameters,
            self.log_dir,
        )

        # Try to load the model if specified
        if self.load_model:
            try:
                self.rl_agent.load(self.pytorch_models_dir, self.file_name)
                self.get_logger().info("Model loaded successfully")
            except Exception as e:
                self.get_logger().warning(f"Failed to load the model: {e}")

        # Reset LSTM hidden states for new episode
        if hasattr(self.rl_agent, "reset_hidden_states"):
            self.rl_agent.reset_hidden_states(batch_size=1)
        
        # Training completion flag
        self.done_training = False
        
        # Episode tracking for LSTM reset
        self.current_episode_steps = 0

        self.log_training_setting_data()

    def log_training_setting_data(self):
        """Log general info at the start of training"""
        self.border = "+" + "-" * 80 + "+"
        self.get_logger().info(self.border)
        self.get_logger().info(f"| A3C-LSTM Navigation Training")
        self.get_logger().info(f"| File name: {self.file_name} | Seed: {self.seed}")
        self.get_logger().info(self.border)
        self.get_logger().info("| Results will be saved in:")
        self.get_logger().info(f"|  {self.pytorch_models_dir}")
        self.get_logger().info(f"|  {self.final_models_dir}")
        self.get_logger().info(f"|  {self.results_dir}")
        self.get_logger().info(f"|  {self.log_dir}")
        self.get_logger().info(self.border)
        self.get_logger().info("| Environment Configuration")
        self.get_logger().info(self.border)
        self.get_logger().info(f"| State Dim: {self.state_dim}")
        self.get_logger().info(f"| Action Dim: {self.action_dim}")
        self.get_logger().info(f"| Max Action: {self.max_action}")
        self.get_logger().info(f"| Max Episode Steps: {self.max_episode_steps}")
        self.get_logger().info(f"| Use Checkpoints: {self.use_checkpoints}")
        self.get_logger().info(self.border)

    def save_models(self, directory, file_name):
        """Save the models at the given step"""
        self.rl_agent.save(directory, file_name)
        self.get_logger().info("Models saved successfully")

    def train_online(self):
        """TD7와 동일한 외형의 온라인 학습 루프(온/오프폴리시 모두 수용)"""
        start_time = time.time()
        evals = []
        epoch = 1
        timesteps_since_eval = 0
        allow_train = False

        state, ep_finished = self.reset(), False
        ep_total_reward, ep_timesteps, ep_num = 0.0, 0, 1

        # 에피소드 시작 시 LSTM 리셋 보장
        if hasattr(self.rl_agent, "reset_hidden_states"):
            self.rl_agent.reset_hidden_states(batch_size=1)

        for t in range(1, self.max_timesteps + 1):
            # 행동 선택 (TD7와 동일한 시그니처 유지)
            if allow_train:
                action = self.rl_agent.select_action(np.array(state))
            else:
                action = self.sample_action_space()

            # 환경 진행
            next_state, reward, ep_finished, _ = self.step(action)
            ep_total_reward += float(reward)
            ep_timesteps += 1

            done_flag = float(ep_finished) if ep_timesteps < self.max_episode_steps else 0.0

            # 온/오프폴리시 에이전트 모두 수용하는 유연한 콜
            if hasattr(self.rl_agent, "store_transition"):
                self.rl_agent.store_transition(state, action, next_state, reward, done_flag)
            elif hasattr(self.rl_agent, "train_step"):
                # train_step이 (s,a,r,s',done) 단일 전이 기반일 때
                self.rl_agent.train_step(state, action, reward, next_state, done_flag)

            # 🔧 수정: 온폴리시는 항상 train(), 오프폴리시는 기존 로직 유지
            if allow_train:
                if getattr(self.rl_agent, "use_on_policy", False):
                    # A3C/A2C 온폴리시 → 항상 수행
                    self.rl_agent.train()
                elif not self.use_checkpoints:
                    # 오프폴리시형: 에이전트 내부 버퍼를 들고 train()만 호출하는 타입
                    if hasattr(self.rl_agent, "train_on_policy"):
                        self.rl_agent.train_on_policy()
                    else:
                        self.rl_agent.train()

            state = next_state

            # 에피소드 종료 처리
            if ep_finished or ep_timesteps == self.max_episode_steps:
                self.get_logger().info(
                    f"Total T: {t+1} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f}"
                )

                # 🔧 수정: 체크포인트 방식은 오프폴리시에만 호출
                if (allow_train and self.use_checkpoints and
                    not getattr(self.rl_agent, "use_on_policy", False) and
                    hasattr(self.rl_agent, "train_and_checkpoint")):
                    self.rl_agent.train_and_checkpoint(ep_timesteps, ep_total_reward)

                # 평가 타이밍
                if allow_train and timesteps_since_eval >= self.eval_freq:
                    timesteps_since_eval %= self.eval_freq
                    self.save_models(self.pytorch_models_dir, self.file_name)
                    self.evaluate_and_print(evals, epoch, start_time)
                    epoch += 1

                # 학습 시작 시점 도달 체크
                if t >= self.timesteps_before_training:
                    allow_train = True

                # 에피소드 리셋
                state = self.reset()
                ep_total_reward, ep_timesteps = 0.0, 0
                ep_num += 1
                # LSTM 상태 리셋
                if hasattr(self.rl_agent, "reset_hidden_states"):
                    self.rl_agent.reset_hidden_states(batch_size=1)

            timesteps_since_eval += 1

        self.done_training = True

    def evaluate_and_print(self, evals, epoch, start_time):
        """TD7 평가 루틴과 동일한 외형/저장 규약"""
        self.get_logger().info(self.border)
        self.get_logger().info(f"| Evaluation at epoch: {epoch}")
        self.get_logger().info(f"| Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")
    
        total_reward = np.zeros(self.eval_eps, dtype=np.float32)
        for ep in range(self.eval_eps):
            state, done = self.reset(), False
            ep_timesteps = 0
    
            # 평가 시엔 탐색 비활성화 + 체크포인트 플래그 유지
            while not done and ep_timesteps < self.max_episode_steps:
                action = self.rl_agent.select_action(
                    np.array(state),
                    self.use_checkpoints,
                    use_exploration=False
                )
                state, reward, done, _ = self.step(action)
                total_reward[ep] += float(reward)
                ep_timesteps += 1
    
            # LSTM 사용 시 평가 에피소드 경계에서 리셋
            if hasattr(self.rl_agent, "reset_hidden_states"):
                self.rl_agent.reset_hidden_states(batch_size=1)
    
        avg = float(total_reward.mean())
        self.get_logger().info(f"| Average reward over {self.eval_eps} episodes: {avg:.3f}")
        self.get_logger().info(self.border)
        evals.append(avg)
        np.save(f"{self.results_dir}/{self.file_name}", np.array(evals, dtype=np.float32))


def main(args=None):
    # Initialize the ROS2 communication
    rclpy.init(args=args)
    # Initialize the node
    train_a3c_node = TrainA3C()
    # Start training
    train_a3c_node.train_online()
    try:
        while rclpy.ok() and not train_a3c_node.done_training:
            rclpy.spin_once(train_a3c_node)
    except KeyboardInterrupt as e:
        train_a3c_node.get_logger().warning(f"KeyboardInterrupt: {e}")
    finally:
        train_a3c_node.get_logger().info("rclpy, shutting down...")
        train_a3c_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
