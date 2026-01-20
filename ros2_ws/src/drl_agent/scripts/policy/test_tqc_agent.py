#!/usr/bin/env python3

import os
import sys
import time
import torch
import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from nav_msgs.msg import Odometry

# ⇩ 기존과 동일한 인터페이스/유틸
from environment_interface import EnvInterface
from file_manager import load_yaml, save_yaml, save_json

# ⇩ TD7 대신 TQC 사용
from tqc_agent import Agent


class TestTQC(EnvInterface):
    def __init__(self):
        super().__init__("test_tqc_agent")
    
        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter("test_mode", "random_test")
        self.declare_parameter("test_config_file", "")   # file or dir
        self.declare_parameter("run_dir", "")            # default: ~/.ros/drl_agent/tqc_test
        self.declare_parameter("models_dir", "")         # default: <run_dir>/pytorch_models
        self.declare_parameter("num_episodes", 100)        # 0 → infinite
        self.declare_parameter("odom_topic", "/bunker/odom")

        
        # __init__ 맨 위 파라미터 선언부에 추가
        self.declare_parameter("checkpoint_actor_file", "/root/.ros/drl_agent/tqc_state_80_nstactics_5_obstacle_11/final_models/tqc_agent_seed_0_20251018_actor.pth")
        
        # ▶️ 실시간 출력 제어(스텝 주기 출력은 없음)
        self.declare_parameter("print_live", True)
        self.declare_parameter("echo_on_save", True)
    
        self.test_mode = self.get_parameter("test_mode").get_parameter_value().string_value.lower()
        if self.test_mode not in ["test", "random_test"]:
            raise NotImplementedError("test_mode must be 'test' or 'random_test'")
        self.random_train_mode = (self.test_mode == "random_test")
        self.get_logger().info(f"Test run mode: {self.test_mode}")

        # ▶️ 파라미터 값 읽기
        self.print_live = self.get_parameter("print_live").get_parameter_value().bool_value
        self.echo_on_save = self.get_parameter("echo_on_save").get_parameter_value().bool_value
    
        # ----------------------------
        # Run dir & subdirs
        # ----------------------------
        run_dir_param = self.get_parameter("run_dir").get_parameter_value().string_value.strip()
        if run_dir_param:
            base_run_dir = os.path.expanduser(run_dir_param)
        elif os.environ.get("DRL_AGENT_RUN_DIR", "").strip():
            base_run_dir = os.path.expanduser(os.environ["DRL_AGENT_RUN_DIR"])
        else:
            base_run_dir = os.path.join(os.path.expanduser("~"), ".ros", "drl_agent", "tqc_test")
        self.run_dir = base_run_dir
    
        self.pytorch_models_dir = self.get_parameter("models_dir").get_parameter_value().string_value.strip()
        if not self.pytorch_models_dir:
            self.pytorch_models_dir = os.path.join(self.run_dir, "pytorch_models")
        self.results_dir = os.path.join(self.run_dir, "results")
        self.log_dir = os.path.join(self.run_dir, "logs")
        self.test_metric_dir = os.path.join(self.run_dir, "test_tqc_runs")
        for d in (self.run_dir, self.pytorch_models_dir, self.results_dir, self.log_dir, self.test_metric_dir):
            os.makedirs(d, exist_ok=True)
    
        # ----------------------------
        # Robust config discovery (local helpers)
        # ----------------------------
        def _find_config_file(filename: str, user_param_path: str | None) -> str | None:
            tried = []
            # 0) user-provided (file or dir)
            if user_param_path:
                p = os.path.expanduser(user_param_path)
                if os.path.isfile(p):
                    base = os.path.dirname(p); cand = os.path.join(base, filename)
                    if os.path.isfile(cand): return cand
                    tried.append(cand)
                elif os.path.isdir(p):
                    cand = os.path.join(p, filename)
                    if os.path.isfile(cand): return cand
                    tried.append(cand)
                else:
                    tried.append(p)
            # 1) ament share
            try:
                from ament_index_python.packages import get_package_share_directory
                share_dir = get_package_share_directory("drl_agent")
                cand = os.path.join(share_dir, "config", filename)
                if os.path.isfile(cand): return cand
                tried.append(cand)
            except Exception:
                pass
            # 2) env: DRL_AGENT_TRAIN_CONFIG (file or dir)
            env_full = os.environ.get("DRL_AGENT_TRAIN_CONFIG", "").strip()
            if env_full:
                env_full = os.path.expanduser(env_full)
                if os.path.isfile(env_full):
                    base = os.path.dirname(env_full); cand = os.path.join(base, filename)
                    if os.path.isfile(cand): return cand
                    tried.append(cand)
                elif os.path.isdir(env_full):
                    cand = os.path.join(env_full, filename)
                    if os.path.isfile(cand): return cand
                    tried.append(cand)
                else:
                    tried.append(env_full)
            # 3) env: DRL_AGENT_SRC_PATH candidates
            src = os.environ.get("DRL_AGENT_SRC_PATH", "").strip()
            if src:
                src = os.path.expanduser(src)
                for d in [
                    os.path.join(src, "drl_agent", "config"),
                    os.path.join(src, "src", "drl_agent", "config"),
                    os.path.join(src, "src", "drl_agent", "src", "drl_agent", "config"),
                    os.path.join(src, "config"),
                ]:
                    cand = os.path.join(d, filename)
                    if os.path.isfile(cand): return cand
                    tried.append(cand)
            # 4) relative to this script
            here = os.path.dirname(os.path.abspath(__file__))
            for cand in [
                os.path.normpath(os.path.join(here, "..", "config", filename)),
                os.path.normpath(os.path.join(here, "..", "..", "config", filename)),
            ]:
                if os.path.isfile(cand): return cand
                tried.append(cand)
            self.get_logger().error("Could not find config '{}'. Tried:\n  {}".format(filename, "\n  ".join(tried)))
            return None
    
        # ----------------------------
        # Load test config & hyperparameters
        # ----------------------------
        user_param_path = self.get_parameter("test_config_file").get_parameter_value().string_value.strip()
        test_cfg_path = _find_config_file("test_tqc_config.yaml", user_param_path)
        if not test_cfg_path:
            sys.exit(255)
        try:
            self.test_config = load_yaml(test_cfg_path)["test_settings"]
        except Exception as e:
            self.get_logger().error(f"Unable to load test config file: {e}")
            sys.exit(255)
    
        hparams_path = _find_config_file("hyperparameters_tqc.yaml", user_param_path)
        if not hparams_path:
            self.get_logger().error("Could not find 'hyperparameters_tqc.yaml'")
            sys.exit(255)
        try:
            hp = load_yaml(hparams_path)["hyperparameters"]
        except Exception as e:
            self.get_logger().error(f"Unable to load hyperparameters file: {e}")
            sys.exit(255)
    
        # ----------------------------
        # Settings
        # ----------------------------
        self.seed = int(self.test_config.get("seed", 0))
        self.use_checkpoints = bool(self.test_config.get("use_checkpoints", True))
        self.max_episode_steps = int(self.test_config.get("max_episode_steps", 500))
        base_file_name = self.test_config.get("base_file_name", "tqc_agent")
        save_date = str(self.test_config.get("save_date", "")).strip()
        # 경로 생성 모드: actor.pth만 사용하도록 고정
        if self.use_checkpoints:
            self.get_logger().warn("Ignoring 'use_checkpoints' from config: forcing actor.pth-only mode.")
        self.use_checkpoints = False
    
        # ----------------------------
        # Seeding
        # ----------------------------
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.set_env_seed(self.seed)
    
        # ----------------------------
        # Env dims & Agent
        # ----------------------------
        state_dim, action_dim, max_action = self.get_dimensions()
        self.rl_agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            hyperparameters=hp,
        )

        # ----------------------------
        # Load actor-only weights (safe load with fallback)
        # ----------------------------
        ckpt_path = self.get_parameter("checkpoint_actor_file").get_parameter_value().string_value.strip()
        ckpt_path = os.path.expanduser(ckpt_path)
        if not os.path.isfile(ckpt_path):
            self.get_logger().error(f"Actor weight file not found: {ckpt_path}")
            sys.exit(255)

        try:
            # 새 방식: 가중치만 안전하게 로드 (PyTorch 최신)
            sd = torch.load(ckpt_path, map_location=self.rl_agent.device, weights_only=True)  # type: ignore[arg-type]
        except TypeError:
            # 구버전 호환: weights_only 인자가 없으면 경고 감수하고 로드
            obj = torch.load(ckpt_path, map_location=self.rl_agent.device)
            if isinstance(obj, dict) and "state_dict" in obj:
                sd = obj["state_dict"]
            else:
                sd = obj

        try:
            self.rl_agent.actor.load_state_dict(sd, strict=False)
            # (선택) checkpoint_actor도 동일 가중치로 맞춰두면 옵션 전환 시 안전
            if hasattr(self.rl_agent, "checkpoint_actor") and self.rl_agent.checkpoint_actor is not None:
                self.rl_agent.checkpoint_actor.load_state_dict(self.rl_agent.actor.state_dict(), strict=False)

            self.rl_agent.actor.eval()
            self.get_logger().info(f"Loaded ACTOR weights from: {ckpt_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load actor weights: {e}")
            sys.exit(255)
    
        # ----------------------------
        # Odom subscription (for trajectory)
        # ----------------------------
        self.odom_callback_group = MutuallyExclusiveCallbackGroup()
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value.strip() or "odom"
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, qos_profile,
            callback_group=self.odom_callback_group
        )
        self.last_odom = None
    
        # ----------------------------
        # Metrics & episode control
        # ----------------------------
        self.all_episode_times = []
        self.all_episode_distances = []
        self.all_trajectories = []
        self.target_reached_counter = 0.0
        self.num_episodes_counter = 0.0

        # ✅ 에피소드별 요약 누적 리스트 (JSON 저장용)
        self.all_episode_summaries = []

    
        self.num_episodes_target = int(self.get_parameter("num_episodes").get_parameter_value().integer_value)
        if self.num_episodes_target < 0:
            self.num_episodes_target = 0
    
        # Start test
        self.test()

    def odom_callback(self, od_data):
        """Keep latest odometry for trajectory logging."""
        self.last_odom = od_data

    def test(self):
        """Continuous evaluation loop."""
        done = False
        current_trajectory = []
        episode_timesteps = 0
        episode_return = 0.0
        state = self.reset()

        episode_start_time = time.time()
        while True:
            # 탐험 없이 정책 평가 (use_checkpoints 옵션 유지)
            action = self.rl_agent.select_action(
                np.array(state),
                use_checkpoint= False,
                use_exploration=False,
            )
            next_state, reward, done, target = self.step(action)
            done = 1 if episode_timesteps + 1 == self.max_episode_steps else int(done)
            episode_return += float(reward)

            # 현재 위치 기록
            if self.last_odom is not None:
                x = self.last_odom.pose.pose.position.x
                y = self.last_odom.pose.pose.position.y
                current_trajectory.append({"x": float(x), "y": float(y)})

            if done:
                # 궤적 누적 및 카운트
                self.all_trajectories.append(current_trajectory)
                self.num_episodes_counter += 1.0

                # 목표 도달 시 시간/거리 통계 반영
                if target:
                    self.all_episode_distances.append(
                        self.calculate_distance(current_trajectory)
                    )
                    self.all_episode_times.append(time.time() - episode_start_time)
                    self.target_reached_counter += 1.0

                # ✅ 에피소드 요약(steps/return/success) 누적
                self.all_episode_summaries.append({
                    "ep": int(self.num_episodes_counter),          # 1, 2, 3, ...
                    "steps": int(episode_timesteps + 1),           # 종료 스텝 수
                    "return": float(episode_return),               # 누적 보상
                    "success": int(bool(target)),                  # 목표 도달 여부(0/1)
                })

                # 메트릭 저장
                self.save_test_metrics()

                # ▶️ 에피소드 요약 출력(종료 시 한 번)
                if self.print_live:
                    self.get_logger().info(
                        f"[EP DONE] ep={int(self.num_episodes_counter)} "
                        f"steps={episode_timesteps+1} return={episode_return:.3f} "
                        f"success={int(bool(target))} traj_len={len(current_trajectory)}"
                    )

                # ✅ 에피소드 목표 도달 시 종료 (0이면 무한 루프)
                if getattr(self, "num_episodes_target", 0) > 0 and \
                   self.num_episodes_counter >= self.num_episodes_target:
                    # 마지막으로 저장 한 번 더(덮어쓰기)
                    self.save_test_metrics()
                    self.get_logger().info(
                        f"Reached {int(self.num_episodes_counter)} episodes. Exiting."
                    )
                    rclpy.shutdown()
                    return

                # 다음 에피소드 초기화
                state = self.reset()
                done = False
                episode_timesteps = 0
                episode_return = 0.0
                current_trajectory = []
                episode_start_time = time.time()
            else:
                state = next_state
                episode_timesteps += 1

    def calculate_distance(self, traj):
        """Sum of segment lengths in a 2D trajectory."""
        distance = 0.0
        for i in range(1, len(traj)):
            x1, y1 = traj[i - 1]["x"], traj[i - 1]["y"]
            x2, y2 = traj[i]["x"], traj[i]["y"]
            distance += float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        return distance

    def save_test_metrics(self):
        """Save per-episode results (ep, steps, return, success) to JSON; keep YAML as-is."""
        import os, json, math, sys, traceback
        from datetime import datetime, timezone
        import numpy as np

        # -----------------------------
        # Helpers
        # -----------------------------
        def _ensure_dir(path: str):
            os.makedirs(path, exist_ok=True)

        def _to_num(x):
            """Numpy/torch/py types → 순수 파이썬 수치형으로 변환."""
            try:
                import numpy as _np
                if isinstance(x, _np.generic):
                    return x.item()
                if isinstance(x, _np.ndarray) and x.ndim == 0:
                    return x.item()
            except Exception:
                pass
            try:
                import torch as _torch
                if isinstance(x, _torch.Tensor) and x.numel() == 1:
                    return x.item()
            except Exception:
                pass
            return x

        def _to_int(x, default=0):
            try:
                v = _to_num(x)
                return int(v)
            except Exception:
                return int(default)

        def _to_float(x, default=0.0):
            try:
                v = float(_to_num(x))
                if math.isnan(v) or math.isinf(v):
                    return float(default)
                return v
            except Exception:
                return float(default)

        def _as_list(x):
            """list-like/numpy/torch → list; 아니면 None"""
            try:
                import numpy as _np
                import torch as _torch
            except Exception:
                _np = None; _torch = None

            if isinstance(x, list):
                return x
            if isinstance(x, tuple):
                return list(x)
            if _np is not None and isinstance(x, _np.ndarray):
                return x.tolist()
            if _torch is not None and isinstance(x, _torch.Tensor):
                return x.detach().cpu().tolist()
            return None

        def _find_series(keywords, value_pred=None, prefer_episode_named=True):
            """
            self.__dict__에서 키워드 포함 리스트를 찾아 반환.
            prefer_episode_named=True면 'episode'가 들어간 이름을 우선.
            value_pred가 주어지면 값 샘플 검증.
            """
            candidates = []
            for name, val in vars(self).items():
                if not isinstance(name, str):
                    continue
                lname = name.lower()
                if not any(k in lname for k in keywords):
                    continue
                seq = _as_list(val)
                if seq is None or len(seq) == 0:
                    continue
                # 값 샘플 검증
                ok = True
                if value_pred is not None:
                    # 앞쪽 10개 정도만 검사
                    sample = seq[: min(10, len(seq))]
                    try:
                        ok = all(value_pred(s) for s in sample)
                    except Exception:
                        ok = False
                if ok:
                    score = 0
                    if prefer_episode_named and ("episode" in lname or "episodes" in lname or "all_episode" in lname):
                        score += 2
                    # 더 긴 시퀀스를 조금 더 선호
                    score += min(len(seq), 1000) / 1000.0
                    candidates.append((score, name, seq))
            if not candidates:
                return None
            # 점수 높은 순
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][2]

        def _is_steps(x):   # 정수/정수로 해석 가능한 값
            try:
                _ = int(_to_num(x))
                return True
            except Exception:
                return False

        def _is_return(x):  # 실수
            try:
                v = float(_to_num(x))
                return not (math.isnan(v) or math.isinf(v))
            except Exception:
                return False

        def _is_success(x):  # 0/1 또는 bool
            try:
                v = _to_num(x)
                if isinstance(v, bool):
                    return True
                iv = int(v)
                return iv in (0, 1)
            except Exception:
                return False

        def _atomic_write_json(path: str, data_obj):
            """tmp에 쓴 뒤 교체하는 원자적 저장."""
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data_obj, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)

        # -----------------------------
        # Paths
        # -----------------------------
        _ensure_dir(self.test_metric_dir)
        json_filename = os.path.join(self.test_metric_dir, "tqc_test_run_result.json")
        metrics_filename = os.path.join(self.test_metric_dir, "tqc_test_run_result.yaml")

        # -----------------------------
        # YAML 메트릭(원래 로직 유지)
        # -----------------------------
        average_time = round(float(np.mean(getattr(self, "all_episode_times", []) or [0.0])), 4)
        average_distance = round(float(np.mean(getattr(self, "all_episode_distances", []) or [0.0])), 4)
        success_rate = round(
            (getattr(self, "target_reached_counter", 0) / getattr(self, "num_episodes_counter", 0))
            if getattr(self, "num_episodes_counter", 0) > 0 else 0.0, 4
        )
        collision_rate = round(1.0 - success_rate, 4)

        data = {
            "test_metrics": {
                "average_time": average_time,
                "average_distance": average_distance,
                "success_rate": success_rate,
                "collision_rate": collision_rate,
            }
        }

        if getattr(self, "echo_on_save", False) or getattr(self, "print_live", False):
            try:
                self.get_logger().info(
                    "[METRICS] "
                    f"eps={int(getattr(self, 'num_episodes_counter', 0))} "
                    f"succ={int(getattr(self, 'target_reached_counter', 0))}/"
                    f"{int(getattr(self, 'num_episodes_counter', 0))} "
                    f"({success_rate*100:.1f}%), "
                    f"avg_time={average_time:.2f}s, avg_dist={average_distance:.2f}m"
                )
            except Exception:
                pass

        # -----------------------------
        # JSON: 에피소드별 결과만 저장 (ep, steps, return, success)
        #  - self의 다양한 속성 이름을 자동 탐색하여 결합
        # -----------------------------
        episodes = []

        # 0) 가장 이상적: 요약 리스트가 이미 있는 경우
        summaries = getattr(self, "all_episode_summaries", None)
        if isinstance(summaries, list) and len(summaries) > 0:
            for i, rec in enumerate(summaries):
                ep      = rec.get("ep", rec.get("episode", i + 1))
                steps   = rec.get("steps", rec.get("episode_steps", 0))
                ret     = rec.get("return", rec.get("ep_return", rec.get("reward", 0.0)))
                success = rec.get("success", rec.get("target_reached", rec.get("goal_reached", 0)))
                episodes.append({
                    "ep": _to_int(ep, i + 1),
                    "steps": _to_int(steps, 0),
                    "return": _to_float(ret, 0.0),
                    "success": _to_int(success, 0),
                })
        else:
            # 1) 이름 패턴으로 자동 탐색
            steps_list = _find_series(
                keywords=["episode_steps", "steps", "step"],
                value_pred=_is_steps
            )
            returns_list = _find_series(
                keywords=["episode_returns", "returns", "return", "reward", "rewards"],
                value_pred=_is_return
            )
            success_list = _find_series(
                keywords=["success", "goal_reached", "target_reached", "reach"],
                value_pred=_is_success
            )

            # 2) 보정: steps가 없고 time과 time_delta로 추정 가능한 경우
            if steps_list is None:
                times = _find_series(
                    keywords=["episode_times", "times", "time_per_episode"],
                    value_pred=_is_return  # 실수여야 함
                )
                time_delta = getattr(self, "time_delta", None)
                if times and isinstance(time_delta, (int, float)) and time_delta > 0:
                    steps_list = [max(1, _to_int(round(t / time_delta), 1)) for t in times]

            # 3) 길이 결정 및 값 채우기
            lengths = [len(x) for x in (steps_list or [], returns_list or [], success_list or [])]
            N = max(lengths) if lengths else 0

            if N > 0:
                # 부족한 리스트는 기본값으로 패딩
                if steps_list is None:
                    steps_list = [0] * N
                if returns_list is None:
                    returns_list = [0.0] * N
                if success_list is None:
                    # 전체 성공률이라도 있으면 그 비율로 유추할 수 있지만,
                    # 여기서는 보수적으로 0으로 채움
                    success_list = [0] * N

                # 서로 길이가 다르면 안전하게 자르거나 늘림
                if len(steps_list) < N: steps_list += [0] * (N - len(steps_list))
                if len(returns_list) < N: returns_list += [0.0] * (N - len(returns_list))
                if len(success_list) < N: success_list += [0] * (N - len(success_list))

                for i in range(N):
                    episodes.append({
                        "ep": i + 1,
                        "steps": _to_int(steps_list[i], 0),
                        "return": _to_float(returns_list[i], 0.0),
                        "success": _to_int(success_list[i], 0),
                    })
            else:
                # 찾을 수 있는 게 전혀 없으면 비우되, 경고는 생략(요청사항: 함수 내부 수정만으로 처리)
                pass

        # -----------------------------
        # Save JSON (atomic)  → 오직 episodes만
        # -----------------------------
        try:
            _atomic_write_json(json_filename, episodes)
        except Exception as e:
            try:
                self.get_logger().error(f"Unable to save per-episode JSON: {e}\n{traceback.format_exc()}")
            except Exception:
                pass
            sys.exit(-1)

        # -----------------------------
        # Save YAML (원래 유틸 사용, 변경 금지)
        # -----------------------------
        try:
            save_yaml(metrics_filename, data)  # ← 기존 함수 그대로 사용
        except Exception as e:
            try:
                self.get_logger().error(f"Unable to save metrics YAML: {e}")
            except Exception:
                pass
            sys.exit(-1)

def main():
    rclpy.init(args=None)
    node = TestTQC()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        while rclpy.ok():
            executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"{'Testing is Done':-^35}")
        node.get_logger().info("rclpy, shutingdown...")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
