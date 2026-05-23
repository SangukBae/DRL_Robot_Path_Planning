#!/usr/bin/env python3
"""Curriculum-learning subclass of TrainTQC.

No existing file is modified. This file only adds:
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
import time
import json
import pickle
import random

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

from train_tqc_agent import TrainTQC   # base class — not modified
from file_manager import load_yaml


class TrainTQCCurriculum(TrainTQC):
    """TQC trainer with automatic curriculum stage advancement.

    Inherits all setup, training loop, and model I/O from TrainTQC.
    Adds:
      1. eval metrics (success / collision / timeout rates)
      2. stage-pass checking (consecutive evals threshold)
      3. ROS2 set_parameters call to push new stage to EnvironmentCurriculum
      4. curriculum-aware CSV log
    """

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

        # Runtime state
        self._curriculum_stage       = 0
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
                "final_goal_dist_m", "curriculum_stage",
            ])
        self.get_logger().info(
            f"[Curriculum] Episode log (with stage): {self._curriculum_reward_csv}"
        )

        self.get_logger().info(
            f"[Curriculum] Trainer ready — "
            f"enabled={self.cur_enabled} "
            f"min_steps={self.cur_min_stage_steps} "
            f"min_eps={self.cur_min_stage_eps} "
            f"consec={self.cur_consec_passes}"
        )
        if self.load_model:
            self._load_curriculum_state()

    # ------------------------------------------------------------------ #
    #  Stage control helpers                                                #
    # ------------------------------------------------------------------ #

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

        for _ in range(self.eval_eps):
            state    = self.reset()
            done     = False
            ep_steps = 0
            ep_rew   = 0.0

            while not done and ep_steps < self.max_episode_steps:
                action = self.rl_agent.select_action(
                    state, use_checkpoint=False, use_exploration=False
                )
                state, reward, done, info = self.step(action)
                ep_rew   += reward
                ep_steps += 1

            s = np.asarray(state, dtype=np.float32).ravel()
            final_dists.append(float(s[ENV_DIM]))
            rewards.append(ep_rew)

            if done and info:
                success_count   += 1
            elif done:
                collision_count += 1
            else:
                timeout_count   += 1

        n = self.eval_eps
        metrics = {
            "mean_reward":    float(np.mean(rewards)),
            "std_reward":     float(np.std(rewards)),
            "success_rate":   success_count   / n,
            "collision_rate": collision_count / n,
            "timeout_rate":   timeout_count   / n,
            "mean_goal_dist": float(np.mean(final_dists)),
        }

        self.get_logger().info(
            f"Eval {n} eps | "
            f"Reward {metrics['mean_reward']:.3f}±{metrics['std_reward']:.3f} | "
            f"Success {metrics['success_rate']*100:.1f}% | "
            f"Collision {metrics['collision_rate']*100:.1f}% | "
            f"Timeout {metrics['timeout_rate']*100:.1f}% | "
            f"GoalDist {metrics['mean_goal_dist']:.3f}m"
        )

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
        if self._resume_loaded:
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
        _state0 = np.asarray(state, dtype=np.float32).ravel()
        _ep_initial_goal_dist = float(_state0[ENV_DIM])
        if next_eval_t is not None and self._resume_global_t > 0:
            next_eval_t = ((self._resume_global_t // self.eval_freq) + 1) * self.eval_freq

        for t in range(self._resume_global_t + 1, self.max_timesteps + 1):
            self._last_global_t = t
            train_ready = t >= self.timesteps_before_training
            use_policy  = t >  self.timesteps_before_training
            if train_ready and not training_enabled_logged:
                self.get_logger().info(
                    f"[Curriculum] Warmup done at step {t} — "
                    f"gradient updates + policy actions enabled."
                )
                training_enabled_logged = True

            _s_np         = np.asarray(state, dtype=np.float32).ravel()
            _lidar_before = _s_np[:ENV_DIM]
            _goal_before  = float(_s_np[ENV_DIM])
            _theta_before = float(_s_np[ENV_DIM + 1])

            if use_policy:
                action_source = "policy"
                action = self.rl_agent.select_action(state)
            else:
                action_source = "warmup"
                action = self.sample_action_space()

            next_state, reward, ep_finished, info = self.step(action)

            # Timeout penalty (same as base class)
            if ep_timesteps == self.max_episode_steps - 1 and not ep_finished:
                reward -= 20.0

            done = float(ep_finished) if ep_timesteps < self.max_episode_steps else 0.0
            self.rl_agent.replay_buffer.add(state, action, next_state, reward, done)

            state            = next_state
            ep_total_reward += reward
            ep_timesteps    += 1
            # Mirror to instance vars so Ctrl+C saves correct partial state
            self._partial_ep_timesteps = ep_timesteps
            self._partial_ep_reward    = ep_total_reward

            _s_after = np.asarray(state, dtype=np.float32).ravel()
            _ep_v_buf.append(float(action[0]))
            _ep_w_buf.append(float(action[1]))
            _ep_min_lidar_buf.append(float(np.min(_s_after[:ENV_DIM])))

            # Step-level CSV (same schema as base class)
            with open(self._step_csv, "a", newline="") as _f:
                csv.writer(_f).writerow([
                    ep_num, t, ep_timesteps, action_source,
                    round(float(action[0]), 6), round(float(action[1]), 6),
                    round(_goal_before, 6),
                    round(float(_s_after[ENV_DIM]), 6),
                    round(_theta_before, 6),
                    round(float(_s_after[ENV_DIM + 1]), 6),
                    round(float(np.min(_lidar_before)), 6),
                    round(float(np.min(_s_after[:ENV_DIM])), 6),
                    round(float(np.mean(_lidar_before)), 6),
                    round(float(np.mean(_s_after[:ENV_DIM])), 6),
                    round(float(reward), 6),
                    int(bool(ep_finished)), int(bool(info)),
                ])

            if train_ready and not self.use_checkpoints:
                self.rl_agent.train()

            eval_due       = bool(next_eval_t is not None and t >= next_eval_t)
            episode_limit  = ep_timesteps >= self.max_episode_steps
            force_eval_cut = eval_due and not ep_finished and not episode_limit

            if ep_finished or episode_limit or force_eval_cut:
                # Base-class episode log (reward_csv, driving_csv)
                result = self._write_episode_logs(
                    ep_num=ep_num, global_t=t,
                    ep_timesteps=ep_timesteps, ep_total_reward=ep_total_reward,
                    state=state, info=info,
                    episode_done=ep_finished, episode_limit=episode_limit,
                    ep_v_buf=_ep_v_buf, ep_w_buf=_ep_w_buf,
                    ep_min_lidar_buf=_ep_min_lidar_buf,
                    ep_initial_goal_dist=_ep_initial_goal_dist,
                    eval_cut=force_eval_cut,
                )

                # Curriculum episode log (adds stage column)
                final_dist   = float(np.asarray(state, dtype=np.float32).ravel()[ENV_DIM])
                goal_reached = bool(ep_finished and info) and not force_eval_cut
                collision    = bool(ep_finished and not goal_reached) and not force_eval_cut
                timeout      = bool(episode_limit and not ep_finished) and not force_eval_cut
                with open(self._curriculum_reward_csv, "a", newline="") as _f:
                    csv.writer(_f).writerow([
                        ep_num, t, ep_timesteps,
                        round(ep_total_reward, 4),
                        round(ep_total_reward / max(ep_timesteps, 1), 4),
                        int(goal_reached), int(collision), int(timeout),
                        int(force_eval_cut),
                        round(final_dist, 4),
                        self._curriculum_stage,
                    ])

                self._total_episodes = ep_num
                # Episode is done — next save should reflect a fresh episode start
                self._partial_ep_timesteps = 0
                self._partial_ep_reward    = 0.0
                self._save_curriculum_state(t)
                self.get_logger().info(
                    f"T:{t} | Ep:{ep_num} | Steps:{ep_timesteps} | "
                    f"Reward:{ep_total_reward:.3f} | {result} | "
                    f"Stage:{self._curriculum_stage}"
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
                ep_total_reward = 0.0
                ep_timesteps    = 0
                ep_num         += 1
                ep_finished     = False
                _ep_v_buf.clear()
                _ep_w_buf.clear()
                _ep_min_lidar_buf.clear()
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
