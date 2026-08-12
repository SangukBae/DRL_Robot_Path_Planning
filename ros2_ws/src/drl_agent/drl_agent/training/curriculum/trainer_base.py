"""Shared curriculum-training machinery for the non-TQC paper-comparison
baselines (SAC, TD7, SB3-SAC, SB3-TD3 curriculum trainers).

Same curriculum mechanism as drl_agent.training.train_tqc_curriculum's
TrainTQCCurriculum, minus the TQC-only aux-prediction / counterfactual-risk /
DYN_AVOID / RUN_LAYOUT machinery: stage advancement via GymParameterClient,
curriculum_state.json checkpoint/resume (+ RNG snapshots, via the same
curriculum/state_io.py module TQC uses), the curriculum-stage CSV column,
evaluate_and_print(), and the online training-loop skeleton. This is pure
code motion out of the 4 near-identical baseline files (diffed byte-for-byte
before extraction — the only real differences across them were the eval/
training action-selection call signature, the presence of a
train_and_checkpoint ensembling hook, and one config default), preserved
here as the three hook methods below; every subclass overrides only what
actually differs for its algorithm.

Subclasses (see training/baselines/{sac,td7,sb3_sac,sb3_td3}_curriculum.py):
  1. Set CURRICULUM_CONFIG_FILENAME (and DEFAULT_MIN_STAGE_STEPS if it
     differs from the native-torch-agent default of 10000).
  2. Override _init_csv_loggers if a subclass needs different columns (none
     of the 4 current ones do — they share the base implementation here).
  3. In __init__, call super().__init__() then self._init_curriculum_state().
  4. Override _select_action_eval / _select_action_training /
     _maybe_train_and_checkpoint only if their algorithm's call signature or
     checkpoint-ensembling support differs from the native-torch-agent
     defaults below (TD7/SB3 do; see each subclass).
"""

import os
import csv
import time

import numpy as np

from drl_agent.common.file_manager import load_yaml
from drl_agent.training.episode_metrics import EpisodeMetrics, PaperMetricsCSV
from drl_agent.training.gym_parameter_client import GymParameterClient
import drl_agent.training.curriculum.stage_logic as curriculum_stage_logic
import drl_agent.training.curriculum.state_io as curriculum_state_io


class BaselineCurriculumTrainerMixin:
    """Curriculum stage advancement + resume state + eval/train loop skeleton,
    shared by every non-TQC curriculum baseline trainer."""

    # Subclasses MUST set this to their own train_<algo>_curriculum_config.yaml.
    CURRICULUM_CONFIG_FILENAME = None
    # native-torch agents (SAC/TD7) default to 10000; SB3 wrappers override to
    # 30000 (their own config default before this extraction).
    DEFAULT_MIN_STAGE_STEPS = 10000

    # ------------------------------------------------------------------ #
    #  CSV logging (identical across every subclass)                       #
    # ------------------------------------------------------------------ #
    def _init_csv_loggers(self):
        import datetime as _datetime
        self._csv_run_tag = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._reward_csv  = os.path.join(self.log_dir, f"episode_rewards_{self._csv_run_tag}.csv")
        self._driving_csv = os.path.join(self.log_dir, f"episode_driving_{self._csv_run_tag}.csv")
        self._step_csv    = os.path.join(self.log_dir, f"policy_step_debug_{self._csv_run_tag}.csv")

        reward_header  = ["episode", "global_t", "steps", "total_reward", "mean_reward",
                          "goal_reached", "collision", "timeout", "eval_cut", "final_goal_dist_m"]
        driving_header = ["episode", "global_t", "steps", "mean_v_norm", "mean_abs_w_norm",
                          "initial_goal_dist_m", "final_goal_dist_m", "goal_dist_reduction_m",
                          "min_lidar_m", "mean_min_lidar_m", "goal_reached", "eval_cut",
                          "mean_gazebo_rtf"]
        step_header    = ["episode", "global_t", "episode_step", "action_source",
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

        self.get_logger().info(f"Episode rewards CSV : {self._reward_csv}")
        self.get_logger().info(f"Episode driving CSV : {self._driving_csv}")
        self.get_logger().info(f"Policy step CSV     : {self._step_csv}")

    # ------------------------------------------------------------------ #
    #  __init__ helper — call from each subclass after super().__init__()  #
    # ------------------------------------------------------------------ #
    def _init_curriculum_state(self):
        assert self.CURRICULUM_CONFIG_FILENAME, (
            f"{type(self).__name__} must set CURRICULUM_CONFIG_FILENAME"
        )
        cur_cfg_path = self._find_config_file(self.CURRICULUM_CONFIG_FILENAME)
        if cur_cfg_path:
            cur = load_yaml(cur_cfg_path).get("curriculum_settings", {})
        else:
            self.get_logger().warn(
                f"[Curriculum] {self.CURRICULUM_CONFIG_FILENAME} not found — using defaults."
            )
            cur = {}

        self.cur_enabled         = bool(cur.get("enabled", True))
        self.cur_min_stage_steps = int(cur.get("min_stage_steps", self.DEFAULT_MIN_STAGE_STEPS))
        self.cur_min_stage_eps   = int(cur.get("min_stage_episodes", 20))
        self.cur_pass_sr         = list(cur.get("pass_eval_success_rate",
                                                [0.90, 0.85, 0.75, 0.70]))
        self.cur_pass_cr         = list(cur.get("pass_eval_collision_rate",
                                                [0.05, 0.10, 0.15, 0.20]))
        # SPL/clearance/human-safety/per-map gates are TQC-curriculum-only
        # extras; empty here so curriculum_stage_logic.should_advance_stage
        # treats them as inactive, matching these baselines' sr/cr-only
        # gating exactly (see _check_stage_advance below).
        self.cur_pass_spl   = []
        self.cur_pass_clear = []
        self.cur_consec_passes   = int(cur.get("consecutive_eval_passes", 2))
        # Replay-buffer reset + re-warmup on promotion INTO a contract-changing
        # stage (Stage 5 unseals the yield action). Mirrors
        # drl_agent.training.train_tqc_curriculum so off-contract
        # (yield-inactive) transitions do not poison the new critic.
        self.cur_reset_buffer_stages = set(
            int(s) for s in (cur.get("reset_buffer_on_promote_to", []) or [])
        )
        self.cur_rewarmup_steps = int(
            cur.get("rewarmup_steps", self.timesteps_before_training)
        )
        self._rewarmup_until_t = 0

        self._curriculum_stage       = 0
        self._stage_start_step       = 0
        self._stage_start_ep         = 0
        self._consecutive_pass_count = 0
        self._total_episodes         = 0
        self._resume_global_t        = 0
        self._resume_loaded          = False
        self._last_global_t          = 0
        self._resume_epoch           = 1
        self._partial_ep_timesteps   = 0
        self._partial_ep_reward      = 0.0

        # ROS2 parameter interaction with the gym_node (EnvironmentCurriculum).
        self._gym_params = GymParameterClient(self)
        # Back-compat aliases (kept in case other code paths reference the raw
        # clients); all parameter traffic now goes through self._gym_params.
        self._param_set_client = self._gym_params.set_client
        self._param_get_client = self._gym_params.get_client

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
            ])
        self.get_logger().info(
            f"[Curriculum] Episode log (with stage): {self._curriculum_reward_csv}"
        )

        # Paper metrics (SPL, path length, CTE, jerk, ...) → episode + eval CSVs
        self.declare_parameter("near_collision_dist_m", 0.5)
        self.declare_parameter("metric_time_delta", 0.1)
        _ncd = self.get_parameter("near_collision_dist_m").get_parameter_value().double_value
        _mdt = self.get_parameter("metric_time_delta").get_parameter_value().double_value
        self._em = EpisodeMetrics(
            self.environment_dim,
            time_delta=_mdt if _mdt > 0 else 0.1,
            near_collision_dist_m=_ncd if _ncd > 0 else 0.5,
        )
        self._paper = PaperMetricsCSV(self.log_dir, self._csv_run_tag)
        self.get_logger().info(
            f"[Metrics] Paper CSVs: {self._paper.episode_path} | {self._paper.eval_path}"
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
    #  Stage control helpers — delegate to the same shared modules TQC's    #
    #  TrainTQCCurriculum uses (GymParameterClient / curriculum_state_io /   #
    #  curriculum_stage_logic), rather than each baseline reimplementing    #
    #  raw rcl_interfaces calls + inline pickle/JSON I/O.                   #
    # ------------------------------------------------------------------ #
    def _set_curriculum_stage(self, stage: int) -> bool:
        """Delegate to GymParameterClient; cache the new stage on success."""
        ok = self._gym_params.set_curriculum_stage(stage)
        if ok:
            self._curriculum_stage = stage
        return ok

    def _save_curriculum_state(self, global_t: int):
        """Delegate to curriculum_state_io.save_curriculum_state (format unchanged)."""
        curriculum_state_io.save_curriculum_state(self, global_t)

    def _load_curriculum_state(self) -> bool:
        """Delegate to curriculum_state_io.load_curriculum_state (format unchanged)."""
        return curriculum_state_io.load_curriculum_state(self)

    def _fetch_num_stages(self) -> int:
        """Delegate to GymParameterClient.get_num_stages."""
        return self._gym_params.get_num_stages()

    def _check_stage_advance(self, global_t: int, metrics: dict, num_stages: int) -> bool:
        """Return True when this eval pass should count toward stage promotion.

        The pure decision lives in curriculum_stage_logic.should_advance_stage
        (same function TrainTQCCurriculum uses); with cur_pass_spl/cur_pass_clear
        empty and the human-safety/per-map gates omitted, this reduces to
        exactly these baselines' original sr/cr-only comparison."""
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
    #  Algorithm-specific hooks — override in a subclass only where its     #
    #  agent's call signature / checkpoint support genuinely differs.       #
    # ------------------------------------------------------------------ #
    def _select_action_eval(self, state):
        """Default (SAC/TD7 — native-torch agents with a checkpoint-ensemble
        select_action signature). SB3 wrappers override (see sb3_*_curriculum.py)."""
        return self.rl_agent.select_action(
            np.array(state), use_checkpoint=self.use_checkpoints, use_exploration=False)

    def _select_action_training(self, state):
        """Default (SAC's signature: use_exploration kwarg). TD7 (no kwarg) and
        SB3 (no np.array wrap, no kwarg) override."""
        return self.rl_agent.select_action(np.array(state), use_exploration=True)

    def _maybe_train_and_checkpoint(self, ep_timesteps, ep_total_reward, train_ready):
        """Default (SAC/TD7 — agents with a checkpoint-ensembling hook). SB3
        wrappers (no such hook) override to a no-op."""
        if self.use_checkpoints and train_ready:
            self.rl_agent.train_and_checkpoint(ep_timesteps, ep_total_reward)

    # ------------------------------------------------------------------ #
    #  Override: evaluate_and_print → returns dict of metrics              #
    # ------------------------------------------------------------------ #
    def evaluate_and_print(self, evals, epoch, start_time):
        self.get_logger().info("=" * 55)
        self.get_logger().info(
            f"[Curriculum] Evaluating — Epoch {epoch} | Stage {self._curriculum_stage}"
        )
        self.get_logger().info(f"Elapsed: {time.time() - start_time:.1f}s")
        self.get_logger().info("=" * 55)

        ENV_DIM = self.environment_dim
        rewards, final_dists = [], []
        success_count = collision_count = timeout_count = 0
        per_ep_metrics = []

        for _ in range(self.eval_eps):
            state    = self.reset()
            done     = False
            ep_steps = 0
            ep_rew   = 0.0
            self._em.reset(state)

            while not done and ep_steps < self.max_episode_steps:
                action = self._select_action_eval(state)
                state, reward, done, info = self.step(action)
                self._em.update(state, action)
                ep_rew   += reward
                ep_steps += 1

            s = np.asarray(state, dtype=np.float32).ravel()
            final_dists.append(float(s[ENV_DIM]))
            rewards.append(ep_rew)
            per_ep_metrics.append(self._em.compute(bool(done and info)))

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
        # Aggregate paper metrics (SPL, CTE, jerk, ...) over the eval episodes
        _agg = PaperMetricsCSV.aggregate(per_ep_metrics)
        metrics.update(_agg)
        self._paper.write_eval(
            epoch=epoch, global_t=self._last_global_t,
            stage=self._curriculum_stage, eval_eps=n,
            base=metrics, metrics_mean=_agg,
        )

        self.get_logger().info(
            f"Eval {n} eps | "
            f"Reward {metrics['mean_reward']:.3f}±{metrics['std_reward']:.3f} | "
            f"Success {metrics['success_rate']*100:.1f}% | "
            f"Collision {metrics['collision_rate']*100:.1f}% | "
            f"Timeout {metrics['timeout_rate']*100:.1f}% | "
            f"GoalDist {metrics['mean_goal_dist']:.3f}m | "
            f"SPL {metrics['spl']:.3f} | CTE {metrics['mean_cross_track_error_m']:.3f}m"
        )

        evals.append(metrics["mean_reward"])
        np.save(f"{self.results_dir}/{self.file_name}", evals)
        return metrics

    # ------------------------------------------------------------------ #
    #  Override: train_online — adds stage advancement around eval          #
    # ------------------------------------------------------------------ #
    def train_online(self):
        start_time = time.time()

        evals_path = f"{self.results_dir}/{self.file_name}.npy"
        if self._resume_loaded and os.path.isfile(evals_path):
            evals = list(np.load(evals_path))
            self.get_logger().info(
                f"[Curriculum] Loaded {len(evals)} past eval points from {evals_path}."
            )
        else:
            evals = []
        epoch = len(evals) + 1

        if self.eval_freq > 0:
            # Align first eval boundary to the first eval_freq grid point after warmup.
            # This prevents an immediate force_eval_cut when warmup ends mid-episode.
            next_eval_t = ((self.timesteps_before_training // self.eval_freq) + 1) * self.eval_freq
        else:
            next_eval_t = None
        training_enabled_logged = False

        num_stages = self._fetch_num_stages()
        self._curriculum_stage = max(0, min(self._curriculum_stage, num_stages - 1))

        if self._resume_loaded:
            self.get_logger().info(
                f"[Curriculum] Resuming from stage "
                f"{self._curriculum_stage} at global step {self._resume_global_t}."
            )
            if not self._set_curriculum_stage(self._curriculum_stage):
                raise RuntimeError(
                    "[Curriculum] Cannot restore saved curriculum stage on gym_node."
                )
        else:
            self.get_logger().info(
                f"[Curriculum] Enforcing stage 0 (empty) for warmup "
                f"({self.timesteps_before_training} steps)."
            )
            if not self._set_curriculum_stage(0):
                raise RuntimeError(
                    "[Curriculum] Cannot push stage 0 to gym_node before warmup."
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
        ep_total_reward = 0.0
        ep_timesteps    = 0
        ep_num          = self._total_episodes + 1
        ep_finished     = False
        _ep_v_buf:         list = []
        _ep_w_buf:         list = []
        _ep_min_lidar_buf: list = []
        _state0 = np.asarray(state, dtype=np.float32).ravel()
        _ep_initial_goal_dist = float(_state0[ENV_DIM])

        if next_eval_t is not None and self._resume_global_t > 0:
            next_eval_t = ((self._resume_global_t // self.eval_freq) + 1) * self.eval_freq

        for t in range(self._resume_global_t + 1, self.max_timesteps + 1):
            self._last_global_t = t
            if ep_timesteps == 0:
                self._em.reset(state)   # new episode → reset paper-metric tracker
            train_ready = t >= self.timesteps_before_training
            use_policy  = t >  self.timesteps_before_training
            # Post-promotion re-warmup: after a Stage-5 buffer reset, take random
            # actions and skip gradient updates until the buffer refills on-contract.
            if self._rewarmup_until_t:
                if t <= self._rewarmup_until_t:
                    train_ready = False
                    use_policy  = False
                else:
                    self._rewarmup_until_t = 0
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
                action = self._select_action_training(state)
            else:
                action_source = "warmup"
                action = self.sample_action_space()

            next_state, reward, ep_finished, info = self.step(action)

            if ep_timesteps == self.max_episode_steps - 1 and not ep_finished:
                reward -= 20.0

            done = float(ep_finished) if ep_timesteps < self.max_episode_steps else 0.0
            self.rl_agent.replay_buffer.add(state, action, next_state, reward, done)

            state            = next_state
            self._em.update(state, action)
            ep_total_reward += reward
            ep_timesteps    += 1
            self._partial_ep_timesteps = ep_timesteps
            self._partial_ep_reward    = ep_total_reward

            _s_after = np.asarray(state, dtype=np.float32).ravel()
            _ep_v_buf.append(float(action[0]))
            _ep_w_buf.append(float(action[1]))
            _ep_min_lidar_buf.append(float(np.min(_s_after[:ENV_DIM])))

            with open(self._step_csv, "a", newline="") as _f:
                csv.writer(_f).writerow([
                    ep_num, t, ep_timesteps, action_source,
                    round(float(action[0]), 6), round(float(action[1]), 6),
                    round(_goal_before, 6), round(float(_s_after[ENV_DIM]), 6),
                    round(_theta_before, 6), round(float(_s_after[ENV_DIM + 1]), 6),
                    round(float(np.min(_lidar_before)), 6),
                    round(float(np.min(_s_after[:ENV_DIM])), 6),
                    round(float(np.mean(_lidar_before)), 6),
                    round(float(np.mean(_s_after[:ENV_DIM])), 6),
                    round(float(reward), 6), int(bool(ep_finished)), int(bool(info)),
                ])

            # train_ready is forced False during warmup AND post-promotion
            # re-warmup (Stage-5 buffer reset), so gradient updates pause while the
            # just-cleared buffer refills with on-contract random-action data.
            if train_ready and not self.use_checkpoints:
                self.rl_agent.train()

            eval_due       = bool(next_eval_t is not None and t >= next_eval_t and use_policy)
            episode_limit  = ep_timesteps >= self.max_episode_steps
            force_eval_cut = eval_due and not ep_finished and not episode_limit

            if ep_finished or episode_limit or force_eval_cut:
                final_dist   = float(np.asarray(state, dtype=np.float32).ravel()[ENV_DIM])
                goal_reached = bool(ep_finished and info) and not force_eval_cut
                collision    = bool(ep_finished and not goal_reached) and not force_eval_cut
                timeout      = bool(episode_limit and not ep_finished) and not force_eval_cut
                if not force_eval_cut:   # skip partial (eval-interrupted) episodes
                    self._paper.write_episode(
                        episode=ep_num, global_t=t, stage=self._curriculum_stage,
                        success=goal_reached, collision=collision, timeout=timeout,
                        total_reward=ep_total_reward, steps=ep_timesteps,
                        metrics=self._em.compute(goal_reached),
                    )

                if goal_reached:
                    result = "GOAL"
                elif collision:
                    result = "COLLISION"
                elif force_eval_cut:
                    result = "EVAL_CUT"
                else:
                    result = "TIMEOUT"

                with open(self._reward_csv, "a", newline="") as _f:
                    csv.writer(_f).writerow([
                        ep_num, t, ep_timesteps,
                        round(ep_total_reward, 4),
                        round(ep_total_reward / max(ep_timesteps, 1), 4),
                        int(goal_reached), int(collision), int(timeout),
                        int(force_eval_cut),
                        round(final_dist, 4),
                    ])
                if _ep_v_buf:
                    with open(self._driving_csv, "a", newline="") as _f:
                        csv.writer(_f).writerow([
                            ep_num, t, ep_timesteps,
                            round(float(np.mean(_ep_v_buf)), 4),
                            round(float(np.mean(np.abs(_ep_w_buf))), 4),
                            round(_ep_initial_goal_dist, 4),
                            round(final_dist, 4),
                            round(_ep_initial_goal_dist - final_dist, 4),
                            round(float(np.min(_ep_min_lidar_buf)), 4),
                            round(float(np.mean(_ep_min_lidar_buf)), 4),
                            int(goal_reached), int(force_eval_cut),
                            float("nan"),  # mean_gazebo_rtf: not tracked by this baseline
                        ])

                with open(self._curriculum_reward_csv, "a", newline="") as _f:
                    csv.writer(_f).writerow([
                        ep_num, t, ep_timesteps,
                        round(ep_total_reward, 4),
                        round(ep_total_reward / max(ep_timesteps, 1), 4),
                        int(goal_reached), int(collision), int(timeout),
                        int(force_eval_cut),
                        round(final_dist, 4),
                        self._curriculum_stage,
                        float("nan"),  # mean_gazebo_rtf: not tracked by this baseline
                    ])

                if not force_eval_cut:
                    self._total_episodes = ep_num
                self._partial_ep_timesteps = 0
                self._partial_ep_reward    = 0.0
                self._save_curriculum_state(t)
                self.get_logger().info(
                    f"T:{t} | Ep:{ep_num} | Steps:{ep_timesteps} | "
                    f"Reward:{ep_total_reward:.3f} | {result} | "
                    f"Stage:{self._curriculum_stage}"
                )

                self._maybe_train_and_checkpoint(ep_timesteps, ep_total_reward, train_ready)

                if eval_due:
                    self.save_models(self.pytorch_models_dir, self.file_name)
                    metrics = self.evaluate_and_print(evals, epoch, start_time)
                    epoch  += 1
                    self._resume_epoch = epoch
                    while next_eval_t is not None and next_eval_t <= t:
                        next_eval_t += self.eval_freq

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
                            # Fail-fast: only proceed once the env has ACTUALLY
                            # switched stages. If /gym_node/set_parameters times out
                            # or is rejected, the env stays on the old contract while
                            # the trainer would reset its buffer / re-warmup on the new
                            # one — a silent desync. Abort instead.
                            if not self._set_curriculum_stage(new_stage):
                                raise RuntimeError(
                                    f"[Curriculum] Failed to push stage {new_stage} to "
                                    f"gym_node (/gym_node/set_parameters); aborting "
                                    f"before buffer reset / re-warmup to avoid a "
                                    f"trainer/environment stage desync."
                                )
                            # Contract-changing boundary (Stage 5 unseals yield):
                            # clear the buffer + re-warmup so off-contract data does
                            # not poison the new stage's critic.
                            if new_stage in self.cur_reset_buffer_stages:
                                self.rl_agent.replay_buffer.reset()
                                self._rewarmup_until_t = (
                                    t + self.cur_rewarmup_steps
                                    if self.cur_rewarmup_steps > 0 else 0
                                )
                                self.get_logger().info(
                                    f"[Curriculum] Stage {new_stage} unseals the yield "
                                    f"action — replay buffer cleared; re-warmup "
                                    f"{self.cur_rewarmup_steps} steps."
                                )
                            self._stage_start_step       = t
                            self._stage_start_ep         = self._total_episodes
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

                elif next_eval_t is not None and t >= next_eval_t:
                    # warmup 중 next_eval_t를 진행시켜 학습 시작 직후 즉시 평가 방지
                    while next_eval_t <= t:
                        next_eval_t += self.eval_freq

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
