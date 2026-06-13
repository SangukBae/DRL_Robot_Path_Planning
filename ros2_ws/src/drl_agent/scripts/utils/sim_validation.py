#!/usr/bin/env python3
# SIM_VALIDATION: simulation verification instrumentation (localization-aware RL).
# VALIDATION_ONLY — this whole file exists solely to verify the localization /
# reward-done-separation / reset-consistency / stale-handling logic in sim. It is
# NOT used on the training/deploy hot path unless `enable_sim_validation_logging`
# is true. To remove the validation feature: delete this file, the
# sim_validation_runner.py / sim_validation_summary.py scripts, and grep for
# `SIM_VALIDATION` in environment.py to drop the 3 small guarded hooks there.
"""Per-step + per-reset CSV logger for localization-error validation.

Writes two CSVs in the environment's log dir:
  loc_validation_step_<tag>.csv   — one row per RL step (obs vs gt, sources, stale)
  loc_validation_reset_<tag>.csv  — one row per episode (reset→first-step jump)
"""

import os
import csv
import math

STEP_COLUMNS = [
    "episode", "step", "curriculum_stage",
    "obs_goal_dist", "obs_heading_err", "gt_goal_dist", "gt_heading_err",
    "reward_goal_dist_used", "done_goal_dist_used",
    "loc_raw_x", "loc_raw_y", "loc_raw_yaw",
    "loc_est_x", "loc_est_y", "loc_est_yaw",
    "gt_x", "gt_y", "gt_yaw",
    "use_gt_for_reward", "use_gt_for_done",
    "odom_gt_count", "odom_loc_count", "odom_proprio_count",
    "stale_gt", "stale_loc", "stale_proprio",
    "loc_noise_enabled", "loc_delay_steps", "loc_sigma_xy", "loc_sigma_yaw", "loc_jump_prob",
    "is_first_step",
]

RESET_COLUMNS = [
    "episode", "curriculum_stage",
    "reset_obs_goal_dist", "reset_obs_heading_err",
    # pre_motion = first step's obs BEFORE propagate (excludes robot motion).
    # The reset→first-step jump is computed from THIS, so noise-off ⇒ 0.
    "pre_motion_obs_goal_dist", "pre_motion_obs_heading_err",
    # first_step = obs AFTER propagate (motion + noise); kept for reference only.
    "first_step_obs_goal_dist", "first_step_obs_heading_err",
    "reset_first_step_goal_jump", "reset_first_step_heading_jump",
    "loc_noise_enabled", "loc_delay_steps",
]


class SimValidationLogger:
    """SIM_VALIDATION: writes step/reset validation CSVs. Stateful (tracks prev
    odom counts for stale flags, and the reset observation for jump checks)."""

    def __init__(self, log_dir, run_tag):
        self.step_path = os.path.join(log_dir, f"loc_validation_step_{run_tag}.csv")
        self.reset_path = os.path.join(log_dir, f"loc_validation_reset_{run_tag}.csv")
        with open(self.step_path, "w", newline="") as f:
            csv.writer(f).writerow(STEP_COLUMNS)
        with open(self.reset_path, "w", newline="") as f:
            csv.writer(f).writerow(RESET_COLUMNS)
        self._prev_counts = None                 # last {role: count}
        self._pending = None                     # (episode, stage, reset_obs_d, reset_obs_h, loc)

    def note_reset(self, episode, stage, reset_obs_goal_dist, reset_obs_heading_err, loc_noise):
        """Record the post-reset observation so the first step can measure the jump."""
        self._pending = (int(episode), int(stage),
                         float(reset_obs_goal_dist), float(reset_obs_heading_err),
                         dict(loc_noise))

    @staticmethod
    def _circular_angle_diff(a, b):
        """SIM_VALIDATION: smallest signed angle difference a-b, wrapped to [-pi, pi]."""
        return (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi

    def log_step(self, *, episode, step, stage,
                 obs_goal_dist, obs_heading_err, gt_goal_dist, gt_heading_err,
                 reward_goal_dist_used, done_goal_dist_used,
                 loc_raw, loc_est, gt, role_counts, loc_noise,
                 pre_motion_obs=None):
        # stale = role's odom count did not advance since the previous logged step
        if self._prev_counts is None:
            stale = {r: 0 for r in ("gt", "loc", "proprio")}
        else:
            stale = {r: int(role_counts[r] <= self._prev_counts[r]) for r in ("gt", "loc", "proprio")}
        self._prev_counts = dict(role_counts)

        is_first = bool(self._pending is not None and self._pending[0] == int(episode))

        with open(self.step_path, "a", newline="") as f:
            csv.writer(f).writerow([
                episode, step, stage,
                round(obs_goal_dist, 5), round(obs_heading_err, 5),
                round(gt_goal_dist, 5), round(gt_heading_err, 5),
                round(reward_goal_dist_used, 5), round(done_goal_dist_used, 5),
                round(loc_raw[0], 5), round(loc_raw[1], 5), round(loc_raw[2], 5),
                round(loc_est[0], 5), round(loc_est[1], 5), round(loc_est[2], 5),
                round(gt[0], 5), round(gt[1], 5), round(gt[2], 5),
                int(bool(loc_noise["use_gt_for_reward"])), int(bool(loc_noise["use_gt_for_done"])),
                role_counts["gt"], role_counts["loc"], role_counts["proprio"],
                stale["gt"], stale["loc"], stale["proprio"],
                int(bool(loc_noise["enabled"])), int(loc_noise["delay_steps"]),
                round(float(loc_noise["sigma_xy_m"]), 5), round(float(loc_noise["sigma_yaw_rad"]), 5),
                round(float(loc_noise["jump_prob"]), 5),
                int(is_first),
            ])

        if is_first:
            _, st, r_d, r_h, _loc = self._pending
            # Jump uses the PRE-MOTION obs (before propagate) so real robot
            # motion does not leak in; falls back to post-motion obs if the
            # caller did not supply it (older callers).
            if pre_motion_obs is not None:
                pm_d, pm_h = float(pre_motion_obs[0]), float(pre_motion_obs[1])
            else:
                pm_d, pm_h = float(obs_goal_dist), float(obs_heading_err)
            goal_jump = abs(pm_d - r_d)
            heading_jump = abs(self._circular_angle_diff(pm_h, r_h))
            with open(self.reset_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    episode, st,
                    round(r_d, 5), round(r_h, 5),
                    round(pm_d, 5), round(pm_h, 5),
                    round(obs_goal_dist, 5), round(obs_heading_err, 5),
                    round(goal_jump, 6), round(heading_jump, 6),
                    int(bool(_loc["enabled"])), int(_loc["delay_steps"]),
                ])
            self._pending = None
