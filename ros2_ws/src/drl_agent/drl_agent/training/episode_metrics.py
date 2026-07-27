#!/usr/bin/env python3
"""Per-episode paper metrics for the navigation comparison experiments.

`EpisodeMetrics` accumulates safety / navigation / control-quality statistics
over a single episode purely from the `(state, action)` stream that every
trainer already passes through its loop — no ROS subscription, no environment
change, no TQC change. It is used identically by the training loop and the
evaluation loop so that every algorithm reports the same numbers.

State layout (with `env_dim` = LiDAR bin count, 80):
    state[0:env_dim]      LiDAR ranges [m]
    state[env_dim+0]      goal distance [m]
    state[env_dim+1]      heading error to goal [rad], signed
    state[env_dim+2,+3]   previous normalized action (r, theta)
    state[env_dim+4]      actual signed speed [m/s]      (from odometry)
    state[env_dim+5]      actual yaw rate [rad/s]         (from odometry)
    state[env_dim+6]      center steering angle [rad]     (from joint states)

Metric notes
------------
* ``path_length_m`` integrates |speed| * dt — i.e. true travelled arc length.
* ``mean_cross_track_error_m`` is computed by dead-reckoning the pose from
  (speed, yaw_rate) in the episode-start frame and measuring perpendicular
  distance to the straight start->goal line. It is therefore an
  odometry-integrated *estimate* (small integration drift over long episodes),
  but it is self-consistent and identical across algorithms, so it is valid for
  comparative claims.
* ``spl`` = success * shortest / max(actual, shortest), with shortest = initial
  straight-line goal distance.
"""

import os
import csv
import math
import numpy as np

import drl_agent.training.run_layout as run_layout

# Canonical column order for the paper metrics (header == row guaranteed).
# NOTE: append-only — new metrics go at the END so existing positional readers of
# the older columns keep working (stl / psc were added for the aux experiments).
METRIC_COLUMNS = [
    "path_length_m",
    "spl",
    "mean_heading_error_rad",
    "mean_cross_track_error_m",
    "mean_action_jerk",
    "mean_steering_change_rad",
    "near_collision_count",
    "travel_time_s",
    "mean_speed_mps",
    "stl",                   # Success weighted by (normalized) Time Length
    "lidar_clearance_rate",  # fraction of steps with min-LiDAR >= clearance radius
                             # (generic obstacle-clearance PROXY — NOT human PSC;
                             #  cannot tell a human from a wall/furniture. True
                             #  human PSC is computed from privileged labels in
                             #  the trainer, see docs/aux_metric_schema.md)
]


class EpisodeMetrics:
    """Accumulate one episode's paper metrics from the (state, action) stream."""

    def __init__(self, env_dim: int, time_delta: float = 0.1,
                 near_collision_dist_m: float = 0.5,
                 stl_ref_speed_mps: float = 1.0,
                 lidar_clearance_radius_m: float = 0.5):
        self.env_dim = int(env_dim)
        self.dt = float(time_delta)
        self.near_dist = float(near_collision_dist_m)
        # STL reference speed: optimal time = shortest_path / ref_speed.
        # lidar_clearance_radius: a step is a "clearance intrusion" when min LiDAR
        # < this. This is a generic obstacle-clearance PROXY (cannot tell a human
        # from a wall/furniture) — NOT the human personal-space PSC, which the
        # trainer computes from privileged human-distance labels.
        self.stl_ref_speed = max(float(stl_ref_speed_mps), 1e-6)
        self.clearance_radius = float(lidar_clearance_radius_m)
        self.reset(np.zeros(self.env_dim + 7, dtype=np.float32))

    # -- helpers ------------------------------------------------------------
    def _s(self, state):
        return np.asarray(state, dtype=np.float32).ravel()

    # -- lifecycle ----------------------------------------------------------
    def reset(self, state0):
        """Begin a new episode. Records the shortest path and goal geometry."""
        s = self._s(state0)
        ed = self.env_dim
        self._d0 = float(s[ed])                       # shortest path (straight line)
        theta0 = float(s[ed + 1])                     # initial bearing to goal
        # Goal position in the start frame (initial heading = +x axis).
        self._goal = (self._d0 * math.cos(theta0), self._d0 * math.sin(theta0))
        # Dead-reckoned pose in the start frame.
        self._x = self._y = self._yaw = 0.0
        self._path = 0.0
        self._head_err, self._speeds, self._cte = [], [], []
        self._jerks, self._steer_changes = [], []
        self._near = 0
        self._clearance_intrusions = 0   # steps with min LiDAR < clearance_radius
        self._prev_action = None
        self._prev_steer = None
        self._steps = 0

    def update(self, state, action):
        """Accumulate one environment step."""
        s = self._s(state)
        ed = self.env_dim
        v = float(s[ed + 4])
        yaw_rate = float(s[ed + 5])
        steer = float(s[ed + 6])
        head_err = float(s[ed + 1])
        min_lidar = float(np.min(s[:ed])) if ed > 0 else float("inf")

        # Path length (true arc length) + dead-reckoned pose for CTE.
        self._path += abs(v) * self.dt
        self._yaw += yaw_rate * self.dt
        self._x += v * math.cos(self._yaw) * self.dt
        self._y += v * math.sin(self._yaw) * self.dt

        gx, gy = self._goal
        line_len = math.hypot(gx, gy)
        if line_len > 1e-6:
            cte = abs(gx * self._y - gy * self._x) / line_len
        else:
            cte = math.hypot(self._x, self._y)
        self._cte.append(cte)

        self._head_err.append(abs(head_err))
        self._speeds.append(abs(v))
        if min_lidar < self.near_dist:
            self._near += 1
        if min_lidar < self.clearance_radius:
            self._clearance_intrusions += 1

        a = np.asarray(action, dtype=np.float32).ravel()
        if self._prev_action is not None:
            self._jerks.append(float(np.linalg.norm(a - self._prev_action)))
        if self._prev_steer is not None:
            self._steer_changes.append(abs(steer - self._prev_steer))
        self._prev_action = a
        self._prev_steer = steer
        self._steps += 1

    def compute(self, success: bool) -> dict:
        """Return the per-episode metric dict (keys == METRIC_COLUMNS)."""
        path = self._path
        if self._d0 > 1e-6 and path > 1e-6:
            spl = (1.0 if success else 0.0) * (self._d0 / max(path, self._d0))
        else:
            spl = 1.0 if (success and self._d0 <= 1e-6) else 0.0

        def _mean(xs):
            return float(np.mean(xs)) if xs else 0.0

        # STL: success weighted by normalized time length (analogue of SPL on
        # time). optimal_time = shortest_path / ref_speed.
        travel_time = self._steps * self.dt
        if self._d0 > 1e-6:
            t_opt = self._d0 / self.stl_ref_speed
            stl = (1.0 if success else 0.0) * (t_opt / max(travel_time, t_opt)) if travel_time > 1e-6 else (1.0 if success else 0.0)
        else:
            stl = 1.0 if success else 0.0
        # LiDAR clearance compliance: fraction of steps that kept min-LiDAR above
        # the clearance radius (generic obstacle proxy, NOT human PSC).
        clearance = 1.0 - (self._clearance_intrusions / self._steps) if self._steps > 0 else 1.0

        return {
            "path_length_m":            round(path, 4),
            "spl":                      round(float(spl), 4),
            "mean_heading_error_rad":   round(_mean(self._head_err), 4),
            "mean_cross_track_error_m": round(_mean(self._cte), 4),
            "mean_action_jerk":         round(_mean(self._jerks), 4),
            "mean_steering_change_rad": round(_mean(self._steer_changes), 4),
            "near_collision_count":     int(self._near),
            "travel_time_s":            round(travel_time, 3),
            "mean_speed_mps":           round(_mean(self._speeds), 4),
            "stl":                      round(float(stl), 4),
            "lidar_clearance_rate":     round(float(clearance), 4),
        }

    @staticmethod
    def empty() -> dict:
        """A zero-filled metric dict (for episodes with no steps)."""
        return {c: (0 if c == "near_collision_count" else 0.0) for c in METRIC_COLUMNS}


# Aggregated-over-eval-episodes / per-training-episode CSV columns.
EPISODE_METRIC_HEADER = [
    "episode", "global_t", "curriculum_stage",
    "success", "collision", "timeout", "total_reward", "steps",
] + METRIC_COLUMNS

EVAL_METRIC_HEADER = [
    "epoch", "global_t", "curriculum_stage", "eval_eps",
    "mean_reward", "std_reward", "success_rate", "collision_rate",
    "timeout_rate", "mean_goal_dist",
] + METRIC_COLUMNS


class PaperMetricsCSV:
    """Writes the two comprehensive paper CSVs (episode-level + eval-level).

    Keeping the schema here guarantees header == row across every trainer and
    keeps the per-trainer integration to a single call.
    """

    def __init__(self, log_dir: str, run_tag: str):
        # RUN_LAYOUT: an empty run_tag (new-structure run, see run_layout.py)
        # yields plain "episode_metrics.csv" / "eval_metrics.csv"; a non-empty
        # tag (legacy layout) keeps the old "_<timestamp>" suffix.
        self.episode_path = os.path.join(
            log_dir, run_layout.tagged_filename("episode_metrics", ".csv", run_tag))
        self.eval_path = os.path.join(
            log_dir, run_layout.tagged_filename("eval_metrics", ".csv", run_tag))
        # RUN_LAYOUT: only write the header for a genuinely NEW file. A resume
        # into an existing new-structure run dir (plain filename, same path
        # every process start) must APPEND to its history, not truncate it.
        run_layout.write_csv_header_if_new(self.episode_path, EPISODE_METRIC_HEADER)
        run_layout.write_csv_header_if_new(self.eval_path, EVAL_METRIC_HEADER)

    def write_episode(self, *, episode, global_t, stage, success, collision,
                      timeout, total_reward, steps, metrics):
        row = [
            episode, global_t, stage,
            int(bool(success)), int(bool(collision)), int(bool(timeout)),
            round(float(total_reward), 4), steps,
        ] + [metrics[c] for c in METRIC_COLUMNS]
        with open(self.episode_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def write_eval(self, *, epoch, global_t, stage, eval_eps, base, metrics_mean):
        row = [
            epoch, global_t, stage, eval_eps,
            round(base["mean_reward"], 4), round(base["std_reward"], 4),
            round(base["success_rate"], 4), round(base["collision_rate"], 4),
            round(base["timeout_rate"], 4), round(base["mean_goal_dist"], 4),
        ] + [metrics_mean[c] for c in METRIC_COLUMNS]
        with open(self.eval_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    @staticmethod
    def aggregate(per_ep_metrics: list) -> dict:
        """Mean each metric over a list of per-episode metric dicts.

        Every metric is kept as a float — including ``near_collision_count``,
        whose aggregated value is the *mean near-collision count per episode*.
        Rounding it to an int would collapse safety-relevant differences
        (e.g. 0.4 vs 0.49 -> 0) that the paper safety table relies on.
        """
        if not per_ep_metrics:
            return EpisodeMetrics.empty()
        out = {}
        for c in METRIC_COLUMNS:
            vals = [m[c] for m in per_ep_metrics]
            out[c] = round(float(np.mean(vals)), 4)
        return out
