"""Human (pedestrian) spawn-pose, mode, goal and waypoint sampling.\n\nExtracted from environment.py. Initialises pedestrian start poses, motion modes and goals for the episode; the per-step motion update lives in human_motion_manager."""

import os
import math
import time
import random
import csv
from datetime import datetime

import numpy as np
from collections import deque
from squaternion import Quaternion

from geometry_msgs.msg import Twist, Pose, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2, JointState, LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from drl_agent_interfaces.msg import DrlModelPoseArray
from ros_gz_interfaces.msg import Entity as GzEntity
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose, SpawnEntity, DeleteEntity

import point_cloud2 as pc2
import pure_pursuit
import geometry_utils as geom
import aux_prediction_labels as aux_labels
from map_catalog import (
    STATIC_GLOBALLY_BANNED_KEYS,
    MAP_TYPE_ALLOWED_STATIC_KEYS,
    MAP_TYPES,
    static_size_group,
)


class HumanSpawnMixin:
    def _build_human_spawn_regions(self):
        """Return shuffled spawn regions for distributed human placement.

        Structured-map modes (corridor_lanes / intersection_arms / lobby_crossings)
        draw their regions from the active layout so pedestrians spawn ON the
        navigable lanes. `quadrants` keeps the legacy four-corner split;
        `global_random` (or any other mode) returns None → whole-arena sampling.
        """
        mode = self.human_placement_mode
        # Layout-aware modes: use the current map's human regions.
        if self.current_layout_spec is not None and mode in (
                "corridor_lanes", "intersection_arms", "lobby_crossings"):
            regions = list(self.current_layout_spec.get("human_regions", []))
            if regions:
                self._human_py_rng.shuffle(regions)
                return regions
            return None

        if mode != "quadrants":
            return None

        q_lo = self.goal_obstacle_lower + self.obstacle_wall_margin
        q_hi = self.goal_obstacle_upper - self.obstacle_wall_margin
        quadrants = [
            (q_lo, 0.0, q_lo, 0.0),   # bottom-left
            (0.0,  q_hi, q_lo, 0.0),  # bottom-right
            (q_lo, 0.0, 0.0,  q_hi),  # top-left
            (0.0,  q_hi, 0.0, q_hi),  # top-right
        ]
        self._human_py_rng.shuffle(quadrants)
        return quadrants

    def _point_in_human_regions(self, x: float, y: float) -> bool:
        """True if (x, y) is inside the active map's human (navigable) regions —
        the lane (corridor), the arms (intersection), or the full inner extent
        (lobby / clutter). Used to keep an episode GOAL REACHABLE: a goal outside
        these regions sits across an internal wall, which the simple goal-driven
        pedestrian motion cannot reach — it would just slide into the wall for the
        whole episode (Fix A stops penetration but not the futile pushing). The
        per-step WAYPOINT is intentionally NOT constrained this way: it must be
        able to transit free space between regions (e.g. the intersection centre,
        which belongs to no arm) — its own wall checks keep it off walls.
        Permissive (True) for non-structured maps / layouts without human_regions.
        """
        spec = self.current_layout_spec
        if spec is None:
            return True
        regions = spec.get("human_regions", [])
        if not regions:
            return True
        for (x_lo, x_hi, y_lo, y_hi) in regions:
            if x_lo <= x <= x_hi and y_lo <= y <= y_hi:
                return True
        return False

    def _sample_human_spawn_pose(
        self,
        radius: float,
        placed: list,
        start_x: float,
        start_y: float,
        human_index: int,
        spawn_regions,
    ):
        """Sample a human pose using the active placement mode."""
        if spawn_regions:
            x_lo, x_hi, y_lo, y_hi = spawn_regions[human_index % len(spawn_regions)]
            return self._sample_free_pose_in_region(
                radius, placed, start_x, start_y, x_lo, x_hi, y_lo, y_hi,
                rng=self._human_np_rng,   # pedestrian spawn stays off the global stream
            )
        return self._sample_free_pose(
            radius, placed, start_x, start_y, rng=self._human_np_rng)

    # Per-mode pedestrian motion defaults. Each human keeps one mode for the
    # whole episode. Tunable via human_mode_params[<mode>] in the YAML (or per
    # stage). Distances in metres, durations in seconds, dir_spread in degrees.
    # goal_span_scale (Falcon-lite) scales BOTH the sampled goal distance and the
    # human_goal_min_span_m floor for this mode, so "waiting" keeps its original
    # near-stationary meaning (small goals + lots of rest) even with a large
    # global min span, while crossing/along_path still travel far.
    _HUMAN_MODE_DEFAULTS = {
        # arena-crossing: long, near-straight travel toward the opposite side
        "crossing":   {"speed_scale": 1.0, "stop_prob": 0.005, "pause_duration": 0.8,
                       "wp_dist_min": 3.0, "wp_dist_max": 5.0, "dir_spread_deg": 12.0,
                       "axis": "crossing",  "init_wait_sec": 0.0, "goal_span_scale": 1.0},
        # along a principal (x/y) axis: long straight runs, paces back at walls
        "along_path": {"speed_scale": 1.0, "stop_prob": 0.005, "pause_duration": 0.8,
                       "wp_dist_min": 3.0, "wp_dist_max": 5.0, "dir_spread_deg": 8.0,
                       "axis": "principal", "init_wait_sec": 0.0, "goal_span_scale": 1.0},
        # mostly stationary: initial wait + frequent, long pauses, slow when
        # moving; tiny goal span so it lingers near its spawn ("waiting" sense).
        "waiting":    {"speed_scale": 0.6, "stop_prob": 0.08,  "pause_duration": 2.5,
                       "wp_dist_min": 1.0, "wp_dist_max": 2.0, "dir_spread_deg": 60.0,
                       "axis": "random",    "init_wait_sec": 3.0, "goal_span_scale": 0.25},
        # gentle meander: short waypoints + wide direction spread → smooth turns
        "slow_turn":  {"speed_scale": 0.8, "stop_prob": 0.01,  "pause_duration": 1.0,
                       "wp_dist_min": 1.5, "wp_dist_max": 2.5, "dir_spread_deg": 50.0,
                       "axis": "random",    "init_wait_sec": 0.0, "goal_span_scale": 0.6},
    }

    def _assign_human_mode(self, x: float, y: float) -> dict:
        """Sample one behaviour mode for a pedestrian and return its state fields.

        Called once per human at spawn. The mode is sampled from
        human_mode_weights (empty → uniform); per-mode kinematic knobs come from
        _HUMAN_MODE_DEFAULTS merged with human_mode_params[mode]. The returned
        dict is merged into the human's state so the motion model reads per-human
        (mode-dependent) values for the whole episode.
        """
        modes = list(self._HUMAN_MODE_DEFAULTS.keys())
        weights = [max(0.0, float(self.human_mode_weights.get(m, 1.0))) for m in modes]
        total = sum(weights)
        if total <= 0.0:
            weights = [1.0] * len(modes)
            total = float(len(modes))
        mode = str(self._human_np_rng.choice(modes, p=[w / total for w in weights]))

        p = dict(self._HUMAN_MODE_DEFAULTS[mode])
        p.update(self.human_mode_params.get(mode, {}) or {})

        axis_kind = str(p.get("axis", "random"))
        if axis_kind == "crossing":
            mode_axis = math.atan2(-y, -x)                      # head across the arena
        elif axis_kind == "principal":
            mode_axis = float(self._human_np_rng.choice([0.0, math.pi / 2, math.pi, -math.pi / 2]))
        else:
            mode_axis = None                                    # random direction each retarget
        fields = {
            "mode":               mode,
            "mode_axis":          mode_axis,
            "mode_dir_spread":    math.radians(float(p["dir_spread_deg"])),
            "mode_wp_dist_min":   float(p["wp_dist_min"]),
            "mode_wp_dist_max":   float(p["wp_dist_max"]),
            "mode_stop_prob":     float(p["stop_prob"]),
            "mode_pause_duration": float(p["pause_duration"]),
            "mode_speed_scale":   float(p["speed_scale"]),
            "mode_init_wait":     float(p["init_wait_sec"]),
            # Falcon-lite per-mode goal-distance scale (waiting stays near).
            "mode_goal_span_scale": float(p.get("goal_span_scale", 1.0)),
        }
        # Falcon-lite: give the human an episode goal so it travels goal-directed
        # rather than random-walking. The travel style (mode) sets the goal span /
        # direction via the fields above.
        if self.human_goal_driven_enabled:
            gx, gy = self._sample_human_goal(x, y, fields)
            fields["goal_x"] = gx
            fields["goal_y"] = gy
        return fields

    def _sample_human_goal(self, cx, cy, state):
        """Pick a global episode goal for a pedestrian (Falcon-lite goal-driven).

        The goal distance scales with the human's travel style: nominal reach =
        (mode wp_dist_max * human_goal_span_multiplier * mode_goal_span_scale),
        with the human_goal_min_span_m floor ALSO scaled by mode_goal_span_scale.
        So crossing / along_path travel far along their fixed axis, waiting stays
        near its spawn (lots of rest), slow_turn roams a moderate distance.

        Recovery: if the mode-shaped sampling cannot find a free goal, fall back
        to a free pose anywhere in the arena (not the current position, which
        would make the human think it already arrived and freeze in a rest loop).
        """
        margin = 1.5
        lower = self.goal_obstacle_lower + self.obstacle_wall_margin + margin
        upper = self.goal_obstacle_upper - self.obstacle_wall_margin - margin
        arena_span = max(0.0, upper - lower)
        scale = max(1e-3, float(state.get("mode_goal_span_scale", 1.0)))
        # Effective floor shrinks for near-stationary modes (waiting), never below
        # a small absolute minimum so a goal is still a short walk away.
        floor = max(0.5, self.human_goal_min_span_m * scale)
        span_max = float(np.clip(
            float(state.get("mode_wp_dist_max", 3.0)) * self.human_goal_span_multiplier * scale,
            floor, arena_span if arena_span > 0 else floor))
        span_min = min(max(0.6 * floor, float(state.get("mode_wp_dist_min", 1.0)) * scale),
                       span_max)
        min_reach = 0.5 * span_min   # goal must be at least this far (non-degenerate)
        axis = state.get("mode_axis")
        spread = float(state.get("mode_dir_spread", math.pi))
        for _ in range(60):
            d = self._human_np_rng.uniform(span_min, span_max)
            if axis is None:
                ang = self._human_np_rng.uniform(-math.pi, math.pi)      # waiting / slow_turn
            else:
                ang = float(axis) + self._human_np_rng.uniform(-spread, spread)  # crossing / along_path
            gx = float(np.clip(cx + d * math.cos(ang), lower, upper))
            gy = float(np.clip(cy + d * math.sin(ang), lower, upper))
            if (math.hypot(gx - cx, gy - cy) >= min_reach
                    and not self.check_dead_zone(
                        gx, gy, use_cross_mask=False,
                        lower_bound=lower, upper_bound=upper)
                    # Goal must be off internal walls AND inside the reachable
                    # navigable region (lane/arm) — not across a wall.
                    and not self._point_in_walls(gx, gy, self.human_target_wall_clearance)
                    and not self._point_in_static_obstacle_for_human(
                        gx, gy, self.human_target_wall_clearance)
                    and not self._segment_hits_static_obstacle_for_human(
                        cx, cy, gx, gy, self.human_segment_wall_clearance)
                    and self._point_in_human_regions(gx, gy)):
                return gx, gy
        # Recovery: any free pose in the arena (wider search than the mode cone).
        for _ in range(40):
            gx = float(self._human_np_rng.uniform(lower, upper))
            gy = float(self._human_np_rng.uniform(lower, upper))
            if (math.hypot(gx - cx, gy - cy) >= min_reach
                    and not self.check_dead_zone(
                        gx, gy, use_cross_mask=False,
                        lower_bound=lower, upper_bound=upper)
                    and not self._point_in_walls(gx, gy, self.human_target_wall_clearance)
                    and not self._point_in_static_obstacle_for_human(
                        gx, gy, self.human_target_wall_clearance)
                    and not self._segment_hits_static_obstacle_for_human(
                        cx, cy, gx, gy, self.human_segment_wall_clearance)
                    and self._point_in_human_regions(gx, gy)):
                return gx, gy
        return float(cx), float(cy)   # arena fully blocked -> hold (very rare)

    def _sample_human_waypoint(self, cx=None, cy=None, state=None):
        """Return a target waypoint (x, y) inside the arena for a pedestrian.

        Sampling is mode-aware when a state with mode_wp_dist_max > 0 is given:
        a distance in [mode_wp_dist_min, mode_wp_dist_max] along the human's mode
        axis (± mode_dir_spread). A None axis means a random direction (waiting /
        slow_turn); a fixed axis (crossing / along_path) keeps travel straight.
        Otherwise it falls back to the global bounded sampling, then to legacy
        whole-arena sampling — keeping each segment short so a new waypoint (mode
        step) is reached every few seconds rather than one long straight line.
        """
        margin = 1.5
        lower = self.goal_obstacle_lower + self.obstacle_wall_margin + margin
        upper = self.goal_obstacle_upper - self.obstacle_wall_margin - margin

        # Falcon-lite goal-driven: return a short LOCAL step toward the human's
        # global goal (so the apparent motion is goal-directed travel, not a
        # random walk). The step length is human_local_step_m, shortened so it
        # never overshoots the goal. If the straight step lands in a dead zone,
        # try a few small heading offsets to route around it (keeps turns gentle).
        goal_driven = (
            self.human_goal_driven_enabled and state is not None
            and cx is not None and cy is not None
            and "goal_x" in state and "goal_y" in state
        )
        if goal_driven:
            gx, gy = float(state["goal_x"]), float(state["goal_y"])
            dgoal = math.hypot(gx - cx, gy - cy)
            step = min(self.human_local_step_m, dgoal) if dgoal > 1e-3 else 0.0
            base_ang = math.atan2(gy - cy, gx - cx)
            # Try the straight step, then progressively wider offsets to route
            # around a dead zone (gentle first, then a full fan as a recovery).
            offsets = (0.0, 0.26, -0.26, 0.52, -0.52, 0.79, -0.79,
                       1.05, -1.05, 1.57, -1.57, 2.36, -2.36, math.pi)
            for off in offsets:
                ang = base_ang + off
                x = float(np.clip(cx + step * math.cos(ang), lower, upper))
                y = float(np.clip(cy + step * math.sin(ang), lower, upper))
                # Route the local step around dead zones AND internal walls: reject
                # a target inside a wall or a step whose straight segment crosses
                # one (the offset fan then bends the path around the wall).
                if (not self.check_dead_zone(x, y, use_cross_mask=False,
                                             lower_bound=lower, upper_bound=upper)
                        and not self._point_in_walls(x, y, self.human_target_wall_clearance)
                        and not self._point_in_static_obstacle_for_human(
                            x, y, self.human_target_wall_clearance)
                        and not self._segment_hits_wall(
                            cx, cy, x, y, self.human_segment_wall_clearance)
                        and not self._segment_hits_static_obstacle_for_human(
                            cx, cy, x, y, self.human_segment_wall_clearance)):
                    return x, y
            return float(cx), float(cy)   # boxed in -> hold (caller resamples goal)

        mode_bounded = (
            cx is not None and cy is not None
            and state is not None and float(state.get("mode_wp_dist_max", 0.0)) > 0.0
        )
        bounded = (
            cx is not None and cy is not None and self.human_waypoint_dist_max > 0.0
        )
        for _ in range(100):
            if mode_bounded:
                dmin = float(state.get("mode_wp_dist_min", 0.0))
                dmax = float(state["mode_wp_dist_max"])
                d = self._human_np_rng.uniform(min(dmin, dmax), dmax)
                axis = state.get("mode_axis")
                if axis is None:
                    ang = self._human_np_rng.uniform(-math.pi, math.pi)
                else:
                    spread = float(state.get("mode_dir_spread", math.pi))
                    ang = float(axis) + self._human_np_rng.uniform(-spread, spread)
                x = float(np.clip(cx + d * math.cos(ang), lower, upper))
                y = float(np.clip(cy + d * math.sin(ang), lower, upper))
            elif bounded:
                d = self._human_np_rng.uniform(
                    min(self.human_waypoint_dist_min, self.human_waypoint_dist_max),
                    self.human_waypoint_dist_max,
                )
                ang = self._human_np_rng.uniform(-math.pi, math.pi)
                x = float(np.clip(cx + d * math.cos(ang), lower, upper))
                y = float(np.clip(cy + d * math.sin(ang), lower, upper))
            else:
                x = self._human_np_rng.uniform(lower, upper)
                y = self._human_np_rng.uniform(lower, upper)
            if (not self.check_dead_zone(x, y, use_cross_mask=False,
                                         lower_bound=lower, upper_bound=upper)
                    and not self._point_in_walls(x, y, self.human_target_wall_clearance)
                    and not self._point_in_static_obstacle_for_human(
                        x, y, self.human_target_wall_clearance)
                    and (cx is None or cy is None
                         or not self._segment_hits_static_obstacle_for_human(
                             cx, cy, x, y, self.human_segment_wall_clearance))):
                return x, y
        return (float(cx), float(cy)) if (mode_bounded or bounded) else (0.0, 0.0)
