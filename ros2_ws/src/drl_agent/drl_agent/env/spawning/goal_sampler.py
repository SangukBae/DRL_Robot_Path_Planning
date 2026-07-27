"""Goal-pose sampling under the structured map curriculum.\n\nExtracted from environment.py. Finds a goal that satisfies distance / map-type / clearance constraints relative to the chosen start."""

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

import drl_agent.common.point_cloud2 as pc2
import drl_agent.common.pure_pursuit as pure_pursuit
import drl_agent.common.geometry_utils as geom
import drl_agent.env.observation.aux_prediction_labels as aux_labels
from drl_agent.env.simulation.map_catalog import (
    STATIC_GLOBALLY_BANNED_KEYS,
    MAP_TYPE_ALLOWED_STATIC_KEYS,
    MAP_TYPES,
    static_size_group,
)


class GoalSamplerMixin:
    def _sample_goal_layout(self, start_x, start_y, goal_radius,
                            is_valid, lingering):
        """Structured-map goal sampler. STAYS INSIDE free_regions through EVERY
        phase — it never falls back to legacy goal_obstacle_lower~upper square
        sampling, so a goal can never land in a walled-off / off-lane cell.

        Phases (each strictly inside free_regions):
          A strict clearance + lane consistency (segment clear of walls)
          B strict clearance (drop lane check)
          C relaxed obstacle clearance (still dead-zone + wall clearance + dist)
          D deterministic free-region grid (~1.2 m), scored by clearance/start-dist/lane
          E farthest free-region grid point (~1.2 m) off walls + dead-zone
          F wide net over ALL regions — fine grid (~0.3 m) + random batch,
            off walls + dead-zone (catches narrow valid bands E's coarse grid skips)
          (centre) last-ditch largest-region centre, logged loudly
        Always returns a pose inside the current map's free regions. Phases A–F
        avoid walls + dead-zone; the centre last-ditch is a sampling-based
        best-effort, NOT a proof — it is reached only in very rare cases (e.g. a
        near-degenerate free region whose tiny valid set both the ~0.3 m grid and
        the 1000-draw random net miss). It is not a guarantee that no valid point
        exists, only that none was found.
        """
        spec = self.current_layout_spec
        clr = self.map_wall_clearance

        # Structure-aware goal region: when a corridor/intersection start region
        # is known, CONSTRAIN every phase below to the opposite-end / different-arm
        # bands (not just a preferred first pass). All A–F robustness fallbacks then
        # operate INSIDE those bands, so a hard episode can never yield a same-side /
        # mid-corridor or same-arm goal. Non-structured starts keep full free_regions.
        sr = self._current_start_region
        goal_bands = spec.get("goal_regions", {}).get(sr["name"]) if sr else None
        regions = goal_bands if goal_bands else spec["free_regions"]

        # A: strict clearance + (optional) lane-consistency.
        # The lane check (straight start->goal line stays off internal walls) is the
        # right PREFERENCE for a single straight lane (corridor) or scattered walls
        # (clutter), but it is WRONG for the intersection: only the OPPOSITE arm has
        # a wall-free straight line — the two SIDE arms are reached by routing
        # through the free centre, so their straight line clips a corner block.
        # Enforcing it here meant phase A (800 area-weighted tries over the 3 equal
        # candidate arms) accepted opposite-arm goals immediately and rejected every
        # side-arm sample, so the goal was almost always the opposite arm and phase B
        # (which drops the check) was never reached. We therefore SKIP the lane check
        # on the intersection, so all three non-start arms — weighted equally by
        # goal_regions — are sampled ~uniformly. corridor/clutter keep it unchanged.
        skip_lane_check = (self.current_map_type == "intersection")
        for _ in range(800):
            xy = self._sample_xy_in_regions(regions, goal_radius)
            if xy is None:
                break
            x, y = xy
            if is_valid(x, y, require_clearance=True) and (
                    skip_lane_check
                    or not self._segment_hits_wall(start_x, start_y, x, y, clr)):
                return x, y
        # B: strict clearance, drop the lane check (e.g. intersection cross-arm
        # goals whose straight line clips a corner but are reachable via centre).
        for _ in range(400):
            xy = self._sample_xy_in_regions(regions, goal_radius)
            if xy is None:
                break
            x, y = xy
            if is_valid(x, y, require_clearance=True):
                return x, y
        # C: relaxed obstacle clearance (dead-zone + wall clearance + dist kept).
        for _ in range(300):
            xy = self._sample_xy_in_regions(regions, goal_radius)
            if xy is None:
                break
            x, y = xy
            if is_valid(x, y, require_clearance=False):
                return x, y

        # D: deterministic grid over the free regions, scored.
        grid = self._layout_region_grid(regions, goal_radius)
        best, best_score = None, -float("inf")
        for (x, y) in grid:
            if not is_valid(x, y, require_clearance=False):
                continue
            start_dist = math.hypot(x - start_x, y - start_y)
            if lingering:
                min_obs_clear = min(
                    math.hypot(x - px, y - py) - (goal_radius + pr)
                    for px, py, pr in lingering)
            else:
                min_obs_clear = float("inf")
            if skip_lane_check:
                # Intersection: BOTH the straight-line lane bonus and the
                # start-distance term geometrically favour the opposite arm (the
                # only one with a wall-free straight line, and the farthest), which
                # would re-collapse the fallback onto opposite-arm goals in crowded
                # / exhausted episodes. Score purely by obstacle clearance so the
                # least-crowded of the three candidate arms wins, with no arm bias.
                score = min_obs_clear
            else:
                lane_bonus = (0.0 if self._segment_hits_wall(start_x, start_y, x, y, clr)
                              else 2.0)
                score = min_obs_clear + 0.1 * start_dist + lane_bonus
            if score > best_score:
                best_score, best = score, (x, y)
        if best is not None:
            self.get_logger().warn(
                "change_goal[layout]: using deterministic free-region grid fallback")
            return best

        # E: last resort — farthest grid point that is merely off walls + dead-zone.
        best, best_d = None, -1.0
        for (x, y) in grid:
            if self.check_dead_zone(x, y, use_cross_mask=False,
                                    lower_bound=self.goal_obstacle_lower,
                                    upper_bound=self.goal_obstacle_upper):
                continue
            if self._point_in_walls(x, y, clr + goal_radius):
                continue
            d = math.hypot(x - start_x, y - start_y)
            if d > best_d:
                best_d, best = d, (x, y)
        if best is not None:
            self.get_logger().warn(
                "change_goal[layout]: last-resort free-region point (off-wall, off-dead-zone)")
            return best

        # F: wide net over ALL free regions for an off-wall + off-dead-zone point.
        # A fine deterministic grid (~0.3 m, centre-out per region) PLUS a random
        # batch, so a valid point in a narrow band BETWEEN the coarse phase D/E
        # grid lines is very likely still found. This is best-effort sampling, not
        # a proof: it drives the miss probability for any non-tiny valid set down
        # to ~0, but cannot guarantee finding a sub-grid, sub-random-net sliver.
        def _f_valid(px, py):
            if self.check_dead_zone(px, py, use_cross_mask=False,
                                    lower_bound=self.goal_obstacle_lower,
                                    upper_bound=self.goal_obstacle_upper):
                return False
            return not self._point_in_walls(px, py, clr + goal_radius)

        fine = []
        for (x_lo, x_hi, y_lo, y_hi) in regions:
            ax_lo, ax_hi = x_lo + goal_radius, x_hi - goal_radius
            ay_lo, ay_hi = y_lo + goal_radius, y_hi - goal_radius
            if ax_hi <= ax_lo or ay_hi <= ay_lo:
                continue
            rcx, rcy = 0.5 * (ax_lo + ax_hi), 0.5 * (ay_lo + ay_hi)
            nx = max(3, int((ax_hi - ax_lo) / 0.3) + 1)
            ny = max(3, int((ay_hi - ay_lo) / 0.3) + 1)
            for x in np.linspace(ax_lo, ax_hi, nx):
                for y in np.linspace(ay_lo, ay_hi, ny):
                    fine.append((float(x), float(y),
                                 math.hypot(x - rcx, y - rcy)))
        fine.sort(key=lambda p: p[2])   # centre-out within each region
        for (x, y, _d) in fine:
            if _f_valid(x, y):
                self.get_logger().warn(
                    "change_goal[layout]: fine-grid fallback (off-wall, off-dead-zone)")
                return x, y
        # Random net catches any sub-0.3 m valid sliver the grid still skipped.
        for _ in range(1000):
            xy = self._sample_xy_in_regions(regions, goal_radius)
            if xy is None:
                break
            x, y = xy
            if _f_valid(x, y):
                self.get_logger().warn(
                    "change_goal[layout]: random-net fallback (off-wall, off-dead-zone)")
                return x, y

        # Last-ditch: neither the ~0.3 m grid nor the 1000-draw random net found
        # an off-wall/off-dead-zone point. This is almost always a near-degenerate
        # free region (its valid set, if any, is tiny) — keep the run alive with
        # the largest region's centre, logged loudly so it is visible. NOTE: this
        # does NOT prove no valid point exists, only that sampling found none.
        x_lo, x_hi, y_lo, y_hi = max(
            regions, key=lambda r: (r[1] - r[0]) * (r[3] - r[2]))
        cx, cy = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
        self.get_logger().warn(
            "change_goal[layout]: no off-wall/off-dead-zone point FOUND in any "
            "region (grid + random net exhausted; likely near-degenerate) — "
            "using region centre as a last-ditch best-effort")
        return cx, cy
