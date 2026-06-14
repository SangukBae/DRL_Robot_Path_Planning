"""Start / free-pose sampling under the structured map curriculum.\n\nExtracted from environment.py. Samples collision-free robot start poses (and generic free poses) within the active layout regions."""

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


class StartSamplerMixin:
    def _sample_free_pose_layout(self, radius: float, placed: list,
                                 start_x: float, start_y: float,
                                 avoid_open_area: bool = False):
        """Layout-aware obstacle pose: sample inside the current map's free
        regions, keep robot/goal/mutual margins + wall clearance (+ optional
        lobby open-area keep-out for large items)."""
        spec = self.current_layout_spec
        regions = spec["free_regions"]
        clr = self.map_wall_clearance
        for _ in range(200):
            xy = self._sample_xy_in_regions(regions, radius)
            if xy is None:
                return None
            x, y = xy
            if self._point_in_walls(x, y, clr + radius):
                continue
            # Keep the reserved corridor/intersection passage(s) traversable.
            if self._pose_blocks_reserved_passage(x, y, radius):
                continue
            if avoid_open_area and self._in_open_area(
                    x, y, margin=radius + self.map_large_footprint_margin):
                continue
            if math.hypot(x - start_x, y - start_y) < self.obstacle_robot_margin + radius:
                continue
            if math.hypot(x - self.goal_x, y - self.goal_y) < self.obstacle_goal_margin + radius:
                continue
            if not self._pose_collides_with_placed(
                x, y, self.obstacle_mutual_margin + radius, placed
            ):
                return x, y
        return None

    def _sample_train_start_pose(self):
        """
        Sample a collision-free start pose (x, y, yaw) for training episodes.

        Checks (in order for each candidate):
          1. Dead-zone exclusion
          2. Lingering-obstacle overlap
          3. Yaw sampled inside the loop — then:
          4. Heading-toward-near-wall rejection
          5. Front-cone immediate-collision rejection
        """
        robot_radius = self._robot_collision_radius()
        lingering    = list(self.spawned_obstacle_records.values())
        edge_margin  = self.start_edge_heading_margin
        clearance    = self.start_front_clearance
        fov_deg      = self.start_front_fov_deg
        layout       = self.current_layout_spec
        # Structured corridor/intersection maps: spawn inside an end/arm band and
        # head down the lane. Other maps keep the legacy free-region sampling.
        start_regions = layout.get("start_regions") if layout else None
        self._current_start_region = None

        def _sample_structured_start():
            """Pick a start region and a uniform pose inside it (footprint kept off
            the lane walls). Returns (x, y, region) or None."""
            region = start_regions[int(np.random.randint(len(start_regions)))]
            x_lo, x_hi, y_lo, y_hi = region["rect"]
            mx = my = robot_radius
            if region["axis"] == "x":         # cross axis = y → side margin on y
                my += self.map_start_side_margin
            else:
                mx += self.map_start_side_margin
            ax_lo, ax_hi = x_lo + mx, x_hi - mx
            ay_lo, ay_hi = y_lo + my, y_hi - my
            if ax_hi <= ax_lo or ay_hi <= ay_lo:
                return None
            return (float(np.random.uniform(ax_lo, ax_hi)),
                    float(np.random.uniform(ay_lo, ay_hi)), region)

        def _sample_start_xy():
            """Layout-aware when a structured map is active, else legacy box."""
            if layout is not None:
                xy = self._sample_xy_in_regions(layout["free_regions"], robot_radius)
                if xy is not None:
                    return xy
            return (np.random.uniform(self.lower, self.upper),
                    np.random.uniform(self.lower, self.upper))

        for _ in range(500):
            region = None
            if start_regions:
                got = _sample_structured_start()
                if got is not None:
                    start_x, start_y, region = got
                else:
                    start_x, start_y = _sample_start_xy()
            else:
                start_x, start_y = _sample_start_xy()

            # 1. Dead-zone
            if self.check_dead_zone(start_x, start_y, use_cross_mask=False):
                continue
            # 1b. Internal-wall clearance (structured maps only)
            if layout is not None and self._point_in_walls(
                    start_x, start_y, self.map_wall_clearance + robot_radius):
                continue
            # 2. Obstacle overlap
            if self._pose_collides_with_placed(start_x, start_y, robot_radius, lingering):
                continue

            # 3. Heading: lane-aligned for structured starts, else random.
            if region is not None:
                angle = self._sample_lane_aligned_yaw(region, start_x, start_y)
            else:
                angle = np.random.uniform(-np.pi, np.pi)

            # 4. Heading-toward-wall rejection
            if self._is_heading_toward_near_wall(start_x, start_y, angle, edge_margin):
                continue
            # 5. Front-cone immediate-collision rejection (obstacles + outer walls)
            if self._has_front_immediate_collision_risk(
                start_x, start_y, angle, lingering, clearance, fov_deg
            ):
                continue
            # 5b. Internal-wall front check (structured maps): _has_front_immediate_
            # collision_risk only sees lingering obstacles + the OUTER arena wall,
            # so without this the robot could start facing a corridor/intersection
            # wall a step away.
            if layout is not None and self._front_hits_internal_wall(
                start_x, start_y, angle, clearance + robot_radius
            ):
                continue

            self._current_start_region = region
            return start_x, start_y, angle

        # Fallback: relax only the front-cone / heading-toward-wall rejections.
        # Structured maps KEEP the end/arm start band + lane-aligned yaw (those
        # constraints are the requirement, not a nicety), so a crowded episode
        # never reverts a corridor/intersection start to mid-lane / random yaw.
        self.get_logger().warn(
            "Start-pose heading checks exhausted 500 tries; "
            "falling back to relaxed front-clearance pose"
        )
        for _ in range(200):
            region = None
            if start_regions:
                got = _sample_structured_start()
                if got is not None:
                    start_x, start_y, region = got
                else:
                    start_x, start_y = _sample_start_xy()
            else:
                start_x, start_y = _sample_start_xy()
            if self.check_dead_zone(start_x, start_y, use_cross_mask=False):
                continue
            if layout is not None and self._point_in_walls(
                    start_x, start_y, self.map_wall_clearance + robot_radius):
                continue
            if self._pose_collides_with_placed(start_x, start_y, robot_radius, lingering):
                continue
            # Structured: lane-aligned yaw (never into a wall); else random.
            angle = (self._sample_lane_aligned_yaw(region, start_x, start_y)
                     if region is not None else np.random.uniform(-np.pi, np.pi))
            self._current_start_region = region
            return start_x, start_y, angle

        # Last-ditch. Structured maps MUST still start inside an end/arm band so
        # position, yaw and goal pairing stay consistent — never the arena centre.
        # Use a deterministic safe point: the band mid-point ON the lane centre
        # line (cross-axis = 0), which is inside the band and off the side walls.
        # Prefer a band whose centre passes the position checks; else take the
        # first band's centre regardless (still structurally correct).
        if start_regions:
            self.get_logger().warn(
                "Start-pose sampling exhausted all 700 tries (500 + 200 fallback); "
                "using deterministic start-region centre (structured map)."
            )

            def _band_centre(region):
                x_lo, x_hi, y_lo, y_hi = region["rect"]
                if region["axis"] == "x":      # lane runs along x → centre y on the lane
                    return 0.5 * (x_lo + x_hi), 0.0
                return 0.0, 0.5 * (y_lo + y_hi)  # lane runs along y → centre x on the lane

            chosen = None
            for region in start_regions:
                cx, cy = _band_centre(region)
                if self.check_dead_zone(cx, cy, use_cross_mask=False):
                    continue
                if self._point_in_walls(cx, cy, self.map_wall_clearance + robot_radius):
                    continue
                if self._pose_collides_with_placed(cx, cy, robot_radius, lingering):
                    continue
                chosen = (region, cx, cy)
                break
            if chosen is None:
                region = start_regions[0]
                cx, cy = _band_centre(region)
                chosen = (region, cx, cy)
            region, cx, cy = chosen
            self._current_start_region = region
            return cx, cy, self._sample_lane_aligned_yaw(region, cx, cy)

        self.get_logger().warn(
            "Start-pose sampling exhausted all 700 tries (500 + 200 fallback); "
            "returning origin (0, 0). Check dead-zone / obstacle configuration."
        )
        return 0.0, 0.0, np.random.uniform(-np.pi, np.pi)

    def _sample_free_pose(self, radius: float, placed: list, start_x: float, start_y: float):
        """Sample a collision-free (x, y) for one obstacle.

        placed: list of (x, y, radius) already committed this episode.
        Returns (x, y) or None if no free position found within 200 tries.
        """
        # Structured map curriculum: delegate to the layout-aware sampler so the
        # obstacle lands inside the current map's free regions (off the walls).
        if self.current_layout_spec is not None:
            return self._sample_free_pose_layout(radius, placed, start_x, start_y)
        arena_lower = self.goal_obstacle_lower + self.obstacle_wall_margin
        arena_upper = self.goal_obstacle_upper - self.obstacle_wall_margin
        for _ in range(200):
            x = np.random.uniform(arena_lower, arena_upper)
            y = np.random.uniform(arena_lower, arena_upper)
            if math.hypot(x - start_x, y - start_y) < self.obstacle_robot_margin + radius:
                continue
            if math.hypot(x - self.goal_x, y - self.goal_y) < self.obstacle_goal_margin + radius:
                continue
            if not self._pose_collides_with_placed(
                x, y, self.obstacle_mutual_margin + radius, placed
            ):
                return x, y
        return None

    def _sample_free_pose_in_region(
        self, radius: float, placed: list, start_x: float, start_y: float,
        x_lo: float, x_hi: float, y_lo: float, y_hi: float
    ):
        """Like _sample_free_pose but constrained to [x_lo, x_hi] × [y_lo, y_hi]."""
        for _ in range(200):
            x = np.random.uniform(x_lo, x_hi)
            y = np.random.uniform(y_lo, y_hi)
            # Structured maps: keep clear of internal walls.
            if self.current_layout_spec is not None and self._point_in_walls(
                    x, y, self.map_wall_clearance):
                continue
            # Keep the reserved corridor/intersection passage clear at SPAWN, so a
            # traversable lane exists at t=0 including humans (they are dynamic and
            # may walk onto the lane later). Lighter keep-out (no static safety
            # margin) since a narrow corridor leaves little off-lane room; humans
            # that still don't fit are parked. No-op on maps without passages.
            if self._pose_blocks_reserved_passage(x, y, radius, safety_margin=0.0):
                continue
            if math.hypot(x - start_x, y - start_y) < self.obstacle_robot_margin + radius:
                continue
            if math.hypot(x - self.goal_x, y - self.goal_y) < self.obstacle_goal_margin + radius:
                continue
            if not self._pose_collides_with_placed(
                x, y, self.obstacle_mutual_margin + radius, placed
            ):
                return x, y
        return None

