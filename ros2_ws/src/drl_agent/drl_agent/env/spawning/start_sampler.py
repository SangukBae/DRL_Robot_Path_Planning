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


class StartSamplerMixin:
    def _sample_free_pose_layout(self, radius: float, placed: list,
                                 start_x: float, start_y: float,
                                 avoid_open_area: bool = False, rng=None):
        """Layout-aware obstacle pose: sample inside the current map's free
        regions, keep robot/goal/mutual margins + wall clearance (+ optional
        lobby open-area keep-out for large items).

        ``rng`` is threaded into _sample_xy_in_regions so HUMAN callers draw from
        their dedicated sub-stream; None keeps the global stream (static/robot)."""
        spec = self.current_layout_spec
        regions = spec["free_regions"]
        clr = self.map_wall_clearance
        for _ in range(200):
            xy = self._sample_xy_in_regions(regions, radius, rng=rng)
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

    def _choose_start_yaw(self, region, x, y):
        """Pick the spawn yaw for a start candidate (single source of truth).

        * structured lane region (corridor / intersection)
              → ``_sample_lane_aligned_yaw`` (unchanged contract).
        * open map (lobby / clutter) AND ``open_map_safe_start_yaw_enabled``
              → ``_sample_open_map_safe_yaw`` (inward-safe, avoids facing the
                outer wall while keeping yaw diversity away from walls).
        * otherwise (flag off, or any other map)
              → legacy uniform random yaw in [-pi, pi] (byte-identical default).
        """
        if region is not None:
            return self._sample_lane_aligned_yaw(region, x, y)
        if (getattr(self, "open_map_safe_start_yaw_enabled", False)
                and getattr(self, "current_map_type", "") in ("lobby", "clutter")):
            return self._sample_open_map_safe_yaw(x, y)
        return float(np.random.uniform(-np.pi, np.pi))

    def _sample_train_start_pose(self):
        """
        Sample a collision-free start pose (x, y, yaw) for training episodes.

        Checks (in order for each candidate):
          1. Dead-zone exclusion
          2. Obstacle overlap — only against GENUINELY-PRESENT entities (see
             `lingering` below); empty in pool mode, so usually a no-op
          3. Yaw sampled inside the loop — then:
          4. Heading-toward-near-wall rejection
          5. Front-cone immediate-collision rejection (obstacles + outer wall)
        """
        robot_radius = self._robot_collision_radius()
        # `lingering` = obstacles the robot must avoid AT SAMPLING TIME. Its
        # contents depend on the obstacle backend, because the same
        # spawned_obstacle_records dict means different things:
        #
        #  * Pool mode (use_obstacle_pool=True, the curriculum default): obstacles
        #    are TELEPORTED, never deleted. spawned_obstacle_records still holds the
        #    PREVIOUS episode's placements here (it is refreshed by
        #    _activate_random_obstacles AFTER this call) and the human-motion timer
        #    even rewrites human entries to wherever they drifted last episode.
        #    Those entities are about to be teleported away — _activate_random_
        #    obstacles re-places every active obstacle keeping >= obstacle_robot_
        #    margin (2.82 m) + radius clear of THIS start, and parks the rest. So
        #    consulting the records is a STALE constraint: prev-episode obstacles
        #    that this episode will not reproduce carve disks out of the new start
        #    band, exhausting the 500/200-try budgets and forcing the deterministic-
        #    centre fallback — collapsing start diversity. We therefore IGNORE them
        #    (empty list) and rely on the placer keeping clear of the chosen start.
        #
        #  * Non-pool mode (use_obstacle_pool=False): obstacles are deleted +
        #    re-spawned each reset. _delete_spawned_obstacles() ran before this call
        #    and dropped a record only once its entity is CONFIRMED gone (a human's
        #    body footprint is retained while ANY of its 8 parts still lingers), so
        #    whatever remains in spawned_obstacle_records is genuinely still in the
        #    world (a delete that timed out / failed) at its body footprint. Those
        #    are REAL leftovers and must be avoided.
        #
        # Either way the front-cone check below still rejects a start facing an
        # OUTER wall within front-clearance (empty placed list -> wall ray-cast
        # only); internal walls are handled by _front_hits_internal_wall.
        lingering    = ([] if getattr(self, "use_obstacle_pool", False)
                        else list(self.spawned_obstacle_records.values()))
        edge_margin  = self.start_edge_heading_margin
        clearance    = self.start_front_clearance
        fov_deg      = self.start_front_fov_deg
        layout       = self.current_layout_spec
        # Structured corridor/intersection maps: spawn inside an end/arm band and
        # head down the lane. Other maps keep the legacy free-region sampling.
        start_regions = layout.get("start_regions") if layout else None
        self._current_start_region = None

        # Dead-zone "in-map" bound for THIS sampler. Structured maps build their
        # end/arm start bands inside the NAVIGABLE extent (e.g. corridor right band
        # x∈[9.3, 11.8]); the legacy default bound (self.lower/upper = ±9.0) lies
        # INSIDE those bands, so check_dead_zone rejected every band candidate,
        # exhausting the 500/200 budgets and forcing the deterministic-centre fallback
        # EACH episode (the warnings in the training log). We pass the navigable extent
        # (the exact box the bands were built within), which can never reject a valid
        # band candidate. goal_obstacle_lower/upper would also stop the storm in THIS
        # config (the footprint shrink keeps samples inside ±11.5), but it is < map_inner
        # and not the band-construction box, so it is not robust to lane/band/robot
        # changes. Legacy scatter (layout is None) keeps self.lower/upper unchanged.
        if layout is not None:
            dz_lower = getattr(self, "map_navigable_lower", self.goal_obstacle_lower)
            dz_upper = getattr(self, "map_navigable_upper", self.goal_obstacle_upper)
        else:
            dz_lower, dz_upper = self.lower, self.upper

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
            if self.check_dead_zone(start_x, start_y, use_cross_mask=False,
                                    lower_bound=dz_lower, upper_bound=dz_upper):
                continue
            # 1b. Internal-wall clearance (structured maps only)
            if layout is not None and self._point_in_walls(
                    start_x, start_y, self.map_wall_clearance + robot_radius):
                continue
            # 2. Obstacle overlap
            if self._pose_collides_with_placed(start_x, start_y, robot_radius, lingering):
                continue

            # 3. Heading: lane-aligned (structured), inward-safe (open map), else random.
            angle = self._choose_start_yaw(region, start_x, start_y)

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
        self._start_relaxed_fallback_count = getattr(
            self, "_start_relaxed_fallback_count", 0) + 1
        self.get_logger().warn(
            "Start-pose heading checks exhausted 500 tries; "
            "falling back to relaxed front-clearance pose "
            f"(relaxed fallbacks so far: {self._start_relaxed_fallback_count})"
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
            if self.check_dead_zone(start_x, start_y, use_cross_mask=False,
                                    lower_bound=dz_lower, upper_bound=dz_upper):
                continue
            if layout is not None and self._point_in_walls(
                    start_x, start_y, self.map_wall_clearance + robot_radius):
                continue
            if self._pose_collides_with_placed(start_x, start_y, robot_radius, lingering):
                continue
            # Structured: lane-aligned; open map: inward-safe; else random.
            angle = self._choose_start_yaw(region, start_x, start_y)
            self._current_start_region = region
            return start_x, start_y, angle

        # Last-ditch. Structured maps MUST still start inside an end/arm band so
        # position, yaw and goal pairing stay consistent — never the arena centre.
        # Use a deterministic safe point: the band mid-point ON the lane centre
        # line (cross-axis = 0), which is inside the band and off the side walls.
        # Prefer a band whose centre passes the position checks; else take the
        # first band's centre regardless (still structurally correct).
        if start_regions:
            self._start_centre_fallback_count = getattr(
                self, "_start_centre_fallback_count", 0) + 1
            self.get_logger().warn(
                "Start-pose sampling exhausted all 700 tries (500 + 200 fallback); "
                "using deterministic start-region centre (structured map). "
                f"(centre fallbacks so far: {self._start_centre_fallback_count})"
            )

            def _band_centre(region):
                x_lo, x_hi, y_lo, y_hi = region["rect"]
                if region["axis"] == "x":      # lane runs along x → centre y on the lane
                    return 0.5 * (x_lo + x_hi), 0.0
                return 0.0, 0.5 * (y_lo + y_hi)  # lane runs along y → centre x on the lane

            chosen = None
            for region in start_regions:
                cx, cy = _band_centre(region)
                if self.check_dead_zone(cx, cy, use_cross_mask=False,
                                        lower_bound=dz_lower, upper_bound=dz_upper):
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

    def _sample_free_pose(self, radius: float, placed: list, start_x: float, start_y: float,
                          rng=None):
        """Sample a collision-free (x, y) for one obstacle.

        placed: list of (x, y, radius) already committed this episode.
        Returns (x, y) or None if no free position found within 200 tries.

        ``rng`` selects the RNG sub-stream: None (default) draws from the GLOBAL
        np.random (static-obstacle / robot / goal callers, unchanged), while
        HUMAN callers pass ``self._human_np_rng`` so pedestrian spawn never
        advances the global stream (reproducibility split).
        """
        r = rng if rng is not None else np.random
        # Structured map curriculum: delegate to the layout-aware sampler so the
        # obstacle lands inside the current map's free regions (off the walls).
        if self.current_layout_spec is not None:
            return self._sample_free_pose_layout(radius, placed, start_x, start_y, rng=rng)
        arena_lower = self.goal_obstacle_lower + self.obstacle_wall_margin
        arena_upper = self.goal_obstacle_upper - self.obstacle_wall_margin
        for _ in range(200):
            x = r.uniform(arena_lower, arena_upper)
            y = r.uniform(arena_lower, arena_upper)
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
        x_lo: float, x_hi: float, y_lo: float, y_hi: float, rng=None
    ):
        """Like _sample_free_pose but constrained to [x_lo, x_hi] × [y_lo, y_hi].

        ``rng`` as in _sample_free_pose: None → global np.random; human callers
        pass ``self._human_np_rng`` to keep pedestrian spawn off the global stream.
        """
        r = rng if rng is not None else np.random
        for _ in range(200):
            x = r.uniform(x_lo, x_hi)
            y = r.uniform(y_lo, y_hi)
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
