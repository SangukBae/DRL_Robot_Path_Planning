"""Structured map-layout runtime: selection, wall activation, placement geometry.\n\nExtracted from environment.py. Chooses the per-episode map_type, activates the matching wall pool, and provides the spatial predicates (wall/passage/open-area tests) used by start/goal/obstacle placement. Pure layout DATA is built by map_layout_registry.build_map_layouts (called from _build_map_layouts)."""

import os
import math
import time
import random
import csv
from datetime import datetime

import numpy as np
from collections import deque
from squaternion import Quaternion

from rclpy.parameter import Parameter
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
import map_layout_registry
from map_catalog import (
    STATIC_GLOBALLY_BANNED_KEYS,
    MAP_TYPE_ALLOWED_STATIC_KEYS,
    MAP_TYPES,
    static_size_group,
)


class MapLayoutMixin:
    def _static_key_fits_map_geometry(self, map_type: str, key: str) -> bool:
        """Return whether a catalog key can ever fit the current map geometry.

        This is stricter than the semantic policy in ``map_catalog``: a key may
        be conceptually allowed for corridor/intersection yet still be
        geometrically impossible once the active lane width, reserved passage and
        wall-clearance settings are applied.
        """
        entry = self._catalog_by_key.get(key)
        if entry is None:
            return False
        if map_type == "corridor":
            return map_layout_registry.structured_lane_footprint_fits(
                radius=float(entry.get("radius", 0.5)),
                lane_width=self.map_corridor_width,
                passage_width=self.map_corridor_passage_width,
                wall_clearance=self.map_wall_clearance,
                passage_safety_margin=self.map_passage_safety_margin,
            )
        if map_type == "intersection":
            return map_layout_registry.structured_lane_footprint_fits(
                radius=float(entry.get("radius", 0.5)),
                lane_width=self.map_intersection_width,
                passage_width=self.map_intersection_passage_width,
                wall_clearance=self.map_wall_clearance,
                passage_safety_margin=self.map_passage_safety_margin,
            )
        return True

    def _robot_collision_radius(self) -> float:
        """Conservative 2D radius for start/goal clearance checks."""
        return max(
            self.sr_d_front + self.sr_margin_front,
            self.sr_d_rear + self.sr_margin_rear,
            self.sr_d_left + self.sr_margin_left,
            self.sr_d_right + self.sr_margin_right,
        )

    def _pose_collides_with_placed(self, x: float, y: float, radius: float, placed: list) -> bool:
        """Return True if a circle at (x, y, radius) overlaps any placed item."""
        for px, py, pr in placed:
            if math.hypot(x - px, y - py) < radius + pr:
                return True
        return False

    def _is_heading_toward_near_wall(self, x: float, y: float, yaw: float, margin: float) -> bool:
        """
        Return True when the robot is within *margin* of an actual arena wall AND its
        heading points toward that wall.

        Uses self._arena_wall_lower / _arena_wall_upper (≈ ±12.5 m on the 25×25
        arena), derived from goal_obstacle bounds + obstacle_wall_margin. This is
        the arena wall CENTRELINE (the SDF wall pose), NOT the start-sampling box
        (self.lower/upper). The ~0.15 m wall half-thickness is a negligible
        over-reach for this heading-rejection heuristic.

        "Heading toward the wall" means the dot-product of the heading vector
        with the outward wall normal is positive (angle < 90° from outward normal).

        Only the wall(s) the robot is close to are evaluated — being near the
        right wall but facing left is not rejected.
        """
        cx, cy = math.cos(yaw), math.sin(yaw)
        lower, upper = self._arena_wall_lower, self._arena_wall_upper

        # right wall (x = upper): outward normal = (+1, 0)
        if x > upper - margin and cx > 0.0:
            return True
        # left wall (x = lower): outward normal = (-1, 0)
        if x < lower + margin and cx < 0.0:
            return True
        # top wall (y = upper): outward normal = (0, +1)
        if y > upper - margin and cy > 0.0:
            return True
        # bottom wall (y = lower): outward normal = (0, -1)
        if y < lower + margin and cy < 0.0:
            return True
        return False

    def _has_front_immediate_collision_risk(
        self,
        x: float,
        y: float,
        yaw: float,
        placed: list,
        front_clearance: float,
        fov_deg: float,
    ) -> bool:
        """
        Return True when the front cone [±fov_deg] within *front_clearance* metres
        contains an obstacle or an arena wall.

        placed: list of (px, py, pr) — obstacle circles already in the scene.
        The arena wall is checked via a simple ray cast in the heading direction.
        """
        fov_half = math.radians(fov_deg)
        # Use actual arena wall inner face, not the start-sampling box boundary.
        lower, upper = self._arena_wall_lower, self._arena_wall_upper
        cx, cy = math.cos(yaw), math.sin(yaw)

        # --- obstacle cone check ---
        for px, py, pr in placed:
            dx, dy = px - x, py - y
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                return True
            rel_angle = math.atan2(dy, dx) - yaw
            # wrap to [-pi, pi]
            rel_angle = (rel_angle + math.pi) % (2.0 * math.pi) - math.pi
            if dist < front_clearance + pr and abs(rel_angle) < fov_half:
                return True

        # --- wall ray cast in heading direction ---
        # Find distance to nearest arena boundary along heading vector.
        t_min = float("inf")
        if cx > 1e-9:
            t_min = min(t_min, (upper - x) / cx)
        elif cx < -1e-9:
            t_min = min(t_min, (lower - x) / cx)
        if cy > 1e-9:
            t_min = min(t_min, (upper - y) / cy)
        elif cy < -1e-9:
            t_min = min(t_min, (lower - y) / cy)
        if t_min < front_clearance:
            return True

        return False

    # ──────────────────────────────────────────────────────────────────
    #  Structured map curriculum — layout build / selection / sampling
    #  (docs/map_curriculum_plan.md §5–§7).  All extents derive from
    #  self.map_inner_* so "안 A" (world expansion) only changes config.
    # ──────────────────────────────────────────────────────────────────

    def _build_map_layouts(self) -> dict:
        """Precompute the wall boxes + sampling regions for each map_type.

        Delegates to the pure, ROS-free ``map_layout_registry.build_map_layouts``;
        only the node's ``map_*`` geometry scalars are forwarded so the layout DATA
        construction stays unit-testable and free of node state.

        Outer-wall correction: ``map_inner_*`` is the arena wall CENTRELINE (e.g.
        ±12.5 for the 25×25 world). The OUTER world walls are NOT part of any
        layout's internal-wall list, so ``_point_in_walls`` does not keep poses off
        them; the structured samplers only shrink a free region by the footprint
        radius. We therefore feed the registry the NAVIGABLE extent = wall inner
        FACE (centre − half the wall thickness) shrunk by ``map_wall_clearance`` —
        the same clearance internal walls already enforce — so every sampled
        footprint edge clears the outer wall instead of overestimating the gap by
        the wall's half-thickness. (The world outer walls share map_wall_thickness;
        see drl_arena.world.) ``map_inner_*`` itself stays the wall centreline so
        parking exclusion and the "25×25" arena size remain intuitive."""
        outer_margin = 0.5 * self.map_wall_thickness + self.map_wall_clearance
        return map_layout_registry.build_map_layouts(
            map_inner_lower=self.map_inner_lower + outer_margin,
            map_inner_upper=self.map_inner_upper - outer_margin,
            map_wall_thickness=self.map_wall_thickness,
            map_corridor_width=self.map_corridor_width,
            map_corridor_passage_width=self.map_corridor_passage_width,
            map_intersection_width=self.map_intersection_width,
            map_intersection_passage_width=self.map_intersection_passage_width,
            map_lobby_open_half_extent=self.map_lobby_open_half_extent,
            map_start_band_depth=self.map_start_band_depth,
        )

    def _build_static_pool_coverage(self) -> list:
        """Greedy coverage key list so EVERY (map_type, size_group) pair has
        >= coverage target activatable static entries pre-spawned (no per-episode
        create/remove).

        Targeting (map, size_group) — not just map — guarantees that a stage's
        size filter (e.g. allowed_static_groups: [large]) still has obstacles to
        activate on every map that allows that size; a per-map-only target would
        happily fill the pool with small/medium and leave large items unspawned.
        Keys allowed by the most still-deficient pairs are picked first to keep
        the pool compact.
        """
        avail = [k for k in self._catalog_by_key
                 if k not in STATIC_GLOBALLY_BANNED_KEYS]
        if not avail:
            return []
        size_of = {k: static_size_group(float(self._catalog_by_key[k].get("radius", 0.5)))
                   for k in avail}
        allowed = {
            m: {
                k for k in (MAP_TYPE_ALLOWED_STATIC_KEYS[m] & set(avail))
                if self._static_key_fits_map_geometry(m, k)
            }
            for m in MAP_TYPES
        }
        # (map, size_group) pairs that actually have candidates, with their target.
        pairs = {}
        for m in MAP_TYPES:
            for g in ("small", "medium", "large"):
                n_avail = sum(1 for k in allowed[m] if size_of[k] == g)
                if n_avail > 0:
                    pairs[(m, g)] = min(self.map_static_coverage_per_group, n_avail)
        counts = {p: 0 for p in pairs}
        chosen = []
        remaining = set(avail)

        def _deficient():
            return {p for p in pairs if counts[p] < pairs[p]}

        while _deficient():
            need = _deficient()
            best_key, best_gain = None, -1
            for k in sorted(remaining):
                g = size_of[k]
                gain = sum(1 for (m, gg) in need if gg == g and k in allowed[m])
                if gain > best_gain:
                    best_gain, best_key = gain, k
            if best_key is None or best_gain <= 0:
                break  # no remaining key helps a still-deficient pair
            chosen.append(best_key)
            remaining.discard(best_key)
            g = size_of[best_key]
            for m in MAP_TYPES:
                if (m, g) in counts and best_key in allowed[m]:
                    counts[(m, g)] += 1
        return sorted(chosen)

    def _ensure_parking_slots(self, needed: int):
        """Guarantee len(parking_slots) >= needed (+margin) and that every slot
        sits OUTSIDE the arena, regenerating a ring grid if the config list is
        too small (pool grows with group-union coverage → more slots needed)."""
        want = int(math.ceil(needed * 1.15)) + 2
        excl = max(abs(self.map_inner_lower), abs(self.map_inner_upper),
                   abs(self._arena_wall_lower), abs(self._arena_wall_upper)) + 1.0
        # Drop any configured slot that fell inside the (possibly enlarged) arena.
        self.parking_slots = [
            s for s in self.parking_slots
            if abs(s[0]) >= excl or abs(s[1]) >= excl
        ]
        if len(self.parking_slots) >= want:
            return
        # Ring extent scales with the arena: reach a few metres past the exclusion
        # band so the corner bands have enough slots, but stay on the ground plane
        # (drl_arena.world floor is 50x50 -> |xy| <= 25). Previously hard-coded to
        # ±17 (only valid for the old 19x19 arena).
        lim = round(min(24.0, excl + 4.5), 1)
        vals = list(np.arange(-lim, lim + 1e-6, 1.6))
        ring = [(float(x), float(y), self.parking_z)
                for x in vals for y in vals
                if abs(x) >= excl or abs(y) >= excl]
        # Keep config slots first, then fill from the ring (dedup by rounded xy).
        seen = {(round(s[0], 2), round(s[1], 2)) for s in self.parking_slots}
        for s in ring:
            key = (round(s[0], 2), round(s[1], 2))
            if key not in seen:
                self.parking_slots.append(s)
                seen.add(key)
            if len(self.parking_slots) >= want:
                break
        if not self.parking_slots:
            self.parking_slots = [(16.0, 16.0, self.parking_z)]

    def _choose_map_type(self) -> str:
        """Pick this episode's map type.

        Evaluation (trainer raised curriculum_eval_mode, OR a pure test node) with
        eval_map_types set: deterministic round-robin so every eval map is covered
        evenly — independent of the training map distribution. Otherwise: sample
        from allowed_map_types using map_type_probs (uniform if lengths mismatch).
        """
        if (self._curriculum_eval_mode or not self.train_mode) and self.eval_map_types:
            pool = [m for m in self.eval_map_types if m in self._map_layouts] \
                   or self.allowed_map_types
            mt = pool[self._eval_map_cursor % len(pool)]
            self._eval_map_cursor += 1
            return mt
        pool = self.allowed_map_types or ["clutter"]
        probs = self.map_type_probs
        if probs and len(probs) == len(pool) and sum(probs) > 0:
            p = np.asarray(probs, dtype=float)
            p = p / p.sum()
            return str(np.random.choice(pool, p=p))
        return str(np.random.choice(pool))

    def _select_episode_layout(self):
        """Choose map_type, activate its walls, publish the current_map_type
        parameter. No-op (clears state) when the curriculum is disabled."""
        if not self.map_layout_enabled or not self._map_layouts:
            self.current_map_type = ""
            self.current_layout_spec = None
            return
        # Pick up the trainer's eval-mode flag; restart the round-robin whenever
        # it toggles so each evaluation cycles eval_map_types from the start.
        try:
            ev = bool(self.get_parameter("curriculum_eval_mode").value)
        except Exception:
            ev = False
        if ev != self._curriculum_eval_mode:
            self._curriculum_eval_mode = ev
            self._eval_map_cursor = 0
        mt = self._choose_map_type()
        if mt not in self._map_layouts:
            mt = self.allowed_map_types[0] if self.allowed_map_types else "clutter"
        self.current_map_type = mt
        self.current_layout_spec = self._map_layouts[mt]
        # Derive the pedestrian distribution from the map structure so humans
        # spawn on the navigable lanes. Re-derived every reset (the curriculum
        # restores human_placement_mode from base first), so no cross-stage leak.
        self.human_placement_mode = {
            "lobby":        "lobby_crossings",
            "corridor":     "corridor_lanes",
            "intersection": "intersection_arms",
            "clutter":      "global_random",
        }.get(mt, self.human_placement_mode)
        self._activate_layout_walls(mt)
        try:
            self.set_parameters([
                Parameter("current_map_type", Parameter.Type.STRING, mt)
            ])
        except Exception as e:
            self.get_logger().warn(f"[MapCurriculum] could not set current_map_type: {e}")

    def _activate_layout_walls(self, map_type: str):
        """Move the chosen map's wall boxes into place; park all others deep
        underground (out of LiDAR + collision range). Pure set_pose — no spawn."""
        if not self.pool_walls:
            return
        active = set(self._map_layouts.get(map_type, {}).get("wall_names", []))
        z_active = self.map_wall_height / 2.0
        for w in self.pool_walls:
            name = w["name"]
            if name in active:
                spec = w["spec"]
                self.set_entity_pose_ignition(
                    name, float(spec["cx"]), float(spec["cy"]), z_active,
                    0.0, 0.0, 0.0, 1.0)
            else:
                # Park underground, spread by index so boxes never overlap.
                self.set_entity_pose_ignition(
                    name, float(w["park_x"]), float(w["park_y"]), float(w["park_z"]),
                    0.0, 0.0, 0.0, 1.0)

    def _point_in_walls(self, x: float, y: float, clearance: float) -> bool:
        """True if (x, y) lies within any active internal wall box + clearance."""
        spec = self.current_layout_spec
        if not spec:
            return False
        for w in spec["walls"]:
            if (abs(x - w["cx"]) <= w["sx"] / 2.0 + clearance
                    and abs(y - w["cy"]) <= w["sy"] / 2.0 + clearance):
                return True
        return False

    def _front_hits_internal_wall(self, x: float, y: float, yaw: float,
                                  distance: float, steps: int = 6) -> bool:
        """Ray-march the heading direction up to `distance` and report whether it
        passes through any active internal wall box. Used by start-pose sampling
        so the robot never starts pointing straight at a corridor/intersection
        wall (which the obstacle/outer-wall front check does not cover)."""
        if self.current_layout_spec is None or not self.current_layout_spec["walls"]:
            return False
        cx, cy = math.cos(yaw), math.sin(yaw)
        for i in range(1, steps + 1):
            d = distance * i / steps
            if self._point_in_walls(x + cx * d, y + cy * d, 0.0):
                return True
        return False

    def _in_open_area(self, x: float, y: float, margin: float = 0.0) -> bool:
        """True if (x, y) is within the lobby central open area inflated by
        `margin` (large-obstacle keep-out). The margin lets a long obstacle's
        whole footprint — not just its centre — stay out of the open space.
        False when the current layout has no open area."""
        spec = self.current_layout_spec
        oa = spec.get("open_area") if spec else None
        if not oa:
            return False
        x_lo, x_hi, y_lo, y_hi = oa
        return (x_lo - margin <= x <= x_hi + margin
                and y_lo - margin <= y <= y_hi + margin)

    # ── Reserved-passage helpers (corridor / intersection lanes) ───────────
    def _circle_intersects_passage(self, x: float, y: float, radius: float,
                                   passage: dict) -> bool:
        """True if a circle (x, y, radius) overlaps one reserved passage strip.
        A passage is an infinite strip about its centre line on the cross axis;
        `radius` should already include any safety margin."""
        hw = passage["half_width"]
        if passage["axis"] == "x":            # strip runs along x; cross axis = y
            return abs(y - passage["y_center"]) < hw + radius
        return abs(x - passage["x_center"]) < hw + radius   # runs along y

    def _point_in_reserved_passage(self, x: float, y: float) -> bool:
        """True if (x, y) lies inside any reserved passage of the active layout."""
        spec = self.current_layout_spec
        for p in (spec.get("reserved_passages", []) if spec else []):
            if self._circle_intersects_passage(x, y, 0.0, p):
                return True
        return False

    def _pose_blocks_reserved_passage(self, x: float, y: float, radius: float,
                                      safety_margin: float = None) -> bool:
        """True if a footprint (radius + safety margin) intrudes any reserved
        passage. No-op for layouts without reserved passages (lobby / clutter /
        non-structured). safety_margin defaults to map_passage_safety_margin
        (static obstacles); humans pass 0.0 — they are dynamic and may step onto
        the lane later, so only their body must clear it at spawn."""
        spec = self.current_layout_spec
        passages = spec.get("reserved_passages", []) if spec else []
        if not passages:
            return False
        sm = self.map_passage_safety_margin if safety_margin is None else safety_margin
        rr = radius + sm
        for p in passages:
            if self._circle_intersects_passage(x, y, rr, p):
                return True
        return False

    def _sample_lane_aligned_yaw(self, region: dict, x: float, y: float) -> float:
        """Spawn yaw aligned with the lane the start region belongs to.

        Nominal heading points along the region axis toward the map centre; an
        offset is then added that ONLY ever rotates the heading back toward the
        lane centre line (never into the near wall). The offset grows with the
        lateral distance from the centre line (center bias) plus a small random
        jitter, so the robot at a wall starts angled gently inward."""
        axis = region["axis"]
        d = float(region["dir"])
        if axis == "x":
            fx, fy, c = d, 0.0, y          # cross coordinate = y
        else:
            fx, fy, c = 0.0, d, x          # cross coordinate = x
        nominal = math.atan2(fy, fx)
        half = max(float(region.get("lane_half", 1.0)), 1e-6)
        s = 0.0 if abs(c) < 1e-6 else math.copysign(1.0, c)
        # Cross-axis unit vector pointing toward the centre line.
        crx, cry = (0.0, -s) if axis == "x" else (-s, 0.0)
        # Sign of the rotation (from forward toward centre) via 2D cross product.
        rot_sign = fx * cry - fy * crx
        u = min(1.0, abs(c) / half)                       # 0 at centre, 1 at wall
        offset = self.map_spawn_yaw_center_bias * u + \
            float(np.random.uniform(0.0, self.map_spawn_yaw_jitter))
        yaw = nominal + rot_sign * offset
        return geom.wrap_to_pi(yaw)

    def _sample_xy_in_regions(self, regions, radius: float, rng=None):
        """Uniformly sample (x, y) in an area-weighted random region, shrinking
        each region by `radius` so the footprint stays inside it.

        ``rng`` selects the RNG: None (default) → global np.random (goal / robot
        start / static-obstacle callers, unchanged); a HUMAN caller threads in
        ``self._human_np_rng`` so pedestrian spawn stays off the global stream."""
        r = rng if rng is not None else np.random
        usable = []
        weights = []
        for (x_lo, x_hi, y_lo, y_hi) in regions:
            ax_lo, ax_hi = x_lo + radius, x_hi - radius
            ay_lo, ay_hi = y_lo + radius, y_hi - radius
            if ax_hi <= ax_lo or ay_hi <= ay_lo:
                continue
            usable.append((ax_lo, ax_hi, ay_lo, ay_hi))
            weights.append((ax_hi - ax_lo) * (ay_hi - ay_lo))
        if not usable:
            return None
        w = np.asarray(weights, dtype=float)
        idx = int(r.choice(len(usable), p=w / w.sum()))
        ax_lo, ax_hi, ay_lo, ay_hi = usable[idx]
        return float(r.uniform(ax_lo, ax_hi)), float(r.uniform(ay_lo, ay_hi))

    def _segment_hits_wall(self, x0: float, y0: float, x1: float, y1: float,
                           clearance: float, samples: int = 12) -> bool:
        """Lane-consistency heuristic: True if the straight start->goal segment
        passes through any active internal wall (sampled). Used to PREFER goals
        reachable along the map structure (corridor stays straight; clutter goals
        directly blocked by a wall are avoided). Endpoints are excluded."""
        spec = self.current_layout_spec
        if not spec or not spec["walls"]:
            return False
        for i in range(1, samples):
            t = i / float(samples)
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            if self._point_in_walls(x, y, clearance):
                return True
        return False

    def _layout_region_grid(self, regions, margin: float, step: float = 1.2):
        """Deterministic candidate points covering the free regions (shrunk by
        `margin`), used as the structured-map goal fallback so we never resort to
        legacy whole-arena coordinates."""
        pts = []
        for (x_lo, x_hi, y_lo, y_hi) in regions:
            ax_lo, ax_hi = x_lo + margin, x_hi - margin
            ay_lo, ay_hi = y_lo + margin, y_hi - margin
            if ax_hi <= ax_lo or ay_hi <= ay_lo:
                continue
            nx = max(2, int((ax_hi - ax_lo) / step) + 1)
            ny = max(2, int((ay_hi - ay_lo) / step) + 1)
            for gx in np.linspace(ax_lo, ax_hi, nx):
                for gy in np.linspace(ay_lo, ay_hi, ny):
                    pts.append((float(gx), float(gy)))
        return pts
