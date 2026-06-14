"""Static-obstacle + wall pool initialisation, per-episode activation and spawn.\n\nExtracted from environment.py. Builds the obstacle/wall pools from the catalog, selects per-episode obstacles honouring the map-type/size policy (map_catalog), and places them via the Gazebo entity helpers."""

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


class ObstacleMixin:
    def _initialize_obstacle_pool(self):
        """Spawn all pool entities once on ground-level parking slots.

        Called on the first reset when use_obstacle_pool=True.  After this call
        obstacles are only repositioned via set_pose; no further create/remove
        calls are needed during normal training.

        Model types are distributed evenly across slots by cycling through the
        shuffled catalog, ensuring good type variety when the pool is larger than
        the catalog. Pool entities start on the floor plane, outside the arena
        walls, and are later either activated in the arena or returned to
        parking slots.
        """
        # Each human slot parks all 6 part models at one shared parking position.
        total_pool_slots = (
            self.obstacle_pool_static_size
            + self.obstacle_pool_human_size
        )
        if len(self.parking_slots) < total_pool_slots:
            self.get_logger().warn(
                f"Only {len(self.parking_slots)} parking slots for {total_pool_slots} pool entities; "
                "some parked obstacles will reuse slots."
            )

        init_slots = list(self.parking_slots)
        random.shuffle(init_slots)
        # Use a one-element list so nested functions share a single mutable counter.
        slot_index = [0]

        def _build_pool(catalog, pool_size, name_prefix, label, key_list=None):
            pool = []
            if pool_size <= 0:
                return pool
            # Structured map curriculum: spawn an explicit coverage key list so
            # every map_type has enough activatable entries (group-aware). The
            # model identity of each slot is fixed here (startup only); per
            # episode we just teleport a map/size-filtered SUBSET — never
            # create/remove. Otherwise: cycle the shuffled catalog as before.
            if key_list:
                build_entries = [self._catalog_by_key[k] for k in key_list
                                 if k in self._catalog_by_key]
            else:
                if not catalog:
                    return pool
                build_entries = list(catalog)
                random.shuffle(build_entries)
                build_entries = [build_entries[i % len(build_entries)]
                                 for i in range(pool_size)]
            for i, entry in enumerate(build_entries):
                model_name = f"{name_prefix}_{i:03d}"
                park_x, park_y, park_z = init_slots[slot_index[0] % len(init_slots)]
                slot_index[0] += 1
                ok = self._spawn_entity_sdf(
                    model_name, entry["uri"],
                    park_x, park_y,
                    yaw=0.0, z=park_z,
                )
                if ok:
                    radius = float(entry.get("radius", 0.5))
                    pool.append({
                        "name":       model_name,
                        "key":        entry.get("key", ""),
                        "size_group": static_size_group(radius),
                        "radius":     radius,
                        "yaw_random": bool(entry.get("yaw_random", True)),
                        "park_x":     park_x,
                        "park_y":     park_y,
                        "park_z":     park_z,
                    })
                else:
                    self.get_logger().warn(
                        f"Pool init: could not spawn {model_name} — slot skipped"
                    )
            self.get_logger().info(
                f"Pool init: {len(pool)}/{pool_size} {label} slots ready"
            )
            return pool

        def _build_human_pool(catalog, pool_size, name_prefix, label):
            """Spawn 8 part models per human slot (v_torso, v_ll, v_rl, v_la, v_ra, p_torso, p_ll, p_rl)."""
            pool = []
            if not catalog or pool_size <= 0:
                return pool
            shuffled_cat = list(catalog)
            random.shuffle(shuffled_cat)
            for i in range(pool_size):
                entry = shuffled_cat[i % len(shuffled_cat)]
                # All 8 parts share a single parking slot
                park_x, park_y, park_z = init_slots[slot_index[0] % len(init_slots)]
                slot_index[0] += 1

                arm_uri = entry.get("visual_arm_uri", entry["visual_torso_uri"])
                part_defs = [
                    ("v_torso", entry["visual_torso_uri"]),
                    ("v_ll",    entry["visual_leg_uri"]),
                    ("v_rl",    entry["visual_leg_uri"]),
                    ("v_la",    arm_uri),
                    ("v_ra",    arm_uri),
                    ("p_torso", entry["proxy_torso_uri"]),
                    ("p_ll",    entry["proxy_leg_uri"]),
                    ("p_rl",    entry["proxy_leg_uri"]),
                ]
                names = {}
                all_ok = True
                for part_key, uri in part_defs:
                    part_name = f"{name_prefix}_{i:03d}_{part_key}"
                    ok = self._spawn_entity_sdf(part_name, uri, park_x, park_y, yaw=0.0, z=park_z)
                    if ok:
                        names[part_key] = part_name
                    else:
                        all_ok = False
                        break

                if all_ok:
                    pool.append({
                        "v_torso": names["v_torso"],
                        "v_ll":    names["v_ll"],
                        "v_rl":    names["v_rl"],
                        "v_la":    names["v_la"],
                        "v_ra":    names["v_ra"],
                        "p_torso": names["p_torso"],
                        "p_ll":    names["p_ll"],
                        "p_rl":    names["p_rl"],
                        "radius":     float(entry.get("radius", 0.30)),
                        "yaw_random": bool(entry.get("yaw_random", True)),
                        "speed_min":  float(entry.get("speed_min", 0.3)),
                        "speed_max":  float(entry.get("speed_max", 0.8)),
                        "park_x": park_x, "park_y": park_y, "park_z": park_z,
                        "torso_z":          float(entry.get("torso_z", 1.25)),
                        "leg_length":       float(entry.get("leg_length", 0.90)),
                        "hip_z":            float(entry.get("hip_z", 0.90)),
                        "leg_y_offset":     float(entry.get("leg_y_offset", 0.13)),
                        "leg_swing_amp_deg": float(entry.get("leg_swing_amp_deg", 18.0)),
                        "arm_length":       float(entry.get("arm_length", 0.60)),
                        "shoulder_z":       float(entry.get("shoulder_z", 1.55)),
                        "arm_y_offset":     float(entry.get("arm_y_offset", 0.22)),
                        "arm_swing_amp_deg": float(entry.get("arm_swing_amp_deg", 20.0)),
                        "gait_freq_hz_min": float(entry.get("gait_freq_hz_min", 0.8)),
                        "gait_freq_hz_max": float(entry.get("gait_freq_hz_max", 1.6)),
                    })
                else:
                    for n in names.values():
                        try:
                            req = DeleteEntity.Request()
                            req.entity.name = n
                            req.entity.type = GzEntity.MODEL
                            self.delete_entity_client.call_async(req)
                        except Exception:
                            pass
                    self.get_logger().warn(
                        f"Pool init: human slot {i} ({name_prefix}) partially failed; cleaned up"
                    )
            self.get_logger().info(
                f"Pool init: {len(pool)}/{pool_size} {label} (8-part) slots ready"
            )
            return pool

        self.pool_static = _build_pool(
            self.static_obstacle_catalog,
            self.obstacle_pool_static_size,
            "rl_sta_pool", "static",
            key_list=(self._static_pool_coverage if self.map_layout_enabled else None),
        )
        self.pool_human = _build_human_pool(
            self.human_catalog,
            self.obstacle_pool_human_size,
            "rl_human_pool", "human",
        )
        # Structured map curriculum: spawn every map's internal wall boxes once,
        # parked deep underground; per episode only the active map's walls are
        # teleported into place (no create/remove → reset wall-clock unaffected).
        if self.map_layout_enabled and self._map_layouts:
            self._initialize_wall_pool()
        self.pool_initialized = True

    def _initialize_wall_pool(self):
        """Spawn all map-layout wall boxes once and park them underground."""
        self.pool_walls = []
        park_idx = 0
        for mt in MAP_TYPES:
            spec = self._map_layouts.get(mt)
            if not spec:
                continue
            for name, wspec in zip(spec["wall_names"], spec["walls"]):
                park_x = 0.0
                park_y = 0.0
                park_z = -20.0 - float(park_idx)   # spread so boxes never overlap
                park_idx += 1
                ok = self._spawn_box_entity(
                    name, float(wspec["sx"]), float(wspec["sy"]),
                    float(self.map_wall_height),
                    park_x, park_y, park_z,
                )
                if ok:
                    self.pool_walls.append({
                        "name": name, "spec": wspec,
                        "park_x": park_x, "park_y": park_y, "park_z": park_z,
                    })
        self.get_logger().info(
            f"[MapCurriculum] wall pool ready: {len(self.pool_walls)} boxes "
            f"across {len([m for m in MAP_TYPES if self._map_layouts.get(m, {}).get('walls')])} structured maps"
        )

    def _activate_random_obstacles(self, start_x: float, start_y: float):
        """Teleport pool entities each reset: active subset → arena, rest → parking.

        A random shuffle of pool slots determines which models appear each episode,
        providing type and position diversity without any create/remove service calls.
        Inactive entities are returned to randomized ground-level parking slots
        outside the arena walls.
        Updates spawned_obstacle_records so change_goal / _sample_train_start_pose
        see the correct occupied cells on the *next* reset.
        """
        new_records: dict = {}
        placed: list = []  # (x, y, radius) of entities placed in the arena this episode
        parking_slots = list(self.parking_slots)
        random.shuffle(parking_slots)
        parking_index = 0
        self.human_states = {}  # clear previous episode's pedestrian state

        def _next_parking_slot():
            nonlocal parking_index
            slot = parking_slots[parking_index % len(parking_slots)]
            parking_index += 1
            return slot

        def _place_pool_group(pool, count, label):
            if not pool:
                return
            layout = self.current_layout_spec
            # Structured map curriculum: restrict the activatable subset to this
            # map_type's allowed keys AND (if the stage sets it) the allowed size
            # groups. Filtered-out entries are parked so none linger in the arena.
            if layout is not None:
                allowed_keys = MAP_TYPE_ALLOWED_STATIC_KEYS.get(self.current_map_type, frozenset())
                groups = set(self.allowed_static_groups) if self.allowed_static_groups else None

                def _candidate(e):
                    if e["key"] in STATIC_GLOBALLY_BANNED_KEYS:
                        return False
                    if e["key"] not in allowed_keys:
                        return False
                    if groups is not None and e["size_group"] not in groups:
                        return False
                    return True

                candidates = [e for e in pool if _candidate(e)]
                others     = [e for e in pool if not _candidate(e)]
            else:
                candidates = list(pool)
                others     = []
            random.shuffle(candidates)
            avoid_open = (self.current_map_type == "lobby")
            activated = 0
            for entry in candidates:
                if activated < count:
                    if layout is not None:
                        pose = self._sample_free_pose_layout(
                            entry["radius"], placed, start_x, start_y,
                            avoid_open_area=(avoid_open and entry["size_group"] == "large"),
                        )
                    else:
                        pose = self._sample_free_pose(
                            entry["radius"], placed, start_x, start_y
                        )
                    if pose is not None:
                        x, y = pose
                        yaw = (
                            np.random.uniform(-math.pi, math.pi)
                            if entry["yaw_random"]
                            else 0.0
                        )
                        q = Quaternion.from_euler(0.0, 0.0, yaw)
                        self.set_entity_pose_ignition(
                            entry["name"], x, y, 0.0,
                            q.x, q.y, q.z, q.w,
                        )
                        placed.append((x, y, entry["radius"]))
                        new_records[entry["name"]] = (x, y, entry["radius"])
                        activated += 1
                        continue
                    self.get_logger().warn(
                        f"Pool: no free pose for {label} — parking {entry['name']}"
                    )
                # Return this slot to a randomized parking location outside the walls.
                px, py, pz = _next_parking_slot()
                self.set_entity_pose_ignition(
                    entry["name"],
                    px, py, pz,
                    0.0, 0.0, 0.0, 1.0,
                )
            # Park every filtered-out entry (wrong map_type / size group).
            for entry in others:
                px, py, pz = _next_parking_slot()
                self.set_entity_pose_ignition(
                    entry["name"], px, py, pz, 0.0, 0.0, 0.0, 1.0,
                )

        def _place_human_pool_group(pool, count):
            """Place 6-part pedestrian models; initialise human_states with gait params."""
            if not pool or count <= 0:
                return
            spawn_regions = self._build_human_spawn_regions()
            shuffled = list(pool)
            random.shuffle(shuffled)
            activated = 0
            for entry in shuffled:
                all_names = [entry["v_torso"], entry["v_ll"], entry["v_rl"],
                             entry["v_la"],    entry["v_ra"],
                             entry["p_torso"], entry["p_ll"],  entry["p_rl"]]
                if activated < count:
                    pose = self._sample_human_spawn_pose(
                        entry["radius"],
                        placed,
                        start_x,
                        start_y,
                        activated,
                        spawn_regions,
                    )
                    if pose is not None:
                        x, y = pose
                        yaw = (
                            np.random.uniform(-math.pi, math.pi)
                            if entry["yaw_random"]
                            else 0.0
                        )
                        mode_fields = self._assign_human_mode(x, y)
                        tx, ty = self._sample_human_waypoint(x, y, state=mode_fields)
                        speed = np.random.uniform(entry["speed_min"], entry["speed_max"]) * mode_fields["mode_speed_scale"]
                        name_p_torso = entry["p_torso"]
                        heading_jitter = np.random.uniform(
                            -self.human_heading_jitter, self.human_heading_jitter
                        )
                        desired_yaw = math.atan2(ty - y, tx - x) + heading_jitter
                        desired_yaw = (desired_yaw + math.pi) % (2.0 * math.pi) - math.pi
                        state = {
                            "visual_torso":      entry["v_torso"],
                            "visual_left_leg":   entry["v_ll"],
                            "visual_right_leg":  entry["v_rl"],
                            "visual_left_arm":   entry["v_la"],
                            "visual_right_arm":  entry["v_ra"],
                            "proxy_torso":       name_p_torso,
                            "proxy_left_leg":    entry["p_ll"],
                            "proxy_right_leg":   entry["p_rl"],
                            "x": x, "y": y, "yaw": yaw,
                            "radius":            entry["radius"],
                            "speed":             speed,
                            "v": 0.0, "w": 0.0,
                            "target_x": tx, "target_y": ty,
                            "heading_jitter":    heading_jitter if self.human_heading_jitter_on_retarget_only else 0.0,
                            "desired_yaw":       desired_yaw,
                            "retarget_cooldown": self.human_min_retarget_interval,
                            "segment_elapsed":   0.0,
                            "pause_left":        mode_fields["mode_init_wait"],
                            "stopping": False,
                            **mode_fields,   # per-human behaviour mode (kept for the episode)
                            "gait_phase":        np.random.uniform(0.0, 2.0 * math.pi),
                            "gait_freq_hz":      np.random.uniform(
                                                     entry["gait_freq_hz_min"],
                                                     entry["gait_freq_hz_max"]),
                            "leg_swing_amp_rad": math.radians(entry["leg_swing_amp_deg"]),
                            "leg_length":        entry["leg_length"],
                            "leg_y_offset":      entry["leg_y_offset"],
                            "hip_z":             entry["hip_z"],
                            "torso_z":           entry["torso_z"],
                            "arm_swing_amp_rad": math.radians(entry["arm_swing_amp_deg"]),
                            "arm_length":        entry["arm_length"],
                            "shoulder_z":        entry["shoulder_z"],
                            "arm_y_offset":      entry["arm_y_offset"],
                        }
                        self.human_states[name_p_torso] = state
                        self._set_human_part_poses(state)
                        placed.append((x, y, entry["radius"]))
                        new_records[name_p_torso] = (x, y, entry["radius"])
                        activated += 1
                        continue
                    self.get_logger().warn(
                        f"Pool: no free pose for human — parking all 8 parts"
                    )
                # Park all 8 parts at the stored parking slot
                px, py, pz = entry["park_x"], entry["park_y"], entry["park_z"]
                for n in all_names:
                    self.set_entity_pose_ignition(n, px, py, pz, 0.0, 0.0, 0.0, 1.0)

        _place_pool_group(self.pool_static,  self.num_of_static_obstacles,  "static")
        _place_human_pool_group(self.pool_human, self.num_of_humans)

        self.spawned_obstacle_records = new_records
        self.spawned_obstacle_names   = list(new_records.keys())

    def _spawn_random_obstacles(self, start_x: float, start_y: float):
        """Delete previous episode's obstacles, then spawn num_of_obstacles new ones.

        Each model is named with the current episode counter so that a timed-out
        delete from the previous episode can never block a new spawn.
        """
        self._delete_spawned_obstacles()
        if (
            (not self.static_obstacle_catalog  or self.num_of_static_obstacles  <= 0)
            and (not self.human_catalog or self.num_of_humans <= 0)
        ):
            return
        ep = self._episode_count
        placed = list(self.spawned_obstacle_records.values())

        def _spawn_from_catalog(entries, count, name_prefix, log_label):
            spawned = 0
            for i in range(count):
                entry = random.choice(entries)
                radius = float(entry.get("radius", 0.5))
                result = self._sample_free_pose(radius, placed, start_x, start_y)
                if result is None:
                    self.get_logger().warn(f"Could not place {log_label} {i + 1} — skipping")
                    continue
                x, y = result
                yaw = np.random.uniform(-math.pi, math.pi) if entry.get("yaw_random", True) else 0.0
                model_name = f"{name_prefix}_{ep:04d}_{i + 1:03d}"
                if self._spawn_entity_sdf(model_name, entry["uri"], x, y, yaw):
                    self.spawned_obstacle_names.append(model_name)
                    self.spawned_obstacle_records[model_name] = (x, y, radius)
                    placed.append((x, y, radius))
                    spawned += 1
            return spawned

        if self.static_obstacle_catalog and self.num_of_static_obstacles > 0:
            _spawn_from_catalog(self.static_obstacle_catalog, self.num_of_static_obstacles, "rl_sta", "static obstacle")

        if self.human_catalog and self.num_of_humans > 0:
            self.human_states = {}
            spawn_regions = self._build_human_spawn_regions()
            for i in range(self.num_of_humans):
                entry = random.choice(self.human_catalog)
                radius = float(entry.get("radius", 0.30))
                result = self._sample_human_spawn_pose(
                    radius, placed, start_x, start_y, i, spawn_regions
                )
                if result is None:
                    self.get_logger().warn(f"Could not place human {i + 1} — skipping")
                    continue
                x, y = result
                yaw = np.random.uniform(-math.pi, math.pi) if entry.get("yaw_random", True) else 0.0
                prefix = f"rl_human_{ep:04d}_{i + 1:03d}"
                arm_uri = entry.get("visual_arm_uri", entry["visual_torso_uri"])
                part_defs = [
                    ("v_torso", entry["visual_torso_uri"]),
                    ("v_ll",    entry["visual_leg_uri"]),
                    ("v_rl",    entry["visual_leg_uri"]),
                    ("v_la",    arm_uri),
                    ("v_ra",    arm_uri),
                    ("p_torso", entry["proxy_torso_uri"]),
                    ("p_ll",    entry["proxy_leg_uri"]),
                    ("p_rl",    entry["proxy_leg_uri"]),
                ]
                names = {}
                spawned_so_far = []
                all_ok = True
                for part_key, uri in part_defs:
                    pname = f"{prefix}_{part_key}"
                    if self._spawn_entity_sdf(pname, uri, x, y, yaw):
                        names[part_key] = pname
                        spawned_so_far.append(pname)
                    else:
                        all_ok = False
                        break

                if all_ok:
                    mode_fields = self._assign_human_mode(x, y)
                    tx, ty = self._sample_human_waypoint(x, y, state=mode_fields)
                    speed = np.random.uniform(
                        float(entry.get("speed_min", 0.3)),
                        float(entry.get("speed_max", 0.8)),
                    ) * mode_fields["mode_speed_scale"]
                    heading_jitter = np.random.uniform(
                        -self.human_heading_jitter, self.human_heading_jitter
                    )
                    desired_yaw = math.atan2(ty - y, tx - x) + heading_jitter
                    desired_yaw = (desired_yaw + math.pi) % (2.0 * math.pi) - math.pi
                    state = {
                        "visual_torso":      names["v_torso"],
                        "visual_left_leg":   names["v_ll"],
                        "visual_right_leg":  names["v_rl"],
                        "visual_left_arm":   names["v_la"],
                        "visual_right_arm":  names["v_ra"],
                        "proxy_torso":       names["p_torso"],
                        "proxy_left_leg":    names["p_ll"],
                        "proxy_right_leg":   names["p_rl"],
                        "x": x, "y": y, "yaw": yaw,
                        "radius":            radius,
                        "speed":             speed,
                        "v": 0.0, "w": 0.0,
                        "target_x": tx, "target_y": ty,
                        "heading_jitter":    heading_jitter if self.human_heading_jitter_on_retarget_only else 0.0,
                        "desired_yaw":       desired_yaw,
                        "retarget_cooldown": self.human_min_retarget_interval,
                        "segment_elapsed":   0.0,
                        "pause_left":        mode_fields["mode_init_wait"],
                        "stopping": False,
                        **mode_fields,   # per-human behaviour mode (kept for the episode)
                        "gait_phase":        np.random.uniform(0.0, 2.0 * math.pi),
                        "gait_freq_hz":      np.random.uniform(
                                                 float(entry.get("gait_freq_hz_min", 0.8)),
                                                 float(entry.get("gait_freq_hz_max", 1.6))),
                        "leg_swing_amp_rad": math.radians(float(entry.get("leg_swing_amp_deg", 18.0))),
                        "leg_length":        float(entry.get("leg_length", 0.90)),
                        "leg_y_offset":      float(entry.get("leg_y_offset", 0.13)),
                        "hip_z":             float(entry.get("hip_z", 0.90)),
                        "torso_z":           float(entry.get("torso_z", 1.25)),
                        "arm_swing_amp_rad": math.radians(float(entry.get("arm_swing_amp_deg", 20.0))),
                        "arm_length":        float(entry.get("arm_length", 0.60)),
                        "shoulder_z":        float(entry.get("shoulder_z", 1.55)),
                        "arm_y_offset":      float(entry.get("arm_y_offset", 0.22)),
                    }
                    self.human_states[names["p_torso"]] = state
                    self._set_human_part_poses(state)
                    self.spawned_obstacle_names.extend(spawned_so_far)
                    self.spawned_obstacle_records[names["p_torso"]] = (x, y, radius)
                    placed.append((x, y, radius))
                else:
                    for n in spawned_so_far:
                        try:
                            req = DeleteEntity.Request()
                            req.entity.name = n
                            req.entity.type = GzEntity.MODEL
                            self.delete_entity_client.call_async(req)
                        except Exception:
                            pass
                    self.get_logger().warn(
                        f"Human {i + 1}: partial spawn failure; {len(spawned_so_far)}/8 parts cleaned up"
                    )

