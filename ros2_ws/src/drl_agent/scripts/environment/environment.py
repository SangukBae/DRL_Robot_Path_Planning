#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import hashlib   # AUX_ABLATION: env config content hash for run manifests
import threading
import random
import time
import csv
from datetime import datetime
import numpy as np
from collections import deque
from squaternion import Quaternion

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import Twist, Pose, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2, JointState
from visualization_msgs.msg import Marker, MarkerArray

from drl_agent_interfaces.srv import Step, Reset, Seed, GetDimensions, SampleActionSpace
from drl_agent_interfaces.msg import DrlModelPoseArray

import point_cloud2 as pc2
from file_manager import load_yaml
import pure_pursuit
# AUX_PRED: privileged future-risk label generation (training-only).
import aux_prediction_labels as aux_labels
from sensor_msgs.msg import LaserScan

from ros_gz_interfaces.msg import Contacts
from ros_gz_interfaces.msg import Entity as GzEntity
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose, SpawnEntity, DeleteEntity


# ──────────────────────────────────────────────────────────────────────────────
# Structured map curriculum — static-obstacle policy
# (source of truth: docs/map_curriculum_plan.md §4)
#
# Two-stage filter, applied per episode:
#   1. globally banned keys are never spawned/activated anywhere
#   2. per-map-type allowed keys decide what a given map_type may activate
# The stage's `allowed_static_groups` is an ORTHOGONAL size filter (small/medium/
# large) that further narrows the per-episode candidates.
# ──────────────────────────────────────────────────────────────────────────────

# §4.3 — sensor-unfriendly / shape-mismatched obstacles excluded everywhere.
STATIC_GLOBALLY_BANNED_KEYS = frozenset({
    "house_cooking_bench",   # ~0.10 m tall → disappears under the LiDAR height filter
})

# §4.2 corridor — small/medium that do not block a single travel lane.
_CORRIDOR_ALLOWED_KEYS = frozenset({
    "warehouse_bucket", "warehouse_trash_can", "hospital_chair", "bookstore_chair",
    "house_chair_a", "bookstore_column_a", "hospital_bedside_table", "hospital_drawer",
    "hospital_instrument_cart1", "hospital_mop_cart", "hospital_surgical_trolley",
    "warehouse_cluttering_c", "warehouse_cluttering_d", "house_kitchen_table",
    "house_refrigerator",
})

# §4.2 intersection — corridor set + a few larger items (edge-anchor in spirit;
# v1 places them in the free regions, large-first, so they settle on arm edges).
_INTERSECTION_ALLOWED_KEYS = _CORRIDOR_ALLOWED_KEYS | frozenset({
    "hospital_metal_cabinet", "hospital_vending_machine", "hospital_parking_trolley_max",
    "hospital_wheelchair", "warehouse_cluttering_a", "house_fitness_equipment",
    "bookstore_desk_a", "bookstore_shelf_a", "bookstore_shelf_b", "bookstore_shelf_c",
    "house_kitchen_cabinet", "hospital_xray_machine", "warehouse_shelf", "warehouse_shelf_e",
})

# §4.2 clutter — complexity is the point; everything except desk_b and the banned.
_CLUTTER_ALLOWED_KEYS = frozenset({
    "warehouse_bucket", "warehouse_cluttering_a", "warehouse_cluttering_c",
    "warehouse_cluttering_d", "warehouse_trash_can", "warehouse_shelf", "warehouse_shelf_e",
    "hospital_instrument_cart1", "hospital_drawer", "hospital_bedside_table",
    "hospital_metal_cabinet", "hospital_vending_machine", "hospital_chair",
    "hospital_mop_cart", "hospital_parking_trolley_max", "hospital_surgical_trolley",
    "hospital_wheelchair", "hospital_trolley_bed", "hospital_xray_machine",
    "bookstore_desk_a", "bookstore_shelf_a", "bookstore_shelf_b", "bookstore_shelf_c",
    "bookstore_info_desk", "bookstore_chair", "bookstore_column_a", "house_kitchen_table",
    "house_bed", "house_kitchen_cabinet", "house_refrigerator", "house_wardrobe",
    "house_chair_a", "house_fitness_equipment", "house_sofa",
})

# §4.2 lobby — corridor set + perimeter-friendly large; desk_b only here (lobby-
# only per §4.3). Open-space character preserved by keeping large items out of the
# central open area at placement time.
_LOBBY_ALLOWED_KEYS = _CORRIDOR_ALLOWED_KEYS | frozenset({
    "hospital_metal_cabinet", "hospital_vending_machine", "hospital_parking_trolley_max",
    "hospital_xray_machine", "bookstore_desk_a", "bookstore_shelf_a", "bookstore_shelf_b",
    "bookstore_shelf_c", "bookstore_info_desk", "house_kitchen_cabinet", "house_wardrobe",
    "house_sofa", "warehouse_shelf", "warehouse_shelf_e",
    "hospital_trolley_bed", "house_bed", "bookstore_desk_b",
})

MAP_TYPE_ALLOWED_STATIC_KEYS = {
    "corridor":     _CORRIDOR_ALLOWED_KEYS,
    "intersection": _INTERSECTION_ALLOWED_KEYS,
    "clutter":      _CLUTTER_ALLOWED_KEYS,
    "lobby":        _LOBBY_ALLOWED_KEYS,
}

MAP_TYPES = ("lobby", "corridor", "intersection", "clutter")


def static_size_group(radius: float) -> str:
    """Map a catalog radius to a size group (docs/map_curriculum_plan.md §4.1)."""
    if radius <= 0.40:
        return "small"
    if radius <= 0.65:
        return "medium"
    return "large"


class Environment(Node):
    """Environment Node for providing services required for DRL.

    This class provides functionalities to interact with an environment through ROS2 services.
    The services include:
    - step: Take an action and get the resulting situation from the environment.
    - reset: Reset the environment and get initial observation.
    - get_dimensions: Get the dimensions of the state, action, and maximum action value.
    """

    def __init__(self):
        super().__init__("gym_node")

        # Determine if the environment is to be run in training or testing mode
        self.declare_parameter("environment_mode", "train")
        self.environment_mode = (
            self.get_parameter("environment_mode")
            .get_parameter_value()
            .string_value.lower()
        )
        if not self.environment_mode in ["train", "test", "random_test"]:
            raise NotImplementedError
        # Environment run mode
        self.train_mode = (
            self.environment_mode == "train" or self.environment_mode == "random_test"
        )
        self.get_logger().info(f"Environment run mode: {self.environment_mode}")

        # Load environment config file (robust)
        self.declare_parameter("config_file", "")
        cfg_param = self.get_parameter("config_file").get_parameter_value().string_value.strip()
        
        env_config_file_name = "environment.yaml"
        start_goal_pairs_file = "test_config.yaml"
        
        candidates = []
        tried = []
        
        # 1) 사용자 파라미터(전체 경로) 우선
        if cfg_param:
            p = os.path.expanduser(cfg_param)
            if os.path.isfile(p):
                cfg_dir = os.path.dirname(p)
                env_config_file_path = p
            else:
                tried.append(p)
        
        # 2) 설치된 share 경로
        if "cfg_dir" not in locals():
            try:
                from ament_index_python.packages import get_package_share_directory
                share_dir = os.path.join(get_package_share_directory("drl_agent"), "config")
                candidates.append(share_dir)
            except Exception:
                pass
            
        # 3) 환경변수: DRL_AGENT_CONFIG (전체 파일 경로)
        if "cfg_dir" not in locals():
            env_full = os.environ.get("DRL_AGENT_CONFIG", "")
            if env_full:
                env_full = os.path.expanduser(env_full)
                if os.path.isfile(env_full):
                    cfg_dir = os.path.dirname(env_full)
                    env_config_file_path = env_full
                else:
                    tried.append(env_full)
        
        # 4) 환경변수: DRL_AGENT_SRC_PATH 기반 후보들
        if "cfg_dir" not in locals():
            drl_agent_src_path = os.environ.get("DRL_AGENT_SRC_PATH", "")
            if drl_agent_src_path:
                candidates += [
                    os.path.join(drl_agent_src_path, "drl_agent", "config"),
                    os.path.join(drl_agent_src_path, "src", "drl_agent", "config"),
                    os.path.join(drl_agent_src_path, "src", "drl_agent", "src", "drl_agent", "config"),
                    os.path.join(drl_agent_src_path, "config"),
                ]
        
            # 5) 소스 트리 상대 경로(개발 중 편의)
            here = os.path.dirname(os.path.abspath(__file__))
            candidates += [
                os.path.normpath(os.path.join(here, "..", "..", "config")),  # .../drl_agent/config
                os.path.normpath(os.path.join(here, "..", "config")),        # .../scripts/config (혹시)
            ]
        
            for d in candidates:
                p = os.path.join(d, env_config_file_name)
                if os.path.isfile(p):
                    cfg_dir = d
                    break
                tried.append(p)
        
        if "cfg_dir" not in locals():
            self.get_logger().error(
                "Could not find '{}'. Tried:\n  {}".format(
                    env_config_file_name, "\n  ".join(tried)
                )
            )
            sys.exit(-1)
        
        if "env_config_file_path" not in locals():
            env_config_file_path = os.path.join(cfg_dir, env_config_file_name)
        start_goal_pairs_file_path = os.path.join(cfg_dir, start_goal_pairs_file)
        self.get_logger().info(f"Using config: {env_config_file_path}")
        # AUX_ABLATION: remember the ACTUAL loaded config path + content hash so
        # a trainer can record exactly which env config produced a run (instead
        # of re-discovering a file that may differ from what the env loaded).
        self._loaded_config_path = os.path.abspath(env_config_file_path)
        try:
            with open(env_config_file_path, "rb") as _cf:
                self._loaded_config_sha1 = hashlib.sha1(_cf.read()).hexdigest()
        except Exception:
            self._loaded_config_sha1 = ""
        # Define the dimensions of the state, action, and maximum action value
        try:
            self.config = load_yaml(env_config_file_path)
        except Exception as e:
            self.get_logger().info(f"Unable to load config file: {e}")
            sys.exit(-1)
        self.environment_config = self.config["environment"]
        self.lower = self.environment_config["lower"]
        self.upper = self.environment_config["upper"]
        self.goal_obstacle_lower = float(
            self.environment_config.get("goal_obstacle_lower", self.lower)
        )
        self.goal_obstacle_upper = float(
            self.environment_config.get("goal_obstacle_upper", self.upper)
        )
        self.environment_dim = self.environment_config["environment_state_dim"]
        self.agent_dim = self.environment_config["agent_state_dim"]
        self.agent_name = self.environment_config["agent_name"]
        # Dynamic obstacles removed: moving obstacles are humans only; static obstacles are fixed.
        self.num_of_static_obstacles  = int(self.environment_config.get("num_of_static_obstacles", 0))

        self.action_dim = self.environment_config["action_dim"]
        self.max_action = self.environment_config["max_action"]
        self.actions_low = self.environment_config["actions_low"]
        self.actions_high = self.environment_config["actions_high"]
        self.vehicle_wheelbase_m = float(
            self.environment_config.get("vehicle_wheelbase_m", 0.547696)
        )
        self.vehicle_steering_limit_deg = float(
            self.environment_config.get("vehicle_steering_limit_deg", 21.58)
        )
        self.vehicle_min_speed_for_steering_mps = float(
            self.environment_config.get("vehicle_min_speed_for_steering_mps", 0.15)
        )
        self.vehicle_steering_limit_rad = math.radians(self.vehicle_steering_limit_deg)

        self.controller_cruise_speed_mps = float(
            self.environment_config.get("controller_cruise_speed_mps", 1.0)
        )
        self.controller_min_speed_mps = float(
            self.environment_config.get("controller_min_speed_mps", 0.3)
        )
        self.controller_speed_steer_factor = float(
            self.environment_config.get("controller_speed_steer_factor", 0.6)
        )
        self.spawn_z = self.environment_config.get("spawn_z", 0.4)
        self.obs_z_min_sensor_m = float(self.environment_config.get("obs_z_min_sensor_m", -0.555))
        self.obs_z_max_sensor_m = float(self.environment_config.get("obs_z_max_sensor_m",  0.250))

        # Rectangular Safety Region params (paper Algorithm 1)
        sr = self.config.get("safety_region", {})
        self.sr_d_front = float(sr.get("d_front",  0.41))
        self.sr_d_rear  = float(sr.get("d_rear",   0.41))
        self.sr_d_left  = float(sr.get("d_left",   0.30))
        self.sr_d_right = float(sr.get("d_right",  0.30))
        # Per-direction margins; fall back to legacy single safety_margin if present
        _fb = float(sr.get("safety_margin", 0.22))
        self.sr_margin_front  = float(sr.get("margin_front",  _fb if "safety_margin" in sr else 0.09))
        self.sr_margin_rear   = float(sr.get("margin_rear",   _fb if "safety_margin" in sr else 0.14))
        self.sr_margin_left   = float(sr.get("margin_left",   _fb if "safety_margin" in sr else 0.22))
        self.sr_margin_right  = float(sr.get("margin_right",  _fb if "safety_margin" in sr else 0.22))
        # Per-direction warning scales; fall back to legacy global reward_warning_scale
        _fb_warn = float(sr.get("reward_warning_scale", 1.5))
        self.reward_warning_scale_front = float(
            sr.get("reward_warning_scale_front", _fb_warn if "reward_warning_scale" in sr else 1.5)
        )
        self.reward_warning_scale_rear = float(
            sr.get("reward_warning_scale_rear", _fb_warn if "reward_warning_scale" in sr else 1.5)
        )
        self.reward_warning_scale_left = float(
            sr.get("reward_warning_scale_left", _fb_warn if "reward_warning_scale" in sr else 1.2)
        )
        self.reward_warning_scale_right = float(
            sr.get("reward_warning_scale_right", _fb_warn if "reward_warning_scale" in sr else 1.2)
        )
        self.sr_scan_resolution   = float(sr.get("scan_resolution", 0.05))
        # Populated after self.bins is set up below
        self._rect_safety_ranges:  np.ndarray = None
        self._rect_warning_ranges: np.ndarray = None

        # Start-pose heading/clearance safety parameters
        self.start_edge_heading_margin = float(
            self.environment_config.get("start_edge_heading_margin_m", 1.0))
        self.start_front_clearance = float(
            self.environment_config.get("start_front_clearance_m", 1.2))
        self.start_front_fov_deg = float(
            self.environment_config.get("start_front_fov_deg", 35.0))

        # Obstacle spawn margin parameters
        self.num_of_humans = int(self.environment_config.get("num_of_humans", 0))

        # AUX_PRED: future-risk label config.  When enabled, step()/reset()
        # append a fixed-size privileged label to the returned state vector; the
        # TQC trainer slices it off (the 87-D RL state itself is unchanged).
        # Disabled by default, so every non-aux config / algorithm is untouched.
        self._aux_label_cfg = aux_labels.AuxLabelConfig(
            self.environment_config.get("aux_prediction", {})
        )
        self._aux_pred_enabled = self._aux_label_cfg.enabled
        if self._aux_pred_enabled:
            self.get_logger().info(
                f"[AUX_PRED] future-risk labels ON: "
                f"horizons={self._aux_label_cfg.horizons_sec}, "
                f"sectors={self._aux_label_cfg.num_sectors}, "
                f"D_c={self._aux_label_cfg.risk_distance_scale}, "
                f"label_dim={self._aux_label_cfg.label_dim}"
            )

        # AUX_ABLATION: expose the ACTUALLY-loaded env config (path, content
        # hash) and the running aux label geometry as read-only ROS parameters,
        # so a trainer can record the true env settings in run_manifest.json via
        # /gym_node/get_parameters (no fragile file re-discovery).
        self.declare_parameter("loaded_config_path", self._loaded_config_path)
        self.declare_parameter("loaded_config_sha1", self._loaded_config_sha1)
        self.declare_parameter("aux_enabled", bool(self._aux_pred_enabled))
        self.declare_parameter("aux_num_sectors", int(self._aux_label_cfg.num_sectors))
        self.declare_parameter(
            "aux_horizons_sec", [float(h) for h in self._aux_label_cfg.horizons_sec])
        self.declare_parameter(
            "aux_risk_distance_scale", float(self._aux_label_cfg.risk_distance_scale))

        # Human proxy motion / domain-randomization parameters
        self.human_update_rate       = float(self.environment_config.get("human_update_rate",        20.0))
        # Per-second probabilities (converted to per-tick inside the timer callback)
        self.human_stop_prob_per_sec = float(self.environment_config.get("human_stop_prob_per_sec",
                                             self.environment_config.get("human_stop_prob", 0.05)))
        self.human_pause_duration    = float(self.environment_config.get("human_pause_duration",     2.0))
        self.human_heading_jitter    = math.radians(float(self.environment_config.get("human_heading_jitter_deg", 15.0)))
        self.human_retarget_prob_per_sec = float(self.environment_config.get("human_retarget_prob_per_sec",
                                                  self.environment_config.get("human_retarget_prob", 0.02)))
        self.human_scan_dropout_prob = float(self.environment_config.get("human_scan_dropout_prob",  0.03))
        self.human_scan_noise_std    = float(self.environment_config.get("human_scan_noise_std",     0.05))
        # Kinematic model limits
        self.human_max_accel         = float(self.environment_config.get("human_max_accel",          0.6))
        self.human_max_yaw_rate      = float(self.environment_config.get("human_max_yaw_rate",       0.9))
        self.human_max_yaw_accel     = float(self.environment_config.get("human_max_yaw_accel",      1.5))
        self.human_k_yaw             = float(self.environment_config.get("human_k_yaw",              2.0))
        # Human heading smoothing. These reduce sudden "snap" turns by
        # persisting heading jitter for the whole waypoint segment, enforcing
        # a minimum retarget interval, and rate-limiting desired-yaw changes.
        self.human_heading_jitter_on_retarget_only = bool(
            self.environment_config.get("human_heading_jitter_on_retarget_only", True)
        )
        self.human_min_retarget_interval = float(
            self.environment_config.get("human_min_retarget_interval", 1.5)
        )
        self.human_desired_yaw_rate_limit = float(
            self.environment_config.get("human_desired_yaw_rate_limit", 1.2)
        )
        # Mode-transition control for Auxiliary-Prediction-friendly pedestrians.
        # These are GLOBAL FALLBACKS: per-human waypoint distance is normally set
        # by the behaviour mode (Environment._HUMAN_MODE_DEFAULTS), so dist_min/max
        # here apply only to a waypoint sampled without a mode. A waypoint is also
        # force-retargeted after human_max_segment_sec of ACTIVE-motion time — the
        # counter does not advance during pause/stop, so a long pause (waiting
        # mode) can make the real interval exceed it.
        # 0 disables the feature (legacy whole-arena sampling, no timeout).
        self.human_waypoint_dist_min = float(
            self.environment_config.get("human_waypoint_dist_min", 0.0)
        )
        self.human_waypoint_dist_max = float(
            self.environment_config.get("human_waypoint_dist_max", 0.0)
        )
        self.human_max_segment_sec = float(
            self.environment_config.get("human_max_segment_sec", 0.0)
        )
        # Episode-level pedestrian behaviour modes (Auxiliary Prediction): each
        # human is assigned ONE mode at spawn (crossing / along_path / waiting /
        # slow_turn) and keeps it for the whole episode. human_mode_weights sets
        # the sampling mix (empty → uniform over the modes); human_mode_params
        # overrides the per-mode defaults in _assign_human_mode(). Both are
        # stage-overridable from environment_curriculum.yaml.
        self.human_mode_weights = dict(self.environment_config.get("human_mode_weights", {}) or {})
        self.human_mode_params  = dict(self.environment_config.get("human_mode_params", {}) or {})

        # Falcon-lite goal-driven pedestrian motion. When enabled, each human is
        # given an episode-level GLOBAL goal at spawn and walks toward it (the
        # waypoint sampler returns short steps along the goal direction); on
        # arrival it rests then picks a new goal. Humans still NEVER react to the
        # robot, and motion stays near-constant-velocity between goals so the
        # constant-velocity aux labels remain valid. Disabled -> legacy random
        # waypoint walk (unchanged behaviour).
        self.human_goal_driven_enabled = bool(
            self.environment_config.get("human_goal_driven_enabled", False))
        self.human_goal_reach_threshold = float(      # [m] arrived at a goal
            self.environment_config.get("human_goal_reach_threshold", 0.6))
        self.human_goal_pause_duration = float(       # [s] rest at a reached goal
            self.environment_config.get("human_goal_pause_duration", 2.0))
        self.human_local_step_m = float(              # [m] local step toward the goal
            self.environment_config.get("human_local_step_m", 2.0))
        self.human_goal_span_multiplier = float(      # goal distance = mode wp_dist * this
            self.environment_config.get("human_goal_span_multiplier", 2.0))
        self.human_goal_min_span_m = float(           # [m] min distance for a new goal
            self.environment_config.get("human_goal_min_span_m", 4.0))
        # Weak human-human local avoidance (lightweight, NOT full ORCA; robot is
        # ignored). Nearby humans nudge each other's desired heading by a small,
        # capped amount so they overlap less without sharp turns.
        self.human_social_avoid_enabled = bool(
            self.environment_config.get("human_social_avoid_enabled", False))
        self.human_social_avoid_radius = float(       # [m] neighbour influence radius
            self.environment_config.get("human_social_avoid_radius", 1.5))
        self.human_social_avoid_strength = float(     # blend weight of repulsion vs goal dir
            self.environment_config.get("human_social_avoid_strength", 0.6))
        self.human_social_avoid_max_heading_offset = math.radians(float(  # cap on the nudge
            self.environment_config.get("human_social_avoid_max_heading_offset_deg", 25.0)))

        self.obstacle_wall_margin   = self.environment_config.get("obstacle_wall_margin",   1.0)
        self.obstacle_robot_margin  = self.environment_config.get("obstacle_robot_margin",  1.5)
        self.obstacle_goal_margin   = self.environment_config.get("obstacle_goal_margin",   1.5)
        self.obstacle_mutual_margin = self.environment_config.get("obstacle_mutual_margin", 1.2)

        # Actual arena wall inner-face boundary (used by start-pose heading checks).
        # Derived from goal_obstacle bounds + obstacle_wall_margin so that
        # obstacle placement "wall_margin" lines up with the true wall face.
        # e.g. goal_obstacle_upper=8.5 + obstacle_wall_margin=1.0 → 9.5 m
        self._arena_wall_lower = float(self.goal_obstacle_lower) - float(self.obstacle_wall_margin)
        self._arena_wall_upper = float(self.goal_obstacle_upper) + float(self.obstacle_wall_margin)

        # Pool mode — spawn all obstacles once at startup, teleport per episode
        self.use_obstacle_pool = bool(self.environment_config.get("use_obstacle_pool", False))
        self.obstacle_pool_static_size  = int(self.environment_config.get(
            "obstacle_pool_static_size",  self.num_of_static_obstacles))
        self.obstacle_pool_human_size   = int(self.environment_config.get(
            "obstacle_pool_human_size",   self.num_of_humans))
        self.parking_z = float(self.environment_config.get("parking_z", 0.0))
        parking_slot_xs = self.environment_config.get(
            "parking_slot_xs", [-16.0, -14.0, -12.0, 12.0, 14.0, 16.0]
        )
        parking_slot_ys = self.environment_config.get(
            "parking_slot_ys", [-16.0, -14.0, -12.0, 12.0, 14.0, 16.0]
        )
        self.parking_slots = [
            (float(px), float(py), self.parking_z)
            for px in parking_slot_xs
            for py in parking_slot_ys
        ]
        if not self.parking_slots:
            self.parking_slots = [(16.0, 16.0, self.parking_z)]

        # Load obstacle catalog — supports either a .yaml filename (relative to cfg_dir)
        # or a package name resolved via ament_index.
        catalog_spec = self.environment_config.get("obstacle_catalog", "obstacle_catalog.yaml")
        if catalog_spec.endswith(".yaml"):
            catalog_path = os.path.join(cfg_dir, catalog_spec)
        else:
            try:
                from ament_index_python.packages import get_package_share_directory
                catalog_path = os.path.join(
                    get_package_share_directory(catalog_spec), "config", "obstacle_catalog.yaml"
                )
            except Exception as e:
                self.get_logger().warn(f"Could not resolve catalog package '{catalog_spec}': {e}")
                catalog_path = ""
        self.static_obstacle_catalog  = []
        self.human_catalog = []
        if catalog_path and os.path.isfile(catalog_path):
            try:
                cat = load_yaml(catalog_path)
                all_obs = cat.get("obstacles", [])
                # Dynamic obstacles removed: every catalog obstacle is treated as
                # static (entries without motion_type default to static).
                self.static_obstacle_catalog  = [e for e in all_obs if e.get("motion_type", "static") != "dynamic"]
                self.human_catalog = cat.get("humans", [])
                self.get_logger().info(
                    f"Loaded {len(self.static_obstacle_catalog)} static obstacle types and "
                    f"{len(self.human_catalog)} human types from {catalog_path}"
                )
            except Exception as e:
                self.get_logger().warn(f"Failed to load obstacle catalog: {e}")

        # Obstacles this node believes may still be present in the world.
        # If deletion times out, keep the last known pose/radius so the next
        # episode avoids spawning the robot or new obstacles on top of it.
        self.spawned_obstacle_names: list = []
        self.spawned_obstacle_records = {}
        # Pool bookkeeping — populated by _initialize_obstacle_pool on first reset
        self.pool_static:  list = []
        self.pool_human:   list = []
        self.pool_initialized = False
        self.human_states: dict = {}  # keyed by proxy entity name; active during each episode
        # Mutual exclusion between the 20 Hz human timer and reset_callback.
        # The timer holds this lock for the full _update_humans_kinematic() iteration.
        # reset_callback acquires it (blocking) before clearing human_states, which
        # guarantees any in-flight timer has finished before we touch shared state.
        self._human_lock = threading.Lock()
        # Secondary fast-path flag: False while reset is rebuilding human_states.
        # Timer checks this before trying to acquire the lock (cheap early-out).
        self._human_updates_enabled: bool = True
        self.human_placement_mode: str = "quadrants"
        # Monotonically increasing episode counter — used to generate unique obstacle names
        # so a timed-out delete from the previous episode never collides with a new spawn.
        self._episode_count = 0

        # ───────────────────────────────────────────────────────────────────
        # Structured map curriculum (docs/map_curriculum_plan.md)
        # Default OFF → every existing scatter config behaves exactly as before.
        # When enabled, each episode samples a map_type (lobby/corridor/
        # intersection/clutter), activates that map's internal walls, and makes
        # start/goal/obstacle/human sampling layout-aware. This block defaults to
        # "안 B": the structured layout lives INSIDE the existing 19×19 world
        # (outer walls untouched); all extents derive from config so "안 A"
        # (world expansion) only needs new map_inner_* / parking values.
        # ───────────────────────────────────────────────────────────────────
        mc = self.environment_config
        self.map_layout_enabled = bool(mc.get("map_layout_enabled", False))
        # Inner navigable extent (default = current obstacle bounds → 안 B).
        self.map_inner_lower = float(mc.get("map_inner_lower", self.goal_obstacle_lower))
        self.map_inner_upper = float(mc.get("map_inner_upper", self.goal_obstacle_upper))
        self.map_wall_thickness = float(mc.get("map_wall_thickness", 0.30))
        self.map_wall_height = float(mc.get("map_wall_height", 1.20))
        self.map_corridor_width = float(mc.get("map_corridor_width", 3.5))
        self.map_intersection_width = float(mc.get("map_intersection_width", 3.5))
        self.map_lobby_open_half_extent = float(mc.get("map_lobby_open_half_extent", 4.0))
        # Extra keep-out (beyond the obstacle's catalog radius) used ONLY for the
        # lobby central open area. Catalog radius badly underestimates the real
        # XY footprint of shelves/desks (radius ~0.6 vs length ~4-6 m), so a
        # centre-only test would let a long item eat the open space. This margin
        # approximates the worst-case half-footprint so large items settle on the
        # perimeter instead. Tune up to push them further out.
        self.map_large_footprint_margin = float(mc.get("map_large_footprint_margin", 2.0))
        # Minimum clearance any sampled pose must keep from an internal wall face.
        self.map_wall_clearance = float(mc.get("map_wall_clearance", 0.55))
        # ── Structured spawn policy (corridor / intersection only) ──────────
        # End-band depth (along the lane axis) and side margin (off the lane
        # walls) for the start regions; lane-aligned spawn yaw bias/jitter; and
        # the reserved central passage widths + safety margin that static
        # obstacles must never block. Defaults are intentionally gentle.
        self.map_start_band_depth = float(mc.get("map_start_band_depth", 2.5))
        self.map_start_side_margin = float(mc.get("map_start_side_margin", 0.4))
        self.map_spawn_yaw_center_bias = math.radians(
            float(mc.get("map_spawn_yaw_center_bias_deg", 12.0)))
        self.map_spawn_yaw_jitter = math.radians(
            float(mc.get("map_spawn_yaw_jitter_deg", 8.0)))
        self.map_corridor_passage_width = float(mc.get("map_corridor_passage_width", 1.6))
        self.map_intersection_passage_width = float(mc.get("map_intersection_passage_width", 1.6))
        self.map_passage_safety_margin = float(mc.get("map_passage_safety_margin", 0.25))
        # Minimum activatable pool coverage PER (map_type, size_group) so a
        # stage's size filter always has obstacles to activate on each map.
        self.map_static_coverage_per_group = int(mc.get(
            "map_static_coverage_per_group",
            mc.get("map_static_coverage_per_type", 5)))
        # Stage-controlled fields (overridden per episode by the curriculum
        # subclass; defaults let the plain env run a single map type too).
        self.allowed_map_types = [
            m for m in (mc.get("allowed_map_types", ["clutter"]) or ["clutter"])
            if m in MAP_TYPE_ALLOWED_STATIC_KEYS
        ] or ["clutter"]
        self.map_type_probs = list(mc.get("map_type_probs", []) or [])
        self.allowed_static_groups = list(mc.get("allowed_static_groups", []) or [])
        self.eval_map_types = list(mc.get("eval_map_types", []) or [])
        # Episode-level layout state (populated by _select_episode_layout()).
        self.current_map_type = ""
        self.current_layout_spec = None
        # Start region chosen this episode (corridor/intersection structured
        # spawn); drives the structure-aware goal region. None for other maps.
        self._current_start_region = None
        # Round-robin pointer for deterministic eval map cycling.
        self._eval_map_cursor = 0
        # Wall-box pool (spawned once, parked underground, activated per episode).
        self.pool_walls = []        # list of {name, sx, sy}
        self._map_layouts = {}      # map_type -> layout spec (built below)
        # Catalog keyed by `key` for group-aware coverage spawning.
        self._catalog_by_key = {
            e["key"]: e for e in self.static_obstacle_catalog if "key" in e
        }
        if self.map_layout_enabled:
            self._map_layouts = self._build_map_layouts()
            self._static_pool_coverage = self._build_static_pool_coverage()
            if self._static_pool_coverage:
                # Group-aware coverage drives the static pool size so every map
                # type has >= coverage activatable entries pre-spawned (no
                # per-episode create/remove).
                self.obstacle_pool_static_size = len(self._static_pool_coverage)
            self._ensure_parking_slots(
                self.obstacle_pool_static_size + self.obstacle_pool_human_size
            )
            self.get_logger().info(
                f"[MapCurriculum] enabled | maps={self.allowed_map_types} | "
                f"static_pool={self.obstacle_pool_static_size} "
                f"(coverage of {len(self._static_pool_coverage)} keys) | "
                f"wall_boxes={sum(len(l['walls']) for l in self._map_layouts.values())} | "
                f"parking_slots={len(self.parking_slots)} | "
                f"inner=[{self.map_inner_lower}, {self.map_inner_upper}]"
            )
        else:
            self._static_pool_coverage = []
        # Read-only ROS parameter so the trainer can log the per-episode map_type
        # (and aggregate per-map evaluation) via /gym_node/get_parameters.
        self.declare_parameter("current_map_type", "")
        # Writable flag the trainer raises around its evaluation episodes so the
        # SAME training env switches to the eval_map_types round-robin (instead of
        # the training map distribution) while STILL activating obstacles/humans.
        # Decouples "which maps are evaluated" from the train/test node mode.
        self._curriculum_eval_mode = False
        self.declare_parameter("curriculum_eval_mode", False)

        self.threshold_params_config = self.config["threshold_parameters"]
        self.goal_threshold = self.threshold_params_config["goal_threshold"]
        self.collision_threshold = self.threshold_params_config["collision_threshold"]
        self.time_delta = self.threshold_params_config["time_delta"]
        self.inter_entity_distance = self.threshold_params_config[
            "inter_entity_distance"
        ]

        self.lidar_max_range = self.threshold_params_config["lidar_max_range"]

        # ── Goal-conditioned observation corruption (localization uncertainty) ──
        # NOTE ON SCOPE: this does NOT model "the whole localization stack". It
        # reflects LOCALIZATION UNCERTAINTY in the policy's GOAL observation only —
        # i.e. the robot's noisy self-pose estimate corrupts the GOAL DISTANCE
        # (state[80]) and HEADING ERROR (state[81]). LiDAR (state[0:80]) and the
        # proprioception slots (state[84:87]) are handled separately (proprio_noise
        # below), and reward/done stay on ground truth (use_gt_for_*).
        #
        # This file is the noise EXECUTOR: it interprets the parameters and injects
        # the perturbation. The curriculum (environment_curriculum.py) is only a
        # per-stage PROFILE SELECTOR — it never interprets the noise maths.
        #
        # Error model (each axis self-gates to a no-op when its flag is off / its
        # magnitude is 0):
        #   • bias            per-episode constant registration offset.
        #   • OU(sigma, tau)  time-correlated measurement error. sigma is the
        #                     STATIONARY std; tau (corr_time_*_s) the correlation
        #                     time. tau=0 ⇒ exp(-dt/tau)=0 ⇒ EXACTLY the legacy
        #                     per-step white Gaussian of std sigma.
        #   • drift           slow random walk. drift_xy_mps / drift_yaw_radps are
        #                     PER-SECOND intensities (accumulated std ≈ rate·√t,
        #                     Brownian √dt scaling). Back-compat aliases:
        #                     random_walk_xy_mps / random_walk_yaw_rps.
        #   • delay_steps     observation latency (separate ablation axis).
        #   • jumps           rare relocalization snaps: small + optional large.
        #   • yaw flip        ±π mirror-relocalization in symmetric maps (corridor).
        #                     STRESS-TEST axis: OFF unless noise_flip_enabled.
        #   • map_type_multipliers  scale sigma/drift/jump per structured-map type
        #                     (corridor localizes worst → larger / anisotropic).
        # Ablation flags (noise_*_enabled) turn whole axes on/off independently;
        # their defaults reproduce the previous behaviour (flip default OFF).
        #
        # EXTENSION POINT (NOT enabled here): exposing localization meta-info
        # (confidence / delay / jump flag) to the policy would change the state
        # dimension, so it is intentionally left out of the base observation and
        # reserved for a separate ablation that also widens the network input.
        _loc = dict(self.config.get("localization", {}) or {})
        # drift_* are the preferred physical names; random_walk_* kept as aliases.
        _drift_xy  = float(_loc.get("drift_xy_mps",   _loc.get("random_walk_xy_mps",  0.0)))
        _drift_yaw = float(_loc.get("drift_yaw_radps", _loc.get("random_walk_yaw_rps", 0.0)))
        self.loc_noise = {
            "enabled":            bool(_loc.get("enabled", False)),
            "mode":               str(_loc.get("mode", "off")),
            # ── per-axis ablation flags (independent on/off) ──
            "noise_goal_enabled":    bool(_loc.get("noise_goal_enabled", True)),
            "noise_jump_enabled":    bool(_loc.get("noise_jump_enabled", True)),
            "noise_flip_enabled":    bool(_loc.get("noise_flip_enabled", False)),
            "noise_delay_enabled":   bool(_loc.get("noise_delay_enabled", True)),
            "sigma_xy_m":         float(_loc.get("sigma_xy_m", 0.0)),
            "sigma_yaw_rad":      float(_loc.get("sigma_yaw_rad", 0.0)),
            # Correlation times for the OU measurement-error process [s].
            # 0 → legacy white-noise behaviour (backward compatible).
            "corr_time_xy_s":     float(_loc.get("corr_time_xy_s", 0.0)),
            "corr_time_yaw_s":    float(_loc.get("corr_time_yaw_s", 0.0)),
            "bias_xy_m":          float(_loc.get("bias_xy_m", 0.0)),
            "bias_yaw_rad":       float(_loc.get("bias_yaw_rad", 0.0)),
            # Per-second drift intensity (Brownian √dt scaling; std ≈ rate·√t).
            # Stored under the legacy keys so the emulator maths is unchanged.
            "random_walk_xy_mps":  _drift_xy,
            "random_walk_yaw_rps": _drift_yaw,
            "delay_steps":        int(_loc.get("delay_steps", 0)),
            # Small relocalization snap.
            "jump_prob":          float(_loc.get("jump_prob", 0.0)),
            "jump_xy_m":          float(_loc.get("jump_xy_m", 0.0)),
            "jump_yaw_rad":       float(_loc.get("jump_yaw_rad", 0.0)),
            # Large relocalization failure (kept very rare; hard stages only).
            "big_jump_prob":      float(_loc.get("big_jump_prob", 0.0)),
            "big_jump_xy_m":      float(_loc.get("big_jump_xy_m", 0.0)),
            "big_jump_yaw_rad":   float(_loc.get("big_jump_yaw_rad", 0.0)),
            # ±π yaw flip in symmetric maps (corridor). STRESS-TEST axis: requires
            # noise_flip_enabled=true (default false) so normal training never flips.
            "yaw_flip_prob":      float(_loc.get("yaw_flip_prob", 0.0)),
            "yaw_flip_map_types": list(_loc.get("yaw_flip_map_types", ["corridor"]) or []),
            # Per-map-type scaling of sigma / drift / jump (+ optional anisotropy).
            "map_type_multipliers": dict(_loc.get("map_type_multipliers", {}) or {}),
            "use_gt_for_reward":  bool(_loc.get("use_gt_for_reward", True)),
            "use_gt_for_done":    bool(_loc.get("use_gt_for_done", True)),
        }
        # Per-episode localization-noise state (initialised in _reset_localization).
        self._loc_bias_x = self._loc_bias_y = self._loc_bias_yaw = 0.0
        self._loc_rw_x = self._loc_rw_y = self._loc_rw_yaw = 0.0
        # OU (time-correlated) measurement-error state.
        self._loc_ou_x = self._loc_ou_y = self._loc_ou_yaw = 0.0
        # Effective per-episode params (base × map-type multiplier + anisotropy),
        # recomputed each reset; passthrough defaults keep step() safe pre-reset.
        self._loc_eff = {
            "sigma_x": 0.0, "sigma_y": 0.0, "sigma_yaw": 0.0,
            "drift_x": 0.0, "drift_y": 0.0, "drift_yaw": 0.0,
            "alpha_xy": 0.0, "alpha_yaw": 0.0, "jump_mult": 1.0,
        }
        self._loc_buf = deque([(0.0, 0.0, 0.0)], maxlen=1)
        self.loc_est_x = self.loc_est_y = self.loc_est_yaw = 0.0

        # ── Proprioception observation noise (separate axis from goal-obs) ──────
        # Perturbs ONLY the proprio observation slots state[84]=signed speed,
        # state[85]=yaw rate, state[86]=steering — NEVER the ground-truth caches
        # used for reward/done/collision. Default OFF → exact passthrough, so the
        # baseline is unchanged. Physical-unit params; a per-episode bias + scale
        # error + white Gaussian (+ optional latency), mirroring real odom/IMU.
        _pp = dict(self.config.get("proprio_noise", {}) or {})
        self.proprio_noise = {
            "enabled":               bool(_pp.get("enabled", False)),
            "speed_sigma_mps":       float(_pp.get("speed_sigma_mps", 0.0)),
            "speed_bias_mps":        float(_pp.get("speed_bias_mps", 0.0)),
            "speed_scale_sigma":     float(_pp.get("speed_scale_sigma", 0.0)),
            "yaw_rate_sigma_radps":  float(_pp.get("yaw_rate_sigma_radps", 0.0)),
            "yaw_rate_bias_radps":   float(_pp.get("yaw_rate_bias_radps", 0.0)),
            "steer_sigma_rad":       float(_pp.get("steer_sigma_rad", 0.0)),
            "delay_steps":           int(_pp.get("delay_steps", 0)),
        }
        self._pp_speed_bias = self._pp_yaw_bias = 0.0
        self._pp_speed_scale = 1.0
        self._pp_buf = deque([(0.0, 0.0, 0.0)], maxlen=1)

        # Callback groups for handling sensors and services in parallel
        self.odom_callback_group = MutuallyExclusiveCallbackGroup()
        self.filtered_cmd_callback_group = MutuallyExclusiveCallbackGroup()
        self.velodyne_callback_group = MutuallyExclusiveCallbackGroup()
        self.clients_callback_group = MutuallyExclusiveCallbackGroup()
        self.laser_callback_group = MutuallyExclusiveCallbackGroup()
        self.joint_state_callback_group = MutuallyExclusiveCallbackGroup()
        self.contact_callback_group = MutuallyExclusiveCallbackGroup()
        self.human_timer_callback_group = MutuallyExclusiveCallbackGroup()
        self.use_contact_collision = False
        self.contact_collision_latched = False
        self.contact_event_count = 0

        # Initialize publishers
        # ★ 토픽 파라미터 (기본값을 Hunter SE Ignition 시스템에 맞춤)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("cmd_vel_filtered_topic", "/cmd_vel_filtered")
        self.declare_parameter("odom_topic", "/odometry")
        # Separate pose sources (default = the single odom_topic → legacy behaviour):
        #   gt_odom_topic      : ground-truth pose (reward / done geometry)
        #   loc_odom_topic     : localization-estimate pose (policy goal observation)
        #   proprio_odom_topic : proprioception (actual speed, yaw rate)
        self.declare_parameter("gt_odom_topic", "")
        self.declare_parameter("loc_odom_topic", "")
        self.declare_parameter("proprio_odom_topic", "")
        self.declare_parameter("joint_states_topic", "/hunter_se/joint_states")
        self.declare_parameter("use_contact_collision", True)
        self.declare_parameter("contact_topic", "/hunter_se/chassis_contacts")
        self.declare_parameter("preserve_hunav_on_reset", True)

        cmd_vel_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        cmd_vel_filtered_topic = (
            self.get_parameter("cmd_vel_filtered_topic").get_parameter_value().string_value
        )
        odom_topic    = self.get_parameter("odom_topic").get_parameter_value().string_value
        joint_states_topic = (
            self.get_parameter("joint_states_topic").get_parameter_value().string_value
        )
        self.use_contact_collision = bool(
            self.get_parameter("use_contact_collision").get_parameter_value().bool_value
        )
        contact_topic = self.get_parameter("contact_topic").get_parameter_value().string_value
        self.preserve_hunav_on_reset = bool(
            self.get_parameter("preserve_hunav_on_reset").get_parameter_value().bool_value
        )
        
        # self.velocity_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.velocity_publisher = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.goal_point_marker_pub = self.create_publisher(
            MarkerArray, "goal_point", 10
        )
        self.wp_r_marker_pub = self.create_publisher(
            MarkerArray, "wp_r_norm", 10
        )
        self.wp_theta_marker_pub = self.create_publisher(
            MarkerArray, "wp_theta_norm", 10
        )
        self.robot_path_pub = self.create_publisher(
            Path, "robot_path", 10
        )
        # Kinematic obstacle motion: single non-blocking publish replaces per-model
        # set_entity_pose_ignition service calls inside the 20 Hz timer callback.
        # best_effort matches the plugin subscriber — avoids DDS QoS mismatch.
        _kinematic_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._model_pose_pub = self.create_publisher(
            DrlModelPoseArray, "/drl/model_poses", _kinematic_qos
        )

        # Create services
        self.srv_seed = self.create_service(Seed, "seed", self.seed_callback)
        self.srv_step = self.create_service(Step, "step", self.step_callback)
        self.srv_reset = self.create_service(Reset, "reset", self.reset_callback)
        self.srv_dimentions = self.create_service(
            GetDimensions, "get_dimensions", self.get_dimensions_callback
        )
        self.srv_action_space_sample = self.create_service(
            SampleActionSpace, "action_space_sample", self.sample_action_callback
        )

        # ----------------------------------------------------------------------------------------------
        # ====================================Ignition Start============================================
        # ----------------------------------------------------------------------------------------------
        # Initialize clients
        self.declare_parameter("world_name", "default")
        self.world_name = (
            self.get_parameter("world_name")
            .get_parameter_value()
            .string_value
        )
        # /world/<world_name>/control  (pause / reset 등)
        self.world_control = self.create_client(
            ControlWorld,
            f"/world/{self.world_name}/control",
            callback_group=self.clients_callback_group,
        )
        # /world/<world_name>/set_pose (모델 텔레포트)
        self.set_entity_pose = self.create_client(
            SetEntityPose,
            f"/world/{self.world_name}/set_pose",
            callback_group=self.clients_callback_group,
        )
        # /world/<world_name>/create  (runtime obstacle spawn)
        self.spawn_entity_client = self.create_client(
            SpawnEntity,
            f"/world/{self.world_name}/create",
            callback_group=self.clients_callback_group,
        )
        # /world/<world_name>/remove  (runtime obstacle delete)
        self.delete_entity_client = self.create_client(
            DeleteEntity,
            f"/world/{self.world_name}/remove",
            callback_group=self.clients_callback_group,
        )
        # ----------------------------------------------------------------------------------------------
        # ====================================Ignition Finish===========================================
        # ----------------------------------------------------------------------------------------------

        # Sensor subscriptions QoS
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        qos_best = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Odometry subscriptions — gt / loc / proprio. Each defaults to the single
        # odom_topic, so with no extra config one subscription serves all three
        # roles and behaviour is identical to the original single /odometry path.
        gt_topic  = (self.get_parameter("gt_odom_topic").get_parameter_value().string_value.strip()  or odom_topic)
        loc_topic = (self.get_parameter("loc_odom_topic").get_parameter_value().string_value.strip() or odom_topic)
        pro_topic = (self.get_parameter("proprio_odom_topic").get_parameter_value().string_value.strip() or odom_topic)
        self._odom_topics = {"gt": gt_topic, "loc": loc_topic, "proprio": pro_topic}
        roles_by_topic = {}
        roles_by_topic.setdefault(gt_topic,  set()).add("gt")
        roles_by_topic.setdefault(loc_topic, set()).add("loc")
        roles_by_topic.setdefault(pro_topic, set()).add("proprio")
        self._odom_subs = []
        for topic, roles in roles_by_topic.items():
            self._odom_subs.append(self.create_subscription(
                Odometry, topic,
                (lambda msg, r=frozenset(roles): self._on_odom(msg, r)),
                qos_profile,
                callback_group=self.odom_callback_group,
            ))
        self.get_logger().info(
            f"  odom sources — gt:{gt_topic} loc:{loc_topic} proprio:{pro_topic}"
        )
        self.filtered_cmd_sub = self.create_subscription(
            Twist,
            cmd_vel_filtered_topic,
            self._update_filtered_cmd,
            qos_profile,
            callback_group=self.filtered_cmd_callback_group,
        )
        self.joint_states_sub = self.create_subscription(
            JointState,
            joint_states_topic,
            self._update_steering_joints,
            qos_profile,
            callback_group=self.joint_state_callback_group,
        )
        self.contact_sub = None
        if self.use_contact_collision:
            self.contact_sub = self.create_subscription(
                Contacts,
                contact_topic,
                self._update_contact_collision,
                qos_profile,
                callback_group=self.contact_callback_group,
            )

        # === 관측 소스 선택: LaserScan vs PointCloud2 ===
        self.declare_parameter("obs_source", "scan")      # "scan" 또는 "pointcloud"
        self.declare_parameter("scan_topic", "/scan")     # pointcloud_to_laserscan 출력 토픽
        self.declare_parameter("pointcloud_topic", "/points")  # PointCloud2 기본 토픽

        obs_source    = self.get_parameter("obs_source").get_parameter_value().string_value.lower()
        scan_topic    = self.get_parameter("scan_topic").get_parameter_value().string_value
        cloud_topic   = self.get_parameter("pointcloud_topic").get_parameter_value().string_value

        self.laser    = None
        self.velodyne = None

        if obs_source == "scan":
            self.get_logger().info(f"Observation source: LaserScan ({scan_topic})")
            self.laser = self.create_subscription(
                LaserScan,
                scan_topic,
                self.update_environment_state_from_scan,
                qos_best,
                callback_group=self.laser_callback_group,
            )
        elif obs_source == "pointcloud":
            self.get_logger().info(f"Observation source: PointCloud2 ({cloud_topic})")
            self.velodyne = self.create_subscription(
                PointCloud2,
                cloud_topic,
                self.update_environment_state_from_cloud,
                qos_profile,
                callback_group=self.velodyne_callback_group,
            )
        else:
            self.get_logger().warn(
                f"Unknown obs_source '{obs_source}', falling back to LaserScan."
            )
            self.laser = self.create_subscription(
                LaserScan,
                scan_topic,
                self.update_environment_state_from_scan,
                qos_best,
                callback_group=self.laser_callback_group,
            )

        # Define bins for collision detection (FULL 360°)
        eps = 0.03
        width = 2*np.pi / self.environment_dim
        start = -np.pi - eps
        self.bins = [[start + i*width, start + (i+1)*width] for i in range(self.environment_dim)]
        self.bins[-1][-1] += eps

        # Define bins for RL observation input only (FRONT 180°: -π/2 to +π/2)
        obs_eps = 0.03
        obs_width = np.pi / self.environment_dim
        obs_start = -np.pi / 2 - obs_eps
        self.obs_bins = [[obs_start + i*obs_width, obs_start + (i+1)*obs_width]
                         for i in range(self.environment_dim)]
        self.obs_bins[-1][-1] += obs_eps

        # Precompute per-bin safety ranges (rectangular footprint, paper Algorithm 1)
        self._rect_safety_ranges  = self._precompute_rect_safety_ranges()
        self._rect_warning_ranges = self._precompute_rect_safety_ranges(
            front_scale=self.reward_warning_scale_front,
            rear_scale=self.reward_warning_scale_rear,
            left_scale=self.reward_warning_scale_left,
            right_scale=self.reward_warning_scale_right,
        )

        # ----------------------------------------------------------------------------------------------
        # ====================================Ignition Start============================================
        # ----------------------------------------------------------------------------------------------
        # Initialize commands
        self.velocity_command = Twist()
        # ----------------------------------------------------------------------------------------------
        # ====================================Ignition Finish===========================================
        # ----------------------------------------------------------------------------------------------

        # Independent timer for obstacle kinematic updates.
        # Uses its own MutuallyExclusiveCallbackGroup so it never blocks
        # the RL step/reset service callbacks.
        obstacle_update_rate = self.human_update_rate
        if self.num_of_humans > 0:
            self.human_timer = self.create_timer(
                1.0 / obstacle_update_rate,
                self._human_timer_callback,
                callback_group=self.human_timer_callback_group,
            )

        # Initialize environment and agent state
        self.environment_state = None
        self.agent_state = None
        # Initialize lock to protect environment_state and agent sate from race condition
        self.environment_state_lock = threading.Lock()
        self.agent_state_lock = threading.Lock()
        self.path_lock = threading.Lock()
        self.robot_path = Path()

        # ...locks 생성 이후, config 값들 로드가 끝난 시점에 안전 초기값 세팅
        self.environment_state = np.ones(self.environment_dim, dtype=float) * self.lidar_max_range
        self.obs_state = np.ones(self.environment_dim, dtype=float) * self.lidar_max_range
        self.agent_state = np.array(
            [np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float
        )
        self.scan_update_count = 0
        self.odom_update_count = 0
        # Per-role odom update counters — so reset can require freshness from each
        # configured source (gt/loc/proprio), even when they are separate topics.
        self._odom_role_count = {"gt": 0, "loc": 0, "proprio": 0}
        self.current_episode_step = 0
        self.latest_actual_speed = 0.0
        self.latest_actual_signed_speed = 0.0
        self.latest_actual_yaw_rate = 0.0
        self.latest_odom_x = 0.0
        self.latest_odom_y = 0.0
        self.latest_odom_yaw = 0.0
        # Pose caches per source (gt = reward/done geometry, loc_raw = pre-noise
        # localization pose for the policy observation).
        self.gt_x = self.gt_y = self.gt_yaw = 0.0
        self.loc_raw_x = self.loc_raw_y = self.loc_raw_yaw = 0.0
        self.latest_filtered_cmd_v = 0.0
        self.latest_filtered_cmd_w = 0.0
        self.latest_front_left_steering = 0.0
        self.latest_front_right_steering = 0.0
        self.latest_center_steering = 0.0
        self._init_debug_csv()

        # SIM_VALIDATION: optional localization-validation logging (default OFF →
        # zero effect on training/deploy). When on, writes loc_validation_*.csv
        # in the env log dir. Remove by deleting this block + grep SIM_VALIDATION.
        self.declare_parameter("enable_sim_validation_logging", False)
        self._sim_val = None
        self._sim_val_pre_motion = None   # SIM_VALIDATION: (goal_dist, heading) before this step's motion
        if self.get_parameter("enable_sim_validation_logging").get_parameter_value().bool_value:
            try:
                from sim_validation import SimValidationLogger
                _tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._sim_val = SimValidationLogger(self._env_log_dir, _tag)
                self.get_logger().info(
                    f"[SIM_VALIDATION] logging ON → {self._sim_val.step_path}"
                )
            except Exception as _e:
                self.get_logger().warn(f"[SIM_VALIDATION] init failed: {_e}")

        # Load start-goal pairs
        if not self.train_mode:
            try:
                self.start_goal_pairs = deque(
                    load_yaml(start_goal_pairs_file_path)["start_goal_pairs"]
                )
            except Exception as e:
                self.get_logger().error(f"Unable to load start-goal pairs: {e}")
                sys.exit(-1)
            self.current_pairs = None

        # Define initial goal pos
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_marker_model_name = "rl_goal_marker"
        self.goal_marker_spawned = False

        self._angle_min = float('nan')
        self._angle_max = float('nan')
        self._angle_inc = float('nan')

        # --- simple 5-zone collision (no min-beam, no hysteresis/speed scaling) ---
        self.declare_parameter("use_zone_collision", True)
        # 8 zones as PAIRS [a0,b0,a1,b1,...] in degrees, domain [-180,180)
        self.declare_parameter("zone_angles_deg",
            [-30, 30,   -50, -30,   30, 50,   -130, -50,  50, 130,  -150, -130, 130, 150,  150, -150]
        )
        self.declare_parameter("zone_thresholds",
            [0.71, 0.78, 0.78, 0.65, 0.65, 0.78, 0.78, 0.71]  # [FC,FRi,FLi,RF,LF,RRi,RLi,RC]
        )

        self.use_zone_collision = bool(self.get_parameter("use_zone_collision").value)
        self.zone_angles_deg    = list(self.get_parameter("zone_angles_deg").value)
        self.zone_thresholds    = list(self.get_parameter("zone_thresholds").value)

        # 내부 캐시
        self._zone_indices = None   # [(i0,i1), ...] 5개 존 빔 인덱스 범위(포함)
        self._zone_mins    = None   # [zmin5..zmin1]

        self._debug_dump_params_once()

    def _rad2deg(self, x):
        return x * 180.0 / math.pi

    def _robot_deg_signed(self, theta_std_rad):
        """
        LaserScan 표준각(0:+x, CCW+) → 로봇각(전방 +x=0°), [-180, 180)
        """
        deg = self._rad2deg(theta_std_rad)  # 표준 각도 자체 사용
        if deg >= 180.0:
            deg -= 360.0
        return deg

    def _compute_zone_indices_simple(self, scan):
        """
        zone_angles_deg 해석:
          - 레거시: [b0,b1,...,bN] → N개 연속 구역 [bk, bk+1]
          - 새 방식: [a0,b0,a1,b1,...] → N개 구역 각각 [ak,bk] (wrap-around 허용: a>b)
        결과: self._zone_indices = [ [i,i2,...], ... ]  # 각 존의 빔 인덱스 리스트
        """
        n    = len(scan.ranges)
        ang0 = float(scan.angle_min)
        inc  = float(scan.angle_increment)

        # 모든 빔의 로봇 기준 signed 각도(도)
        rdeg = [self._robot_deg_signed(ang0 + i*inc) for i in range(n)]

        angles = list(self.zone_angles_deg or [])
        thrs   = list(self.zone_thresholds or [])
        zones_pairs = []

        if len(angles) == len(thrs) + 1 and len(thrs) >= 1:
            # 레거시: 경계열 → 연속 구역
            bounds = angles
            for k in range(len(thrs)):
                a, b = bounds[k], bounds[k+1]
                zones_pairs.append((a, b))
        elif len(angles) == 2 * len(thrs) and len(thrs) >= 1:
            # 새 방식: (a,b) 쌍
            zones_pairs = list(zip(angles[::2], angles[1::2]))
        else:
            # 형식 오류 시, 빈 리스트로 두고 종료
            self._zone_indices = [[] for _ in range(len(thrs))]
            return

        def in_range(d, a, b):
            # [-180,180)에서 [a,b] 포함. a<=b 일반, a>b는 wrap-around
            return (a <= d <= b) if (a <= b) else (d >= a or d <= b)

        idx_lists = [[] for _ in range(len(zones_pairs))]
        for i, d in enumerate(rdeg):
            for zi, (a, b) in enumerate(zones_pairs):
                if in_range(d, float(a), float(b)):
                    idx_lists[zi].append(i)
                    break
        self._zone_indices = idx_lists

    def _update_zone_mins_simple(self, scan):
        """
        존별 최소거리(min). 유효빔 없으면 inf.
        self._zone_indices: 각 존의 빔 인덱스 리스트
        """
        if self._zone_indices is None:
            self._compute_zone_indices_simple(scan)

        zmins = []
        for idx_list in (self._zone_indices or []):
            if not idx_list:
                zmins.append(float('inf')); continue
            vals = []
            for i in idx_list:
                r = scan.ranges[i]
                if math.isfinite(r) and (scan.range_min <= r <= scan.range_max):
                    vals.append(min(r, self.lidar_max_range))
            zmins.append(min(vals) if vals else float('inf'))
        self._zone_mins = zmins
    
    def _update_zone_mins_from_env_state(self):
        """
        환경 상태 벡터(self.environment_state)를 기반으로 존별 최소거리(self._zone_mins)를 계산.
        LaserScan / PointCloud 어느 입력이든 공통으로 사용하기 위해,
        env_state를 '가짜 LaserScan'으로 감싸서 기존 _update_zone_mins_simple() 로직을 재활용한다.
        """
        # 존 충돌 기능을 안 쓰면 바로 종료
        if not getattr(self, "use_zone_collision", False):
            self._zone_mins = None
            return

        # env_state가 아직 준비되지 않았으면 스킵
        if self.environment_state is None or len(self.environment_state) != self.environment_dim:
            self._zone_mins = None
            return

        # bins 경계로부터 빔 중심 각도 / 간격을 근사
        try:
            width = float(self.bins[0][1] - self.bins[0][0])   # 각 bin 폭 (rad)
            ang0  = float(self.bins[0][0] + 0.5 * width)       # 첫 번째 빔 중심각
        except Exception:
            # bins 설정이 이상하면 존 충돌 비활성화
            self._zone_mins = None
            return

        from types import SimpleNamespace
        fake_scan = SimpleNamespace(
            angle_min       = ang0,
            angle_increment = width,
            range_min       = 0.0,
            range_max       = float(self.lidar_max_range),
            ranges          = list(self.environment_state),
        )

        # env_state 기준으로 존 인덱스/최소값 다시 계산
        self._zone_indices = None  # 강제로 재계산
        self._update_zone_mins_simple(fake_scan)

    def _fmt_arr(self, arr):
        import numpy as np
        try:
            a = np.asarray(arr, dtype=float)
            return np.array2string(a, precision=3, suppress_small=True)
        except Exception:
            return str(arr)

    def _check_lengths(self):
        msgs = []
        try:
            if len(self.actions_low) != self.action_dim:
                msgs.append(f"⚠ actions_low length {len(self.actions_low)} != action_dim {self.action_dim}")
            if len(self.actions_high) != self.action_dim:
                msgs.append(f"⚠ actions_high length {len(self.actions_high)} != action_dim {self.action_dim}")
        except Exception as e:
            msgs.append(f"⚠ actions length check failed: {e}")

        try:
            n_angles = len(self.zone_angles_deg)
            n_thr    = len(self.zone_thresholds)
            # 허용 모드:
            #  (A) 경계 N→존 N-1  (레거시 5존)
            #  (B) (a,b) 쌍 2N → 존 N (새 8존)
            if not (n_angles == n_thr + 1 or n_angles == 2 * n_thr):
                msgs.append(f"⚠ zone_angles_deg({n_angles}) should be (zone_thresholds+1) or (2*zone_thresholds).")
        except Exception as e:
            msgs.append(f"⚠ zone arrays check failed: {e}")

        return msgs

    def _debug_dump_params_once(self):
        """Prints a clear, one-shot debug dump of YAML vs. effective params."""
        # YAML 원본 섹션들(없으면 {})
        cfg = getattr(self, "config", {}) or {}
        env = dict(cfg.get("environment", {}))
        thr = dict(cfg.get("threshold_parameters", {}))

        self.get_logger().info("========== [ENVIRONMENT CONFIG DEBUG DUMP] ==========")
        # 파일 경로(있으면)
        try:
            # env_config_file_path은 네 코드에서 지역변수였으니, 가져올 수 있으면 출력
            # 못 가져오면 skip
            self.get_logger().info(f"YAML file loaded OK (see previous 'Using config:' line)")
        except Exception:
            pass

        # --- YAML에서 읽은 값 (원본) ---
        self.get_logger().info("[YAML] environment:")
        self.get_logger().info(f"  lower/upper           : {env.get('lower')} / {env.get('upper')}")
        self.get_logger().info(f"  dims(state/agent/act) : {env.get('environment_state_dim')} / {env.get('agent_state_dim')} / {env.get('action_dim')}")
        self.get_logger().info(f"  agent_name            : {env.get('agent_name')}")
        self.get_logger().info(f"  num_of_static_obstacles : {env.get('num_of_static_obstacles')}")
        self.get_logger().info(f"  max_action            : {env.get('max_action')}")
        self.get_logger().info(f"  actions_low/high      : {env.get('actions_low')} / {env.get('actions_high')}")
        self.get_logger().info(
            f"  vehicle wb/steer/minv : {env.get('vehicle_wheelbase_m')} / "
            f"{env.get('vehicle_steering_limit_deg')} / "
            f"{env.get('vehicle_min_speed_for_steering_mps')}"
        )

        self.get_logger().info("[YAML] threshold_parameters:")
        self.get_logger().info(f"  goal_threshold        : {thr.get('goal_threshold')}")
        self.get_logger().info(f"  collision_threshold   : {thr.get('collision_threshold')}")
        self.get_logger().info(f"  time_delta            : {thr.get('time_delta')}")
        self.get_logger().info(f"  inter_entity_distance : {thr.get('inter_entity_distance')}")
        self.get_logger().info(f"  lidar_max_range       : {thr.get('lidar_max_range')}")

        self.get_logger().info("[YAML] zones (top-level):")
        self.get_logger().info(f"  use_zone_collision    : {cfg.get('use_zone_collision')}")
        self.get_logger().info(f"  zone_angles_deg       : {cfg.get('zone_angles_deg')}")
        self.get_logger().info(f"  zone_thresholds       : {cfg.get('zone_thresholds')}")

        # --- 최종 적용값 (YAML + ROS 파라미터 반영 후) ---
        self.get_logger().info("-----------------------------------------------------")
        self.get_logger().info("[EFFECTIVE] Scalars:")
        self.get_logger().info(f"  lower/upper           : {self.lower} / {self.upper}  (type: {type(self.lower).__name__}/{type(self.upper).__name__})")
        self.get_logger().info(f"  dims(state/agent/act) : {self.environment_dim} / {self.agent_dim} / {self.action_dim}")
        self.get_logger().info(f"  agent_name            : {self.agent_name}")
        self.get_logger().info(f"  num_of_static_obstacles : {self.num_of_static_obstacles}")
        self.get_logger().info(f"  max_action            : {self.max_action}")
        self.get_logger().info(f"  goal/collision thr    : {self.goal_threshold} / {self.collision_threshold}")
        self.get_logger().info(f"  dt / inter_d / lidar  : {self.time_delta} / {self.inter_entity_distance} / {self.lidar_max_range}")
        self.get_logger().info(
            f"  vehicle wb/steer/minv : {self.vehicle_wheelbase_m} / "
            f"{self.vehicle_steering_limit_deg} / "
            f"{self.vehicle_min_speed_for_steering_mps}"
        )

        self.get_logger().info("[EFFECTIVE] Arrays:")
        self.get_logger().info(f"  actions_low           : {self._fmt_arr(self.actions_low)}  (len={len(self.actions_low) if hasattr(self.actions_low,'__len__') else 'n/a'})")
        self.get_logger().info(f"  actions_high          : {self._fmt_arr(self.actions_high)} (len={len(self.actions_high) if hasattr(self.actions_high,'__len__') else 'n/a'})")
        self.get_logger().info(f"  zone_angles_deg       : {self.zone_angles_deg} (len={len(self.zone_angles_deg) if hasattr(self.zone_angles_deg,'__len__') else 'n/a'})")
        self.get_logger().info(f"  zone_thresholds       : {self.zone_thresholds} (len={len(self.zone_thresholds) if hasattr(self.zone_thresholds,'__len__') else 'n/a'})")
        self.get_logger().info(f"  use_zone_collision    : {self.use_zone_collision}")

        # --- 토픽 설정도 함께 표시 (헷갈리는 경우가 많아서) ---
        try:
            cmd_vel_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
            odom_topic    = self.get_parameter("odom_topic").get_parameter_value().string_value
            scan_topic_   = self.get_parameter("scan_topic").get_parameter_value().string_value
            obs_source_   = self.get_parameter("obs_source").get_parameter_value().string_value
            contact_topic_ = self.get_parameter("contact_topic").get_parameter_value().string_value
            use_contact_collision_ = self.get_parameter("use_contact_collision").get_parameter_value().bool_value
        except Exception:
            cmd_vel_topic = "/cmd_vel"
            odom_topic    = "/odometry"
            scan_topic_   = "/scan"
            obs_source_   = "scan"
            contact_topic_ = "/hunter_se/chassis_contacts"
            use_contact_collision_ = False

        self.get_logger().info("[TOPICS]")
        self.get_logger().info(f"  cmd_vel_topic         : {cmd_vel_topic}")
        self.get_logger().info(f"  odom_topic            : {odom_topic}")
        self.get_logger().info(f"  obs_source            : {obs_source_}")
        self.get_logger().info(f"  scan_topic            : {scan_topic_}  (← actual subscription)")
        self.get_logger().info(f"  use_contact_collision : {use_contact_collision_}")
        self.get_logger().info(f"  contact_topic         : {contact_topic_}")
        self.get_logger().info("-----------------------------------------------------")


        # --- 간단한 일관성/유효성 검사 ---
        issues = self._check_lengths()
        if issues:
            for m in issues:
                self.get_logger().warn(m)
        else:
            self.get_logger().info("Sanity checks: OK")

        self.get_logger().info("=====================================================")

    def _map_action_to_waypoint(self, action):
        """
        action: shape (2,) in [-1, 1]
        action[0] → waypoint distance r [actions_low[0], actions_high[0]] m
        action[1] → waypoint angle theta [actions_low[1], actions_high[1]] rad
                    (robot frame: positive = left/CCW)
        returns: (r [m], theta [rad], x_wp [m], y_wp [m])
          x_wp = r * cos(theta)  (forward in robot frame)
          y_wp = r * sin(theta)  (left in robot frame)
        """
        return pure_pursuit.action_to_waypoint(action, self.actions_low, self.actions_high)

    def _controller_waypoint_to_command(self, x_wp, y_wp):
        """
        Pure Pursuit: robot-frame waypoint → (speed [m/s], steering [rad]).
        x_wp: forward component [m], y_wp: lateral component (positive = left) [m].
        steering: center steering angle, clipped to vehicle_steering_limit_rad.
        speed: reduced for tighter turns (controller_speed_steer_factor).
        """
        return pure_pursuit.waypoint_to_command(
            x_wp, y_wp,
            self.vehicle_wheelbase_m,
            self.vehicle_steering_limit_rad,
            self.controller_cruise_speed_mps,
            self.controller_min_speed_mps,
            self.controller_speed_steer_factor,
        )
    
    def terminate_session(self):
        """Destroy the node and shut down rclpy when done"""
        self.get_logger().info("gym_node shutting down...")
        self.destroy_node()

    def seed_callback(self, request, response):
        """Sets environment seed for reproducibility of the training process."""
        np.random.seed(request.seed)
        self._rotate_debug_csv()
        response.success = True
        return response

    def sample_action_callback(self, _, response):
        """Samples an action from the action space.

        Returns actions in the normalized policy space [-1, 1] for each dimension.
        _map_action_to_waypoint() then maps [-1, 1] → [r, theta] waypoint.
        This is consistent with the action range that policy agents output and
        with EnvInterface.step() which no longer remaps the first dimension.
        """
        action = np.random.uniform(-1.0, 1.0, size=self.action_dim)
        response.action = np.array(action, dtype=np.float32).tolist()
        return response

    def get_dimensions_callback(self, _, response):
        """Returns the dimensions of the state, action, and maximum action value"""
        response.state_dim = self.environment_dim + self.agent_dim
        response.action_dim = self.action_dim
        response.max_action = self.max_action
        response.environment_dim = self.environment_dim
        response.agent_dim = self.agent_dim
        return response

    def update_environment_state_from_cloud(self, cloud_msg):
        """Updates environment state using 360° LiDAR PointCloud2 data.

        Fills two separate arrays per point in one pass:
        - self.environment_state: full 360° bins (for collision detection)
        - self.obs_state: front 180° bins (for RL observation input only)
        """
        with self.environment_state_lock:
            self.environment_state = (
                np.ones(self.environment_dim, dtype=float) * self.lidar_max_range
            )
            self.obs_state = np.ones(self.environment_dim, dtype=float) * self.lidar_max_range

            data = list(
                pc2.read_points(
                    cloud_msg, skip_nans=False, field_names=("x", "y", "z")
                )
            )

            for x, y, z in data:
                if self.obs_z_min_sensor_m <= z <= self.obs_z_max_sensor_m:
                    beta = math.atan2(y, x)
                    dist = math.sqrt(x * x + y * y + z * z)
                    dist = min(dist, self.lidar_max_range)

                    # Full 360° bins: collision detection
                    for j in range(len(self.bins)):
                        if self.bins[j][0] <= beta < self.bins[j][1]:
                            if dist < self.environment_state[j]:
                                self.environment_state[j] = dist
                            break

                    # Front 180° bins: RL observation input only
                    for j in range(len(self.obs_bins)):
                        if self.obs_bins[j][0] <= beta < self.obs_bins[j][1]:
                            if dist < self.obs_state[j]:
                                self.obs_state[j] = dist
                            break

            try:
                self._update_zone_mins_from_env_state()
            except Exception as e:
                self.get_logger().warn(f"zone mins update (cloud) failed: {e}")
            self.scan_update_count += 1


    def update_environment_state_from_scan(self, scan):
        """Updates environment state using LaserScan data (from pointcloud_to_laserscan)

        Fills two separate arrays per beam in one pass:
        - self.environment_state: full 360° bins (for collision detection)
        - self.obs_state: front 180° bins (for RL observation input only)
        """
        with self.environment_state_lock:
            self.environment_state = np.ones(self.environment_dim) * self.lidar_max_range
            self.obs_state = np.ones(self.environment_dim) * self.lidar_max_range

            self._angle_min = float(scan.angle_min)
            self._angle_max = float(scan.angle_max)
            self._angle_inc = float(scan.angle_increment)

            angle = scan.angle_min
            inc = scan.angle_increment

            for r in scan.ranges:
                if not math.isfinite(r) or r < scan.range_min or r > scan.range_max:
                    angle += inc
                    continue

                beta = angle
                dist = min(r, self.lidar_max_range)

                # Full 360° bins: collision detection
                for j in range(len(self.bins)):
                    if self.bins[j][0] <= beta < self.bins[j][1]:
                        if dist < self.environment_state[j]:
                            self.environment_state[j] = dist
                        break

                # Front 180° bins: RL observation input only
                for j in range(len(self.obs_bins)):
                    if self.obs_bins[j][0] <= beta < self.obs_bins[j][1]:
                        if dist < self.obs_state[j]:
                            self.obs_state[j] = dist
                        break

                angle += inc

            try:
                self._update_zone_mins_from_env_state()
            except Exception as e:
                self.get_logger().warn(f"zone mins update (scan) failed: {e}")
            self.scan_update_count += 1

    def get_environment_state(self):
        """Returns a copy of the full 360° environment state (for collision detection)."""
        with self.environment_state_lock:
            if self.environment_state is None:
                return np.ones(self.environment_dim, dtype=float) * self.lidar_max_range
            return self.environment_state.copy()

    def _human_obs_bin_mask(self, obs: np.ndarray) -> np.ndarray:
        """Return a boolean mask of obs_bins whose returns likely originate from a human proxy.

        A bin is marked True only when BOTH conditions hold:
          1. Bearing: the human's bearing in the robot local frame falls in the bin's
             angular range (±1 neighbour bin on each side).
          2. Range: the scan return in that bin is within the expected distance window
             [human_dist - human_radius - margin, human_dist + human_radius + margin].
             If a wall or static obstacle is closer (occluding the human), its return
             will be much shorter than human_dist and the bin will NOT be marked.

        This avoids overt-contaminating non-human returns that happen to share a bearing
        with an active human proxy (occlusion case).
        """
        n = self.environment_dim
        mask = np.zeros(n, dtype=bool)
        if not self.human_states:
            return mask

        rx, ry, ryaw = self.latest_odom_x, self.latest_odom_y, self.latest_odom_yaw
        obs_low  = self.obs_bins[0][0]
        obs_high = self.obs_bins[-1][1]
        bin_width = (obs_high - obs_low) / n
        # Extra range margin beyond the physical proxy radius to absorb odometry drift
        # and the gap between proxy centre and its LiDAR-facing surface.
        range_margin = 0.3  # m

        for state in self.human_states.values():
            hx, hy = state["x"], state["y"]
            human_dist   = math.hypot(hx - rx, hy - ry)
            human_radius = float(state.get("radius", 0.30))
            dist_tol     = human_radius + range_margin

            # World-frame bearing → robot local frame, wrapped to [-π, π]
            world_angle = math.atan2(hy - ry, hx - rx)
            local_angle = (world_angle - ryaw + math.pi) % (2 * math.pi) - math.pi

            # Skip humans outside the front 180° obs window
            if local_angle < obs_low or local_angle >= obs_high:
                continue

            # Centre bin index for this human
            idx = int((local_angle - obs_low) / bin_width)
            idx = max(0, min(n - 1, idx))

            # Mark ±1 bins only when the scan reading is in the human's range window
            for di in (-1, 0, 1):
                j = idx + di
                if 0 <= j < n and abs(obs[j] - human_dist) <= dist_tol:
                    mask[j] = True

        return mask

    def get_obs_state(self):
        """Returns a copy of the front 180° observation state (for RL input only).

        When human proxies are active in train mode, applies noise and/or dropout ONLY
        to bins whose scan return is close to an active human proxy's distance
        (bearing AND range both match). Noise and dropout are independently controlled.
        """
        with self.environment_state_lock:
            if self.obs_state is None:
                return np.ones(self.environment_dim, dtype=float) * self.lidar_max_range
            obs = self.obs_state.copy()

        if self.train_mode and self.human_states:
            do_noise   = self.human_scan_noise_std   > 0.0
            do_dropout = self.human_scan_dropout_prob > 0.0
            if do_noise or do_dropout:
                human_mask = self._human_obs_bin_mask(obs)
                if human_mask.any():
                    if do_noise:
                        noise = np.random.normal(0.0, self.human_scan_noise_std, obs.shape)
                        obs[human_mask] = np.clip(
                            obs[human_mask] + noise[human_mask], 0.05, self.lidar_max_range
                        )
                    if do_dropout:
                        drop = np.random.rand(human_mask.sum()) < self.human_scan_dropout_prob
                        human_indices = np.where(human_mask)[0]
                        obs[human_indices[drop]] = self.lidar_max_range

        return obs

    @staticmethod
    def _odom_xyyaw(odom):
        """Extract (x, y, yaw[-pi,pi]) from an Odometry message."""
        x = float(odom.pose.pose.position.x)
        y = float(odom.pose.pose.position.y)
        q = Quaternion(
            odom.pose.pose.orientation.w,
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
        )
        return x, y, q.to_euler(degrees=False)[2]

    def _goal_metrics(self, x, y, yaw):
        """Goal-relative (distance [m], heading error [-pi,pi]) from a pose."""
        dx = self.goal_x - x
        dy = self.goal_y - y
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return 0.0, 0.0
        theta = math.atan2(dy, dx) - yaw
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        return dist, theta

    def _on_odom(self, odom, roles):
        """Single odom callback shared by gt / loc / proprio roles.

        Roles determine which caches a message feeds (a single /odometry topic
        with all three roles reproduces the original behaviour). The policy
        observation (agent_state[0:2]) is built from the *raw localization* pose;
        the per-step localization-noise emulator is applied later in
        step_callback so it advances at the RL rate, not the odom rate.
        """
        x, y, yaw = self._odom_xyyaw(odom)
        if "gt" in roles:
            self.gt_x, self.gt_y, self.gt_yaw = x, y, yaw
            # Collision geometry / path use ground truth.
            self.latest_odom_x, self.latest_odom_y, self.latest_odom_yaw = x, y, yaw
            self._append_pose_to_path(odom)
        if "loc" in roles:
            self.loc_raw_x, self.loc_raw_y, self.loc_raw_yaw = x, y, yaw
        if "proprio" in roles:
            vx = float(odom.twist.twist.linear.x)
            vy = float(odom.twist.twist.linear.y)
            self.latest_actual_signed_speed = vx
            self.latest_actual_speed = math.hypot(vx, vy)
            self.latest_actual_yaw_rate = float(odom.twist.twist.angular.z)
        for _role in roles:
            self._odom_role_count[_role] += 1
        self._rebuild_agent_state()
        self.odom_update_count += 1

    def _rebuild_agent_state(self):
        """Rebuild agent_state from caches (goal obs from raw localization pose).

        Slots 2,3 (previous action) are filled in step_callback; slots 0,1 are
        overwritten there with the localization-noise-emulated observation.
        """
        dist, theta = self._goal_metrics(self.loc_raw_x, self.loc_raw_y, self.loc_raw_yaw)
        with self.agent_state_lock:
            self.agent_state = np.array(
                [
                    dist,
                    theta,
                    0.0,
                    0.0,
                    self.latest_actual_speed,
                    self.latest_actual_yaw_rate,
                    self.latest_center_steering,
                ],
                dtype=float,
            )

    # ------------------------------------------------------------------ #
    #  Localization-error emulator (sim-to-real)                            #
    # ------------------------------------------------------------------ #

    def _loc_map_multiplier(self):
        """Return per-map-type localization multipliers for the CURRENT episode's
        map_type from loc_noise['map_type_multipliers'].

        Returns (m_sigma_xy, m_sigma_yaw, m_drift, m_jump, along_axis, along_extra).
        Safe fallback (all 1.0, no anisotropy) for an empty / legacy / unmapped
        map_type, so non-structured runs are unaffected.

        `along_axis` ("x"/"y") + `along_extra` give anisotropy: the corridor runs
        along WORLD-X in _build_map_layouts (free band |y|<=w/2, full x), so a
        corridor entry uses along_axis: x to blow up the ALONG-corridor position
        (and drift) uncertainty — the real aperture-problem failure mode — while
        the cross-corridor axis keeps the base multiplier."""
        mt = getattr(self, "current_map_type", "") or ""
        table = self.loc_noise.get("map_type_multipliers", {}) or {}
        e = table.get(mt, {}) if isinstance(table, dict) else {}
        if not isinstance(e, dict):
            e = {}
        return (
            float(e.get("sigma_xy", 1.0)),
            float(e.get("sigma_yaw", 1.0)),
            float(e.get("drift", 1.0)),
            float(e.get("jump", 1.0)),
            str(e.get("along_axis", "") or ""),
            float(e.get("along_extra", 1.0)),
        )

    def _reset_localization(self, x0, y0, yaw0):
        """Reset ALL per-episode localization-noise state and seed the latency
        buffer / estimate with the (bias-applied) initial pose.

        Seeding with the episode bias — not the clean pose — means the reset
        observation already matches the noise model, so the policy does not jump
        from a clean initial observation to a biased one on the first step. The
        zero-mean correlated (OU) error and the drift start from step 1, seeded at
        zero. The per-episode EFFECTIVE sigmas / drifts (base × current map-type
        multiplier + corridor anisotropy) and the OU memory factors are computed
        here, since current_map_type is set by _select_episode_layout() earlier in
        the reset. Disabled → clean passthrough."""
        n = self.loc_noise
        x0, y0, yaw0 = float(x0), float(y0), float(yaw0)
        # Always reset the OU state so correlated error never leaks across episodes.
        self._loc_ou_x = self._loc_ou_y = self._loc_ou_yaw = 0.0
        if not n["enabled"]:
            self._loc_bias_x = self._loc_bias_y = self._loc_bias_yaw = 0.0
            self._loc_rw_x = self._loc_rw_y = self._loc_rw_yaw = 0.0
            self._loc_buf = deque([(x0, y0, yaw0)], maxlen=1)
            self.loc_est_x, self.loc_est_y, self.loc_est_yaw = x0, y0, yaw0
            return
        # Per-episode bias (constant registration offset) — part of the goal-obs
        # corruption axis, so only sampled when that axis is enabled.
        _goal_on = n["noise_goal_enabled"]
        self._loc_bias_x = float(np.random.normal(0.0, n["bias_xy_m"]))  if (_goal_on and n["bias_xy_m"]  > 0.0) else 0.0
        self._loc_bias_y = float(np.random.normal(0.0, n["bias_xy_m"]))  if (_goal_on and n["bias_xy_m"]  > 0.0) else 0.0
        self._loc_bias_yaw = float(np.random.normal(0.0, n["bias_yaw_rad"])) if (_goal_on and n["bias_yaw_rad"] > 0.0) else 0.0
        self._loc_rw_x = self._loc_rw_y = self._loc_rw_yaw = 0.0

        # Effective per-axis sigma / drift = base × map-type multiplier (+ corridor
        # anisotropy on the along-corridor axis).
        m_sxy, m_syaw, m_drift, m_jump, along_axis, along_extra = self._loc_map_multiplier()
        # Resolve the per-second drift intensity at READ time, preferring the
        # physical-name key `drift_*` when present (curriculum profiles set it as
        # a raw key via deep-merge, which bypasses the __init__ alias resolution)
        # and falling back to the legacy `random_walk_*` key otherwise.
        rate_xy  = float(n["drift_xy_mps"])    if "drift_xy_mps"    in n else float(n["random_walk_xy_mps"])
        rate_yaw = float(n["drift_yaw_radps"]) if "drift_yaw_radps" in n else float(n["random_walk_yaw_rps"])
        sx = n["sigma_xy_m"] * m_sxy
        sy = n["sigma_xy_m"] * m_sxy
        dx = rate_xy * m_drift
        dy = rate_xy * m_drift
        if along_axis == "x":
            sx *= along_extra; dx *= along_extra
        elif along_axis == "y":
            sy *= along_extra; dy *= along_extra
        dt = self.time_delta
        # OU memory factor alpha = exp(-dt/tau); tau=0 → alpha=0 → white noise
        # of std sigma (legacy behaviour). Stationary std of the OU process = sigma.
        ax   = math.exp(-dt / n["corr_time_xy_s"])  if n["corr_time_xy_s"]  > 0.0 else 0.0
        ayaw = math.exp(-dt / n["corr_time_yaw_s"]) if n["corr_time_yaw_s"] > 0.0 else 0.0
        self._loc_eff = {
            "sigma_x": sx, "sigma_y": sy, "sigma_yaw": n["sigma_yaw_rad"] * m_syaw,
            "drift_x": dx, "drift_y": dy, "drift_yaw": rate_yaw * m_drift,
            "alpha_xy": ax, "alpha_yaw": ayaw, "jump_mult": m_jump,
        }

        ix = x0 + self._loc_bias_x
        iy = y0 + self._loc_bias_y
        iyaw = (yaw0 + self._loc_bias_yaw + math.pi) % (2 * math.pi) - math.pi
        # Latency is its own ablation axis (noise_delay_enabled).
        delay = int(n["delay_steps"]) if n["noise_delay_enabled"] else 0
        maxlen = max(1, delay + 1)
        self._loc_buf = deque([(ix, iy, iyaw)] * maxlen, maxlen=maxlen)
        self.loc_est_x, self.loc_est_y, self.loc_est_yaw = ix, iy, iyaw

    def _loc_emulator_step(self, x, y, yaw):
        """Advance the localization-noise emulator ONE RL step and return the
        (delayed, noisy) estimated pose. No-op (passthrough) when disabled.

        Error = bias + accumulated drift + time-correlated OU measurement error,
        plus rare jumps. The OU term reduces to legacy white Gaussian when the
        correlation time is 0 (alpha=0). Effective per-axis magnitudes come from
        self._loc_eff (base × map-type multiplier + corridor anisotropy)."""
        n = self.loc_noise
        if not n["enabled"]:
            return x, y, yaw
        dt = self.time_delta
        eff = self._loc_eff
        sqrt_dt = math.sqrt(dt)

        ex, ey, eyaw = x, y, yaw

        # ── Goal-obs corruption axis (bias + OU + drift) ── gated by noise_goal_enabled.
        if n["noise_goal_enabled"]:
            # Time-correlated (OU) measurement error: e <- alpha*e + sqrt(1-alpha^2)*sigma*N.
            # alpha=0 ⇒ e = sigma*N(0,1) (identical to the previous per-step white noise).
            ax, ayaw = eff["alpha_xy"], eff["alpha_yaw"]
            bx = math.sqrt(max(0.0, 1.0 - ax * ax))
            byaw = math.sqrt(max(0.0, 1.0 - ayaw * ayaw))
            self._loc_ou_x   = ax * self._loc_ou_x   + bx   * eff["sigma_x"]   * float(np.random.normal())
            self._loc_ou_y   = ax * self._loc_ou_y   + bx   * eff["sigma_y"]   * float(np.random.normal())
            self._loc_ou_yaw = ayaw * self._loc_ou_yaw + byaw * eff["sigma_yaw"] * float(np.random.normal())

            # Random-walk drift — per-second intensity (Brownian √dt scaling so the
            # accumulated std ≈ rate·√t, matching the *_mps / *_rps unit names).
            self._loc_rw_x   += float(np.random.normal(0.0, eff["drift_x"]   * sqrt_dt))
            self._loc_rw_y   += float(np.random.normal(0.0, eff["drift_y"]   * sqrt_dt))
            self._loc_rw_yaw += float(np.random.normal(0.0, eff["drift_yaw"] * sqrt_dt))

            # Episode bias + accumulated drift + correlated measurement error.
            ex   += self._loc_bias_x   + self._loc_rw_x   + self._loc_ou_x
            ey   += self._loc_bias_y   + self._loc_rw_y   + self._loc_ou_y
            eyaw += self._loc_bias_yaw + self._loc_rw_yaw + self._loc_ou_yaw

        # ── Jump axis ── rare relocalization snaps, gated by noise_jump_enabled.
        # A large failure (big_jump) takes precedence over the small snap; both
        # scale with the map-type jump multiplier.
        if n["noise_jump_enabled"]:
            jm = eff["jump_mult"]
            r = np.random.rand()
            bp = n["big_jump_prob"]
            sp = n["jump_prob"]
            if bp > 0.0 and r < bp:
                ex   += float(np.random.uniform(-n["big_jump_xy_m"],  n["big_jump_xy_m"]))  * jm
                ey   += float(np.random.uniform(-n["big_jump_xy_m"],  n["big_jump_xy_m"]))  * jm
                eyaw += float(np.random.uniform(-n["big_jump_yaw_rad"], n["big_jump_yaw_rad"])) * jm
            elif sp > 0.0 and r < bp + sp:
                ex   += float(np.random.uniform(-n["jump_xy_m"],  n["jump_xy_m"]))  * jm
                ey   += float(np.random.uniform(-n["jump_xy_m"],  n["jump_xy_m"]))  * jm
                eyaw += float(np.random.uniform(-n["jump_yaw_rad"], n["jump_yaw_rad"])) * jm

        # ── Yaw-flip axis (STRESS-TEST) ── ±π mirror relocalization in symmetric
        # maps (corridor). OFF unless noise_flip_enabled — never in normal training.
        if (n["noise_flip_enabled"]
                and n["yaw_flip_prob"] > 0.0
                and (getattr(self, "current_map_type", "") or "") in n["yaw_flip_map_types"]
                and np.random.rand() < n["yaw_flip_prob"]):
            eyaw += math.pi

        eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi
        # Latency buffer: return the pose delay_steps ago.
        self._loc_buf.append((ex, ey, eyaw))
        return self._loc_buf[0]

    # ------------------------------------------------------------------ #
    #  Proprioception observation noise (separate axis from goal-obs)       #
    # ------------------------------------------------------------------ #
    def _reset_proprio_noise(self, speed0=0.0, yaw_rate0=0.0, steer0=0.0):
        """Reset per-episode proprio-noise state (bias + scale error sampled per
        episode) and SEED the latency buffer with a FULL initial noisy sample of
        the proprio, so the reset observation already reflects the noise model on
        EVERY axis — including steering, which has only a per-step Gaussian (no
        bias/scale). This removes the clean→noisy jump on the first step for all
        three proprio axes (speed / yaw-rate / steering). No-op when disabled."""
        p = self.proprio_noise
        speed0, yaw_rate0, steer0 = float(speed0), float(yaw_rate0), float(steer0)
        if not p["enabled"]:
            self._pp_speed_bias = self._pp_yaw_bias = 0.0
            self._pp_speed_scale = 1.0
            self._pp_buf = deque([(speed0, yaw_rate0, steer0)], maxlen=1)
            return
        self._pp_speed_bias = float(np.random.normal(0.0, p["speed_bias_mps"]))    if p["speed_bias_mps"]   > 0.0 else 0.0
        self._pp_yaw_bias   = float(np.random.normal(0.0, p["yaw_rate_bias_radps"])) if p["yaw_rate_bias_radps"] > 0.0 else 0.0
        self._pp_speed_scale = 1.0 + (float(np.random.normal(0.0, p["speed_scale_sigma"])) if p["speed_scale_sigma"] > 0.0 else 0.0)
        # Seed = one full noise sample (bias + scale + per-step Gaussian) on each
        # axis, so steering (Gaussian-only) is seeded noisy too — not left clean.
        seed_speed = self._pp_speed_scale * speed0 + self._pp_speed_bias
        if p["speed_sigma_mps"] > 0.0:
            seed_speed += float(np.random.normal(0.0, p["speed_sigma_mps"]))
        seed_yaw = yaw_rate0 + self._pp_yaw_bias
        if p["yaw_rate_sigma_radps"] > 0.0:
            seed_yaw += float(np.random.normal(0.0, p["yaw_rate_sigma_radps"]))
        seed_steer = steer0
        if p["steer_sigma_rad"] > 0.0:
            seed_steer += float(np.random.normal(0.0, p["steer_sigma_rad"]))
        seed = (seed_speed, seed_yaw, seed_steer)
        maxlen = max(1, int(p["delay_steps"]) + 1)
        self._pp_buf = deque([seed] * maxlen, maxlen=maxlen)

    def _proprio_emulator_step(self, speed, yaw_rate, steer):
        """Return the noisy (speed, yaw_rate, steer) proprio OBSERVATION. Affects
        ONLY the policy observation slots state[84:87] — the ground-truth caches
        used for reward / done / collision are untouched. No-op when disabled."""
        p = self.proprio_noise
        if not p["enabled"]:
            return speed, yaw_rate, steer
        es = self._pp_speed_scale * float(speed) + self._pp_speed_bias
        if p["speed_sigma_mps"] > 0.0:
            es += float(np.random.normal(0.0, p["speed_sigma_mps"]))
        ew = float(yaw_rate) + self._pp_yaw_bias
        if p["yaw_rate_sigma_radps"] > 0.0:
            ew += float(np.random.normal(0.0, p["yaw_rate_sigma_radps"]))
        est = float(steer)
        if p["steer_sigma_rad"] > 0.0:
            est += float(np.random.normal(0.0, p["steer_sigma_rad"]))
        # Latency buffer (separate from the goal-obs delay).
        self._pp_buf.append((es, ew, est))
        return self._pp_buf[0]

    def get_agent_state(self):
        """Return a copy of the agent state"""
        with self.agent_state_lock:
            if self.agent_state is None:
                return np.array(
                    [np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float
                )
            return self.agent_state.copy()

    def _update_filtered_cmd(self, msg: Twist):
        """Track the prefilter output to compare cmd -> filtered -> odom."""
        self.latest_filtered_cmd_v = float(msg.linear.x)
        self.latest_filtered_cmd_w = float(msg.angular.z)

    def _update_steering_joints(self, msg: JointState):
        """Track realized front steering joint angles from joint_states."""
        try:
            left_index = msg.name.index("front_left_steering")
            right_index = msg.name.index("front_right_steering")
        except ValueError:
            return

        try:
            left = float(msg.position[left_index])
            right = float(msg.position[right_index])
        except (IndexError, TypeError, ValueError):
            return

        self.latest_front_left_steering = left
        self.latest_front_right_steering = right
        self.latest_center_steering = 0.5 * (left + right)

    def _update_contact_collision(self, msg: Contacts):
        """Latch any chassis-contact event for definitive collision termination."""
        if not self.use_contact_collision or not msg.contacts:
            return
        if not self.contact_collision_latched:
            self.contact_event_count += 1
        self.contact_collision_latched = True

    def _init_debug_csv(self):
        """Create a fresh step-by-step execution CSV for the current run."""
        run_dir = os.environ.get("DRL_AGENT_RUN_DIR", "").strip()
        if run_dir:
            base_run_dir = os.path.expanduser(run_dir)
        else:
            package_root = self._resolve_drl_agent_source_root()
            base_run_dir = os.path.join(
                package_root,
                "runtime",
                "tqc",
            )
        self._env_log_dir = os.path.join(base_run_dir, "logs")
        os.makedirs(self._env_log_dir, exist_ok=True)
        self._rotate_debug_csv()

    def _rotate_debug_csv(self):
        """Rotate environment step CSV so each training start gets a new file."""
        csv_run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._env_step_csv = os.path.join(
            self._env_log_dir, f"environment_step_debug_{csv_run_tag}.csv"
        )
        header = [
            "episode", "episode_step",
            "action_0_norm", "action_1_norm",
            "cmd_v_mps", "cmd_w_rads", "cmd_steering_rad", "wp_r_m", "wp_theta_rad",
            "filtered_cmd_v_mps", "filtered_cmd_w_rads",
            "front_left_steering_rad", "front_right_steering_rad", "center_steering_rad",
            "actual_speed_mps", "actual_signed_speed_mps", "actual_yaw_rate_rads",
            "odom_x", "odom_y",
            "goal_dist_m", "theta_err_rad",
            "lidar_min_m", "lidar_mean_m",
            "rect_proximity",
            "reward_delta_goal_m",
            "reward_progress",
            "reward_heading",
            "penalty_curv",
            "penalty_obstacle",
            "penalty_step",
            "penalty_smooth",
            "penalty_wp_smooth",
            "reward_terminal",
            "reward",
            "collision", "target", "done",
        ]
        with open(self._env_step_csv, "w", newline="") as f:
            csv.writer(f).writerow(header)
        self.get_logger().info(f"Environment step CSV: {self._env_step_csv}")

    def _resolve_drl_agent_source_root(self) -> str:
        """Resolve the source-package root even when this script is run from install/."""
        here = os.path.abspath(__file__)
        candidates = []

        src_env = os.environ.get("DRL_AGENT_SRC_PATH", "").strip()
        if src_env:
            src_env = os.path.expanduser(src_env)
            candidates.extend([
                os.path.join(src_env, "drl_agent"),
                os.path.join(src_env, "src", "drl_agent"),
                src_env,
            ])

        if "/install/" in here:
            ws_root = here.split("/install/")[0]
            candidates.append(os.path.join(ws_root, "src", "drl_agent"))

        cwd = os.path.abspath(os.getcwd())
        candidates.extend([
            os.path.join(cwd, "src", "drl_agent"),
            os.path.normpath(os.path.join(os.path.dirname(here), "..", "..")),
        ])

        for cand in candidates:
            if os.path.isdir(cand) and os.path.basename(cand) == "drl_agent":
                return os.path.normpath(cand)

        return os.path.normpath(os.path.join(os.path.dirname(here), "..", ".."))

    def _reset_robot_path(self):
        """Clear the trajectory for the current episode and publish an empty path."""
        with self.path_lock:
            self.robot_path = Path()
            self.robot_path.header.frame_id = "odom"
            self.robot_path.header.stamp = self.get_clock().now().to_msg()
            self.robot_path_pub.publish(self.robot_path)

    def _append_pose_to_path(self, odom: Odometry):
        """Append the current odometry pose to the episode path."""
        pose_stamped = PoseStamped()
        pose_stamped.header = odom.header
        pose_stamped.pose = odom.pose.pose

        with self.path_lock:
            self.robot_path.header = odom.header
            self.robot_path.header.frame_id = odom.header.frame_id or "odom"
            self.robot_path.poses.append(pose_stamped)
            self.robot_path_pub.publish(self.robot_path)
        
    # ----------------------------------------------------------------------------------------------
    # ====================================Ignition Start============================================
    # ----------------------------------------------------------------------------------------------
    def _wait_for_srv(self, client, name: str):
        """공통: 서비스가 뜰 때까지 대기"""
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                f"Service {name} not available, waiting again..."
            )

    def pause_world(self, pause: bool):
        """Ignition 월드 일시정지 / 재개"""
        srv_name = f"/world/{self.world_name}/control"
        self._wait_for_srv(self.world_control, srv_name)

        req = ControlWorld.Request()
        req.world_control.pause = bool(pause)
        try:
            future = self.world_control.call_async(req)
            result = self._await_future(future)
            if result is None:
                self.get_logger().warn(f"{srv_name} (pause={pause}): timed out")
            elif not result.success:
                self.get_logger().warn(f"{srv_name} (pause={pause}): success=false")
        except Exception as e:
            self.get_logger().error(f"{srv_name} service call failed: {e}")
            sys.exit(-1)

    def reset_world(self):
        """Ignition 월드 리셋 (모델만, 시간은 유지)."""
        srv_name = f"/world/{self.world_name}/control"
        self._wait_for_srv(self.world_control, srv_name)

        req = ControlWorld.Request()
        req.world_control.reset.model_only = True
        req.world_control.pause = True
        try:
            future = self.world_control.call_async(req)
            result = self._await_future(future)
            if result is None:
                self.get_logger().warn(f"{srv_name} (reset): timed out")
            elif not result.success:
                self.get_logger().warn(f"{srv_name} (reset): success=false")
        except Exception as e:
            self.get_logger().error(f"{srv_name} (reset) service call failed: {e}")
            sys.exit(-1)

    def _publish_zero_command(self):
        """Stop the robot command stream before teleporting models during reset."""
        self.velocity_command.linear.x = 0.0
        self.velocity_command.linear.y = 0.0
        self.velocity_command.linear.z = 0.0
        self.velocity_command.angular.x = 0.0
        self.velocity_command.angular.y = 0.0
        self.velocity_command.angular.z = 0.0
        self.velocity_publisher.publish(self.velocity_command)

    def _prepare_episode_reset(self):
        """Pause the world and optionally skip the expensive global model reset."""
        self._publish_zero_command()
        self.pause_world(True)
        if self.preserve_hunav_on_reset:
            return
        self.reset_world()
        self.goal_marker_spawned = False

    def set_entity_pose_ignition(self, name, x, y, z, qx, qy, qz, qw):
        """Ignition 월드에서 특정 모델을 텔레포트"""
        srv_name = f"/world/{self.world_name}/set_pose"
        self._wait_for_srv(self.set_entity_pose, srv_name)

        req = SetEntityPose.Request()
        req.entity.name = str(name)
        req.entity.type = GzEntity.MODEL

        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(z)
        req.pose.orientation.x = float(qx)
        req.pose.orientation.y = float(qy)
        req.pose.orientation.z = float(qz)
        req.pose.orientation.w = float(qw)

        try:
            future = self.set_entity_pose.call_async(req)
            result = self._await_future(future)
            if result is None:
                self.get_logger().warn(f"{srv_name} (entity={name}): timed out")
            elif not result.success:
                self.get_logger().warn(f"{srv_name} (entity={name}): success=false")
        except Exception as e:
            self.get_logger().error(f"{srv_name} service call failed: {e}")
            sys.exit(-1)

    def propagate_state(self, time_delta):
        """Ignition 월드를 time_delta초 동안 돌렸다가 다시 pause"""
        # 시뮬레이션 재개
        self.pause_world(False)
        time.sleep(time_delta)
        # 다시 일시정지
        self.pause_world(True)
    # ----------------------------------------------------------------------------------------------
    # ====================================Ignition Finish===========================================
    # ----------------------------------------------------------------------------------------------

    def _append_aux_labels(self, state_array):
        """AUX_PRED: return state_array (list) with the privileged future-risk
        label appended.  When disabled, returns the plain state list so the
        wire format is identical to baseline.

        Robot pose uses the ground-truth pose (privileged); pedestrian motion is
        read from self.human_states (privileged sim state).  Nothing here is
        needed at inference -- the trainer simply stops slicing when labels are
        absent, and the deployed policy is encoder + actor only.
        """
        state_list = np.asarray(state_array, dtype=np.float32).ravel().tolist()
        if not self._aux_pred_enabled:
            return state_list

        try:
            with self._human_lock:
                humans = [
                    {
                        "x": s["x"], "y": s["y"],
                        "yaw": s.get("yaw", 0.0), "v": s.get("v", 0.0),
                    }
                    for s in self.human_states.values()
                ]
            label = aux_labels.compute_future_risk_labels(
                humans,
                (self.gt_x, self.gt_y, self.gt_yaw),
                self._aux_label_cfg,
            )
        except Exception as exc:  # never let label gen break the RL step
            self.get_logger().warn(f"[AUX_PRED] label generation failed: {exc}")
            label = [0.0] * self._aux_label_cfg.label_dim

        # AUX_PRED: prepend the geometry header [VERSION, K, H, h_0..h_{H-1}] so
        # the consumer can verify the EXACT structure (not just total length),
        # then the label.  Always send the header (even on the zero-label
        # fallback) so the trainer can still validate the contract.
        state_list.extend(float(v) for v in self._aux_label_cfg.wire_header())
        state_list.extend(float(v) for v in label)
        return state_list

    def step_callback(self, request, response):
        target = False
        action = request.action  # 정규화 [-1,1]
        self.current_episode_step += 1

        # SIM_VALIDATION: capture the goal observation BEFORE this step's motion
        # and BEFORE the emulator advances (no side effect — reads the current
        # localization estimate). For step 1 this equals the reset observation,
        # so the reset→first-step jump excludes real robot motion and is truly 0
        # when reset-seeding is correct (any value → a clean→noisy regression).
        if self._sim_val is not None:
            self._sim_val_pre_motion = self._goal_metrics(
                self.loc_est_x, self.loc_est_y, self.loc_est_yaw)

        # 1) 액션 → 로컬 웨이포인트 → Pure Pursuit 제어 명령
        r, theta, x_wp, y_wp = self._map_action_to_waypoint(action)
        v, cmd_steering = self._controller_waypoint_to_command(x_wp, y_wp)

        # 2) Twist publish:
        #   linear.x  = speed from Pure Pursuit [m/s]
        #   angular.z = center steering angle [rad]  ← prefilter expects this
        self.velocity_command.linear.x  = v
        self.velocity_command.angular.z = cmd_steering
        self.velocity_publisher.publish(self.velocity_command)

        # Kinematic yaw rate at commanded speed (zero when v=0, for reward/log)
        w_reward = v * math.tan(cmd_steering) / max(self.vehicle_wheelbase_m, 1e-6)

        # (선택) 마커는 정규화 액션 기준 유지
        self.publish_markers(action)

        # 3) 보행자 이동: 별도 20 Hz 타이머(_human_timer_callback)에서 처리

        # 4) 시뮬레이션 진행
        self.propagate_state(self.time_delta)

        # 5) 상태 구성
        # environment_state (360°): 충돌 판정 전용
        # obs_state (전방 180°):    RL 입력 전용
        environment_state = self.get_environment_state()
        obs_state = self.get_obs_state()
        agent_state = self.get_agent_state()
        # agent_state layout:
        #   [0]: goal_dist, [1]: theta_err
        #   [2:4]: previous normalized action (r_norm, theta_norm)
        #   [4]: actual_speed, [5]: actual_yaw_rate, [6]: center_steering
        # Localization-noise emulation: advance one RL step and overwrite the
        # goal observation (slots 0,1) with the noisy/delayed localization
        # estimate. Disabled → passthrough (identical to ground truth).
        _lx, _ly, _lyaw = self._loc_emulator_step(self.loc_raw_x, self.loc_raw_y, self.loc_raw_yaw)
        self.loc_est_x, self.loc_est_y, self.loc_est_yaw = _lx, _ly, _lyaw
        _obs_dist, _obs_theta = self._goal_metrics(_lx, _ly, _lyaw)
        agent_state[0], agent_state[1] = _obs_dist, _obs_theta
        # Proprioception observation noise (separate axis): perturb ONLY the
        # observed speed / yaw-rate / steering slots [4:7]. Reward/done/collision
        # use the ground-truth caches, not these. No-op when disabled.
        _pe = self._proprio_emulator_step(agent_state[4], agent_state[5], agent_state[6])
        agent_state[4], agent_state[5], agent_state[6] = _pe
        # Ground-truth goal metrics for reward / done (default; configurable).
        _gt_dist, _gt_theta = self._goal_metrics(self.gt_x, self.gt_y, self.gt_yaw)

        agent_state[2], agent_state[3] = float(action[0]), float(action[1])
        state = np.append(obs_state, agent_state)

        # 6) 충돌/완료 판단 (full 360° environment_state 사용)
        done, collision, min_used = self.check_collision(environment_state)
        if self.use_contact_collision and self.contact_collision_latched:
            done = True
            collision = True
            min_used = min(min_used, 0.0) if math.isfinite(min_used) else 0.0

        # Pose-based source selection: GT by default (training stability);
        # estimated (loc) selectable via use_gt_for_reward / use_gt_for_done.
        goal_dist_done = _gt_dist if self.loc_noise["use_gt_for_done"] else _obs_dist
        curr_goal_dist = _gt_dist if self.loc_noise["use_gt_for_reward"] else _obs_dist
        _pdist = getattr(self, "_prev_goal_dist", None)
        prev_goal_dist = float(curr_goal_dist if _pdist is None else _pdist)
        theta_err = _gt_theta if self.loc_noise["use_gt_for_reward"] else _obs_theta

        if goal_dist_done < self.goal_threshold:
            self.get_logger().info(f"{'GOAL REACHED':-^50}")
            target = True
            done = True

        # SIM_VALIDATION: per-step localization-validation log (default OFF).
        if self._sim_val is not None:
            self._sim_val.log_step(
                episode=self._episode_count, step=self.current_episode_step,
                stage=int(getattr(self, "_current_stage", -1)),
                obs_goal_dist=_obs_dist, obs_heading_err=_obs_theta,
                gt_goal_dist=_gt_dist, gt_heading_err=_gt_theta,
                reward_goal_dist_used=curr_goal_dist, done_goal_dist_used=goal_dist_done,
                loc_raw=(self.loc_raw_x, self.loc_raw_y, self.loc_raw_yaw),
                loc_est=(self.loc_est_x, self.loc_est_y, self.loc_est_yaw),
                gt=(self.gt_x, self.gt_y, self.gt_yaw),
                role_counts=dict(self._odom_role_count), loc_noise=self.loc_noise,
                pre_motion_obs=self._sim_val_pre_motion,
            )

        # 7) 직사각형 근접도 — 충돌/보상 기하 통일
        rect_proximity = self._compute_rect_proximity(environment_state)
        lidar_min = float(np.min(environment_state)) if len(environment_state) else float("inf")
        lidar_mean = float(np.mean(environment_state)) if len(environment_state) else float("inf")

        # 8) 보상 계산
        # v_max, w_max: Pure Pursuit controller 기준 (actions_low/high는 웨이포인트 범위)
        v_max = self.controller_cruise_speed_mps
        w_max = v_max * math.tan(self.vehicle_steering_limit_rad) / max(self.vehicle_wheelbase_m, 1e-6)
        prev_waypoint_theta = float(getattr(self, "_prev_waypoint_theta", 0.0))

        reward, reward_terms = self.get_reward(
            target, collision,
            v, w_reward,
            prev_goal_dist, curr_goal_dist,
            theta_err=theta_err,
            rect_proximity=rect_proximity,
            min_laser=min_used,
            v_max=v_max, w_max=w_max,
            waypoint_theta=theta,
            prev_waypoint_theta=prev_waypoint_theta,
            return_terms=True,
        )
        self._prev_waypoint_theta = theta
    
        # 9) 다음 스텝 대비 기록
        self._prev_goal_dist = curr_goal_dist
        self._prev_v, self._prev_w = v, w_reward
        with open(self._env_step_csv, "a", newline="") as _f:
            csv.writer(_f).writerow([
                self._episode_count, self.current_episode_step,
                round(float(action[0]), 6), round(float(action[1]), 6),
                round(float(v), 6), round(float(w_reward), 6), round(float(cmd_steering), 6),
                round(float(r), 6), round(float(theta), 6),
                round(float(self.latest_filtered_cmd_v), 6),
                round(float(self.latest_filtered_cmd_w), 6),
                round(float(self.latest_front_left_steering), 6),
                round(float(self.latest_front_right_steering), 6),
                round(float(self.latest_center_steering), 6),
                round(float(self.latest_actual_speed), 6),
                round(float(self.latest_actual_signed_speed), 6),
                round(float(self.latest_actual_yaw_rate), 6),
                round(float(self.latest_odom_x), 6), round(float(self.latest_odom_y), 6),
                round(float(curr_goal_dist), 6),
                round(float(theta_err) if theta_err is not None else 0.0, 6),
                round(lidar_min, 6), round(lidar_mean, 6),
                round(float(rect_proximity), 6),
                round(float(reward_terms["delta_d"]), 6),
                round(float(reward_terms["progress"]), 6),
                round(float(reward_terms["heading"]), 6),
                round(float(reward_terms["curv_pen"]), 6),
                round(float(reward_terms["obstacle"]), 6),
                round(float(reward_terms["step_pen"]), 6),
                round(float(reward_terms["smooth"]), 6),
                round(float(reward_terms["wp_smooth"]), 6),
                round(float(reward_terms["terminal"]), 6),
                round(float(reward), 6),
                int(bool(collision)), int(bool(target)), int(bool(done)),
            ])

        # 10) 응답
        # AUX_PRED: append privileged future-risk label (no-op when disabled).
        response.state  = self._append_aux_labels(state)
        response.reward = float(reward)
        response.done   = bool(done)
        response.target = bool(target)
        return response

    def reset_callback(self, _, response):
        """Resets the state of the environment and returns an initial observation, state"""
        # Stop the obstacle-motion timer and wait for any in-flight iteration to finish.
        # 1) Set the flag so the timer won't enter a new iteration.
        self._human_updates_enabled = False
        # 2) Acquire the lock: if a timer callback is mid-iteration, this blocks
        #    until it releases the lock (i.e. finishes obstacle kinematic updates).
        #    Once we hold the lock, no timer is touching shared obstacle state.
        with self._human_lock:
            self.human_states = {}

        self._episode_count += 1
        self.current_episode_step = 0
        self.contact_collision_latched = False
        # Clear per-episode reward memory so the first step of the new episode
        # does not inherit the last state of the previous episode.
        self._prev_goal_dist   = None
        self._prev_v           = 0.0
        self._prev_w           = 0.0
        self._prev_waypoint_theta = 0.0
        self._reset_robot_path()
        prev_scan_updates = self.scan_update_count
        prev_role_updates = dict(self._odom_role_count)
        with self.environment_state_lock:
            self.environment_state = None
        with self.agent_state_lock:
            self.agent_state = None

        """*****************************************************
        ** Start by resetting Ignition world
        *****************************************************"""
        self._prepare_episode_reset()
        time.sleep(self.time_delta)
        if self.use_obstacle_pool:
            if not self.pool_initialized:
                self._initialize_obstacle_pool()
        else:
            self._delete_spawned_obstacles()

        # Structured map curriculum: pick this episode's map_type and activate
        # its internal walls BEFORE start/goal/obstacle/human sampling so every
        # sampler is layout-aware. No-op when the curriculum is disabled.
        self._select_episode_layout()

        """*****************************************************
		** Determine start positions for the agent
		*****************************************************"""
        if self.train_mode:
            start_x, start_y, angle = self._sample_train_start_pose()
        else:
            if not self.start_goal_pairs:
                self.get_logger().info(f"{'All start-goal pairs are visited':-^50}")
                self.terminate_session()
            self.current_pairs = self.start_goal_pairs.popleft()
            start_x = self.current_pairs["start"]["x"]
            start_y = self.current_pairs["start"]["y"]
            angle = self.current_pairs["start"]["theta"]

        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        # Ignition 월드에서 로봇 모델 텔레포트
        self.set_entity_pose_ignition(
            self.agent_name,
            start_x,
            start_y,
            self.spawn_z,             # environment.yaml spawn_z 값 사용
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w,
        )

        """*****************************************************
		** Change goal and randomize obstacles
		*****************************************************"""
        self.change_goal(start_x, start_y)
        if self.train_mode:
            if self.use_obstacle_pool:
                self._activate_random_obstacles(start_x, start_y)
            else:
                self._spawn_random_obstacles(start_x, start_y)
        # Obstacle motion state is now fully populated — re-enable the timer
        self._human_updates_enabled = True
        # Publish markers for rviz
        self.publish_markers([0.0, 0.0])
        # Propagate state for 2*time_delta seconds
        self.propagate_state(2 * self.time_delta)

        # 첫 "새" 관측이 들어올 때까지 짧게 대기 (최대 1.5초)
        # Require a fresh update from EACH configured odom role (gt/loc/proprio),
        # not just a shared counter — so a dead loc/proprio topic is detected.
        t0 = time.time()
        while (
            (
                self.environment_state is None
                or self.agent_state is None
                or self.scan_update_count <= prev_scan_updates
                or any(self._odom_role_count[r] <= prev_role_updates[r]
                       for r in ("gt", "loc", "proprio"))
            )
            and (time.time() - t0 < 1.5)
        ):
            rclpy.spin_once(self, timeout_sec=0.05)
        _stale_roles = [r for r in ("gt", "loc", "proprio")
                        if self._odom_role_count[r] <= prev_role_updates[r]]
        if _stale_roles:
            self.get_logger().warn(
                f"[reset] odom source(s) did not refresh: {_stale_roles} "
                f"(topics {[self._odom_topics[r] for r in _stale_roles]}) — caches may be stale."
            )

        """*****************************************************
		** Compute state after reset
		*****************************************************"""
        # Reset ALL localization-noise state (bias, random walk, latency buffer,
        # jump) and seed it with the settled initial pose.
        self._reset_localization(self.loc_raw_x, self.loc_raw_y, self.loc_raw_yaw)
        obs_state = self.get_obs_state()
        agent_state = self.get_agent_state()
        # Patch the goal observation (slots 0,1) with the just-reset emulated
        # localization pose so the initial observation matches what step() will
        # produce — no clean→noisy jump on the first step. (Disabled → unchanged.)
        _d, _t = self._goal_metrics(self.loc_est_x, self.loc_est_y, self.loc_est_yaw)
        agent_state[0], agent_state[1] = _d, _t
        # Same for proprioception (slots 4,5,6): seed the proprio-noise buffer from
        # the settled initial proprio and patch the reset observation with that
        # seeded (bias/scale-applied) estimate, so there is no clean→noisy proprio
        # jump on the first step either. (Disabled → unchanged passthrough.)
        self._reset_proprio_noise(agent_state[4], agent_state[5], agent_state[6])
        if self.proprio_noise["enabled"]:
            agent_state[4], agent_state[5], agent_state[6] = self._pp_buf[0]
        # SIM_VALIDATION: record the reset observation for the reset→first-step
        # jump check (default OFF).
        if self._sim_val is not None:
            self._sim_val.note_reset(self._episode_count, int(getattr(self, "_current_stage", -1)),
                                     float(agent_state[0]), float(agent_state[1]), self.loc_noise)
        # AUX_PRED: append privileged future-risk label (no-op when disabled).
        response.state = self._append_aux_labels(np.append(obs_state, agent_state))
        return response

    def change_goal(self, start_x=0.0, start_y=0.0):
        """Places a new goal that is not in a dead zone and is far enough from start."""
        if self.train_mode:
            min_start_goal_dist = 3.0
            goal_radius = max(self.goal_threshold, 0.25)
            lingering = list(self.spawned_obstacle_records.values())

            def _is_valid_goal(x: float, y: float, require_clearance: bool = True) -> bool:
                if self.check_dead_zone(
                    x,
                    y,
                    use_cross_mask=False,
                    lower_bound=self.goal_obstacle_lower,
                    upper_bound=self.goal_obstacle_upper,
                ):
                    return False
                if math.hypot(x - start_x, y - start_y) < min_start_goal_dist:
                    return False
                if require_clearance and self._pose_collides_with_placed(
                    x, y, goal_radius, lingering
                ):
                    return False
                if self.current_layout_spec is not None and self._point_in_walls(
                        x, y, self.map_wall_clearance + goal_radius):
                    return False
                return True

            # Structured maps: the goal is sampled ENTIRELY inside the active
            # map's free regions, through every fallback phase. It NEVER drops to
            # the legacy whole-arena (goal_obstacle_lower~upper) sampling below,
            # so a goal can't land in a walled-off / off-lane cell. The sampler
            # always returns a pose inside the free regions.
            if self.current_layout_spec is not None:
                self.goal_x, self.goal_y = self._sample_goal_layout(
                    start_x, start_y, goal_radius, _is_valid_goal, lingering)
                self._update_gazebo_goal_marker()
                return

            # Phase 1: original strict sampling with full obstacle clearance.
            for _ in range(1000):
                x = random.uniform(self.goal_obstacle_lower, self.goal_obstacle_upper)
                y = random.uniform(self.goal_obstacle_lower, self.goal_obstacle_upper)
                if _is_valid_goal(x, y, require_clearance=True):
                    self.goal_x, self.goal_y = x, y
                    self._update_gazebo_goal_marker()
                    return

            # Phase 2: still keep dead-zone and start-distance constraints, but
            # relax lingering-obstacle exclusion to avoid hanging the reset loop.
            self.get_logger().warn(
                "change_goal: strict sampling failed after 1000 tries; "
                "retrying with relaxed obstacle-clearance constraint"
            )
            for _ in range(300):
                x = random.uniform(self.goal_obstacle_lower, self.goal_obstacle_upper)
                y = random.uniform(self.goal_obstacle_lower, self.goal_obstacle_upper)
                if _is_valid_goal(x, y, require_clearance=False):
                    self.goal_x, self.goal_y = x, y
                    self._update_gazebo_goal_marker()
                    return

            # Phase 3: deterministic fallback on a coarse grid. Prefer points that
            # are far from the start and maximize clearance from lingering obstacles.
            best = None
            best_score = -float("inf")
            grid_size = 11
            xs = np.linspace(self.goal_obstacle_lower, self.goal_obstacle_upper, grid_size)
            ys = np.linspace(self.goal_obstacle_lower, self.goal_obstacle_upper, grid_size)
            for x in xs:
                for y in ys:
                    if not _is_valid_goal(float(x), float(y), require_clearance=False):
                        continue
                    start_dist = math.hypot(float(x) - start_x, float(y) - start_y)
                    if lingering:
                        min_obs_clearance = min(
                            math.hypot(float(x) - px, float(y) - py) - (goal_radius + pr)
                            for px, py, pr in lingering
                        )
                    else:
                        min_obs_clearance = float("inf")
                    score = min_obs_clearance + 0.1 * start_dist
                    if score > best_score:
                        best_score = score
                        best = (float(x), float(y))

            if best is not None:
                self.goal_x, self.goal_y = best
                self.get_logger().warn(
                    "change_goal: using deterministic fallback goal after sampling exhaustion"
                )
                self._update_gazebo_goal_marker()
                return

            # Last resort: keep the episode moving with a simple offset from the start.
            # This should be rare and is preferable to deadlocking the entire run.
            fallback_x = float(np.clip(start_x + min_start_goal_dist, self.goal_obstacle_lower, self.goal_obstacle_upper))
            fallback_y = float(np.clip(start_y, self.goal_obstacle_lower, self.goal_obstacle_upper))
            self.goal_x, self.goal_y = fallback_x, fallback_y
            self.get_logger().warn(
                "change_goal: all sampling phases failed; using last-resort fallback goal"
            )
        else:
            self.goal_x = self.current_pairs["goal"]["x"]
            self.goal_y = self.current_pairs["goal"]["y"]
        self._update_gazebo_goal_marker()

    def _compute_rect_safety_hit(
        self,
        angle: float,
        front_scale: float = 1.0,
        rear_scale: float = 1.0,
        left_scale: float = 1.0,
        right_scale: float = 1.0,
    ):
        """Ray–rectangle intersection distance for the inflated robot footprint.

        Returns the first hit distance and the face name. Optional per-face scales
        are applied to the returned distance so the same routine can be reused for
        both the hard boundary and the soft warning boundary.
        """
        d_f  = self.sr_d_front  + self.sr_margin_front
        d_r  = self.sr_d_rear   + self.sr_margin_rear
        d_l  = self.sr_d_left   + self.sr_margin_left
        d_ri = self.sr_d_right  + self.sr_margin_right
        ca, sa = math.cos(angle), math.sin(angle)
        candidates = []
        # Front face  x = +d_f
        if ca > 1e-9:
            t = d_f / ca
            if -d_ri - 1e-6 <= sa * t <= d_l + 1e-6:
                candidates.append((t, "front", front_scale))
        # Rear face   x = -d_r
        elif ca < -1e-9:
            t = d_r / (-ca)
            if -d_ri - 1e-6 <= sa * t <= d_l + 1e-6:
                candidates.append((t, "rear", rear_scale))
        # Left face   y = +d_l
        if sa > 1e-9:
            t = d_l / sa
            if -d_r - 1e-6 <= ca * t <= d_f + 1e-6:
                candidates.append((t, "left", left_scale))
        # Right face  y = -d_ri
        elif sa < -1e-9:
            t = d_ri / (-sa)
            if -d_r - 1e-6 <= ca * t <= d_f + 1e-6:
                candidates.append((t, "right", right_scale))
        if not candidates:
            fallback = max(d_f, d_r, d_l, d_ri)
            return fallback, "none"
        dist, face, scale = min(candidates, key=lambda item: item[0])
        return dist * scale, face

    def _precompute_rect_safety_ranges(
        self,
        front_scale: float = 1.0,
        rear_scale: float = 1.0,
        left_scale: float = 1.0,
        right_scale: float = 1.0,
    ) -> np.ndarray:
        """Precompute V_range for every observation bin.

        The earlier phase-sampled version could leave central or boundary bins
        unselected depending on the face sampling phase, which caused missed
        collisions for head-on wall contact. Here every bin center gets its own
        ray-rectangle intersection distance, so the hard and soft boundaries are
        defined continuously across the full 360-degree collision bins (self.bins).
        """
        bin_centers = np.array([0.5 * (lo + hi) for lo, hi in self.bins], dtype=float)
        ranges = np.empty(self.environment_dim, dtype=float)
        for idx, angle in enumerate(bin_centers):
            ranges[idx], _ = self._compute_rect_safety_hit(
                float(angle),
                front_scale=front_scale,
                rear_scale=rear_scale,
                left_scale=left_scale,
                right_scale=right_scale,
            )
        return ranges

    def _compute_rect_proximity(self, laser_data) -> float:
        """Proximity to the rectangular safety boundary for reward shaping.

        For each phase-selected bin (finite V_range), computes:
          deficit[i] = max(0, 1 - obs[i] / warning_range[i])
        where warning_range[i] is derived from the same rectangular geometry but
        with per-face warning scales.
        Returns the maximum deficit (0.0 = fully safe, 1.0 = at hard boundary).
        """
        if self._rect_warning_ranges is None or self._rect_safety_ranges is None:
            return 0.0
        obs      = np.asarray(laser_data, dtype=float)
        selected = np.isfinite(self._rect_safety_ranges)
        if not np.any(selected):
            return 0.0
        obs_sel     = obs[selected]
        warning_sel = self._rect_warning_ranges[selected]
        valid = (obs_sel > 0.0) & np.isfinite(obs_sel)
        if not np.any(valid):
            return 0.0
        deficits = np.maximum(0.0, 1.0 - obs_sel[valid] / np.maximum(warning_sel[valid], 1e-6))
        return float(np.max(deficits))

    def check_collision(self, laser_data):
        """Rectangular Safety Region collision detection (paper Algorithm 1).

        Collision is triggered when any LiDAR bin reads strictly less than the
        per-bin safety range V_range[i] (ray–rectangle intersection distance of
        the inflated robot footprint).  Zone infrastructure (_zone_mins etc.) is
        kept intact for the reward function's proximity penalty.

        Returns: (done, collision, min_laser_used)
        """
        if self._rect_safety_ranges is None:
            # Fallback: global-min rule (should not happen after __init__)
            min_laser = float(np.min(laser_data)) if len(laser_data) else float('inf')
            hit = min_laser < self.collision_threshold
            return hit, hit, min_laser

        obs = np.asarray(laser_data, dtype=float)
        # Only consider beams that returned a finite, positive, sub-max reading
        valid = (obs > 0.0) & np.isfinite(obs) & (obs < self.lidar_max_range)
        if not np.any(valid):
            return False, False, float('inf')

        # Phase-selected bins have a finite safety range; unselected bins carry
        # np.inf.  numpy evaluates (obs <= np.inf) as True for any finite obs,
        # so we must AND with the selected mask to avoid false collisions on
        # every unselected bin that receives a valid scan return.
        selected = np.isfinite(self._rect_safety_ranges)
        collision_mask = valid & selected & (obs <= self._rect_safety_ranges)
        if np.any(collision_mask):
            min_used = float(np.min(obs[collision_mask]))
            return True, True, min_used

        return False, False, float(np.min(obs[valid]))

    # ------------------------------------------------------------------
    # Obstacle spawn/delete  (static obstacles + humans; replaces legacy shuffle_obstacles)
    # ------------------------------------------------------------------

    def _await_future(self, future, timeout: float = 3.0):
        """Poll a service future until done or timed out.

        Safe to call from inside a service callback when the node runs under
        MultiThreadedExecutor: the other executor threads keep processing ROS
        callbacks (including the service response) while this thread sleeps.
        Returns the result, or None on timeout.
        """
        deadline = time.time() + timeout
        while not future.done():
            if time.time() > deadline:
                self.get_logger().warn(f"Service future timed out after {timeout:.1f}s")
                return None
            time.sleep(0.05)
        return future.result()

    def _make_obstacle_sdf(self, model_name: str, uri: str) -> str:
        """Return a minimal SDF string that includes a catalog model as static."""
        return (
            '<sdf version="1.8">'
            f'<model name="{model_name}">'
            "<static>true</static>"
            f"<include><uri>{uri}</uri></include>"
            "</model>"
            "</sdf>"
        )

    def _make_box_sdf(self, model_name: str, sx: float, sy: float, sz: float) -> str:
        """Return a static box SDF used as an internal map-layout wall segment."""
        return (
            '<sdf version="1.8">'
            f'<model name="{model_name}">'
            "<static>true</static>"
            f'<link name="wall_link">'
            f'<collision name="wall_collision">'
            f"<geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>"
            "</collision>"
            f'<visual name="wall_visual">'
            f"<geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>"
            "<material>"
            "<ambient>0.55 0.55 0.58 1</ambient>"
            "<diffuse>0.55 0.55 0.58 1</diffuse>"
            "</material>"
            "</visual>"
            "</link>"
            "</model>"
            "</sdf>"
        )

    def _spawn_box_entity(self, model_name: str, sx: float, sy: float, sz: float,
                          x: float, y: float, z: float) -> bool:
        """Spawn one static box wall and wait for Gazebo confirmation."""
        req = SpawnEntity.Request()
        req.entity_factory.name = model_name
        req.entity_factory.allow_renaming = False
        req.entity_factory.sdf = self._make_box_sdf(model_name, sx, sy, sz)
        req.entity_factory.pose.position.x = float(x)
        req.entity_factory.pose.position.y = float(y)
        req.entity_factory.pose.position.z = float(z)
        req.entity_factory.pose.orientation.w = 1.0
        try:
            future = self.spawn_entity_client.call_async(req)
            result = self._await_future(future)
            if result is None or not result.success:
                self.get_logger().warn(f"Spawn wall {model_name}: failed/timed out")
                return False
            return True
        except Exception as e:
            self.get_logger().warn(f"SpawnEntity wall {model_name}: {e}")
            return False

    def _make_goal_marker_sdf(self, model_name: str) -> str:
        """Return a thin ground disc used to visualize the goal in Gazebo."""
        radius = max(float(self.goal_threshold), 0.25)
        height = 0.004
        return (
            '<sdf version="1.8">'
            f'<model name="{model_name}">'
            "<static>true</static>"
            "<link name=\"goal_marker_link\">"
            "<visual name=\"goal_marker_visual\">"
            f"<geometry><cylinder><radius>{radius:.3f}</radius><length>{height:.3f}</length></cylinder></geometry>"
            "<material>"
            "<ambient>0.95 0.15 0.15 0.90</ambient>"
            "<diffuse>0.95 0.15 0.15 0.90</diffuse>"
            "<specular>0.05 0.05 0.05 0.90</specular>"
            "<emissive>0.35 0.05 0.05 0.90</emissive>"
            "</material>"
            "</visual>"
            "</link>"
            "</model>"
            "</sdf>"
        )

    def _spawn_entity_sdf(self, model_name: str, uri: str, x: float, y: float, yaw: float, z: float = 0.0) -> bool:
        """Spawn one static obstacle and wait for Gazebo's confirmation.

        Returns True if Gazebo confirmed the spawn, False on failure or timeout.
        """
        req = SpawnEntity.Request()
        req.entity_factory.name = model_name
        req.entity_factory.allow_renaming = False
        req.entity_factory.sdf = self._make_obstacle_sdf(model_name, uri)
        req.entity_factory.pose.position.x = float(x)
        req.entity_factory.pose.position.y = float(y)
        req.entity_factory.pose.position.z = float(z)
        q = Quaternion.from_euler(0.0, 0.0, float(yaw))
        req.entity_factory.pose.orientation.x = float(q.x)
        req.entity_factory.pose.orientation.y = float(q.y)
        req.entity_factory.pose.orientation.z = float(q.z)
        req.entity_factory.pose.orientation.w = float(q.w)
        try:
            future = self.spawn_entity_client.call_async(req)
            result = self._await_future(future)
            if result is None:
                self.get_logger().warn(f"Spawn {model_name}: timed out")
                return False
            if not result.success:
                self.get_logger().warn(f"Spawn {model_name}: Gazebo returned success=false")
                return False
            return True
        except Exception as e:
            self.get_logger().warn(f"SpawnEntity for {model_name}: {e}")
            return False

    def _spawn_goal_marker(self, x: float, y: float, z: float = 0.002) -> bool:
        """Spawn the Gazebo goal marker once and keep moving it with set_pose afterwards."""
        req = SpawnEntity.Request()
        req.entity_factory.name = self.goal_marker_model_name
        req.entity_factory.allow_renaming = False
        req.entity_factory.sdf = self._make_goal_marker_sdf(self.goal_marker_model_name)
        req.entity_factory.pose.position.x = float(x)
        req.entity_factory.pose.position.y = float(y)
        req.entity_factory.pose.position.z = float(z)
        req.entity_factory.pose.orientation.w = 1.0
        try:
            future = self.spawn_entity_client.call_async(req)
            result = self._await_future(future)
            if result is None:
                self.get_logger().warn("Spawn goal marker: timed out")
                return False
            if not result.success:
                self.get_logger().warn("Spawn goal marker: Gazebo returned success=false")
                return False
            self.goal_marker_spawned = True
            return True
        except Exception as e:
            self.get_logger().warn(f"Spawn goal marker failed: {e}")
            return False

    def _update_gazebo_goal_marker(self):
        """Ensure the current goal is visible inside Gazebo as a thin ground disc."""
        marker_z = 0.002
        if not self.goal_marker_spawned:
            if not self._spawn_goal_marker(self.goal_x, self.goal_y, marker_z):
                return
        self.set_entity_pose_ignition(
            self.goal_marker_model_name,
            self.goal_x,
            self.goal_y,
            marker_z,
            0.0,
            0.0,
            0.0,
            1.0,
        )

    def _delete_spawned_obstacles(self):
        """Delete all obstacles from the previous episode and wait for each confirmation."""
        if not self.spawned_obstacle_names:
            return
        remaining_names = []
        for name in list(self.spawned_obstacle_names):
            req = DeleteEntity.Request()
            req.entity.name = name
            req.entity.type = GzEntity.MODEL
            try:
                future = self.delete_entity_client.call_async(req)
                result = self._await_future(future)
                if result is None:
                    self.get_logger().warn(f"Delete {name}: timed out (entity may linger)")
                    remaining_names.append(name)
                elif not result.success:
                    self.get_logger().warn(f"Delete {name}: Gazebo returned success=false")
                    remaining_names.append(name)
                else:
                    self.spawned_obstacle_records.pop(name, None)
            except Exception as e:
                self.get_logger().warn(f"DeleteEntity for {name}: {e}")
                remaining_names.append(name)
        self.spawned_obstacle_names = remaining_names

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

        Uses self._arena_wall_lower / _arena_wall_upper (≈ ±9.5 m), which are
        derived from goal_obstacle bounds + obstacle_wall_margin.  This is the
        true arena inner-face boundary, NOT the start-sampling box (self.lower/upper).

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

        Walls are axis-aligned boxes {cx, cy, sx, sy}. free_regions /
        human_regions are (x_lo, x_hi, y_lo, y_hi) rectangles. open_area (lobby)
        is a central rectangle kept clear of LARGE obstacles. Wall entity names
        are assigned here and spawned once by _initialize_obstacle_pool().
        """
        lo, hi = self.map_inner_lower, self.map_inner_upper
        span = hi - lo
        t = self.map_wall_thickness
        layouts = {}

        # lobby — open space, no internal walls; large items avoid the centre.
        ohe = self.map_lobby_open_half_extent
        layouts["lobby"] = {
            "walls": [],
            "free_regions": [(lo, hi, lo, hi)],
            "human_regions": [(lo, hi, lo, hi)],
            "open_area": (-ohe, ohe, -ohe, ohe),
        }

        # corridor — one horizontal travel lane bounded by two long walls.
        half = self.map_corridor_width / 2.0
        band = min(self.map_start_band_depth, 0.45 * span)  # end-band depth
        c_ph = self.map_corridor_passage_width / 2.0        # reserved-strip half width
        left_band  = (lo, lo + band, -half, half)
        right_band = (hi - band, hi, -half, half)
        layouts["corridor"] = {
            "walls": [
                {"cx": 0.0, "cy":  (half + t / 2.0), "sx": span, "sy": t},
                {"cx": 0.0, "cy": -(half + t / 2.0), "sx": span, "sy": t},
            ],
            "free_regions": [(lo, hi, -half, half)],
            "human_regions": [(lo, hi, -half, half)],
            "open_area": None,
            # Structured spawn metadata: start at either end band, head inward.
            "start_regions": [
                {"name": "left",  "rect": left_band,  "axis": "x",
                 "dir": 1.0,  "lane_half": half},
                {"name": "right", "rect": right_band, "axis": "x",
                 "dir": -1.0, "lane_half": half},
            ],
            "goal_regions": {"left": [right_band], "right": [left_band]},
            # Central along-lane strip kept clear of static obstacles.
            "reserved_passages": [
                {"axis": "x", "y_center": 0.0, "half_width": c_ph},
            ],
        }

        # intersection — plus-shaped lanes; four corner blocks wall off the rest.
        half = self.map_intersection_width / 2.0
        blk = hi - half          # corner block side length
        cmid = (half + hi) / 2.0  # corner block centre offset
        band = min(self.map_start_band_depth, 0.9 * (hi - half))
        i_ph = self.map_intersection_passage_width / 2.0
        arm_l = (lo, lo + band, -half, half)
        arm_r = (hi - band, hi, -half, half)
        arm_b = (-half, half, lo, lo + band)
        arm_t = (-half, half, hi - band, hi)
        layouts["intersection"] = {
            "walls": [
                {"cx":  cmid, "cy":  cmid, "sx": blk, "sy": blk},
                {"cx": -cmid, "cy":  cmid, "sx": blk, "sy": blk},
                {"cx":  cmid, "cy": -cmid, "sx": blk, "sy": blk},
                {"cx": -cmid, "cy": -cmid, "sx": blk, "sy": blk},
            ],
            "free_regions": [(lo, hi, -half, half), (-half, half, lo, hi)],
            "human_regions": [
                (half, hi, -half, half), (lo, -half, -half, half),
                (-half, half, half, hi), (-half, half, lo, -half),
            ],
            "open_area": None,
            # Start at any of the 4 arm-end bands, head toward the centre.
            "start_regions": [
                {"name": "left",   "rect": arm_l, "axis": "x",
                 "dir": 1.0,  "lane_half": half},
                {"name": "right",  "rect": arm_r, "axis": "x",
                 "dir": -1.0, "lane_half": half},
                {"name": "bottom", "rect": arm_b, "axis": "y",
                 "dir": 1.0,  "lane_half": half},
                {"name": "top",    "rect": arm_t, "axis": "y",
                 "dir": -1.0, "lane_half": half},
            ],
            # Goal lives in a DIFFERENT arm than the start arm.
            "goal_regions": {
                "left":   [arm_r, arm_t, arm_b],
                "right":  [arm_l, arm_t, arm_b],
                "bottom": [arm_t, arm_l, arm_r],
                "top":    [arm_b, arm_l, arm_r],
            },
            # Cross-shaped central lanes kept clear of static obstacles.
            "reserved_passages": [
                {"axis": "x", "y_center": 0.0, "half_width": i_ph},
                {"axis": "y", "x_center": 0.0, "half_width": i_ph},
            ],
        }

        # clutter — a few short internal walls (wide gaps keep connectivity).
        layouts["clutter"] = {
            "walls": [
                {"cx": -0.28 * span, "cy":  0.18 * span, "sx": t,            "sy": 0.30 * span},
                {"cx":  0.26 * span, "cy": -0.10 * span, "sx": 0.30 * span,  "sy": t},
                {"cx":  0.10 * span, "cy":  0.30 * span, "sx": 0.26 * span,  "sy": t},
                {"cx": -0.10 * span, "cy": -0.30 * span, "sx": 0.26 * span,  "sy": t},
            ],
            "free_regions": [(lo, hi, lo, hi)],
            "human_regions": [(lo, hi, lo, hi)],
            "open_area": None,
        }

        for mt, spec in layouts.items():
            spec["wall_names"] = [f"rl_wall_{mt}_{i:02d}" for i in range(len(spec["walls"]))]
        return layouts

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
        allowed = {m: (MAP_TYPE_ALLOWED_STATIC_KEYS[m] & set(avail)) for m in MAP_TYPES}
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
        vals = list(np.arange(-17.0, 17.0 + 1e-6, 1.6))
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
        return (yaw + math.pi) % (2.0 * math.pi) - math.pi

    def _sample_xy_in_regions(self, regions, radius: float):
        """Uniformly sample (x, y) in an area-weighted random region, shrinking
        each region by `radius` so the footprint stays inside it."""
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
        idx = int(np.random.choice(len(usable), p=w / w.sum()))
        ax_lo, ax_hi, ay_lo, ay_hi = usable[idx]
        return float(np.random.uniform(ax_lo, ax_hi)), float(np.random.uniform(ay_lo, ay_hi))

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

        # A: strict + lane-consistent (preferred).
        for _ in range(800):
            xy = self._sample_xy_in_regions(regions, goal_radius)
            if xy is None:
                break
            x, y = xy
            if is_valid(x, y, require_clearance=True) and not self._segment_hits_wall(
                    start_x, start_y, x, y, clr):
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
            lane_bonus = 0.0 if self._segment_hits_wall(start_x, start_y, x, y, clr) else 2.0
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
                random.shuffle(regions)
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
        random.shuffle(quadrants)
        return quadrants

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
                radius, placed, start_x, start_y, x_lo, x_hi, y_lo, y_hi
            )
        return self._sample_free_pose(radius, placed, start_x, start_y)

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
        mode = str(np.random.choice(modes, p=[w / total for w in weights]))

        p = dict(self._HUMAN_MODE_DEFAULTS[mode])
        p.update(self.human_mode_params.get(mode, {}) or {})

        axis_kind = str(p.get("axis", "random"))
        if axis_kind == "crossing":
            mode_axis = math.atan2(-y, -x)                      # head across the arena
        elif axis_kind == "principal":
            mode_axis = float(np.random.choice([0.0, math.pi / 2, math.pi, -math.pi / 2]))
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
            d = np.random.uniform(span_min, span_max)
            if axis is None:
                ang = np.random.uniform(-math.pi, math.pi)      # waiting / slow_turn
            else:
                ang = float(axis) + np.random.uniform(-spread, spread)  # crossing / along_path
            gx = float(np.clip(cx + d * math.cos(ang), lower, upper))
            gy = float(np.clip(cy + d * math.sin(ang), lower, upper))
            if (math.hypot(gx - cx, gy - cy) >= min_reach
                    and not self.check_dead_zone(
                        gx, gy, use_cross_mask=False,
                        lower_bound=lower, upper_bound=upper)):
                return gx, gy
        # Recovery: any free pose in the arena (wider search than the mode cone).
        for _ in range(40):
            gx = float(np.random.uniform(lower, upper))
            gy = float(np.random.uniform(lower, upper))
            if (math.hypot(gx - cx, gy - cy) >= min_reach
                    and not self.check_dead_zone(
                        gx, gy, use_cross_mask=False,
                        lower_bound=lower, upper_bound=upper)):
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
                if not self.check_dead_zone(x, y, use_cross_mask=False,
                                            lower_bound=lower, upper_bound=upper):
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
                d = np.random.uniform(min(dmin, dmax), dmax)
                axis = state.get("mode_axis")
                if axis is None:
                    ang = np.random.uniform(-math.pi, math.pi)
                else:
                    spread = float(state.get("mode_dir_spread", math.pi))
                    ang = float(axis) + np.random.uniform(-spread, spread)
                x = float(np.clip(cx + d * math.cos(ang), lower, upper))
                y = float(np.clip(cy + d * math.sin(ang), lower, upper))
            elif bounded:
                d = np.random.uniform(
                    min(self.human_waypoint_dist_min, self.human_waypoint_dist_max),
                    self.human_waypoint_dist_max,
                )
                ang = np.random.uniform(-math.pi, math.pi)
                x = float(np.clip(cx + d * math.cos(ang), lower, upper))
                y = float(np.clip(cy + d * math.sin(ang), lower, upper))
            else:
                x = np.random.uniform(lower, upper)
                y = np.random.uniform(lower, upper)
            if not self.check_dead_zone(x, y, use_cross_mask=False,
                                        lower_bound=lower, upper_bound=upper):
                return x, y
        return (float(cx), float(cy)) if (mode_bounded or bounded) else (0.0, 0.0)

    def _publish_model_poses(self, batch: list):
        """Publish (name, x, y, z, qx, qy, qz, qw) tuples to DrlModelPosePlugin."""
        msg = DrlModelPoseArray()
        for name, x, y, z, qx, qy, qz, qw in batch:
            msg.names.append(str(name))
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            p.position.z = float(z)
            p.orientation.x = float(qx)
            p.orientation.y = float(qy)
            p.orientation.z = float(qz)
            p.orientation.w = float(qw)
            msg.poses.append(p)
        self._model_pose_pub.publish(msg)

    def _collect_human_part_poses(self, state: dict, batch: list):
        """Compute world poses for torso + 2 leg + 2 arm cylinders and append to batch.

        Replaces _set_human_part_poses() service calls with batch entries that
        are published in one shot by _publish_model_poses() at the end of the
        timer callback.
        """
        x       = state["x"]
        y       = state["y"]
        yaw     = state["yaw"]
        phase   = state["gait_phase"]
        amp     = state["leg_swing_amp_rad"]
        leg_len = state["leg_length"]
        hip_z   = state["hip_z"]
        leg_y   = state["leg_y_offset"]
        torso_z = state["torso_z"]

        left_pitch  =  amp * math.sin(phase)
        right_pitch = -left_pitch

        cy = math.cos(yaw)
        sy = math.sin(yaw)

        def world_xy(lx, ly):
            return x + cy * lx - sy * ly, y + sy * lx + cy * ly

        # Torso — always upright at human centre
        torso_q = Quaternion.from_euler(0.0, 0.0, yaw)
        twx, twy = world_xy(0.0, 0.0)
        for model_name in (state["visual_torso"], state["proxy_torso"]):
            batch.append((model_name, twx, twy, torso_z,
                          torso_q.x, torso_q.y, torso_q.z, torso_q.w))

        # Legs — pendulum swing: hip end (-Z) is fixed, foot end (+Z) swings.
        # Using (π - pitch) makes -Z end sit at the hip pivot and +Z end swing freely.
        for vis_name, prx_name, pitch, side_y in (
            (state["visual_left_leg"],  state["proxy_left_leg"],  left_pitch,  +leg_y),
            (state["visual_right_leg"], state["proxy_right_leg"], right_pitch, -leg_y),
        ):
            lx_local = math.sin(pitch) * leg_len * 0.5
            lz       = hip_z - math.cos(pitch) * leg_len * 0.5
            wx, wy   = world_xy(lx_local, side_y)
            q = Quaternion.from_euler(0.0, math.pi - pitch, yaw)
            batch.append((vis_name, wx, wy, lz, q.x, q.y, q.z, q.w))
            batch.append((prx_name, wx, wy, lz, q.x, q.y, q.z, q.w))

        # Arms — pendulum swing: shoulder end (-Z) fixed, hand end (+Z) swings.
        # Arms swing opposite to the same-side leg (natural cross-body gait).
        arm_amp     = state.get("arm_swing_amp_rad", 0.0)
        arm_len     = state.get("arm_length", 0.60)
        shoulder_z  = state.get("shoulder_z", torso_z + 0.40)
        arm_y       = state.get("arm_y_offset", 0.22)
        # left arm opposes left leg; right arm opposes right leg
        left_arm_pitch  = -left_pitch  * (arm_amp / max(amp, 1e-6))
        right_arm_pitch = -right_pitch * (arm_amp / max(amp, 1e-6))
        for vis_name, pitch, side_y in (
            (state.get("visual_left_arm"),  left_arm_pitch,  +arm_y),
            (state.get("visual_right_arm"), right_arm_pitch, -arm_y),
        ):
            if vis_name is None:
                continue
            ax_local = math.sin(pitch) * arm_len * 0.5
            az       = shoulder_z - math.cos(pitch) * arm_len * 0.5
            wx, wy   = world_xy(ax_local, side_y)
            q = Quaternion.from_euler(0.0, math.pi - pitch, yaw)
            batch.append((vis_name, wx, wy, az, q.x, q.y, q.z, q.w))

    def _set_human_part_poses(self, state: dict):
        """Compute and apply world poses for torso + 2 leg cylinders (visual and proxy).

        The torso is placed upright at the human centre.  Each leg cylinder swings
        around its hip pivot: centre offset = (sin(pitch)*leg_len/2, ±leg_y, hip_z - cos(pitch)*leg_len/2)
        in the human local frame.  Orientation = Quaternion.from_euler(0, pitch, yaw).
        """
        x       = state["x"]
        y       = state["y"]
        yaw     = state["yaw"]
        phase   = state["gait_phase"]
        amp     = state["leg_swing_amp_rad"]
        leg_len = state["leg_length"]
        hip_z   = state["hip_z"]
        leg_y   = state["leg_y_offset"]
        torso_z = state["torso_z"]

        left_pitch  =  amp * math.sin(phase)
        right_pitch = -left_pitch

        cy = math.cos(yaw)
        sy = math.sin(yaw)

        def world_xy(lx, ly):
            return x + cy * lx - sy * ly, y + sy * lx + cy * ly

        # Torso — always upright at human centre
        torso_q = Quaternion.from_euler(0.0, 0.0, yaw)
        twx, twy = world_xy(0.0, 0.0)
        self.set_entity_pose_ignition(
            state["visual_torso"], twx, twy, torso_z,
            torso_q.x, torso_q.y, torso_q.z, torso_q.w,
        )
        self.set_entity_pose_ignition(
            state["proxy_torso"], twx, twy, torso_z,
            torso_q.x, torso_q.y, torso_q.z, torso_q.w,
        )

        # Legs — pendulum: hip end (-Z) fixed, foot end (+Z) swings
        for vis_name, prx_name, pitch, side_y in (
            (state["visual_left_leg"],  state["proxy_left_leg"],  left_pitch,  +leg_y),
            (state["visual_right_leg"], state["proxy_right_leg"], right_pitch, -leg_y),
        ):
            lx_local = math.sin(pitch) * leg_len * 0.5
            lz       = hip_z - math.cos(pitch) * leg_len * 0.5
            wx, wy   = world_xy(lx_local, side_y)
            q = Quaternion.from_euler(0.0, math.pi - pitch, yaw)
            self.set_entity_pose_ignition(vis_name, wx, wy, lz, q.x, q.y, q.z, q.w)
            self.set_entity_pose_ignition(prx_name, wx, wy, lz, q.x, q.y, q.z, q.w)

        # Arms — pendulum: shoulder end (-Z) fixed, hand end (+Z) swings
        amp_leg     = state["leg_swing_amp_rad"]
        arm_amp     = state.get("arm_swing_amp_rad", 0.0)
        arm_len     = state.get("arm_length", 0.60)
        shoulder_z  = state.get("shoulder_z", torso_z + 0.40)
        arm_y       = state.get("arm_y_offset", 0.22)
        left_arm_pitch  = -left_pitch  * (arm_amp / max(amp_leg, 1e-6))
        right_arm_pitch = -right_pitch * (arm_amp / max(amp_leg, 1e-6))
        for vis_name, pitch, side_y in (
            (state.get("visual_left_arm"),  left_arm_pitch,  +arm_y),
            (state.get("visual_right_arm"), right_arm_pitch, -arm_y),
        ):
            if vis_name is None:
                continue
            ax_local = math.sin(pitch) * arm_len * 0.5
            az       = shoulder_z - math.cos(pitch) * arm_len * 0.5
            wx, wy   = world_xy(ax_local, side_y)
            q = Quaternion.from_euler(0.0, math.pi - pitch, yaw)
            self.set_entity_pose_ignition(vis_name, wx, wy, az, q.x, q.y, q.z, q.w)

    def _human_timer_callback(self):
        """ROS timer callback — drives moving obstacles independently of RL steps."""
        # Fast-path flag check (no lock cost on the common path during reset).
        if not self._human_updates_enabled or not self.human_states:
            return
        # Hold the lock for the entire iteration so reset_callback can guarantee
        # no in-flight update is running before it clears human_states.
        with self._human_lock:
            if not self._human_updates_enabled or not self.human_states:
                return
            dt = 1.0 / self.human_update_rate
            pose_batch = []
            self._update_humans_kinematic(dt, pose_batch)
            if pose_batch:
                self._publish_model_poses(pose_batch)

    def _social_avoidance_offset(self, key, x, y, desired_yaw, human_pos):
        """Falcon-lite weak human-human avoidance: a small, capped heading nudge.

        Sums a repulsion vector from neighbouring HUMANS within
        human_social_avoid_radius (linear falloff: closer = stronger), blends it
        with the current goal-directed heading using human_social_avoid_strength,
        and returns the resulting heading change CLAMPED to
        human_social_avoid_max_heading_offset.  The robot is never considered, so
        humans stay non-reactive to it; the cap + the caller's yaw-rate limiter
        keep turns gentle so the constant-velocity aux labels stay valid.
        Returns 0.0 when disabled or no neighbour is close.
        """
        R = self.human_social_avoid_radius
        if not self.human_social_avoid_enabled or R <= 0.0:
            return 0.0
        rx = ry = 0.0
        for k, (ox, oy) in human_pos.items():
            if k == key:
                continue
            dx, dy = x - ox, y - oy
            d = math.hypot(dx, dy)
            if 1e-3 < d < R:
                wgt = 1.0 - d / R           # linear falloff (1 at contact, 0 at R)
                rx += (dx / d) * wgt
                ry += (dy / d) * wgt
        if rx == 0.0 and ry == 0.0:
            return 0.0
        # Blend the unit goal heading with the repulsion vector, then cap the
        # angular deviation so avoidance only ever bends the path slightly.
        bx = math.cos(desired_yaw) + self.human_social_avoid_strength * rx
        by = math.sin(desired_yaw) + self.human_social_avoid_strength * ry
        delta = (math.atan2(by, bx) - desired_yaw + math.pi) % (2.0 * math.pi) - math.pi
        cap = self.human_social_avoid_max_heading_offset
        return float(np.clip(delta, -cap, cap))

    def _update_humans_kinematic(self, dt: float, pose_batch: list):
        """Kinematic pedestrian motion with acceleration limits, smooth stop/resume, and gait.

        Stop/pause state machine (per agent):
          Normal  → stop_prob fires → Stopping (decelerate v,w → 0; position still integrates)
          Stopping → v<0.05 and |w|<0.05 → Pause  (hold x,y; v=w=0; gait frozen)
          Pause   → pause_left expires → Normal

        Gait phase advances at rate proportional to current speed.
        All poses are collected into pose_batch via _collect_human_part_poses() and
        published in one shot by the timer callback to DrlModelPosePlugin.
        """
        arena_lower = self.goal_obstacle_lower + self.obstacle_wall_margin
        arena_upper = self.goal_obstacle_upper - self.obstacle_wall_margin
        wall_buffer = 0.8

        # stop probability is per-human (mode-dependent), computed in the loop
        retarget_prob_tick = 1.0 - (1.0 - self.human_retarget_prob_per_sec) ** dt

        # Falcon-lite: snapshot current human positions once per tick so weak
        # human-human avoidance uses a consistent frame (and never the robot).
        human_pos = {k: (s["x"], s["y"]) for k, s in self.human_states.items()} \
            if self.human_social_avoid_enabled else {}

        for _key, state in list(self.human_states.items()):
            x, y   = state["x"],   state["y"]
            yaw    = state["yaw"]
            v      = state["v"]
            w      = state["w"]
            tx, ty = state["target_x"], state["target_y"]

            # ── True pause: hold position; zero v and w; freeze gait phase ──────
            pause_left = state.get("pause_left", 0.0)
            if pause_left > 0.0:
                state["pause_left"] = max(0.0, pause_left - dt)
                state["v"] = 0.0
                state["w"] = 0.0
                self._collect_human_part_poses(state, pose_batch)
                self.spawned_obstacle_records[_key] = (x, y, state["radius"])
                continue

            # ── Falcon-lite goal arrival: trigger a SMOOTH stop, then rest + new
            #    goal. We pre-sample the next goal now and flag the arrival; the
            #    stopping branch below decelerates this tick (no velocity jump),
            #    and the stop->pause transition adopts the new goal + goal pause. ─
            if (self.human_goal_driven_enabled and "goal_x" in state
                    and not state.get("goal_reached_pending", False)
                    and math.hypot(state["goal_x"] - x, state["goal_y"] - y)
                        < self.human_goal_reach_threshold):
                state["stopping"] = True
                state["goal_reached_pending"] = True
                ngx, ngy = self._sample_human_goal(x, y, state)
                state["next_goal_x"] = ngx
                state["next_goal_y"] = ngy

            # ── Stopping: decelerate v and w to 0; position still integrates ────
            # Per-human, mode-dependent stop probability → mode-level stop/go.
            stop_prob_tick = 1.0 - (1.0 - float(
                state.get("mode_stop_prob", self.human_stop_prob_per_sec))) ** dt
            stopping = state.get("stopping", False)
            if not stopping and np.random.rand() < stop_prob_tick:
                stopping = True
                state["stopping"] = True

            if stopping:
                max_dv = self.human_max_accel    * dt
                max_dw = self.human_max_yaw_accel * dt
                v = max(0.0, v - max_dv)
                w = float(np.clip(0.0, w - max_dw, w + max_dw))

                yaw   = (yaw + w * dt + math.pi) % (2.0 * math.pi) - math.pi
                new_x = max(arena_lower, min(arena_upper, x + v * math.cos(yaw) * dt))
                new_y = max(arena_lower, min(arena_upper, y + v * math.sin(yaw) * dt))

                state["x"]   = new_x
                state["y"]   = new_y
                state["yaw"] = yaw
                state["v"]   = v
                state["w"]   = w

                # Advance gait phase at current (decelerating) speed
                freq = state["gait_freq_hz"] * max(0.3, v / max(0.01, state["speed"]))
                state["gait_phase"] = (state["gait_phase"] + 2.0 * math.pi * freq * dt) % (2.0 * math.pi)

                if v < 0.05 and abs(w) < 0.05:
                    state["stopping"]   = False
                    if state.pop("goal_reached_pending", False):
                        # Arrived at the global goal: adopt the pre-sampled next
                        # goal, rest for the (longer) goal pause, and aim the local
                        # target at the new goal -> goal-directed travel resumes.
                        state["goal_x"] = state.pop("next_goal_x", state.get("goal_x"))
                        state["goal_y"] = state.pop("next_goal_y", state.get("goal_y"))
                        state["pause_left"] = float(self.human_goal_pause_duration)
                        ntx, nty = self._sample_human_waypoint(
                            state["x"], state["y"], state=state)
                        state["target_x"], state["target_y"] = ntx, nty
                        state["segment_elapsed"] = 0.0
                    else:
                        state["pause_left"] = float(
                            state.get("mode_pause_duration", self.human_pause_duration))
                    state["v"]          = 0.0
                    state["w"]          = 0.0

                self._collect_human_part_poses(state, pose_batch)
                self.spawned_obstacle_records[_key] = (new_x, new_y, state["radius"])
                continue

            # ── Normal motion: waypoint following with kinematic limits ──────────
            dist_to_target = math.hypot(tx - x, ty - y)
            near_wall = (
                x <= arena_lower + wall_buffer or x >= arena_upper - wall_buffer or
                y <= arena_lower + wall_buffer or y >= arena_upper - wall_buffer
            )
            retarget_cooldown = max(0.0, float(state.get("retarget_cooldown", 0.0)) - dt)
            state["retarget_cooldown"] = retarget_cooldown
            # Track ACTIVE-motion time on the current segment ("mode") and force
            # a new waypoint once it exceeds human_max_segment_sec, bounding the
            # transition period instead of leaving it to a rare 1%/s draw.
            # NOTE: this counter only advances during normal motion — the pause
            # and stopping branches `continue` above, so a long pause (e.g. the
            # waiting mode) makes the real wall-clock interval exceed this bound.
            segment_elapsed = float(state.get("segment_elapsed", 0.0)) + dt
            state["segment_elapsed"] = segment_elapsed
            segment_timeout = (
                self.human_max_segment_sec > 0.0
                and segment_elapsed >= self.human_max_segment_sec
            )
            do_prob_retarget = (
                retarget_cooldown <= 0.0 and np.random.rand() < retarget_prob_tick
            )
            if dist_to_target < 0.5 or near_wall or do_prob_retarget or segment_timeout:
                # For fixed-axis modes (crossing / along_path), reverse the travel
                # axis at a wall so the human paces back across instead of pushing
                # into it — keeps motion straight and predictable (no reaction).
                # Skipped for goal-driven humans: their waypoint sampler aims at
                # an in-arena goal and ignores mode_axis.
                if (near_wall and not self.human_goal_driven_enabled
                        and state.get("mode_axis") is not None):
                    state["mode_axis"] = (
                        (float(state["mode_axis"]) + math.pi) + math.pi
                    ) % (2.0 * math.pi) - math.pi
                tx, ty = self._sample_human_waypoint(x, y, state=state)
                # Recovery: if goal-driven and the local step is ~0 the path
                # toward the current goal is boxed in -> pick a new (reachable)
                # goal so the human escapes instead of freezing in a 0-length
                # retarget loop.
                if (self.human_goal_driven_enabled and "goal_x" in state
                        and math.hypot(tx - x, ty - y) < 1e-2):
                    state["goal_x"], state["goal_y"] = self._sample_human_goal(x, y, state)
                    tx, ty = self._sample_human_waypoint(x, y, state=state)
                state["target_x"] = tx
                state["target_y"] = ty
                state["retarget_cooldown"] = self.human_min_retarget_interval
                state["segment_elapsed"] = 0.0
                if self.human_heading_jitter_on_retarget_only:
                    state["heading_jitter"] = np.random.uniform(
                        -self.human_heading_jitter, self.human_heading_jitter
                    )

            dx_t, dy_t = tx - x, ty - y
            desired_yaw_raw = math.atan2(dy_t, dx_t)
            if self.human_heading_jitter_on_retarget_only:
                desired_yaw_raw += float(state.get("heading_jitter", 0.0))
            else:
                desired_yaw_raw += np.random.uniform(
                    -self.human_heading_jitter, self.human_heading_jitter
                )
            # Falcon-lite weak avoidance: nudge the desired heading away from
            # nearby humans by a small capped amount (then the rate limiter below
            # smooths it). Robot is never considered.
            if self.human_social_avoid_enabled:
                desired_yaw_raw += self._social_avoidance_offset(
                    _key, x, y, desired_yaw_raw, human_pos)
            desired_yaw_raw = (desired_yaw_raw + math.pi) % (2.0 * math.pi) - math.pi

            prev_desired_yaw = float(state.get("desired_yaw", yaw))
            desired_yaw_delta = (
                (desired_yaw_raw - prev_desired_yaw + math.pi) % (2.0 * math.pi)
            ) - math.pi
            max_desired_yaw_delta = max(1e-3, self.human_desired_yaw_rate_limit) * dt
            desired_yaw = prev_desired_yaw + float(np.clip(
                desired_yaw_delta,
                -max_desired_yaw_delta,
                max_desired_yaw_delta,
            ))
            desired_yaw = (desired_yaw + math.pi) % (2.0 * math.pi) - math.pi
            state["desired_yaw"] = desired_yaw

            yaw_error = (desired_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
            w_cmd = float(np.clip(
                self.human_k_yaw * yaw_error,
                -self.human_max_yaw_rate, self.human_max_yaw_rate,
            ))
            max_dw = self.human_max_yaw_accel * dt
            w = float(np.clip(w_cmd, w - max_dw, w + max_dw))

            abs_err = abs(yaw_error)
            if abs_err > 0.6:
                v_des = state["speed"] * 0.4
            elif abs_err > 0.3:
                v_des = state["speed"] * 0.7
            else:
                v_des = state["speed"]

            max_dv = self.human_max_accel * dt
            v = float(np.clip(v_des, v - max_dv, v + max_dv))
            v = max(0.0, v)

            yaw   = (yaw + w * dt + math.pi) % (2.0 * math.pi) - math.pi
            new_x = max(arena_lower, min(arena_upper, x + v * math.cos(yaw) * dt))
            new_y = max(arena_lower, min(arena_upper, y + v * math.sin(yaw) * dt))

            state["x"]   = new_x
            state["y"]   = new_y
            state["yaw"] = yaw
            state["v"]   = v
            state["w"]   = w

            # Gait frequency scales with speed ratio (min 30% of nominal freq)
            freq = state["gait_freq_hz"] * max(0.3, v / max(0.01, state["speed"]))
            state["gait_phase"] = (state["gait_phase"] + 2.0 * math.pi * freq * dt) % (2.0 * math.pi)

            self._collect_human_part_poses(state, pose_batch)
            self.spawned_obstacle_records[_key] = (new_x, new_y, state["radius"])

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

    def check_dead_zone(
        self,
        x,
        y,
        use_cross_mask: bool = False,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ):
        """True면 금지영역, False면 허용.
           use_cross_mask=False이면 십자 띠 제한을 해제한다."""
        if lower_bound is None:
            lower_bound = self.lower
        if upper_bound is None:
            upper_bound = self.upper

        # 맵 바깥은 항상 금지
        if x < lower_bound or x > upper_bound or y < lower_bound or y > upper_bound:
            return True

        # 십자 띠 제한을 쓰지 않으면 바로 허용
        if not use_cross_mask:
            return False

        # 십자형 내부 띠 금지(기존 로직)
        if 2.0 < abs(x) < upper_bound and abs(y) < 1.0:
            return True
        if abs(x) < 1.0 and 2.0 < abs(y) < upper_bound:
            return True

        return False

    def publish_markers(self, action):
        """Publishes visual data for RViz: goal ground-disc + waypoint action bars.
        action[0] (normalized) → wp_r_norm bar   (waypoint distance, larger = farther)
        action[1] (normalized) → wp_theta_norm bar (waypoint angle, larger = sharper turn)
        """
        goal_diameter = max(2.0 * float(self.goal_threshold), 0.5)
        marker_specs = [
            {
                "frame_id": "odom",
                "marker_type": Marker.CYLINDER,
                "scale": (goal_diameter, goal_diameter, 0.004),
                "color": (0.9, 1.0, 0.1, 0.1),
                "position": (self.goal_x, self.goal_y, 0.002),
                "orientation": (0.0, 0.0, 0.0, 1.0),
                "action": Marker.ADD,
                "ns": "",
                "marker_id": 0,
                "publisher": self.goal_point_marker_pub,
            },
            {
                "frame_id": "odom",
                "marker_type": Marker.CUBE,
                "scale": (abs(action[0]), 0.1, 0.01),  # |r_norm| ∈ [0,1]
                "color": (1.0, 1.0, 0.0, 0.0),
                "position": (5.0, 0.0, 0.0),
                "orientation": (0.0, 0.0, 0.0, 1.0),
                "action": Marker.ADD,
                "ns": "",
                "marker_id": 1,
                "publisher": self.wp_r_marker_pub,
            },
            {
                "frame_id": "odom",
                "marker_type": Marker.CUBE,
                "scale": (abs(action[1]), 0.1, 0.01),  # |theta_norm| ∈ [0,1]
                "color": (1.0, 1.0, 0.0, 0.0),
                "position": (5.0, 0.2, 0.0),
                "orientation": (0.0, 0.0, 0.0, 1.0),
                "action": Marker.ADD,
                "ns": "",
                "marker_id": 2,
                "publisher": self.wp_theta_marker_pub,
            },
        ]
        for spec in marker_specs:
            marker = self.create_marker(**spec)
            marker_array = MarkerArray()
            marker_array.markers.append(marker)
            spec["publisher"].publish(marker_array)

    @staticmethod
    def create_marker(**kwargs):
        """Create marker to be published for visualization"""
        marker = Marker()
        marker.ns = kwargs.get("ns", "")
        marker.id = kwargs.get("marker_id", 0)
        marker.header.frame_id = kwargs.get("frame_id", "odom")
        marker.type = kwargs.get("marker_type", Marker.CYLINDER)
        marker.action = kwargs.get("action", Marker.ADD)
        marker.scale.x, marker.scale.y, marker.scale.z = kwargs.get(
            "scale", (0.1, 0.1, 0.01)
        )
        marker.color.a, marker.color.r, marker.color.g, marker.color.b = kwargs.get(
            "color", (1.0, 0.0, 1.0, 0.0)
        )
        (
            marker.pose.position.x,
            marker.pose.position.y,
            marker.pose.position.z,
        ) = kwargs.get("position", (0.0, 0.0, 0.0))
        (
            marker.pose.orientation.x,
            marker.pose.orientation.y,
            marker.pose.orientation.z,
            marker.pose.orientation.w,
        ) = kwargs.get("orientation", (0.0, 0.0, 0.0, 1.0))
        return marker

    @staticmethod
    def get_reward(
        target, collision,
        v, w,                                  # m/s, rad/s (Pure Pursuit 출력)
        prev_goal_dist, curr_goal_dist,
        theta_err=None,
        rect_proximity=None,
        zmins=None, zthrs=None,
        min_laser=None,
        v_max=1.5, w_max=6.0,

        # ---- 튜닝 파라미터 ----
        k_p=0.5,                 # 진행 보상 게인 (누적 양수 보상 과대 억제)
        progress_clip=0.25,

        # 곡률 페널티 (waypoint RL에서 Pure Pursuit가 처리하므로 기본 0)
        lambda_k=0.0,

        # 장애물 근접 (존 기반)
        z_weights=(0.6, 0.85, 1.0, 0.85, 0.6),
        safety_margin=1.5,
        w_obs=1.5,

        # 장애물 근접 (폴백)
        d_safe_base=0.55,
        d_safe_speed=0.30,

        # 헤딩/시간/스무딩
        k_h=0.03,
        step_pen=0.05,
        k_smooth=0.0,
        prev_v=None, prev_w=None,

        # 웨이포인트 스무딩 (급격한 방향 전환 억제)
        waypoint_theta=0.0,
        prev_waypoint_theta=0.0,
        k_smooth_wp=0.05,
        return_terms=False,
    ):
        terms = {
            "delta_d": 0.0,
            "progress": 0.0,
            "heading": 0.0,
            "curv_pen": 0.0,
            "obstacle": 0.0,
            "step_pen": 0.0,
            "smooth": 0.0,
            "wp_smooth": 0.0,
            "terminal": 0.0,
        }
        # 터미널
        if target:
            terms["terminal"] = 20.0
            return (20.0, terms) if return_terms else 20.0
        if collision:
            terms["terminal"] = -30.0
            return (-30.0, terms) if return_terms else -30.0

        # 정규화
        v_n = v / max(v_max, 1e-6)
        w_n = w / max(w_max, 1e-6)

        # 1) 진행 보상
        delta_d  = np.clip(prev_goal_dist - curr_goal_dist, -progress_clip, progress_clip)
        progress = k_p * delta_d
        terms["delta_d"] = float(delta_d)
        terms["progress"] = float(progress)

        # 2) 곡률 페널티 (lambda_k=0 → disabled for waypoint RL)
        kappa    = abs(w_n) / (abs(v_n) + 1e-3)
        curv_pen = lambda_k * kappa
        terms["curv_pen"] = float(curv_pen)

        # 2b) 웨이포인트 스무딩 (연속 step 간 waypoint 각도 변화 억제)
        dtheta = abs(waypoint_theta - prev_waypoint_theta)
        if dtheta > math.pi:
            dtheta = 2.0 * math.pi - dtheta
        wp_smooth = k_smooth_wp * dtheta / math.pi
        terms["wp_smooth"] = float(wp_smooth)

        # 3) 장애물 근접 페널티 (직사각형 우선 → 레거시 zone → 글로벌 min 폴백)
        obstacle = 0.0
        if rect_proximity is not None:
            # 충돌/보상 기하 통일: _compute_rect_proximity() 값 직접 사용
            obstacle = w_obs * float(rect_proximity)   # 0 ~ w_obs
        elif zmins is not None and zthrs is not None and len(zmins) == 5 and len(zthrs) == 5:
            # 레거시 zone 경로 (호환성 유지, 현재는 use_zone_collision=false로 미사용)
            deficits = []
            for i in range(5):
                thr_expanded = max(1e-6, safety_margin * float(zthrs[i]))
                zmin = float(zmins[i])
                d = max(0.0, 1.0 - (zmin / thr_expanded))
                deficits.append(d)
            wsum = sum(z_weights)
            weighted = sum(wi * di for wi, di in zip(z_weights, deficits)) / max(wsum, 1e-6)
            obstacle = w_obs * weighted
        else:
            # 폴백: 글로벌 min_laser 기반 (속도 의존 안전거리)
            if min_laser is not None and np.isfinite(min_laser):
                d_safe = d_safe_base + d_safe_speed * abs(v)
                if min_laser < d_safe:
                    obstacle = w_obs * (1.0 - min_laser / max(d_safe, 1e-6))
        terms["obstacle"] = float(obstacle)

        # 4) 헤딩 보너스 — goal에 가까워지는 step에서만 부여 (reward hacking 방지)
        heading = (k_h * max(0.0, math.cos(theta_err))
                   if (theta_err is not None and delta_d > 0.0)
                   else 0.0)
        terms["heading"] = float(heading)

        # 5) 스무딩(선택)
        smooth = 0.0
        if k_smooth > 0.0 and prev_v is not None and prev_w is not None:
            dv = abs(v - prev_v) / max(v_max, 1e-6)
            dw = abs(w - prev_w) / max(w_max, 1e-6)
            smooth = k_smooth * 0.5 * (dv + dw)
        terms["smooth"] = float(smooth)
        terms["step_pen"] = float(step_pen)

        # 6) 시간 페널티 및 합산
        reward = progress + heading - curv_pen - obstacle - step_pen - smooth - wp_smooth

        return (float(reward), terms) if return_terms else float(reward)

def main(args=None):
    # Initialize the ROS2 communication
    rclpy.init(args=args)
    # Create the environment node
    environment = Environment()
    # Use MultiThreadedExecutor to handle the two sensor callbacks in parallel.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(environment)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        environment.get_logger().info("gym_node, shutting down...")
        environment.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
