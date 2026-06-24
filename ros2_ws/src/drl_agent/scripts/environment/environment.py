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
# Pure helpers extracted from this file (no ROS deps) — see utils/.
import geometry_utils as geom
import seed_utils
import reward_calculator
from collision_checker import RectSafetyChecker
from localization_noise import LocalizationNoiseModel, ProprioNoiseModel
# AUX_PRED: privileged future-risk label generation (training-only).
import aux_prediction_labels as aux_labels
from obs_time_context import ObsTimeContext
from sensor_msgs.msg import LaserScan

from ros_gz_interfaces.msg import Contacts
from ros_gz_interfaces.msg import Entity as GzEntity
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose, SpawnEntity, DeleteEntity


# Structured map curriculum — static-obstacle catalog policy now lives in the
# neutral, ROS-free map_catalog module (shared with the extracted obstacle /
# map-layout mixins so they need no back-import to this file).
from map_catalog import (
    STATIC_GLOBALLY_BANNED_KEYS,
    MAP_TYPE_ALLOWED_STATIC_KEYS,
    MAP_TYPES,
    static_size_group,
    resolve_active_count,
)

# Extracted responsibility groups (mixins). Each lives in its own module and
# carries one cohesive concern; Environment composes them via inheritance so the
# node body below stays orchestration-focused. The mixins reference shared node
# state through ``self`` (initialised in Environment.__init__), so behaviour is
# identical to the previous single-file implementation.
from zone_tracker import ZoneMixin
from observation_builder import ObservationMixin
from map_layout_runtime import MapLayoutMixin
from start_sampler import StartSamplerMixin
from goal_sampler import GoalSamplerMixin
from human_spawn_sampler import HumanSpawnMixin
from human_motion_manager import HumanMotionMixin
from gazebo_entity_manager import GazeboEntityMixin
from obstacle_catalog_spawner import ObstacleMixin
# Bounded Gazebo-service wait + the failure type the callbacks propagate, so a
# dead Gazebo control/set_pose service never hangs /step or /reset forever.
from gazebo_service_wait import GazeboServiceError, bounded_wait_for_service


class Environment(
    ZoneMixin,
    ObservationMixin,
    MapLayoutMixin,
    StartSamplerMixin,
    GoalSamplerMixin,
    HumanSpawnMixin,
    HumanMotionMixin,
    GazeboEntityMixin,
    ObstacleMixin,
    Node,
):
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

        # ── Observation time-context (actor-visible frame stacking) ────────────
        # Stacks the last N `obs_state` frames (the 80-D front-180° RL scan — NOT
        # the raw point cloud, NOT the 360° collision `environment_state`) so the
        # ACTOR sees short-horizon temporal context through the shared encoder,
        # WITHOUT any recurrence. Layout keeps the current full state FIRST so the
        # 87-D baseline layout is byte-for-byte preserved (state[:80]=current obs,
        # state[80]=goal_dist) and only OLDER obs frames are appended:
        #   [obs_t(80), agent_t(7), obs_{t-1}(80), ..., obs_{t-(N-1)}(80)]
        # By default agent_state is kept current-only (obs-only history); set
        # stack_agent_state=true to store the full 87-D frame per step instead.
        # OFF by default → state stays the exact 87-D vector, so every existing
        # config / checkpoint is unaffected. Reported via get_dimensions so the
        # agent (encoder/buffer/actor/critic, all sized from state_dim) adapts with
        # NO agent-side code change; train and inference share one contract.
        _otc = dict(self.environment_config.get("observation_time_context", {}) or {})
        self._otc = ObsTimeContext(
            self.environment_dim, self.agent_dim,
            enabled=bool(_otc.get("enabled", False)),
            obs_frame_stack=int(_otc.get("obs_frame_stack", 1)),
            stack_agent_state=bool(_otc.get("stack_agent_state", False)),
        )
        self.obs_time_context_enabled = self._otc.enabled
        self.obs_frame_stack = self._otc.obs_frame_stack
        self.obs_stack_agent_state = self._otc.stack_agent_state
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
        # Rectangular-safety geometry lives in RectSafetyChecker, built once below
        # after self.bins / collision_threshold / lidar_max_range are known.
        self._safety_checker: RectSafetyChecker = None

        # Start-pose heading/clearance safety parameters
        self.start_edge_heading_margin = float(
            self.environment_config.get("start_edge_heading_margin_m", 1.0))
        self.start_front_clearance = float(
            self.environment_config.get("start_front_clearance_m", 1.2))
        self.start_front_fov_deg = float(
            self.environment_config.get("start_front_fov_deg", 35.0))

        # Obstacle spawn margin parameters
        self.num_of_humans = int(self.environment_config.get("num_of_humans", 0))

        # Per-episode active-count resolution (map-type-aware curriculum support).
        # The curriculum subclass overwrites _stage_active_* / _stage_active_*_by_map
        # in _apply_curriculum_stage; the base (non-curriculum) env keeps the single
        # config values with EMPTY by-map maps, so _apply_episode_active_counts() is
        # a no-op there (it just re-asserts the base counts). See that method.
        self._stage_active_static       = self.num_of_static_obstacles
        self._stage_active_humans       = self.num_of_humans
        self._stage_active_static_by_map = {}
        self._stage_active_humans_by_map = {}
        self._last_active_counts_logged = None

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

        # ── Human-only dynamic-risk reward shaping (reward_calculator) ─────────
        # Separate from the static obstacle-proximity penalty: penalises closing
        # on pedestrians using GT robot+human kinematics (personal-space + approach
        # rate + TTC). OPT-IN: when the `human_risk_penalty` block is ABSENT this
        # defaults to DISABLED, so every pre-existing config — including legacy
        # human-bearing configs, external configs, and the non-curriculum
        # environment.yaml — keeps its EXACT prior reward (truly backward
        # compatible). The curriculum config opts in explicitly (enabled: true).
        # Even when enabled the penalty is exactly 0 whenever no humans are active,
        # so the human-free stages (0-2) are unaffected either way.
        _hrp = dict(self.environment_config.get("human_risk_penalty", {}) or {})
        self.human_risk_enabled  = bool(_hrp.get("enabled", False))
        self.human_risk_w_ps     = float(_hrp.get("w_personal_space", 0.4))
        self.human_risk_d_ps     = float(_hrp.get("personal_space_radius_m", 1.0))
        self.human_risk_w_app    = float(_hrp.get("w_approach_rate", 0.25))
        self.human_risk_d_app    = float(_hrp.get("approach_radius_m", 2.0))
        self.human_risk_v_ref    = float(_hrp.get("approach_v_ref_mps", 1.0))
        self.human_risk_w_ttc    = float(_hrp.get("w_ttc", 0.3))
        self.human_risk_ttc_safe = float(_hrp.get("ttc_safe_s", 2.0))

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

        # Internal-wall clearances for human GOAL / WAYPOINT sampling (structured
        # maps). Single source of truth so the sampler and the per-step runtime
        # resolver (_resolve_human_wall_collision, which uses each human's own
        # radius) stay consistent. The point clearance uses the MAX human radius,
        # so the sampler never accepts a wall-adjacent target that the runtime
        # would immediately reject — for ANY human, even if radii change. The
        # segment clearance is a small routing tolerance: it only rejects a local
        # step whose straight line clearly crosses a wall (the offset fan then
        # bends the path around it).
        self.human_target_wall_clearance = max(
            (float(e.get("radius", 0.30)) for e in self.human_catalog), default=0.35)
        self.human_segment_wall_clearance = 0.10

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
        # Reproducibility: the static/human pool is built ONCE on the FIRST reset
        # (_initialize_obstacle_pool), which runs AFTER the trainer's /seed
        # service has already re-seeded the env to the run seed. Its three
        # random.shuffle() draws therefore (a) varied the pool COMPOSITION run to
        # run only as a side effect of the run seed, and (b) ADVANCED the global
        # `random` stream by a count that depends on pool size / catalog — so any
        # change to the obstacle set (e.g. the new corridor geometry filter)
        # shifted the very next draws: the first episode's start/goal sampling.
        # We fix this by giving the pool build its OWN local RNG seeded with
        # pool_build_seed (see _initialize_obstacle_pool): pool composition is now
        # reproducible AND the global per-episode stream is left untouched by pool
        # construction. Store the seed here; the actual seeding is local to the
        # build so it cannot be clobbered by the /seed service.
        self.declare_parameter("pool_build_seed", 0)
        self.pool_build_seed = int(
            self.get_parameter("pool_build_seed").get_parameter_value().integer_value
        )
        # Dedicated pedestrian RNG sub-stream (spawn + motion). Kept OFF the
        # global random/np.random streams so wall-clock-paced human-motion ticks
        # never shift the next episode's start/goal/map/static sampling. Base seed
        # is set from the run seed in seed_callback; re-seeded per episode in
        # reset_callback as (base_seed, episode_count). Initialised here so the
        # attributes exist before the first /seed and the first reset.
        self._human_rng_base_seed = self.pool_build_seed
        self._seed_human_rngs(self._human_rng_base_seed, 0)
        # Human-RNG reproducibility policy, exposed as read-only params so the
        # trainer can stamp it into run_manifest.json (single source of truth).
        # RESUME CONTRACT = Option B: the human sub-stream is reseeded each reset
        # from (base_seed, episode_count); base_seed is set from the run /seed.
        # On resume the env process restarts (episode_count from 0) and is re-
        # seeded deterministically from derive_resume_seed(base_seed, global_t),
        # so the human stream is REPRODUCIBLE PER CHECKPOINT but NOT a bit-exact
        # continuation of the pre-interrupt stream (consistent with the env-RNG
        # contract in _reseed_env_for_resume). Exact continuation is intentionally
        # out of scope: a separate env process can't snapshot its RandomState
        # across the boundary, and within-episode motion is wall-clock paced.
        self._HUMAN_RNG_POLICY = (
            "substream_isolated_from_global; "
            "per_episode_reseed=SeedSequence(base_seed,episode_count); "
            "resume=deterministic_per_checkpoint; exact_resume_disabled"
        )
        self.declare_parameter("human_rng_enabled", True)
        self.declare_parameter("human_rng_policy", self._HUMAN_RNG_POLICY)
        self.declare_parameter("human_rng_base_seed", int(self._human_rng_base_seed))
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
        # Per-episode localization-noise state lives in LocalizationNoiseModel,
        # which owns the bias / OU / drift / jump / latency state and is reset()
        # per episode and step()ed per RL step (see _reset_localization /
        # _loc_emulator_step delegates). The node keeps only the loc_est_* cache.
        self._loc_model = LocalizationNoiseModel(self.loc_noise, self.time_delta)
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
        # Per-episode proprio-noise state lives in ProprioNoiseModel.
        self._pp_model = ProprioNoiseModel(self.proprio_noise)

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
        # Bounded-wait budgets for the Gazebo world-control / set_pose service
        # calls made INSIDE the /step and /reset callbacks. These replace the old
        # unbounded `while not wait_for_service` loop: if Gazebo dies the callback
        # raises GazeboServiceError after at most ~wait+call seconds instead of
        # hanging forever. Kept well under the trainer's service_call_timeout_sec
        # (30s) so the gym node fails FIRST and the trainer sees a clean timeout.
        self.declare_parameter("gazebo_service_wait_timeout_sec", 5.0)
        self.declare_parameter("gazebo_service_wait_poll_sec", 1.0)
        self.declare_parameter("gazebo_service_call_timeout_sec", 5.0)
        self._gz_wait_timeout = float(
            self.get_parameter("gazebo_service_wait_timeout_sec").value)
        self._gz_wait_poll = float(
            self.get_parameter("gazebo_service_wait_poll_sec").value)
        self._gz_call_timeout = float(
            self.get_parameter("gazebo_service_call_timeout_sec").value)
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

        # Precompute per-bin safety ranges (rectangular footprint, paper Algorithm 1).
        # All rect-safety geometry now lives in RectSafetyChecker.
        self._safety_checker = RectSafetyChecker(
            d_front=self.sr_d_front, d_rear=self.sr_d_rear,
            d_left=self.sr_d_left, d_right=self.sr_d_right,
            margin_front=self.sr_margin_front, margin_rear=self.sr_margin_rear,
            margin_left=self.sr_margin_left, margin_right=self.sr_margin_right,
            bins=self.bins, environment_dim=self.environment_dim,
            collision_threshold=self.collision_threshold,
            lidar_max_range=self.lidar_max_range,
            warning_scale_front=self.reward_warning_scale_front,
            warning_scale_rear=self.reward_warning_scale_rear,
            warning_scale_left=self.reward_warning_scale_left,
            warning_scale_right=self.reward_warning_scale_right,
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

    def _seed_human_rngs(self, base_seed: int, episode_index: int):
        """(Re)build the dedicated pedestrian RNG sub-stream.

        Creates fresh ``self._human_np_rng`` (np.random.RandomState) and
        ``self._human_py_rng`` (random.Random) seeded from (base_seed,
        episode_index). All human spawn + motion sampling draws from these so the
        GLOBAL random/np.random streams are never advanced by humans — that
        decouples the robot's start/goal/map/static sampling from the wall-clock-
        paced human-motion timer. Re-seeding each episode (index = episode count)
        makes the per-episode human spawn config reproducible regardless of how
        many motion ticks the previous episode happened to run."""
        self._human_np_rng, self._human_py_rng = seed_utils.make_substream_rngs(
            base_seed, episode_index)

    def seed_callback(self, request, response):
        """Sets environment seed for reproducibility of the training process.

        Seeds BOTH global RNGs the env draws from: numpy (obstacle/start/goal
        sampling) and Python's `random` (human spawn / waypoint / curriculum
        sampling). Seeding only numpy left those `random.*` draws unseeded, so
        same-seed runs were not byte-for-byte reproducible.

        Also (re)bases the dedicated human RNG sub-stream on this seed so the
        pedestrian stream is reproducible per run without ever touching the
        global streams above (see _seed_human_rngs)."""
        seed = int(request.seed)
        seeded = seed_utils.seed_basic_rngs(seed)
        self._human_rng_base_seed = seed
        self._seed_human_rngs(seed, getattr(self, "_episode_count", 0))
        # Mirror the actual human base seed into the read-only param so the
        # manifest records the TRUE per-run value (best-effort; never fatal).
        try:
            self.set_parameters(
                [Parameter("human_rng_base_seed", Parameter.Type.INTEGER, int(seed))]
            )
        except Exception:
            pass
        self.get_logger().info(
            f"[Seed] Environment RNGs seeded ({' + '.join(seeded)} + human-substream) "
            f"with {seed}"
        )
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
        """Returns the dimensions of the state, action, and maximum action value.

        state_dim reflects observation time-context stacking when enabled (the
        agent sizes its encoder/buffer/actor/critic from this single number).
        environment_dim / agent_dim stay the PER-FRAME sizes (80 / 7) so paper
        metrics that read the current frame (state[:environment_dim],
        state[environment_dim]) are unaffected — the current frame is always first.
        """
        response.state_dim = self.stacked_state_dim()
        response.action_dim = self.action_dim
        response.max_action = self.max_action
        response.environment_dim = self.environment_dim
        response.agent_dim = self.agent_dim
        return response

    # ------------------------------------------------------------------ #
    #  Observation time-context (frame stacking) — delegate to ObsTimeContext #
    # ------------------------------------------------------------------ #
    def stacked_state_dim(self) -> int:
        """Full RL state width on the wire (current frame + appended history)."""
        return self._otc.stacked_state_dim()

    def _reset_obs_history(self, obs_state, agent_state):
        """Seed the obs history with the first frame at episode start (repeat)."""
        self._otc.reset(obs_state, agent_state)

    def _assemble_state(self, obs_state, agent_state, advance=True):
        """Build the (optionally stacked) RL state and advance the history."""
        return self._otc.assemble(obs_state, agent_state, advance=advance)

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
        dist, theta = geom.goal_distance_and_heading(
            x, y, yaw, self.goal_x, self.goal_y)
        if dist < 1e-9:
            return 0.0, 0.0
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
        """Delegate to LocalizationNoiseModel.map_multiplier for the current map."""
        return self._loc_model.map_multiplier(getattr(self, "current_map_type", "") or "")

    def _reset_localization(self, x0, y0, yaw0):
        """Reset the localization-noise model for a new episode and refresh the
        loc_est_* cache from the (bias-applied) seed. Delegates to
        LocalizationNoiseModel; the current map type drives the per-episode
        sigma / drift multipliers and is read here (set by _select_episode_layout)."""
        self.loc_est_x, self.loc_est_y, self.loc_est_yaw = self._loc_model.reset(
            x0, y0, yaw0, getattr(self, "current_map_type", "") or ""
        )

    def _loc_emulator_step(self, x, y, yaw):
        """Advance the localization-noise model ONE RL step (delegate)."""
        return self._loc_model.step(x, y, yaw)

    # ------------------------------------------------------------------ #
    #  Proprioception observation noise (separate axis from goal-obs)       #
    # ------------------------------------------------------------------ #
    def _reset_proprio_noise(self, speed0=0.0, yaw_rate0=0.0, steer0=0.0):
        """Reset the proprio-noise model for a new episode (delegate)."""
        self._pp_model.reset(speed0, yaw_rate0, steer0)

    def _proprio_emulator_step(self, speed, yaw_rate, steer):
        """Return the noisy (speed, yaw_rate, steer) proprio OBSERVATION (delegate)."""
        return self._pp_model.step(speed, yaw_rate, steer)

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
            "penalty_human_personal_space",
            "penalty_human_approach_rate",
            "penalty_human_ttc",
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
    def _wait_for_srv(self, client, name: str, op: str) -> bool:
        """Bounded wait for a Gazebo service to become available.

        Replaces the old unbounded ``while not wait_for_service`` loop: probes at
        most ``gazebo_service_wait_timeout_sec`` (cadence
        ``gazebo_service_wait_poll_sec``) and ALWAYS returns. Returns True when
        the service is up; on exhaustion logs WHICH op/service timed out after
        how long and returns False (the caller raises GazeboServiceError).
        ``op`` is a short label (pause/unpause/reset/set_pose) for the logs."""
        ok, elapsed = bounded_wait_for_service(
            lambda step: client.wait_for_service(timeout_sec=step),
            self._gz_wait_timeout,
            self._gz_wait_poll,
            on_wait=lambda waited: self.get_logger().warn(
                f"[gazebo] {op}: service {name} not available, waiting "
                f"({waited:.1f}/{self._gz_wait_timeout:.1f}s)..."),
        )
        if not ok:
            self.get_logger().error(
                f"[gazebo] {op}: service {name} UNAVAILABLE after {elapsed:.1f}s "
                f"(wait budget {self._gz_wait_timeout:.1f}s) — failing fast")
        return ok

    def _call_world_service(self, client, req, srv_name: str, op: str):
        """Shared bounded call path for a Gazebo world/set_pose service.

        Raises GazeboServiceError (never hangs, never sys.exit) when the service
        is unavailable within the wait budget, when the response future does not
        arrive within the call budget, or when the call itself raises. A
        ``success=false`` reply is logged but NOT treated as fatal (it is not a
        hang and matches the previous warn-and-continue behaviour). Returns the
        result on success."""
        t0 = time.time()
        if not self._wait_for_srv(client, srv_name, op):
            raise GazeboServiceError(
                f"{srv_name} ({op}): service unavailable after "
                f"{time.time() - t0:.1f}s wait")
        try:
            future = client.call_async(req)
            result = self._await_future(
                future, timeout=self._gz_call_timeout, op=op)
        except Exception as e:
            raise GazeboServiceError(
                f"{srv_name} ({op}): call raised after "
                f"{time.time() - t0:.1f}s: {e}") from e
        if result is None:
            raise GazeboServiceError(
                f"{srv_name} ({op}): no response within "
                f"{self._gz_call_timeout:.1f}s (future timed out after "
                f"{time.time() - t0:.1f}s total)")
        if not result.success:
            self.get_logger().warn(
                f"[gazebo] {op}: {srv_name} returned success=false (continuing)")
        return result

    def pause_world(self, pause: bool):
        """Ignition 월드 일시정지 / 재개 — Gazebo 서비스 실패 시 즉시 상위로 전파."""
        op = "pause" if pause else "unpause"
        srv_name = f"/world/{self.world_name}/control"
        req = ControlWorld.Request()
        req.world_control.pause = bool(pause)
        self._call_world_service(self.world_control, req, srv_name, op)

    def reset_world(self):
        """Ignition 월드 리셋 (모델만, 시간은 유지) — 실패 시 상위로 전파."""
        srv_name = f"/world/{self.world_name}/control"
        req = ControlWorld.Request()
        req.world_control.reset.model_only = True
        req.world_control.pause = True
        self._call_world_service(self.world_control, req, srv_name, "reset")

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
        """Ignition 월드에서 특정 모델을 텔레포트 — 실패 시 상위로 전파."""
        srv_name = f"/world/{self.world_name}/set_pose"
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

        self._call_world_service(
            self.set_entity_pose, req, srv_name, f"set_pose[{name}]")

    def propagate_state(self, time_delta):
        """Ignition 월드를 time_delta초 동안 돌렸다가 다시 pause.

        unpause→sleep→pause 의 각 경계에 로그를 남겨 /step 이 정확히 어느
        Gazebo 호출에서 멎는지 바로 보이게 한다. pause_world 가 실패하면
        GazeboServiceError 가 콜백 밖으로 전파된다(여기서 삼키지 않는다)."""
        self.get_logger().debug(
            f"[gazebo] propagate: unpause → run {time_delta:.3f}s")
        self.pause_world(False)
        time.sleep(time_delta)
        self.pause_world(True)
        self.get_logger().debug("[gazebo] propagate: re-paused")
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
            # GT robot world-frame velocity (for the v2 TTC / hazard targets):
            # signed forward speed projected on the GT heading. No-op for the
            # v1 risk/min_dist blocks (they ignore robot_vel).
            _rv = float(self.latest_actual_signed_speed)
            robot_vel = (_rv * math.cos(self.gt_yaw), _rv * math.sin(self.gt_yaw))
            label = aux_labels.compute_future_risk_labels(
                humans,
                (self.gt_x, self.gt_y, self.gt_yaw),
                self._aux_label_cfg,
                robot_vel=robot_vel,
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
        """/step entrypoint. Wraps the implementation so a Gazebo service failure
        (GazeboServiceError from propagate_state → pause_world) is logged with the
        episode/step context and re-raised. Re-raising means NO response is sent:
        the trainer's bounded /step call times out and raises EnvServiceError,
        triggering its checkpoint-on-failure path. The env then stops cleanly via
        main()'s GazeboServiceError handler — no infinite hang, no sys.exit."""
        try:
            return self._step_callback_impl(request, response)
        except GazeboServiceError as e:
            self.get_logger().error(
                f"[gym_node] /step ABORTED (episode {self._episode_count}, "
                f"step {self.current_episode_step}): {e}")
            raise

    def _step_callback_impl(self, request, response):
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
        # Observation time-context: append the last (N-1) obs frames and advance
        # the history (no-op 87-D passthrough when disabled).
        state = self._assemble_state(obs_state, agent_state, advance=True)

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

        # Human-only dynamic-risk penalty (privileged GT human + robot state).
        # None when disabled; the penalty is naturally 0 when no humans are active
        # (human_states empty), so early/human-free stages are unaffected.
        human_risk_terms = None
        if self.human_risk_enabled:
            with self._human_lock:
                _humans = [
                    {"x": s["x"], "y": s["y"],
                     "yaw": s.get("yaw", 0.0), "v": s.get("v", 0.0)}
                    for s in self.human_states.values()
                ]
            human_risk_terms = reward_calculator.compute_human_risk_penalty(
                self.gt_x, self.gt_y, self.gt_yaw,
                self.latest_actual_signed_speed,
                _humans,
                w_ps=self.human_risk_w_ps, d_ps=self.human_risk_d_ps,
                w_app=self.human_risk_w_app, d_app=self.human_risk_d_app,
                v_ref=self.human_risk_v_ref,
                w_ttc=self.human_risk_w_ttc, ttc_safe=self.human_risk_ttc_safe,
            )

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
            human_risk_terms=human_risk_terms,
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
                round(float(reward_terms["human_personal_space_penalty"]), 6),
                round(float(reward_terms["human_approach_rate_penalty"]), 6),
                round(float(reward_terms["human_ttc_penalty"]), 6),
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

    def _apply_episode_active_counts(self):
        """Pick THIS episode's active static/human counts for the current map_type.

        Priority (per axis):
          1. stage's ``active_*_by_map[map_type]``  (set by the curriculum)
          2. stage's single ``active_*``            (or the base config value)
        Values are coerced to int, floored at 0, and capped at the pre-allocated
        pool size, so a per-map override can never exceed obstacle_pool_*_size or
        the structured-map geometry guarantees (the placer still parks anything it
        cannot fit). Called from reset_callback AFTER _select_episode_layout() has
        set self.current_map_type and BEFORE obstacle activation/spawn reads these
        counts. No-op for the plain (non-curriculum) env: the by-map maps are empty
        and _stage_active_* equal the base config, so this just re-asserts them.
        """
        mt = self.current_map_type or ""
        self.num_of_static_obstacles = resolve_active_count(
            self._stage_active_static_by_map, self._stage_active_static,
            self.obstacle_pool_static_size, mt)
        self.num_of_humans = resolve_active_count(
            self._stage_active_humans_by_map, self._stage_active_humans,
            self.obstacle_pool_human_size, mt)
        # Log only when the (map, counts) tuple changes, so transitions are visible
        # without spamming every episode.
        key = (mt, self.num_of_static_obstacles, self.num_of_humans)
        if key != self._last_active_counts_logged:
            self._last_active_counts_logged = key
            self.get_logger().info(
                f"[Episode active] map_type='{mt or 'none'}' -> "
                f"static={self.num_of_static_obstacles} humans={self.num_of_humans}"
            )

    def reset_callback(self, request, response):
        """/reset entrypoint. Wraps the implementation so a Gazebo service failure
        (GazeboServiceError from _prepare_episode_reset / set_entity_pose_ignition
        / propagate_state) is logged with the episode context and re-raised. As
        with /step, re-raising sends no response → the trainer's bounded /reset
        call times out → checkpoint-on-failure → clean shutdown via main()."""
        try:
            return self._reset_callback_impl(request, response)
        except GazeboServiceError as e:
            self.get_logger().error(
                f"[gym_node] /reset ABORTED (episode {self._episode_count}): {e}")
            raise

    def _reset_callback_impl(self, _, response):
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
        # Re-seed the dedicated human RNG sub-stream for THIS episode. Done while
        # the motion timer is disabled (above) and human_states is empty, so no
        # concurrent reader. This makes the episode's human spawn config a pure
        # function of (run seed, episode_count) — independent of the previous
        # episode's wall-clock-paced motion tick count — and keeps every human
        # draw off the global streams used for start/goal/map/static sampling.
        self._seed_human_rngs(
            getattr(self, "_human_rng_base_seed", self.pool_build_seed),
            self._episode_count,
        )
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
        # Resolve THIS episode's active static/human counts now that map_type is
        # known: a stage may set per-map counts (active_*_by_map) so e.g. the
        # narrow corridor gets fewer obstacles than the intersection in the SAME
        # stage. Must run AFTER _select_episode_layout() and BEFORE obstacle
        # activation/spawn below (which read num_of_static_obstacles / num_of_humans).
        self._apply_episode_active_counts()

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
            agent_state[4], agent_state[5], agent_state[6] = self._pp_model.peek()
        # SIM_VALIDATION: record the reset observation for the reset→first-step
        # jump check (default OFF).
        if self._sim_val is not None:
            self._sim_val.note_reset(self._episode_count, int(getattr(self, "_current_stage", -1)),
                                     float(agent_state[0]), float(agent_state[1]), self.loc_noise)
        # Observation time-context: (re)seed the obs history with the first frame
        # (first-frame repeat), then assemble WITHOUT advancing — the seeded deque
        # already represents this episode's "past". No-op 87-D when disabled.
        self._reset_obs_history(obs_state, agent_state)
        state0 = self._assemble_state(obs_state, agent_state, advance=False)
        # AUX_PRED: append privileged future-risk label (no-op when disabled).
        response.state = self._append_aux_labels(state0)
        return response

    def change_goal(self, start_x=0.0, start_y=0.0):
        """Places a new goal that is not in a dead zone and is far enough from start."""
        if self.train_mode:
            min_start_goal_dist = 3.0
            goal_radius = max(self.goal_threshold, 0.25)
            # `lingering` follows the SAME mode-aware contract as start sampling
            # (see start_sampler._sample_train_start_pose): in pool mode the
            # records are the PREVIOUS episode's about-to-be-teleported placements
            # (STALE → ignore, since obstacles are re-placed keeping
            # obstacle_goal_margin clear of this goal), while in non-pool mode they
            # are genuinely-present failed-delete leftovers at their body footprint
            # (humans kept while any part lingers — see _delete_spawned_obstacles)
            # which are REAL → avoid.
            lingering = ([] if getattr(self, "use_obstacle_pool", False)
                         else list(self.spawned_obstacle_records.values()))

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

    def _compute_rect_proximity(self, laser_data) -> float:
        """Delegate to RectSafetyChecker.compute_proximity (reward shaping)."""
        return self._safety_checker.compute_proximity(laser_data)

    def check_collision(self, laser_data):
        """Delegate to RectSafetyChecker.check_collision (paper Algorithm 1).

        Returns: (done, collision, min_laser_used)"""
        return self._safety_checker.check_collision(laser_data)

    # ------------------------------------------------------------------
    # Obstacle spawn/delete  (static obstacles + humans; replaces legacy shuffle_obstacles)
    # ------------------------------------------------------------------

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
    def get_reward(*args, **kwargs):
        """Thin wrapper delegating to :func:`reward_calculator.compute_reward`.

        The reward shaping was extracted to ``reward_calculator.py`` so it is
        ROS-free and unit-testable. This static method is kept for backward
        compatibility (call sites and ``environment_360.py`` overrides)."""
        return reward_calculator.compute_reward(*args, **kwargs)

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
    except GazeboServiceError as e:
        # A /step or /reset callback hit a dead Gazebo world-control / set_pose
        # service. rclpy re-raises callback exceptions out of spin(); catch it
        # here so the gym node stops with a clear FATAL line instead of an opaque
        # traceback. The trainer's in-flight call times out and its
        # checkpoint-on-failure path runs — fail-fast on BOTH ends.
        environment.get_logger().error(
            f"[gym_node] FATAL: stopping — Gazebo service failure: {e}. "
            "The trainer's /step or /reset will time out and trigger its "
            "checkpoint-on-failure path.")
    finally:
        environment.get_logger().info("gym_node, shutting down...")
        environment.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
