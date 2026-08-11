#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import json      # DYN_AVOID: serialize per-episode dynamic-avoidance diagnostics
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

import drl_agent.common.point_cloud2 as pc2
from drl_agent.common.file_manager import load_yaml
import drl_agent.common.pure_pursuit as pure_pursuit
# Pure helpers extracted from this file (no ROS deps) — see utils/.
import drl_agent.common.geometry_utils as geom
import drl_agent.common.seed_utils as seed_utils
import drl_agent.config.paths as config_paths
import drl_agent.env.rewards.reward_calculator as reward_calculator
from drl_agent.env.simulation.collision_checker import RectSafetyChecker
from drl_agent.env.simulation.localization_noise import LocalizationNoiseModel, ProprioNoiseModel
# AUX_PRED: privileged future-risk label generation (training-only).
import drl_agent.env.observation.aux_prediction_labels as aux_labels
# DYN_AVOID: privileged per-episode dynamic-obstacle avoidance telemetry.
from drl_agent.env.humans.dynamic_avoidance_telemetry import DynamicAvoidanceEpisodeDiag
from drl_agent.env.observation.obs_time_context import ObsTimeContext
from sensor_msgs.msg import LaserScan

from ros_gz_interfaces.msg import Contacts
from ros_gz_interfaces.msg import Entity as GzEntity
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose, SpawnEntity, DeleteEntity


# Structured map curriculum — static-obstacle catalog policy now lives in the
# neutral, ROS-free map_catalog module (shared with the extracted obstacle /
# map-layout mixins so they need no back-import to this file).
from drl_agent.env.simulation.map_catalog import (
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
from drl_agent.env.simulation.zone_tracker import ZoneMixin
from drl_agent.env.observation.observation_builder import ObservationMixin
from drl_agent.env.simulation.map_layout_runtime import MapLayoutMixin
from drl_agent.env.spawning.start_sampler import StartSamplerMixin
from drl_agent.env.spawning.goal_sampler import GoalSamplerMixin
from drl_agent.env.humans.human_spawn_sampler import HumanSpawnMixin
from drl_agent.env.humans.human_motion_manager import (
    HumanMotionMixin, compute_human_tick_plan)
from drl_agent.env.simulation.gazebo_entity_manager import GazeboEntityMixin
from drl_agent.env.spawning.obstacle_catalog_spawner import ObstacleMixin
# Bounded Gazebo-service wait + the failure type the callbacks propagate, so a
# dead Gazebo control/set_pose service never hangs /step or /reset forever.
from drl_agent.env.simulation.gazebo_service_wait import (
    GazeboServiceError, bounded_wait_for_service, compute_physics_step_count)


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
            # this file: <pkg_root>/drl_agent/env/simulation/environment.py
            here = os.path.dirname(os.path.abspath(__file__))
            candidates += [
                os.path.normpath(os.path.join(here, "..", "..", "..", "config")),  # <pkg_root>/config
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
        # action_mode selects the action->command decoding used in
        # _step_callback_impl. Default is INFERRED from action_dim exactly as
        # the pre-existing dispatch did (action_dim>=3 -> the 3D hybrid
        # waypoint+yield contract, else the legacy 2D waypoint contract), so
        # every existing config (none of which sets this key) resolves to the
        # same mode as before -- byte-identical default behaviour.
        # "speed_steering" (new, opt-in): 2-D continuous speed/steering, no
        # waypoint geometry, no binary yield channel -- see pure_pursuit.
        # speed_steering_action_to_command.
        self.action_mode = str(self.environment_config.get(
            "action_mode",
            "waypoint_yield" if self.action_dim >= 3 else "waypoint",
        )).strip().lower()
        if self.action_mode not in ("waypoint_yield", "waypoint", "speed_steering"):
            raise ValueError(
                f"environment.action_mode={self.action_mode!r} is not one of "
                "'waypoint_yield', 'waypoint', 'speed_steering'."
            )
        if self.action_mode == "speed_steering" and self.action_dim != 2:
            raise ValueError(
                "environment.action_mode=speed_steering requires action_dim=2 "
                f"(got action_dim={self.action_dim})."
            )
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
        # STOP/YIELD capability (opt-in). >0 ramps Pure-Pursuit speed toward 0 as
        # the commanded waypoint distance L drops below this, so a short waypoint
        # (small action[0]) lets the policy creep/stop. Default 0.0 → no-op, and
        # to truly reach 0 m/s also lower actions_low[0] (and/or
        # controller_min_speed_mps). See pure_pursuit.waypoint_to_command.
        self.controller_low_speed_distance_m = float(
            self.environment_config.get("controller_low_speed_distance_m", 0.0)
        )
        # 3D hybrid-action controller (pure_pursuit.hybrid_action_to_command).
        # Non-yield (MOVE) mode is floored so the policy cannot stop unless it
        # commands yield (action[2]); yield mode caps speed at the creep value.
        self.controller_lookahead_min_m = float(
            self.environment_config.get("controller_lookahead_min_m", 0.8)
        )
        self.controller_v_move_min_mps = float(
            self.environment_config.get("controller_v_move_min_mps", 0.35)
        )
        self.controller_yield_creep_mps = float(
            self.environment_config.get("controller_yield_creep_mps", 0.0)
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

        # ── Open-map (lobby / clutter) inward-safe start yaw ────────────────────
        # OPT-IN (default OFF → byte-identical legacy random-yaw sampling). When
        # ON, an open-map start near an OUTER wall samples its yaw inside the
        # inward-admissible sector instead of fully random, removing "spawn facing
        # the wall → instant failure" while keeping yaw diversity away from walls.
        # Only affects lobby/clutter (open maps); corridor/intersection keep their
        # lane-aligned yaw untouched. See _sample_open_map_safe_yaw.
        self.open_map_safe_start_yaw_enabled = bool(
            self.environment_config.get("open_map_safe_start_yaw_enabled", False))
        # "Near a wall" distance threshold. <=0 → reuse start_edge_heading_margin
        # so this sampler and the _is_heading_toward_near_wall rejection share the
        # SAME wall-distance basis (a yaw produced here then always passes it).
        _omwm = float(self.environment_config.get("open_map_safe_yaw_wall_margin_m", 0.0))
        self.open_map_safe_yaw_wall_margin = (
            _omwm if _omwm > 0.0 else self.start_edge_heading_margin)
        # Shrink each admissible-sector edge inward by this so the sampled heading
        # keeps an angular buffer from being exactly wall-parallel.
        self.open_map_safe_yaw_edge_margin = math.radians(
            float(self.environment_config.get("open_map_safe_yaw_edge_margin_deg", 10.0)))
        # Half-width of the fallback inward sector used only if the edge-margin
        # shrink would degenerate the admissible sector (keeps diversity, never a
        # single deterministic yaw).
        self.open_map_safe_yaw_fallback_halfwidth = math.radians(
            float(self.environment_config.get("open_map_safe_yaw_fallback_halfwidth_deg", 25.0)))

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
        # Deterministic (sim-step-synchronized) human motion, default OFF ->
        # byte-identical to before (legacy wall-clock timer). When enabled,
        # NO independent timer is created; propagate_state() instead drives an
        # exact, integer number of fixed-dt human-motion ticks per call itself,
        # interleaved with physics sub-stepping -- eliminating the wall-clock
        # scheduling dependency that made in-episode human trajectories
        # non-reproducible across runs at different real-time speeds (see
        # human_motion_manager.compute_human_tick_plan). This changes the
        # ACTUAL tick count/timing (not just a wait-mechanism latency fix), so
        # per CLAUDE.md it must stay config-isolated, never silently applied.
        # CLI override, same convention as risk_map_reward_enabled. Because
        # this parameter is declared as a STRING sentinel, direct ros2 CLI use
        # must pass a quoted string, e.g.
        #   -p human_deterministic_stepping_enabled:='"true"'
        self.declare_parameter("human_deterministic_stepping_enabled", "")
        self.human_deterministic_stepping = config_paths.parse_bool_override(
            self.get_parameter("human_deterministic_stepping_enabled")
                .get_parameter_value().string_value,
            bool(self.environment_config.get("human_deterministic_stepping", False)),
        )

        # Deterministic Gazebo physics stepping, default OFF -> byte-identical
        # to before (legacy unpause/sleep(duration)/pause, 2 world-control
        # calls). When enabled, a SINGLE WorldControl call (pause=True,
        # multi_step=N) requests an EXACT N * physics_step_size seconds of
        # sim-time advance (empirically confirmed: N=1/50/100 -> sim-time
        # deltas of exactly 0.001/0.05/0.1s, world stays paused afterward) --
        # halving the world-control call count vs. the legacy path, which is
        # the actual measured source of this mode's speedup (live-measured:
        # 203.6ms/step legacy vs. 152.3ms/step here). A per-step bounded
        # /clock wait for exact sim-time CONFIRMATION (not just the request)
        # was attempted but reproducibly hung real /step calls for reasons not
        # fully root-caused (see _advance_physics_deterministic's docstring
        # and the gazebo_multi_step_clock_wait_landmine memory note) --
        # excluded; this mode instead sleeps for the requested duration (same
        # approach the legacy path already uses, just once instead of via two
        # calls) and then verifies sensor freshness via the SAME proven
        # scan/odom-count wait reset_callback already uses. Independent of
        # human_deterministic_stepping; both compose via the shared
        # _advance_physics() primitive.
        self.declare_parameter("gazebo_deterministic_stepping_enabled", "")
        self.gazebo_deterministic_stepping = config_paths.parse_bool_override(
            self.get_parameter("gazebo_deterministic_stepping_enabled")
                .get_parameter_value().string_value,
            bool(self.environment_config.get("gazebo_deterministic_stepping", False)),
        )
        self.gazebo_physics_step_size = float(
            self.environment_config.get("gazebo_physics_step_size", 0.001))
        self.gazebo_sensor_wait_timeout_sec = float(
            self.environment_config.get("gazebo_sensor_wait_timeout_sec", 1.5))

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
        self.human_static_obstacle_clearance = float( # [m] extra keep-out for kinematic humans
            self.environment_config.get("human_static_obstacle_clearance", 0.35))
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

        # ── Stop/Yield reward shaping (reward_calculator) ──────────────────────
        # Conditional shaping layered ON TOP of human_risk_penalty. HARD-GATED on
        # a human being present this step, so it is exactly 0 in human-free
        # stages/timesteps (curriculum stages 0-2 unaffected) — including the idle
        # penalty. When a human IS present it rewards slowing/stopping ONLY while
        # that human is a closing hazard (raw closing-risk signals computed
        # alongside the human-risk penalty), relieves the step penalty in that
        # window, and lightly penalises idling when a present human is not a
        # hazard. OPT-IN: absent block → DISABLED → reward byte-for-byte unchanged
        # (backward compatibility / ablation). Needs a stop-capable controller
        # (controller_low_speed_distance_m > 0 + lowered actions_low[0]) to have
        # anything to reward; requires human_risk_penalty.enabled so the gating
        # signals (incl. nearest-human distance) are populated.
        _yr = dict(self.environment_config.get("yield_reward", {}) or {})
        self.yield_enabled            = bool(_yr.get("enabled", False))
        self.yield_risk_gate_ttc_s    = float(_yr.get("risk_gate_ttc_s", 2.0))
        self.yield_risk_gate_approach = float(_yr.get("risk_gate_approach", 0.15))
        self.yield_w_bonus            = float(_yr.get("w_yield_bonus", 0.05))
        self.yield_max_bonus          = float(_yr.get("max_yield_bonus", 0.1))
        self.yield_idle_w             = float(_yr.get("w_idle_penalty", 0.03))
        self.yield_idle_speed_mps     = float(_yr.get("idle_speed_threshold_mps", 0.1))
        self.yield_step_relief_scale  = float(_yr.get("step_penalty_relief_scale", 1.0))
        # 3D-action yield channel: controller gate (whether action[2] is honoured)
        # and the threshold at which it flips to yield. Stage overrides may seal
        # yielding early in the curriculum via yield_reward.action_enabled=false.
        self.yield_action_enabled     = bool(_yr.get("action_enabled", True))
        self.yield_action_threshold   = float(_yr.get("action_threshold", 0.3))
        # Explicit-yield reward shaping (keyed on the COMMANDED yield, not speed):
        # a strong penalty when yielding while it is SAFE (risk_low), plus a
        # duration escalation for prolonged yields. Bounded; default-safe values.
        self.yield_w_bad              = float(_yr.get("w_bad_yield", 0.15))
        self.yield_streak_grace       = int(_yr.get("yield_streak_grace", 8))
        self.yield_streak_grow_w      = float(_yr.get("yield_streak_grow_w", 0.01))
        self._yield_streak            = 0

        # Generic low-progress (stall) penalty — NOT human-gated. Fires whenever it
        # is safe (no human hazard AND low static proximity) yet the robot makes no
        # progress at low speed. This is the mid-episode timeout pressure that the
        # human-gated anti-freeze cannot express (it covers human-free stalls too).
        _st = dict(self.environment_config.get("stall", {}) or {})
        self.stall_enabled            = bool(_st.get("enabled", False))
        self.stall_w                  = float(_st.get("w_penalty", 0.02))
        self.stall_progress_eps       = float(_st.get("progress_eps", 0.0))
        self.stall_speed_mps          = float(_st.get("speed_threshold_mps", 0.15))
        self.stall_static_risk_thr    = float(_st.get("static_risk_thr", 0.2))

        # Anti-freeze penalty: discourages SUSTAINED, no-progress, low-speed holds
        # while a human hazard is active (the freeze→timeout failure mode). OPT-IN:
        # absent block / enabled:false → reward byte-for-byte unchanged. Needs the
        # same risk gating as yield_reward (human_risk_penalty.enabled). The streak
        # counter is owned here and reset per episode.
        _af = dict(self.environment_config.get("anti_freeze", {}) or {})
        self.antifreeze_enabled     = bool(_af.get("enabled", False))
        self.antifreeze_speed_mps   = float(_af.get("speed_threshold_mps", 0.12))
        self.antifreeze_min_streak  = int(_af.get("min_freeze_steps", 12))
        self.antifreeze_w           = float(_af.get("w_penalty", 0.02))
        self._freeze_streak         = 0

        # ── PHASE2 candidates: directional risk-map reward shaping + the
        # Action-Risk Head's env-side supervision target. Both consume the SAME
        # privileged GT risk_map geometry (directional_risk block) computed from
        # self.human_states, independent of aux_prediction.enabled — a single
        # near-term horizon, sector-binned exactly like aux_prediction_labels'
        # risk_map convention (theta=0 -> middle bin). Each feature has its OWN
        # enabled switch so the 4-way experiment matrix (baseline / reward-shaping
        # only / action-risk-head only / both) is reachable; the shared geometry
        # is just avoiding a duplicated CV-rollout implementation.
        _dr = dict(self.environment_config.get("directional_risk", {}) or {})
        self._directional_risk_cfg = aux_labels.AuxLabelConfig({
            "enabled": True,
            "horizons_sec": [float(_dr.get("horizon_sec", 1.0))],
            "num_sectors": int(_dr.get("num_sectors", 16)),
            "risk_distance_scale": float(_dr.get("risk_distance_scale", 3.0)),
            "min_speed_for_motion": float(_dr.get("min_speed_for_motion", 0.05)),
        })
        # speed_steering-only swept-path rollout dynamics (see pure_pursuit.
        # ackermann_swept_path) -- MUST mirror hunter_se_gazebo/config/
        # hunter_se_cmd_prefilter.yaml's accel_limit_mps2/brake_decel_mps2/
        # steering_rate_deg_s (a stale copy here would silently make the risk
        # target model a different vehicle than the one actually running).
        # Defaults match that file exactly, so an environment_curriculum.yaml
        # that never sets these keys still gets the REAL vehicle dynamics,
        # not an instant-response assumption.
        self._dr_rollout_accel_mps2 = float(_dr.get("rollout_accel_limit_mps2", 6.0))
        self._dr_rollout_brake_decel_mps2 = float(_dr.get("rollout_brake_decel_mps2", 6.0))
        self._dr_rollout_steering_rate_rad_s = math.radians(
            float(_dr.get("rollout_steering_rate_deg_s", 200.0)))
        self._dr_rollout_path_samples = int(_dr.get("rollout_path_samples", 15))
        # TRAJ_RISK: opt-in extension of the swept-path rollout target (built
        # for speed_steering, see _compute_directional_risk below) to the
        # waypoint_yield / legacy waypoint action modes. Default OFF -> those
        # modes keep the existing current-pose-only per-sector lookup, byte-
        # identical to before this flag existed.
        self._waypoint_trajectory_risk_enabled = bool(
            _dr.get("waypoint_trajectory_risk_enabled", False))

        # PRIVILEGED: risk_map_reward feeds the GT risk_map DIRECTLY into the
        # reward (training-time only; never an observation). Default OFF ->
        # reward is byte-identical to before this feature (see reward_calculator
        # .compute_reward's risk_map_reward_enabled docstring for the anti-
        # reward-hacking progress-positive gate).
        _rmr = dict(self.environment_config.get("risk_map_reward", {}) or {})
        # CLI override (PHASE2 experiment matrix): "" (default) -> config value;
        # -p risk_map_reward_enabled:=true/false forces it without editing YAML.
        self.declare_parameter("risk_map_reward_enabled", "")
        self.risk_map_reward_enabled = config_paths.parse_bool_override(
            self.get_parameter("risk_map_reward_enabled").get_parameter_value().string_value,
            bool(_rmr.get("enabled", False)),
        )
        self.risk_map_reward_penalty_w     = float(_rmr.get("penalty_weight", 0.3))
        self.risk_map_reward_bonus_w       = float(_rmr.get("bonus_weight", 0.15))
        self.risk_map_reward_bonus_max     = float(_rmr.get("bonus_max", 0.1))
        self.risk_map_reward_gate_eps      = float(_rmr.get("progress_positive_gate_eps", 0.0))
        self._prev_risk_dir = None  # reset per episode; None on episode start

        # speed_steering-only continuous-control shaping (reward_
        # calculator's continuous_control_reward_enabled block) -- heading-
        # error-delta reward + steering/speed continuous-control penalty.
        # reward_calculator itself re-gates on action_mode=="speed_steering",
        # so leaving this enabled for waypoint/waypoint_yield configs would
        # still be a no-op, but every profile that doesn't use speed_steering
        # simply omits the block (default OFF) -> reward byte-identical to
        # before this feature. Per-episode prev-state mirrors the
        # _prev_risk_dir pattern.
        _ccr = dict(self.environment_config.get("continuous_control_reward", {}) or {})
        self.continuous_control_reward_enabled = bool(_ccr.get("enabled", False))
        self.ccr_heading_delta_weight       = float(_ccr.get("heading_delta_weight", 0.2))
        self.ccr_steering_magnitude_weight  = float(_ccr.get("steering_magnitude_weight", 0.005))
        self.ccr_steering_change_weight     = float(_ccr.get("steering_change_weight", 0.01))
        self.ccr_speed_change_weight        = float(_ccr.get("speed_change_weight", 0.005))
        self._reset_continuous_control_reward_state()

        # Env-side mirror of the agent's action_risk_head.enabled (hyperparameters_
        # tqc.yaml). Independent of risk_map_reward.enabled AND aux_prediction.
        # enabled. When true, the env appends a tiny (risk_dir, min_dist_dir)
        # supervision-target wire block ahead of the (optional) aux label tail —
        # see _prepend_action_risk_target(). Default OFF -> no wire change.
        _arh = dict(self.environment_config.get("action_risk_head", {}) or {})
        # CLI override, same convention as risk_map_reward_enabled above.
        self.declare_parameter("action_risk_head_enabled", "")
        self.action_risk_head_env_enabled = config_paths.parse_bool_override(
            self.get_parameter("action_risk_head_enabled").get_parameter_value().string_value,
            bool(_arh.get("enabled", False)),
        )

        # Optional privileged counterfactual labels.  Each configured normalized
        # action is decoded with the exact same controller contract as a real
        # action, then scored at every horizon from the current pre-step state.
        _cf = dict(self.environment_config.get(
            "counterfactual_multi_horizon_risk", {}) or {})
        self.counterfactual_risk_env_enabled = bool(_cf.get("enabled", False))
        self.counterfactual_risk_horizons = [
            float(v) for v in _cf.get("horizons_sec", [0.5, 1.0, 1.5, 2.0])]
        self.counterfactual_candidate_actions = [
            list(map(float, row)) for row in _cf.get("candidate_actions", [])]
        if self.counterfactual_risk_env_enabled:
            if not self.counterfactual_risk_horizons or any(
                    h <= 0.0 for h in self.counterfactual_risk_horizons):
                raise RuntimeError(
                    "counterfactual_multi_horizon_risk.horizons_sec must be "
                    "non-empty and positive.")
            if not self.counterfactual_candidate_actions or any(
                    len(a) != self.action_dim
                    for a in self.counterfactual_candidate_actions):
                raise RuntimeError(
                    "counterfactual_multi_horizon_risk.candidate_actions must "
                    f"be a non-empty list of {self.action_dim}-D actions.")
            if any(abs(v) > 1.0 for a in self.counterfactual_candidate_actions
                   for v in a):
                raise RuntimeError(
                    "counterfactual candidate actions must be normalized to "
                    "[-1, 1].")

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
        # DYN_AVOID: privileged per-episode dynamic-obstacle (pedestrian)
        # avoidance diagnostics. The accumulator is fed the privileged robot +
        # human ground truth each /step; the resulting flat record is published as
        # a JSON string on the read-only `episode_dynamic_diag` parameter, which
        # the curriculum trainer reads once per episode (like current_map_type)
        # and writes into dynamic_avoidance_metrics_<run_tag>.csv. Thresholds are
        # ROS params so the diagnostic definitions are tunable without code edits.
        self.declare_parameter("dyn_diag_near_human_dist_m", 1.0)
        self.declare_parameter("dyn_diag_interaction_radius_m", 2.0)
        self.declare_parameter("dyn_diag_collision_attrib_radius_m", 0.7)
        self.declare_parameter("dyn_diag_ttc_collision_radius_m", 0.5)
        self.declare_parameter("dyn_diag_static_clutter_lidar_m", 0.6)
        self._dyn_diag = DynamicAvoidanceEpisodeDiag(
            near_human_dist_m=float(self.get_parameter("dyn_diag_near_human_dist_m").value),
            interaction_radius_m=float(self.get_parameter("dyn_diag_interaction_radius_m").value),
            collision_attrib_radius_m=float(self.get_parameter("dyn_diag_collision_attrib_radius_m").value),
            ttc_collision_radius_m=float(self.get_parameter("dyn_diag_ttc_collision_radius_m").value),
            static_clutter_lidar_m=float(self.get_parameter("dyn_diag_static_clutter_lidar_m").value),
        )
        self.declare_parameter("episode_dynamic_diag", "{}")
        # Cache of the last-published diag signature so per-step publishing is
        # skipped whenever the diagnostic CONTENT is unchanged (see
        # _publish_dynamic_diag). The trainer only reads this once per episode, so
        # this keeps the ROS parameter-update + /parameter_events cost off the hot
        # step path except when new dynamic-avoidance information actually appears.
        self._dyn_diag_last_key = None

        # PHASE1B risk-map-dump (eval-only, default OFF): a per-step JSON-string
        # parameter carrying the GT robot pose + a lightweight human-state
        # summary, mirroring the episode_dynamic_diag pattern above but
        # published EVERY step (not change-gated) so an external eval script
        # (risk_map_eval.py) can pair it with the per-step aux label it already
        # receives via the normal /step response (EnvInterface.last_aux_label /
        # .last_aux_meta). risk_map_dump_enabled gates the JSON-encode +
        # set_parameters cost, so training pays nothing when it is off.
        self.declare_parameter("risk_map_dump_enabled", False)
        self.declare_parameter("step_debug_state", "{}")

        # Writable flag the trainer raises around its evaluation episodes so the
        # SAME training env switches to the eval_map_types round-robin (instead of
        # the training map distribution) while STILL activating obstacles/humans.
        # Decouples "which maps are evaluated" from the train/test node mode.
        self._curriculum_eval_mode = False
        self.declare_parameter("curriculum_eval_mode", False)

        # PHASE1B (eval-only, default OFF -- byte-identical to prior behaviour
        # when unused): fixed evaluation scenario suite. When BOTH
        # fixed_eval_suite_enabled and curriculum_eval_mode are true, the reset
        # path (see _reset_callback_impl) reseeds the GLOBAL random/np.random
        # streams AND the dedicated human RNG sub-stream from a SUITE-LOCAL
        # episode index (fixed_eval_suite_base_seed, N) instead of the
        # ever-growing global episode counter, so map/start/goal/static/human
        # sampling for suite episode N is a pure, reproducible function of
        # (base_seed, N) -- independent of prior training/eval history. This
        # lets hard-eval, pilot judging and the final ablation all reuse the
        # SAME suite (same base_seed) and get directly comparable episodes.
        #
        # The episode-index counter resets to 0 on a curriculum_eval_mode
        # False->True TRANSITION -- but that transition is only OBSERVED
        # during an actual reset() call, so it is not a reliable "start a
        # fresh suite run" signal on its own: if a prior eval process left
        # curriculum_eval_mode sitting at True (e.g. it crashed, or simply
        # didn't restore it) and a NEW eval process sets it True again
        # without any reset() happening while it was briefly False, no
        # transition is ever seen and the suite silently continues from
        # wherever the previous process left off. fixed_eval_suite_reset_token
        # is the explicit, unambiguous alternative: an eval script sets it to
        # a fresh value (e.g. a timestamp) once at the start of its run, and
        # ANY change in its value forces the episode index back to 0 on the
        # very next reset() -- independent of eval-mode toggle history.
        self.declare_parameter("fixed_eval_suite_enabled", False)
        self.declare_parameter("fixed_eval_suite_base_seed", 0)
        self.declare_parameter("fixed_eval_suite_reset_token", 0)
        self._fixed_suite_episode_index = 0
        self._fixed_suite_eval_mode_prev = False
        self._fixed_suite_last_reset_token = 0
        self._fixed_suite_last_episode_index = None  # for eval-script logging

        # PHASE1B (eval-only, default OFF): "hard pedestrian" preset. When BOTH
        # hard_pedestrian_eval_enabled and curriculum_eval_mode are true, this
        # episode's human spawn draws from eval_human_mode_weights /
        # eval_human_mode_params (config, below) instead of the current
        # curriculum stage's mix -- reusing the EXISTING mode vocabulary
        # (crossing/along_path/waiting/slow_turn), just re-weighted toward
        # direction-change/stop-heavy modes. Applied+restored around ONE
        # episode's spawn call only (see _apply_hard_pedestrian_eval_override),
        # so it can never leak into training or non-hard-eval episodes.
        self.declare_parameter("hard_pedestrian_eval_enabled", False)
        _hpe = dict(self.environment_config.get("hard_pedestrian_eval", {}) or {})
        self._eval_human_mode_weights = dict(_hpe.get("human_mode_weights", {}) or {})
        self._eval_human_mode_params = dict(_hpe.get("human_mode_params", {}) or {})

        # PHASE1B (eval-only, default OFF): "robot-reactive pedestrian" preset.
        # Reuses the existing Falcon-lite human-human avoidance machinery
        # (_social_avoidance_offset, human_motion_manager.py) with a symmetric
        # robot-repulsion term, gated by this flag so training-time humans stay
        # non-reactive to the robot by default (required for the aux
        # constant-velocity labels to stay valid). Declared as a live ROS
        # parameter (not a plain attr) so an eval script can toggle it without
        # restarting Gazebo, mirroring curriculum_eval_mode.
        self.declare_parameter("human_robot_avoid_enabled", False)
        _hra_default = bool(self.environment_config.get("human_robot_avoid_enabled", False))
        if _hra_default:
            self.set_parameters(
                [Parameter("human_robot_avoid_enabled", Parameter.Type.BOOL, True)]
            )
        self.human_robot_avoid_radius = float(       # [m] robot influence radius
            self.environment_config.get("human_robot_avoid_radius", 2.0))
        self.human_robot_avoid_strength = float(     # blend weight of repulsion vs goal dir
            self.environment_config.get("human_robot_avoid_strength", 0.6))
        self.human_robot_avoid_max_heading_offset = math.radians(float(  # cap on the nudge
            self.environment_config.get("human_robot_avoid_max_heading_offset_deg", 25.0)))

        self.threshold_params_config = self.config["threshold_parameters"]
        self.goal_threshold = self.threshold_params_config["goal_threshold"]
        self.collision_threshold = self.threshold_params_config["collision_threshold"]
        self.time_delta = self.threshold_params_config["time_delta"]
        if self.human_deterministic_stepping:
            # Fail-fast at startup (not lazily at the first step) if time_delta
            # isn't an exact multiple of the human-motion tick period.
            compute_human_tick_plan(self.time_delta, self.human_update_rate)
        if self.gazebo_deterministic_stepping:
            # Fail-fast at startup if time_delta isn't an exact multiple of
            # physics_step_size (see gazebo_service_wait.compute_physics_step_count).
            compute_physics_step_count(self.time_delta, self.gazebo_physics_step_size)
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
        # human_deterministic_stepping (default OFF) drives ticks explicitly
        # from propagate_state() instead -- no independent timer is created,
        # so there is nothing left to decouple from wall-clock scheduling.
        obstacle_update_rate = self.human_update_rate
        if self.num_of_humans > 0 and not self.human_deterministic_stepping:
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
                from drl_agent.evaluation.sim_validation import SimValidationLogger
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
            low_speed_distance_m=self.controller_low_speed_distance_m,
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

    def _apply_hard_pedestrian_eval_override(self):
        """PHASE1B hard-pedestrian-eval (default OFF): if enabled AND
        curriculum_eval_mode is on, swap self.human_mode_weights/params to the
        eval-only preset for the duration of THIS episode's human spawn. Saves
        and returns whatever was set beforehand (may be the current curriculum
        stage's mix, or None if nothing changes) so the caller can restore it
        immediately after spawning -- see _restore_hard_pedestrian_eval_override.
        No-op (returns None) when the flag is off or no preset is configured,
        so training / normal eval episodes are completely unaffected."""
        enabled = bool(self.get_parameter("hard_pedestrian_eval_enabled").value)
        eval_mode_now = bool(self.get_parameter("curriculum_eval_mode").value)
        if not (enabled and eval_mode_now):
            return None
        if not self._eval_human_mode_weights:
            self.get_logger().warn(
                "[HardPedEval] hard_pedestrian_eval_enabled=true but no "
                "hard_pedestrian_eval.human_mode_weights configured -- using "
                "the current stage's mix unchanged this episode."
            )
            return None
        saved = (dict(self.human_mode_weights), dict(self.human_mode_params))
        self.human_mode_weights = dict(self._eval_human_mode_weights)
        if self._eval_human_mode_params:
            merged = {k: dict(v) for k, v in self.human_mode_params.items()}
            for mode, overrides in self._eval_human_mode_params.items():
                merged.setdefault(mode, {}).update(overrides)
            self.human_mode_params = merged
        return saved

    def _restore_hard_pedestrian_eval_override(self, saved):
        """Undo _apply_hard_pedestrian_eval_override (no-op if it was a no-op)."""
        if saved is None:
            return
        self.human_mode_weights, self.human_mode_params = saved

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
            "human_min_ttc_s",
            "reward_yield_bonus",
            "penalty_idle",
            "penalty_freeze",
            "penalty_risk_map",
            "reward_risk_map_bonus",
            "reward_heading_delta",
            "penalty_steering_magnitude",
            "penalty_steering_change",
            "penalty_speed_change",
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
            # this file: <pkg_root>/drl_agent/env/simulation/environment.py
            os.path.normpath(os.path.join(os.path.dirname(here), "..", "..", "..")),
        ])

        for cand in candidates:
            if os.path.isdir(cand) and os.path.basename(cand) == "drl_agent":
                return os.path.normpath(cand)

        return os.path.normpath(os.path.join(os.path.dirname(here), "..", "..", ".."))

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
        GazeboServiceError 가 콜백 밖으로 전파된다(여기서 삼키지 않는다).

        human_deterministic_stepping (default OFF) replaces this single
        physics advance with an integer number of fixed-dt human-motion ticks
        interleaved with equal physics sub-windows (see
        _propagate_state_deterministic_human_ticks). gazebo_deterministic_
        stepping (default OFF, independent of the above) additionally
        replaces each physics sub-window's legacy unpause/sleep/pause with a
        single exact-step-count multi_step call (see _advance_physics).
        Byte-identical to the legacy path when both are disabled."""
        if self.human_deterministic_stepping:
            self._propagate_state_deterministic_human_ticks(time_delta)
            return
        self.get_logger().debug(
            f"[gazebo] propagate: unpause → run {time_delta:.3f}s")
        self._advance_physics(time_delta)
        self.get_logger().debug("[gazebo] propagate: re-paused")

    def _propagate_state_deterministic_human_ticks(self, time_delta):
        """Split time_delta into compute_human_tick_plan()'s exact tick count,
        advancing human motion by one fixed-dt tick immediately before each
        equal physics sub-window -- so the number of human-motion ticks and
        their dt are fixed by (time_delta, human_update_rate) alone, never by
        wall-clock scheduling speed."""
        n_ticks, sub_dt = compute_human_tick_plan(time_delta, self.human_update_rate)
        self.get_logger().debug(
            f"[gazebo] propagate(deterministic): {n_ticks} ticks × {sub_dt:.3f}s "
            f"= {time_delta:.3f}s")
        for _ in range(n_ticks):
            self._advance_humans_one_tick(sub_dt)
            self._advance_physics(sub_dt)
        self.get_logger().debug("[gazebo] propagate(deterministic): re-paused")

    def _advance_physics(self, duration):
        """Advance Gazebo physics by `duration` seconds: the shared primitive
        used by BOTH propagate_state() and its human-tick-interleaved variant.
        Legacy unpause/sleep(duration)/pause (2 world-control calls), or
        (gazebo_deterministic_stepping) a single exact-step-count multi_step
        call + sleep(duration) + sensor-freshness wait (see
        _advance_physics_deterministic; the stricter /clock completion wait
        was excluded after live /step hangs) -- byte-identical to the legacy
        body below when disabled."""
        if self.gazebo_deterministic_stepping:
            self._advance_physics_deterministic(duration)
            return
        self.pause_world(False)
        time.sleep(duration)
        self.pause_world(True)

    def _multi_step_world(self, n_steps: int):
        """One WorldControl call: pause=True + multi_step=n_steps steps physics
        by EXACTLY n_steps * gazebo_physics_step_size seconds of sim time then
        leaves the world paused (empirically confirmed live). Failure
        propagates as GazeboServiceError, matching pause_world/reset_world."""
        srv_name = f"/world/{self.world_name}/control"
        req = ControlWorld.Request()
        req.world_control.pause = True
        req.world_control.multi_step = int(n_steps)
        self._call_world_service(self.world_control, req, srv_name,
                                  f"multi_step[{n_steps}]")

    def _wait_for_sensor_freshness(self, prev_scan_updates, prev_role_updates):
        """Bounded wait for a NEW scan + odom (all roles) update after a
        multi_step call, mirroring reset_callback's existing freshness check.
        Warn-and-continue on timeout (not fail-fast): a one-tick-stale sensor
        reading is an already-tolerated degradation elsewhere in this env,
        unlike the /clock target above."""
        t0 = time.time()
        while (
            self.scan_update_count <= prev_scan_updates
            or any(self._odom_role_count[r] <= prev_role_updates[r]
                   for r in ("gt", "loc", "proprio"))
        ) and (time.time() - t0 < self.gazebo_sensor_wait_timeout_sec):
            rclpy.spin_once(self, timeout_sec=0.01)
        stale = [r for r in ("gt", "loc", "proprio")
                 if self._odom_role_count[r] <= prev_role_updates[r]]
        if stale or self.scan_update_count <= prev_scan_updates:
            self.get_logger().warn(
                f"[gazebo] post-multi_step sensor freshness: scan stale="
                f"{self.scan_update_count <= prev_scan_updates} odom stale roles={stale} "
                f"— caches may be one tick behind.")

    def _advance_physics_deterministic(self, duration):
        """multi_step requests an EXACT physics-step count (n_steps below) in
        ONE world-control call -- replacing the legacy path's two calls
        (unpause + pause) with one, which is the actual measured source of
        this mode's speedup (live-measured: 203.6ms/step legacy vs.
        152.3ms/step here, both averaged over independent runs). A per-step
        bounded /clock wait for exact sim-time confirmation was attempted
        (WorldControl(pause=True, multi_step=N) itself is confirmed exact:
        N=1/50/100 -> sim-time deltas of exactly 0.001/0.05/0.1s in isolation)
        but reproducibly hung real /step calls under the live node's
        MultiThreadedExecutor for reasons not fully root-caused (same failure
        CLASS as the _await_future polling-interval investigation: an
        isolated probe against the same /clock topic worked reliably every
        time, but the identical wait embedded in this node's fuller
        callback-group/threading context did not) -- see the
        gazebo_multi_step_clock_wait_landmine memory note. Excluded; this
        method instead sleeps for `duration` (matching the legacy path's own
        approach, which does the same with NO freshness check at all) and
        then runs _wait_for_sensor_freshness -- which DOES work reliably from
        step_callback (reuses reset_callback's proven scan/odom-count-based
        wait unmodified) -- so a genuinely stale LiDAR/odom reading is still
        caught and logged, even without exact sim-clock confirmation."""
        n_steps = compute_physics_step_count(duration, self.gazebo_physics_step_size)
        prev_scan = self.scan_update_count
        prev_roles = dict(self._odom_role_count)
        self.get_logger().debug(
            f"[gazebo] multi_step: {n_steps} × {self.gazebo_physics_step_size:.4f}s "
            f"= {duration:.3f}s")
        self._multi_step_world(n_steps)
        time.sleep(duration)
        self._wait_for_sensor_freshness(prev_scan, prev_roles)
    # ----------------------------------------------------------------------------------------------
    # ====================================Ignition Finish===========================================
    # ----------------------------------------------------------------------------------------------

    def _swept_path_risk(self, target_v, target_cmd_steering,
                         horizon_sec=None, humans=None):
        """TRAJ_RISK: action-conditioned GLOBAL swept-path (risk, min_dist) for
        an arbitrary candidate command (target_v, target_cmd_steering), reused
        by both the speed_steering action mode and, when
        directional_risk.waypoint_trajectory_risk_enabled is on,
        waypoint_yield/legacy waypoint. NOT tied to any one action_mode or a
        single "selected" action -- a future counterfactual-label caller
        (e.g. evaluating left/weak-left/straight/weak-right/right/yield
        candidates) can call this directly with each candidate's own decoded
        (v, cmd_steering) without touching _compute_directional_risk's
        dispatch at all.

        ``humans``: optional PRE-FETCHED list (same dict-per-human shape this
        method would otherwise build itself under ``self._human_lock``). Pass
        a snapshot taken once by the caller to avoid re-acquiring the lock and
        rebuilding the list on every one of many rollouts against the SAME
        pre-step state (see _compute_counterfactual_risk_targets, which
        evaluates (candidates + the executed action) x horizons from one
        snapshot). None (default) preserves the original single-call
        behaviour used by _compute_directional_risk.

        The robot's trajectory is built by pure_pursuit.ackermann_swept_path
        from its CURRENT actual speed/steering (self.latest_actual_signed_
        speed / self.latest_center_steering) ramping toward (target_v,
        target_cmd_steering) under hunter_se_cmd_prefilter's own accel/brake/
        steering-rate limits -- NOT an instant jump to the target on this same
        tick. This is what lets a decelerating/stopping (or YIELDing) action
        genuinely read as lower risk than driving at full speed toward the
        same person, while still crediting the real residual motion a stop
        command doesn't instantly cancel (e.g. braking from 2 m/s covers
        ~0.33 m before actually stopping, at this vehicle's tuned limits) --
        i.e. YIELD is never scored as an instantaneous stop. Distance is the
        GLOBAL minimum over the whole swept path vs. every human's own
        constant-velocity path (matching sample times), with NO re-filtering
        by bearing sector -- a per-sector lookup would silently drop a
        crossing pedestrian whose OWN bearing sector differs from the action's
        heading sector even though the two paths nearly intersect, and would
        miss a pedestrian who is only close at some time STRICTLY BETWEEN now
        and the horizon (including while the robot itself is stopped).

        Returns (risk, min_dist), both in [0, 1].
        """
        if humans is None:
            with self._human_lock:
                humans = [
                    {"x": s["x"], "y": s["y"],
                     "yaw": s.get("yaw", 0.0), "v": s.get("v", 0.0)}
                    for s in self.human_states.values()
                ]
        cfg = self._directional_risk_cfg
        horizon_sec = (
            float(cfg.horizons_sec[0]) if horizon_sec is None and cfg.horizons_sec
            else float(horizon_sec or 0.0))
        robot_path = pure_pursuit.ackermann_swept_path(
            self.latest_actual_signed_speed, self.latest_center_steering,
            target_v, target_cmd_steering, self.vehicle_wheelbase_m, horizon_sec,
            accel_limit_mps2=self._dr_rollout_accel_mps2,
            brake_decel_mps2=self._dr_rollout_brake_decel_mps2,
            steering_rate_rad_s=self._dr_rollout_steering_rate_rad_s,
            num_samples=self._dr_rollout_path_samples,
        )
        risk, min_dist = aux_labels.compute_action_conditioned_risk(
            humans, (self.gt_x, self.gt_y, self.gt_yaw), cfg, robot_path)
        return float(risk), float(min_dist)

    def _decode_counterfactual_action(self, action):
        """Decode a normalized candidate without publishing or mutating state."""
        if self.action_mode == "speed_steering":
            return pure_pursuit.speed_steering_action_to_command(
                action, self.controller_cruise_speed_mps,
                self.vehicle_steering_limit_rad)
        if self.action_mode == "waypoint_yield":
            v, steering, _theta, _ctl = pure_pursuit.hybrid_action_to_command(
                action, self.actions_low, self.actions_high,
                self.vehicle_wheelbase_m, self.vehicle_steering_limit_rad,
                self.controller_cruise_speed_mps,
                self.controller_speed_steer_factor,
                yield_enabled=self.yield_action_enabled,
                yield_threshold=self.yield_action_threshold,
                lookahead_min_m=self.controller_lookahead_min_m,
                v_move_min_mps=self.controller_v_move_min_mps,
                yield_creep_speed_mps=self.controller_yield_creep_mps)
            return v, steering
        _r, _theta, x_wp, y_wp = self._map_action_to_waypoint(action)
        return self._controller_waypoint_to_command(x_wp, y_wp)

    def _compute_counterfactual_risk_targets(self, executed_v=None,
                                             executed_cmd_steering=None):
        """Return (fixed_candidate_targets, executed_action_target).

        fixed_candidate_targets : (M, H) ndarray, swept-path closeness-risk
            for each configured normalized candidate action, one column per
            configured horizon -- unchanged semantics from before.
        executed_action_target  : (H,) ndarray for the action ACTUALLY
            executed this step (``executed_v``/``executed_cmd_steering``,
            already decoded by the step callback's own controller-contract
            dispatch -- NOT re-decoded here), or None when
            ``executed_v`` is None.

        Both share ONE human-state snapshot taken under self._human_lock --
        previously each of the (candidates x horizons) rollouts re-acquired
        the lock and rebuilt the humans list from scratch. Must be called
        BEFORE propagate_state() (see the call site in
        _step_callback_impl), same pre-step-state contract as
        _compute_directional_risk.
        """
        with self._human_lock:
            humans = [
                {"x": s["x"], "y": s["y"],
                 "yaw": s.get("yaw", 0.0), "v": s.get("v", 0.0)}
                for s in self.human_states.values()
            ]
        rows = []
        for action in self.counterfactual_candidate_actions:
            target_v, target_steering = Environment._decode_counterfactual_action(
                self, np.asarray(action, dtype=np.float32))
            rows.append([
                Environment._swept_path_risk(
                    self, target_v, target_steering, horizon_sec=h,
                    humans=humans)[0]
                for h in self.counterfactual_risk_horizons
            ])
        executed_row = None
        if executed_v is not None:
            executed_row = np.asarray([
                Environment._swept_path_risk(
                    self, executed_v, executed_cmd_steering, horizon_sec=h,
                    humans=humans)[0]
                for h in self.counterfactual_risk_horizons
            ], dtype=np.float32)
        return np.asarray(rows, dtype=np.float32), executed_row

    def _compute_directional_risk(self, theta, target_v=0.0, target_cmd_steering=0.0):
        """PHASE2: pre-step GT risk for the action just decoded. Shared by
        risk_map_reward (fed straight into get_reward) and the Action-Risk
        Head's env-side supervision target (wired onto the response) -- one
        privileged CV-rollout computation, two independent consumers/
        switches.

        MUST be called with self.human_states / GT robot pose as they stand
        BEFORE this step's motion (i.e. before propagate_state()) -- see the
        call site in _step_callback_impl -- so neither consumer ever sees
        post-step information leak into an action-credit signal.

        Two DIFFERENT computations depending on self.action_mode AND (for
        waypoint modes) directional_risk.waypoint_trajectory_risk_enabled:

        * action_mode == "speed_steering", OR (action_mode in
          {"waypoint_yield", "waypoint"} AND
          self._waypoint_trajectory_risk_enabled): action-conditioned GLOBAL
          swept-path risk -- see _swept_path_risk's docstring for the full
          rationale. Isolated to speed_steering by default so
          phase2/waypoint_yield and the legacy waypoint contract's semantics
          are UNCHANGED unless the profile explicitly opts in via that flag.
        * every other case (unchanged from before either feature existed):
          per-SECTOR risk (aux_prediction_labels.compute_directional_risk_map,
          current-pose-only, no target_v/target_cmd_steering dependence at
          all), sliced at the sector `theta` points into.

        Returns (risk_dir, min_dist_dir), both in [0, 1].
        """
        use_swept_path = (
            self.action_mode == "speed_steering"
            or (
                self._waypoint_trajectory_risk_enabled
                and self.action_mode in ("waypoint_yield", "waypoint")
            )
        )
        if use_swept_path:
            # Explicit class-qualified call (not self._swept_path_risk(...)):
            # this method is exercised in tests via
            # Environment._compute_directional_risk(fake_node, ...) where
            # fake_node is a plain types.SimpleNamespace, not a real
            # Environment instance -- self.<method> lookup would fail to find
            # _swept_path_risk on it, since it isn't in the object's MRO.
            return Environment._swept_path_risk(self, target_v, target_cmd_steering)
        with self._human_lock:
            humans = [
                {"x": s["x"], "y": s["y"],
                 "yaw": s.get("yaw", 0.0), "v": s.get("v", 0.0)}
                for s in self.human_states.values()
            ]
        cfg = self._directional_risk_cfg
        risk_row, min_dist_row = aux_labels.compute_directional_risk_map(
            humans, (self.gt_x, self.gt_y, self.gt_yaw), cfg)
        sector = aux_labels.sector_index_for_theta(theta, cfg.num_sectors)
        return float(risk_row[sector]), float(min_dist_row[sector])

    def _append_aux_labels(self, state_array, action_risk_target=None,
                           counterfactual_risk_target=None,
                           counterfactual_executed_target=None):
        """AUX_PRED: return state_array (list) with the privileged future-risk
        label appended.  When disabled, returns the plain state list so the
        wire format is identical to baseline.

        Robot pose uses the ground-truth pose (privileged); pedestrian motion is
        read from self.human_states (privileged sim state).  Nothing here is
        needed at inference -- the trainer simply stops slicing when labels are
        absent, and the deployed policy is encoder + actor only.

        PHASE2: when ``action_risk_head_env_enabled`` and ``action_risk_target``
        (risk_dir, min_dist_dir) is supplied, PREPEND that action-conditioned
        block ahead of the (optional) aux_prediction tail below -- see
        aux_prediction_labels.strip_action_risk_wire for the wire format. The
        two tails are independent: this block is always absent on /reset
        responses (no action has been taken yet at reset).
        """
        state_list = np.asarray(state_array, dtype=np.float32).ravel().tolist()
        if (getattr(self, "counterfactual_risk_env_enabled", False)
                and counterfactual_risk_target is not None):
            state_list.extend(aux_labels.counterfactual_risk_wire(
                counterfactual_risk_target,
                self.counterfactual_risk_horizons))
        if (getattr(self, "counterfactual_risk_env_enabled", False)
                and counterfactual_executed_target is not None):
            state_list.extend(aux_labels.executed_action_risk_wire(
                counterfactual_executed_target,
                self.counterfactual_risk_horizons))
        if self.action_risk_head_env_enabled and action_risk_target is not None:
            risk_dir, min_dist_dir = action_risk_target
            state_list.extend([
                aux_labels.ACTION_RISK_WIRE_SENTINEL,
                float(risk_dir), float(min_dist_dir),
            ])
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

    def _publish_dynamic_diag(self, force=False):
        """DYN_AVOID: push the current per-episode dynamic-avoidance diagnostics
        onto the read-only `episode_dynamic_diag` parameter as a JSON string.

        The curriculum trainer reads this once per episode (right after the final
        step, before /reset). Because the env cannot know which step is the last
        one of a trainer-owned episode (timeouts have no env-side `done`), the
        value must stay current every step — but the actual parameter write + the
        /parameter_events publication are only paid when the diagnostic CONTENT
        changes (cheap ``state_key`` compare), so unchanged steps (e.g. human-free
        stages, or steps with no new proximity/yield information) cost nothing on
        the ROS side. `force=True` publishes unconditionally (used on reset so a
        read before the first step never returns stale data). NaN sentinels
        round-trip via Python json (allow_nan) on both ends."""
        key = self._dyn_diag.state_key()
        if not force and key == self._dyn_diag_last_key:
            return
        self._dyn_diag_last_key = key
        try:
            payload = json.dumps(self._dyn_diag.as_dict())
        except Exception:
            payload = "{}"
        self.set_parameters([
            Parameter("episode_dynamic_diag", Parameter.Type.STRING, payload)
        ])

    def _publish_step_debug_state(self, action_r=None, action_theta=None):
        """PHASE1B risk-map-dump (eval-only, default OFF -- see the
        declare_parameter comment above): publish THIS step's GT robot pose +
        the selected waypoint (r, theta, when called from a step) + a
        lightweight human-state summary as a JSON string on the
        ``step_debug_state`` parameter. Only called when risk_map_dump_enabled
        is true, so it costs nothing otherwise. Never raises: a serialization
        failure falls back to "{}" (best-effort logging, never breaks a step)."""
        try:
            with self._human_lock:
                humans = [
                    {"x": s["x"], "y": s["y"], "yaw": s.get("yaw", 0.0),
                     "v": s.get("v", 0.0), "mode": s.get("mode", "")}
                    for s in self.human_states.values()
                ]
            payload = json.dumps({
                "robot_x": self.gt_x, "robot_y": self.gt_y, "robot_yaw": self.gt_yaw,
                "action_r": None if action_r is None else float(action_r),
                "action_theta": None if action_theta is None else float(action_theta),
                "humans": humans,
            })
        except Exception:
            payload = "{}"
        self.set_parameters([
            Parameter("step_debug_state", Parameter.Type.STRING, payload)
        ])

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

        # 1) 액션 → 제어 명령. Dispatch is keyed on self.action_mode (not just
        #    action_dim, since both "waypoint" and "speed_steering" are 2-D):
        #    - "speed_steering" (new): action=[speed, steer], no waypoint
        #      geometry, no binary yield channel -- speed 0 is a genuine
        #      continuous stop. `theta`/`r` are repurposed as the Ackermann-
        #      rollout heading/distance (see pure_pursuit.ackermann_rollout) so
        #      the SAME downstream directional-risk / reward / CSV code below
        #      (which only reads the shared v/cmd_steering/theta/r/yielding
        #      variables) needs no further branching.
        #    - "waypoint_yield" (3D hybrid): action=[r, theta, yield]. 비-yield
        #      (MOVE)에서는 r>=L_min, speed>=v_move_min 으로 강제되어 물리적으로
        #      정지 불가; yield에서만 creep/정지 허용.
        #    - "waypoint" (legacy 2D ablation): 기존 2-call 경로 그대로 사용 →
        #      2D 계약은 byte-identical 유지.
        if self.action_mode == "speed_steering":
            v, cmd_steering = pure_pursuit.speed_steering_action_to_command(
                action, self.controller_cruise_speed_mps,
                self.vehicle_steering_limit_rad,
            )
            theta, r = pure_pursuit.ackermann_rollout(
                v, cmd_steering, self.vehicle_wheelbase_m,
                float(self._directional_risk_cfg.horizons_sec[0]),
            )
            yielding = False
        elif self.action_mode == "waypoint_yield":
            v, cmd_steering, theta, ctl = pure_pursuit.hybrid_action_to_command(
                action, self.actions_low, self.actions_high,
                self.vehicle_wheelbase_m, self.vehicle_steering_limit_rad,
                self.controller_cruise_speed_mps, self.controller_speed_steer_factor,
                yield_enabled=self.yield_action_enabled,
                yield_threshold=self.yield_action_threshold,
                lookahead_min_m=self.controller_lookahead_min_m,
                v_move_min_mps=self.controller_v_move_min_mps,
                yield_creep_speed_mps=self.controller_yield_creep_mps,
            )
            r = float(ctl["r"])          # MOVE-floored / raw waypoint distance for logging
            yielding = bool(ctl["yielding"])
        else:  # "waypoint" (legacy 2D ablation)
            r, theta, x_wp, y_wp = self._map_action_to_waypoint(action)
            v, cmd_steering = self._controller_waypoint_to_command(x_wp, y_wp)
            yielding = False

        # PHASE2: pre-step (decision-time) directional GT risk for the JUST-
        # DECODED waypoint direction `theta` -- MUST run before propagate_state()
        # below so self.human_states / GT pose reflect the state the policy
        # actually conditioned on, not this step's motion outcome. Shared by
        # risk_map_reward (reward shaping) and the Action-Risk Head's env-side
        # supervision target (wired onto the response further down); each has
        # its own independent enable switch.
        action_risk_dir = action_risk_min_dist = None
        if self.risk_map_reward_enabled or self.action_risk_head_env_enabled:
            action_risk_dir, action_risk_min_dist = self._compute_directional_risk(
                theta, v, cmd_steering)
        counterfactual_risk_target = None
        counterfactual_executed_target = None
        if self.counterfactual_risk_env_enabled:
            counterfactual_risk_target, counterfactual_executed_target = (
                self._compute_counterfactual_risk_targets(v, cmd_steering))

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

        # DYN_AVOID: fold this step's PRIVILEGED pedestrian geometry + the actual
        # (per-stage-gated) yield decision into the per-episode dynamic-avoidance
        # diagnostics, then republish them for the trainer's diagnostic CSV. Uses
        # human_states (privileged), so it works even for an aux-OFF baseline.
        # Contract: human_states holds ONLY the CURRENTLY-active pedestrians (it is
        # cleared on reset and repopulated per episode), so "human_observed_steps"
        # and "min_human_distance_m" are defined over the active set — matching how
        # the reward path and aux labels read the same dict.
        with self._human_lock:
            _diag_humans = [
                {"x": s["x"], "y": s["y"], "v": s.get("v", 0.0),
                 "yaw": s.get("yaw", 0.0), "mode": s.get("mode", "")}
                for s in self.human_states.values()
            ]
        self._dyn_diag.update(
            (self.gt_x, self.gt_y, self.gt_yaw),
            self.latest_actual_signed_speed,
            _diag_humans, lidar_min, bool(collision),
            yielding=bool(yielding),
            yield_available=bool(self.action_dim >= 3 and self.yield_action_enabled),
        )
        self._publish_dynamic_diag()

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
            yield_enabled=self.yield_enabled,
            yield_risk_gate_ttc_s=self.yield_risk_gate_ttc_s,
            yield_risk_gate_approach=self.yield_risk_gate_approach,
            yield_w_bonus=self.yield_w_bonus,
            yield_max_bonus=self.yield_max_bonus,
            yield_idle_w=self.yield_idle_w,
            yield_idle_speed_mps=self.yield_idle_speed_mps,
            yield_step_relief_scale=self.yield_step_relief_scale,
            cruise_speed_mps=self.controller_cruise_speed_mps,
            antifreeze_enabled=self.antifreeze_enabled,
            antifreeze_speed_mps=self.antifreeze_speed_mps,
            antifreeze_min_streak=self.antifreeze_min_streak,
            antifreeze_w=self.antifreeze_w,
            freeze_streak_in=self._freeze_streak,
            # 3D-action explicit yield + generic stall shaping.
            yield_action=yielding,
            yield_w_bad=self.yield_w_bad,
            yield_streak_in=self._yield_streak,
            yield_streak_grace=self.yield_streak_grace,
            yield_streak_grow_w=self.yield_streak_grow_w,
            stall_enabled=self.stall_enabled,
            stall_w=self.stall_w,
            stall_progress_eps=self.stall_progress_eps,
            stall_speed_mps=self.stall_speed_mps,
            stall_static_risk_thr=self.stall_static_risk_thr,
            # PHASE2: directional risk-map reward shaping (0 unless enabled).
            risk_map_reward_enabled=self.risk_map_reward_enabled,
            risk_dir=action_risk_dir,
            min_dist_dir=action_risk_min_dist,
            prev_risk_dir=self._prev_risk_dir,
            risk_penalty_weight=self.risk_map_reward_penalty_w,
            risk_bonus_weight=self.risk_map_reward_bonus_w,
            risk_bonus_max=self.risk_map_reward_bonus_max,
            progress_positive_gate_eps=self.risk_map_reward_gate_eps,
            # speed_steering continuous-control shaping (0 unless
            # enabled AND action_mode=="speed_steering").
            continuous_control_reward_enabled=self.continuous_control_reward_enabled,
            action_mode=self.action_mode,
            heading_delta_weight=self.ccr_heading_delta_weight,
            prev_theta_err=self._prev_theta_err,
            cmd_steering=cmd_steering,
            prev_cmd_steering=self._prev_ccr_cmd_steering,
            prev_cmd_speed=self._prev_ccr_cmd_speed,
            steering_limit_rad=self.vehicle_steering_limit_rad,
            steering_magnitude_weight=self.ccr_steering_magnitude_weight,
            steering_change_weight=self.ccr_steering_change_weight,
            speed_change_weight=self.ccr_speed_change_weight,
            return_terms=True,
        )
        # Carry the anti-freeze / yield streaks into the next step (reset per episode).
        self._freeze_streak = int(reward_terms.get("freeze_streak", 0))
        self._yield_streak  = int(reward_terms.get("yield_streak", 0))
        self._prev_waypoint_theta = theta
        # PHASE2: carry this step's directional risk into the next step's bonus
        # comparison (None on episode start / when risk_map_reward is off and
        # action_risk_head didn't compute it either).
        self._prev_risk_dir = action_risk_dir
        # carry this step's heading error / commanded steering+speed
        # into the next step's continuous-control shaping (None on episode
        # start / no-op unless continuous_control_reward_enabled).
        self._prev_theta_err = theta_err
        self._prev_ccr_cmd_steering = cmd_steering
        self._prev_ccr_cmd_speed = v

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
                round(min(float(reward_terms["human_min_ttc_s"]), 99.0), 6),
                round(float(reward_terms["yield_bonus"]), 6),
                round(float(reward_terms["idle_pen"]), 6),
                round(float(reward_terms["freeze_pen"]), 6),
                round(float(reward_terms["risk_map_penalty"]), 6),
                round(float(reward_terms["risk_map_bonus"]), 6),
                round(float(reward_terms["reward_heading_delta"]), 6),
                round(float(reward_terms["penalty_steering_magnitude"]), 6),
                round(float(reward_terms["penalty_steering_change"]), 6),
                round(float(reward_terms["penalty_speed_change"]), 6),
                round(float(reward_terms["terminal"]), 6),
                round(float(reward), 6),
                int(bool(collision)), int(bool(target)), int(bool(done)),
            ])

        # 10) 응답
        # AUX_PRED: append privileged future-risk label (no-op when disabled).
        # PHASE2: also prepend the Action-Risk Head's action-conditioned target
        # for THIS transition (no-op unless action_risk_head_env_enabled).
        _action_risk_target = (
            (action_risk_dir, action_risk_min_dist)
            if action_risk_dir is not None else None
        )
        response.state = self._append_aux_labels(
            state, action_risk_target=_action_risk_target,
            counterfactual_risk_target=counterfactual_risk_target,
            counterfactual_executed_target=counterfactual_executed_target)
        response.reward = float(reward)
        response.done   = bool(done)
        response.target = bool(target)
        # RISK_BALANCE: the FULL-360 collision verdict + min distance (the
        # SAME `min_used` check_collision() itself used above), so a caller
        # (e.g. the replay buffer's risk_meta) never has to approximate
        # collision proximity from the front-180 obs_state slice of `state`
        # (which shares environment_dim's length with, but is NOT, the
        # full-360 environment_state check_collision() actually evaluates).
        response.collision = bool(collision)
        response.min_obstacle_dist_m = (
            float(min_used) if math.isfinite(min_used) else float(self.lidar_max_range)
        )
        # PHASE1B risk-map-dump (eval-only, default OFF): publish this step's
        # GT pose + selected waypoint + human summary for an external eval
        # script to pair with the aux label above. No-op cost when the flag
        # is off (the default).
        if bool(self.get_parameter("risk_map_dump_enabled").value):
            self._publish_step_debug_state(action_r=r, action_theta=theta)
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

    def _reset_continuous_control_reward_state(self):
        """Clear continuous_control_reward's per-episode carry state
        (heading error + commanded steering/speed from the PREVIOUS step) so
        the new episode's first step sees them as None -- this is what makes
        reward_calculator.compute_reward's heading-delta/change-penalty terms
        exactly 0 on an episode's first step (see its continuous_control_
        reward_enabled docstring). Called once from __init__ (first-init
        default) and once per episode from reset_callback; factored out as
        its own method (mirroring no other per-episode reset field here, but
        the smallest sub-block worth isolating) so it is unit-testable as a
        real code path, not just via pure-function inputs."""
        self._prev_theta_err = None
        self._prev_ccr_cmd_steering = None
        self._prev_ccr_cmd_speed = None

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

        # PHASE1B fixed-eval-suite (default OFF -- see the declare_parameter
        # comment above): when active, reseed the GLOBAL random/np.random
        # streams from a SUITE-LOCAL episode index (not the ever-growing
        # _episode_count) BEFORE any start/goal/static sampling below draws
        # from them, and re-derive the human sub-stream from that same index.
        # No-op / byte-identical to prior behaviour when the flag is off.
        _suite_enabled = bool(self.get_parameter("fixed_eval_suite_enabled").value)
        _eval_mode_now = bool(self.get_parameter("curriculum_eval_mode").value)
        _reset_token = int(self.get_parameter("fixed_eval_suite_reset_token").value)
        # Reset on EITHER an eval-mode toggle OR an explicit reset-token change
        # (see the declare_parameter comment above for why the toggle alone is
        # not reliable across separate eval-script invocations).
        if _suite_enabled and (
            _eval_mode_now != self._fixed_suite_eval_mode_prev
            or _reset_token != self._fixed_suite_last_reset_token
        ):
            self._fixed_suite_episode_index = 0
        self._fixed_suite_eval_mode_prev = _eval_mode_now
        self._fixed_suite_last_reset_token = _reset_token
        if _suite_enabled and _eval_mode_now:
            _suite_seed = int(self.get_parameter("fixed_eval_suite_base_seed").value)
            _suite_idx = self._fixed_suite_episode_index
            self._fixed_suite_episode_index += 1
            _episode_seed = seed_utils.derive_resume_seed(_suite_seed, _suite_idx)
            seed_utils.seed_basic_rngs(_episode_seed)
            _human_seed_base, _human_episode_index = _suite_seed, _suite_idx
            self._fixed_suite_last_episode_index = _suite_idx
        else:
            _human_seed_base = getattr(self, "_human_rng_base_seed", self.pool_build_seed)
            _human_episode_index = self._episode_count
            self._fixed_suite_last_episode_index = None
        # Re-seed the dedicated human RNG sub-stream for THIS episode. Done while
        # the motion timer is disabled (above) and human_states is empty, so no
        # concurrent reader. This makes the episode's human spawn config a pure
        # function of (run seed, episode_count) — independent of the previous
        # episode's wall-clock-paced motion tick count — and keeps every human
        # draw off the global streams used for start/goal/map/static sampling.
        # (Fixed-eval-suite mode substitutes the suite-local index above.)
        self._seed_human_rngs(_human_seed_base, _human_episode_index)
        self.current_episode_step = 0
        self.contact_collision_latched = False
        # DYN_AVOID: start a fresh per-episode dynamic-avoidance record and clear
        # the published diagnostics so a partial read before the first step of the
        # new episode never returns stale (previous-episode) values. force=True so
        # the empty-episode baseline is published even though the signature reset
        # to the same "no data" key as a prior empty episode.
        self._dyn_diag.reset()
        self._publish_dynamic_diag(force=True)
        # Clear per-episode reward memory so the first step of the new episode
        # does not inherit the last state of the previous episode.
        self._prev_goal_dist   = None
        self._prev_v           = 0.0
        self._prev_w           = 0.0
        self._prev_waypoint_theta = 0.0
        self._freeze_streak    = 0
        self._yield_streak     = 0
        self._prev_risk_dir    = None  # PHASE2: no risk-decrease bonus on episode's first step
        self._reset_continuous_control_reward_state()
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
            # PHASE1B hard-pedestrian-eval (default OFF): temporarily swap in the
            # eval-only human_mode_weights/params for THIS episode's spawn only,
            # then restore whatever was set before -- so it can never leak into
            # training or non-hard-eval episodes (see the declare_parameter
            # comment above for why a save/restore this tightly scoped needs no
            # curriculum-stage bookkeeping).
            _saved_hm = self._apply_hard_pedestrian_eval_override()
            try:
                if self.use_obstacle_pool:
                    self._activate_random_obstacles(start_x, start_y)
                else:
                    self._spawn_random_obstacles(start_x, start_y)
            finally:
                self._restore_hard_pedestrian_eval_override(_saved_hm)
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
        # PHASE1B risk-map-dump (eval-only, default OFF): seed step_debug_state
        # for step 0 too, so an eval script reading it right after reset() (before
        # any step()) never sees stale data from a previous episode.
        if bool(self.get_parameter("risk_map_dump_enabled").value):
            self._publish_step_debug_state()
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
