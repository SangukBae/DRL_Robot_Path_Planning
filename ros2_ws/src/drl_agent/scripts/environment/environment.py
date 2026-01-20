#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import threading
import random
import time
import numpy as np
from collections import deque
from squaternion import Quaternion

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

from drl_agent_interfaces.srv import Step, Reset, Seed, GetDimensions, SampleActionSpace

import point_cloud2 as pc2
from file_manager import load_yaml
from sensor_msgs.msg import LaserScan

from geometry_msgs.msg import Pose
from ros_gz_interfaces.msg import Entity as GzEntity
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose


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
        
        env_config_file_path = os.path.join(cfg_dir, env_config_file_name)
        start_goal_pairs_file_path = os.path.join(cfg_dir, start_goal_pairs_file)
        self.get_logger().info(f"Using config: {env_config_file_path}")
        # Define the dimensions of the state, action, and maximum action value
        try:
            self.config = load_yaml(env_config_file_path)
        except Exception as e:
            self.get_logger().info(f"Unable to load config file: {e}")
            sys.exit(-1)
        self.environment_config = self.config["environment"]
        self.lower = self.environment_config["lower"]
        self.upper = self.environment_config["upper"]
        self.environment_dim = self.environment_config["environment_state_dim"]
        self.agent_dim = self.environment_config["agent_state_dim"]
        self.agent_name = self.environment_config["agent_name"]
        self.num_of_obstacles = self.environment_config["num_of_obstacles"]

        self.action_dim = self.environment_config["action_dim"]
        self.max_action = self.environment_config["max_action"]
        self.actions_low = self.environment_config["actions_low"]
        self.actions_high = self.environment_config["actions_high"]

        self.threshold_params_config = self.config["threshold_parameters"]
        self.goal_threshold = self.threshold_params_config["goal_threshold"]
        self.collision_threshold = self.threshold_params_config["collision_threshold"]
        self.time_delta = self.threshold_params_config["time_delta"]
        self.inter_entity_distance = self.threshold_params_config[
            "inter_entity_distance"
        ]

        self.lidar_max_range = self.threshold_params_config["lidar_max_range"]

        # Callback groups for handling sensors and services in parallel
        self.odom_callback_group = MutuallyExclusiveCallbackGroup()
        self.velodyne_callback_group = MutuallyExclusiveCallbackGroup()
        self.clients_callback_group = MutuallyExclusiveCallbackGroup()
        self.laser_callback_group = MutuallyExclusiveCallbackGroup()

        # Initialize publishers
        # ★ 토픽 파라미터 (기본값을 Scout Ignition 시스템에 맞춤)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/odometry")
        self.declare_parameter("laser_topic", "/laser_scan")

        cmd_vel_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        odom_topic    = self.get_parameter("odom_topic").get_parameter_value().string_value
        laser_topic   = self.get_parameter("laser_topic").get_parameter_value().string_value
        
        # self.velocity_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.velocity_publisher = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.goal_point_marker_pub = self.create_publisher(
            MarkerArray, "goal_point", 10
        )
        self.linear_vel_marker_pub = self.create_publisher(
            MarkerArray, "linear_velocity", 10
        )
        self.angular_vel_marker_pub = self.create_publisher(
            MarkerArray, "angular_velocity", 10
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
        # ----------------------------------------------------------------------------------------------
        # ====================================Ignition Finish===========================================
        # ----------------------------------------------------------------------------------------------

        # Sensor subscriptions QoS
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        qos_best = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Odometry subscription
        self.odom = self.create_subscription(
            Odometry,
            # "/odom",
            odom_topic,
            self.update_agent_state,
            qos_profile,
            callback_group=self.odom_callback_group,
        )
        self.odom

        # === 관측 소스 선택: LaserScan vs PointCloud2 ===
        self.declare_parameter("obs_source", "scan")          # "scan" 또는 "pointcloud"
        self.declare_parameter("scan_topic", "/laser_scan")   # LaserScan 기본 토픽
        self.declare_parameter("pointcloud_topic", "/points") # PointCloud2 기본 토픽

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

        # Define bins for grouping LaserScan (FULL 360°)
        eps = 0.03
        width = 2*np.pi / self.environment_dim
        start = -np.pi - eps
        self.bins = [[start + i*width, start + (i+1)*width] for i in range(self.environment_dim)]
        self.bins[-1][-1] += eps

        # ----------------------------------------------------------------------------------------------
        # ====================================Ignition Start============================================
        # ----------------------------------------------------------------------------------------------
        # Initialize commands
        self.velocity_command = Twist()
        # ----------------------------------------------------------------------------------------------
        # ====================================Ignition Finish===========================================
        # ----------------------------------------------------------------------------------------------

        # Initialize environment and agent state
        self.environment_state = None
        self.agent_state = None
        # Initialize lock to protect environment_state and agent sate from race condition
        self.environment_state_lock = threading.Lock()
        self.agent_state_lock = threading.Lock()

        # ...locks 생성 이후, config 값들 로드가 끝난 시점에 안전 초기값 세팅
        self.environment_state = np.ones(self.environment_dim, dtype=float) * self.lidar_max_range
        self.agent_state = np.array([np.inf, 0.0, 0.0, 0.0], dtype=float)

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
        self.get_logger().info(f"  num_of_obstacles      : {env.get('num_of_obstacles')}")
        self.get_logger().info(f"  max_action            : {env.get('max_action')}")
        self.get_logger().info(f"  actions_low/high      : {env.get('actions_low')} / {env.get('actions_high')}")

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
        self.get_logger().info(f"  num_of_obstacles      : {self.num_of_obstacles}")
        self.get_logger().info(f"  max_action            : {self.max_action}")
        self.get_logger().info(f"  goal/collision thr    : {self.goal_threshold} / {self.collision_threshold}")
        self.get_logger().info(f"  dt / inter_d / lidar  : {self.time_delta} / {self.inter_entity_distance} / {self.lidar_max_range}")

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
            laser_topic   = self.get_parameter("laser_topic").get_parameter_value().string_value
        except Exception:
            cmd_vel_topic = "/cmd_vel"
            odom_topic    = "/odometry"
            laser_topic   = "/laser_scan"

        self.get_logger().info("[TOPICS]")
        self.get_logger().info(f"  cmd_vel_topic         : {cmd_vel_topic}")
        self.get_logger().info(f"  odom_topic            : {odom_topic}")
        self.get_logger().info(f"  laser_topic           : {laser_topic}")
        self.get_logger().info("-----------------------------------------------------")


        # --- 간단한 일관성/유효성 검사 ---
        issues = self._check_lengths()
        if issues:
            for m in issues:
                self.get_logger().warn(m)
        else:
            self.get_logger().info("Sanity checks: OK")

        self.get_logger().info("=====================================================")

    def _map_action_to_twist(self, action):
        """
        action: shape (2,) in [-1, 1]
        returns: (v [m/s], w [rad/s]) mapped to [actions_low, actions_high]
        """
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        low  = np.asarray(self.actions_low,  dtype=np.float32)
        high = np.asarray(self.actions_high, dtype=np.float32)
        cmd = 0.5 * (a + 1.0) * (high - low) + low  # [-1,1] → [low, high]
        v = float(np.clip(cmd[0], low[0], high[0]))  # m/s
        w = float(np.clip(cmd[1], low[1], high[1]))  # rad/s
        return v, w
    
    def terminate_session(self):
        """Destroy the node and shut down rclpy when done"""
        self.get_logger().info("gym_node shutting down...")
        self.destroy_node()

    def seed_callback(self, request, response):
        """Sets environment seed for reproducibility of the training process."""
        np.random.seed(request.seed)
        response.success = True
        return response

    def sample_action_callback(self, _, response):
        """Samples an action from the action space."""
        action = np.random.uniform(self.actions_low, self.actions_high)
        response.action = np.array(action, dtype=np.float32).tolist()
        return response

    def get_dimensions_callback(self, _, response):
        """Returns the dimensions of the state, action, and maximum action value"""
        response.state_dim = self.environment_dim + self.agent_dim
        response.action_dim = self.action_dim
        response.max_action = self.max_action
        return response

    def update_environment_state_from_cloud(self, cloud_msg):
        """Updates environment state using 360° LiDAR PointCloud2 data.

        Reads 3D point cloud data (e.g., from Ouster), converts it into
        planar distance data, and fills all 360° angular bins (self.bins)
        with the minimum distance per sector.
        """
        with self.environment_state_lock:
            # 초기값: lidar_max_range로 모두 채움
            self.environment_state = (
                np.ones(self.environment_dim, dtype=float) * self.lidar_max_range
            )

            # PointCloud2 → (x, y, z) 리스트
            data = list(
                pc2.read_points(
                    cloud_msg, skip_nans=False, field_names=("x", "y", "z")
                )
            )

            for x, y, z in data:
                # 바닥/노이즈 필터 (기존 조건 유지)
                if z > -0.2:
                    # 각도(beta): 로봇 기준 평면 각도 [-pi, pi]
                    beta = math.atan2(y, x)

                    # 거리: 3D 거리 그대로 사용, lidar_max_range로 클램프
                    dist = math.sqrt(x * x + y * y + z * z)
                    dist = min(dist, self.lidar_max_range)

                    # 공통 bins(360°)에 투영
                    for j in range(len(self.bins)):
                        if self.bins[j][0] <= beta < self.bins[j][1]:
                            if dist < self.environment_state[j]:
                                self.environment_state[j] = dist
                            break

            # 포인트클라우드 기반 env_state를 이용해 존 최소거리 계산
            try:
                self._update_zone_mins_from_env_state()
            except Exception as e:
                self.get_logger().warn(f"zone mins update (cloud) failed: {e}")


    def update_environment_state_from_scan(self, scan):
        """Updates environment state using LaserScan data (from pointcloud_to_laserscan)

        Reads LaserScan (angle_min, angle_increment, ranges) and fills the
        front-arc bins with the minimum planar distance per sector.
        """
        with self.environment_state_lock:
            # 초기값: lidar_max_range로 채움(관측 없으면 최대거리로 유지)
            self.environment_state = np.ones(self.environment_dim) * self.lidar_max_range

            self._angle_min = float(scan.angle_min)
            self._angle_max = float(scan.angle_max)
            self._angle_inc = float(scan.angle_increment)

            # LaserScan 각도/간격
            angle = scan.angle_min
            inc = scan.angle_increment

            # 각 빔(r) 순회
            for r in scan.ranges:
                # 유효한 측정만 사용 (inf/NaN/범위밖 제외)
                if not math.isfinite(r) or r < scan.range_min or r > scan.range_max:
                    angle += inc
                    continue

                beta = angle          # 평면 각도 (rad)
                dist = min(r, self.lidar_max_range)  # 환경 정의 최대거리로 클램프

                # 섹터(bin) 찾기: 현재 bins는 전방 180° 영역만 커버
                for j in range(len(self.bins)):
                    if self.bins[j][0] <= beta < self.bins[j][1]:
                        # 해당 섹터의 최소 거리 갱신
                        if dist < self.environment_state[j]:
                            self.environment_state[j] = dist
                        break

                angle += inc
            
            # (for r in scan.ranges:) 루프 끝난 직후
            try:
                self._update_zone_mins_from_env_state()
            except Exception as e:
                self.get_logger().warn(f"zone mins update (scan) failed: {e}")

    def get_environment_state(self):
        """Returns a copy of the environment state"""
        with self.environment_state_lock:
            if self.environment_state is None:
                return np.ones(self.environment_dim, dtype=float) * self.lidar_max_range
            return self.environment_state.copy()

    def update_agent_state(self, odom):
        """Update agent state using data from odometry (robust atan2-based version)"""
        with self.agent_state_lock:
            # Robot pose
            odom_x = odom.pose.pose.position.x
            odom_y = odom.pose.pose.position.y

            # Heading (yaw) from quaternion
            q = Quaternion(
                odom.pose.pose.orientation.w,
                odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
            )
            yaw = q.to_euler(degrees=False)[2]  # [-pi, pi]

            # Vector to goal
            dx = self.goal_x - odom_x
            dy = self.goal_y - odom_y
            dist = math.hypot(dx, dy)

            # Heading error: goal bearing - current yaw, wrapped to [-pi, pi]
            if dist < 1e-9:
                theta = 0.0
            else:
                goal_bearing = math.atan2(dy, dx)             # [-pi, pi]
                theta = goal_bearing - yaw
                theta = (theta + math.pi) % (2 * math.pi) - math.pi

            # Store [goal_distance, heading_error, prev_action_0, prev_action_1]
            self.agent_state = np.array([dist, theta, 0.0, 0.0], dtype=float)

    def get_agent_state(self):
        """Return a copy of the agent state"""
        with self.agent_state_lock:
            if self.agent_state is None:
                return np.array([np.inf, 0.0, 0.0, 0.0], dtype=float)
            return self.agent_state.copy()
        
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
            self.world_control.call_async(req)
        except Exception as e:
            self.get_logger().error(
                f"{srv_name} service call failed: {e}"
            )
            sys.exit(-1)

    def reset_world(self):
        """Ignition 월드 리셋 (시간 + 모델)"""
        srv_name = f"/world/{self.world_name}/control"
        self._wait_for_srv(self.world_control, srv_name)

        req = ControlWorld.Request()
        # 모든 것 리셋
        req.world_control.reset.all = True
        # 권장: 리셋하면서 바로 pause 상태로
        req.world_control.pause = True
        try:
            self.world_control.call_async(req)
        except Exception as e:
            self.get_logger().error(
                f"{srv_name} (reset) service call failed: {e}"
            )
            sys.exit(-1)

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
            self.set_entity_pose.call_async(req)
        except Exception as e:
            self.get_logger().error(
                f"{srv_name} service call failed: {e}"
            )
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

    def step_callback(self, request, response):
        target = False
        action = request.action  # 정규화 [-1,1]
    
        # 1) 액션 → 실제 속도
        v, w = self._map_action_to_twist(action)
    
        # 2) Twist 퍼블리시
        self.velocity_command.linear.x  = v
        self.velocity_command.angular.z = w
        self.velocity_publisher.publish(self.velocity_command)
    
        # (선택) 마커는 정규화 액션 기준 유지
        self.publish_markers(action)
    
        # 3) 시뮬레이션 진행
        self.propagate_state(self.time_delta)
    
        # 4) 상태 구성
        environment_state = self.get_environment_state()
        agent_state = self.get_agent_state()
        # agent_state[0]: goal_dist, agent_state[1]: theta_err (가정)
        # agent_state[2:4]: 최근 액션(정규화) 저장
        agent_state[2], agent_state[3] = float(action[0]), float(action[1])
        state = np.append(environment_state, agent_state)
    
        # 5) 충돌/완료 판단
        done, collision, min_used = self.check_collision(environment_state)
    
        curr_goal_dist = float(agent_state[0])
        prev_goal_dist = float(getattr(self, "_prev_goal_dist", curr_goal_dist))
        theta_err = None
        try:
            theta_err = float(agent_state[1])
        except Exception:
            theta_err = None
    
        if curr_goal_dist < self.goal_threshold:
            self.get_logger().info(f"{'GOAL REACHED':-^50}")
            target = True
            done = True
    
        # 6) 존 정보(있으면 사용)
        zmins = getattr(self, "_zone_mins", None) if getattr(self, "use_zone_collision", False) else None
        zthrs = self.zone_thresholds if (zmins is not None) else None
    
        # 7) 보상 계산 (v,w 및 zone/폴백 모두 대응)
        v_max = max(abs(self.actions_low[0]),  abs(self.actions_high[0]))  # 예: 1.5
        w_max = max(abs(self.actions_low[1]),  abs(self.actions_high[1]))  # 예: 6.0
    
        reward = self.get_reward(
            target, collision,
            v, w,
            prev_goal_dist, curr_goal_dist,
            theta_err=theta_err,
            zmins=zmins, zthrs=zthrs,      # 존 기반 근접 페널티
            min_laser=min_used,            # 폴백(min)
            v_max=v_max, w_max=w_max
            # prev_v/getattr(self,'_prev_v',None) 등 스무딩을 쓰려면 여기 인자도 추가
        )
    
        # 8) 다음 스텝 대비 기록
        self._prev_goal_dist = curr_goal_dist
        self._prev_v, self._prev_w = v, w
    
        # 9) 응답
        response.state  = state.tolist()
        response.reward = float(reward)
        response.done   = bool(done)
        response.target = bool(target)
        return response

    def reset_callback(self, _, response):
        """Resets the state of the environment and returns an initial observation, state"""

        """*****************************************************
        ** Start by resetting Ignition world
        *****************************************************"""
        self.reset_world()
        time.sleep(self.time_delta)

        """*****************************************************
		** Determine start positions for the agent
		*****************************************************"""
        if self.train_mode:
            position_ok = False
            angle = np.random.uniform(-np.pi, np.pi)
            while not position_ok:
                start_x = np.random.uniform(self.lower, self.upper)
                start_y = np.random.uniform(self.lower, self.upper)
                position_ok = not self.check_dead_zone(start_x, start_y, use_cross_mask=False)
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
            self.agent_name,          # 예: "scout_v2"
            start_x,
            start_y,
            0.4,                      # z 높이 (현재 world에서 잘 쓰이던 값 유지)
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w,
        )

        """*****************************************************
		** Change goal and randomize obstacles
		*****************************************************"""
        self.change_goal()
        if self.train_mode:
            self.shuffle_obstacles(start_x, start_y)
        # Publish markers for rviz
        self.publish_markers([0.0, 0.0])
        # Propagate state for 2*time_delta seconds
        self.propagate_state(2 * self.time_delta)

        # 첫 관측이 들어올 때까지 짧게 대기 (최대 1.0초)
        t0 = time.time()
        while (self.environment_state is None or self.agent_state is None) and (time.time() - t0 < 1.0):
            rclpy.spin_once(self, timeout_sec=0.05)

        """*****************************************************
		** Compute state after reset
		*****************************************************"""
        environment_state = self.get_environment_state()
        agent_state = self.get_agent_state()
        response.state = np.append(environment_state, agent_state).tolist()
        return response

    def change_goal(self):
        """Places a new goal and ensures its location is not on one of the obstacles"""
        if self.train_mode:
            # 로봇 시작점과의 최소 거리: 장애물 간격과 동일 값 사용
            # min_start_goal_dist = float(getattr(self, "inter_entity_distance", 1.0))
            min_start_goal_dist = 3.0
            sx = float(self.set_agent_state.pose.position.x)
            sy = float(self.set_agent_state.pose.position.y)

            while True:
                self.goal_x = random.uniform(self.upper, self.lower)
                self.goal_y = random.uniform(self.upper, self.lower)

                # 금지영역(필요 시 마스크 해제) + 로봇 시작점과의 최소거리 조건
                if self.check_dead_zone(self.goal_x, self.goal_y, use_cross_mask=False):
                    continue
                if math.hypot(self.goal_x - sx, self.goal_y - sy) < min_start_goal_dist:
                    continue
                break
        else:
            self.goal_x = self.current_pairs["goal"]["x"]
            self.goal_y = self.current_pairs["goal"]["y"]

    def check_collision(self, laser_data):
        """
        Detect a collision.
        - If use_zone_collision: per-zone min vs per-zone threshold (Z5..Z1).
        - Else: fallback to legacy global-min rule.
        Returns: (done, collision, min_laser_used)
        """
        # --- 존 기반(가변 개수) ---
        if getattr(self, "use_zone_collision", False) and getattr(self, "_zone_mins", None) is not None:
            zmins = self._zone_mins
            thrs  = self.zone_thresholds
            n = min(len(zmins), len(thrs))
            flags = [(zmins[i] < thrs[i]) for i in range(n)]
            anycol = any(flags)
            min_used = min(zmins) if zmins else float('inf')
            return anycol, anycol, min_used

        # --- 폴백: 글로벌 min ---
        done = collision = False
        min_laser = float(np.min(laser_data)) if len(laser_data) else float('inf')
        if min_laser < self.collision_threshold:
            done = collision = True
        return done, collision, min_laser

    def shuffle_obstacles(self, start_x, start_y):
        """Randomly changes the location of the obstacles upon reset"""
        prev_obstacle_positions = []
        for i in range(1, self.num_of_obstacles + 1):
            position_ok = False
            obstacle_name = f"obstacle_{i}"
            while not position_ok:
                x = np.random.uniform(self.lower, self.upper)
                y = np.random.uniform(self.lower, self.upper)

                position_ok = not self.check_dead_zone(x, y, use_cross_mask=False)
                distance_to_robot = np.linalg.norm([x - start_x, y - start_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if (
                    distance_to_robot < self.inter_entity_distance
                    or distance_to_goal < self.inter_entity_distance
                ):
                    position_ok = False
                    continue

                for prev_x, prev_y in prev_obstacle_positions:
                    distance_to_other_obstacles = np.linalg.norm(
                        [x - prev_x, y - prev_y]
                    )
                    if distance_to_other_obstacles < self.inter_entity_distance:
                        position_ok = False

            # Ignition 월드에서 obstacle_i 텔레포트
            self.set_entity_pose_ignition(
                obstacle_name,
                x,
                y,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            )
            prev_obstacle_positions.append((x, y))

    def check_dead_zone(self, x, y, use_cross_mask: bool = False):
        """True면 금지영역, False면 허용.
           use_cross_mask=False이면 십자 띠 제한을 해제한다."""
        # 맵 바깥은 항상 금지
        if abs(x) > self.upper or abs(y) > self.upper:
            return True

        # 십자 띠 제한을 쓰지 않으면 바로 허용
        if not use_cross_mask:
            return False

        # 십자형 내부 띠 금지(기존 로직)
        if 2.0 < abs(x) < self.upper and abs(y) < 1.0:
            return True
        if abs(x) < 1.0 and 2.0 < abs(y) < self.upper:
            return True

        return False

    def publish_markers(self, action):
        """Publishes visual data for Rviz to visualize the goal and the robot's actions"""
        marker_specs = [
            {
                "frame_id": "odom",
                "marker_type": Marker.CYLINDER,
                "scale": (0.1, 0.1, 0.01),
                "color": (1.0, 0.0, 1.0, 0.0),
                "position": (self.goal_x, self.goal_y, 0.0),
                "orientation": (0.0, 0.0, 0.0, 1.0),
                "action": Marker.ADD,
                "ns": "",
                "marker_id": 0,
                "publisher": self.goal_point_marker_pub,
            },
            {
                "frame_id": "odom",
                "marker_type": Marker.CUBE,
                "scale": (abs(action[0]), 0.1, 0.01),
                "color": (1.0, 1.0, 0.0, 0.0),
                "position": (5.0, 0.0, 0.0),
                "orientation": (0.0, 0.0, 0.0, 1.0),
                "action": Marker.ADD,
                "ns": "",
                "marker_id": 1,
                "publisher": self.linear_vel_marker_pub,
            },
            {
                "frame_id": "odom",
                "marker_type": Marker.CUBE,
                "scale": (abs(action[1]), 0.1, 0.01),
                "color": (1.0, 1.0, 0.0, 0.0),
                "position": (5.0, 0.2, 0.0),
                "orientation": (0.0, 0.0, 0.0, 1.0),
                "action": Marker.ADD,
                "ns": "",
                "marker_id": 2,
                "publisher": self.angular_vel_marker_pub,
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
        v, w,                                  # m/s, rad/s (Twist로 보낸 값)
        prev_goal_dist, curr_goal_dist,         # 목표까지 직선거리
        theta_err=None,                         # 로봇 헤딩 vs 목표방향 (rad), 없으면 None
        zmins=None, zthrs=None,                 # zone 기반 충돌 사용 시 길이 5 리스트( [Z5,Z4,Z3,Z2,Z1] )
        min_laser=None,                         # 폴백: check_collision()이 준 min_used
        v_max=1.5, w_max=6.0,

        # ---- 튜닝 파라미터 ----
        # 진행/경로 품질
        k_p=2.0,                 # 진행 보상 게인 (거리 0.25m 줄면 +0.5)
        progress_clip=0.25,      # 스텝당 인정 거리 감소 상한

        # 곡률(원운동 억제)
        lambda_k=0.35,           # κ 페널티 계수

        # 장애물 근접 (존 기반)
        z_weights=(0.6, 0.85, 1.0, 0.85, 0.6),  # 중앙(Z1) 가중치를 가장 크게
        safety_margin=1.5,       # 임계치 바깥쪽으로 여유 (1.5배까지 선제 페널티 시작)
        w_obs=0.8,               # 근접 페널티 스케일(최대치)

        # 장애물 근접 (폴백: 글로벌 min)
        d_safe_base=0.55,        # Bunker 전장(~1.03m) 고려, 전방 기본 안전거리
        d_safe_speed=0.30,       # 속도 비례 안전거리: base + d_safe_speed*|v|

        # 헤딩/시간/스무딩
        k_h=0.3,                 # 헤딩 보너스 (cos theta_err)
        step_pen=0.01,           # 시간 페널티
        k_smooth=0.0,            # 스무딩(이전 속도 필요) 기본 사용 안 함
        prev_v=None, prev_w=None
    ):
        # 터미널
        if target:    return 10.0
        if collision: return -10.0

        # 정규화
        v_n = v / max(v_max, 1e-6)
        w_n = w / max(w_max, 1e-6)

        # 1) 진행 보상
        delta_d  = np.clip(prev_goal_dist - curr_goal_dist, -progress_clip, progress_clip)
        progress = k_p * delta_d   # 대략 [-0.5, +0.5]

        # 2) 곡률 페널티 (원운동 억제)
        kappa    = abs(w_n) / (abs(v_n) + 1e-3)
        curv_pen = lambda_k * kappa

        # 3) 장애물 근접 페널티 (존 우선, 폴백 글로벌 min)
        obstacle = 0.0
        if zmins is not None and zthrs is not None and len(zmins) == 5 and len(zthrs) == 5:
            # 각 존별 "여유 부족" 비율: 1 - (zmin / (safety_margin * zthr)), 음수면 0
            deficits = []
            for i in range(5):
                thr_expanded = max(1e-6, safety_margin * float(zthrs[i]))
                zmin = float(zmins[i])
                d = max(0.0, 1.0 - (zmin / thr_expanded))  # 0~1
                deficits.append(d)

            # 중앙(Z1)이 가장 크도록 가중 평균
            wsum = sum(z_weights)
            weighted = sum(wi * di for wi, di in zip(z_weights, deficits)) / max(wsum, 1e-6)
            obstacle = w_obs * weighted   # 0 ~ w_obs
        else:
            # 폴백: 글로벌 min_laser 기반 (속도 의존 안전거리)
            if min_laser is not None and np.isfinite(min_laser):
                d_safe = d_safe_base + d_safe_speed * abs(v)
                if min_laser < d_safe:
                    obstacle = w_obs * (1.0 - min_laser / max(d_safe, 1e-6))  # 0~w_obs

        # 4) 헤딩 보너스(선택)
        heading = k_h * math.cos(theta_err) if theta_err is not None else 0.0

        # 5) 스무딩(선택)
        smooth = 0.0
        if k_smooth > 0.0 and prev_v is not None and prev_w is not None:
            dv = abs(v - prev_v) / max(v_max, 1e-6)
            dw = abs(w - prev_w) / max(w_max, 1e-6)
            smooth = k_smooth * 0.5 * (dv + dw)

        # 6) 시간 페널티 및 합산
        reward = progress + heading - curv_pen - obstacle - step_pen - smooth

        # 스케일 안정화
        return float(np.clip(reward, -1.0, 1.0))

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