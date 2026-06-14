"""Legacy zone min/index tracking (sector-based proximity).\n\nExtracted from environment.py. Kept for the legacy zone collision/reward path (use_zone_collision, default off). Operates on shared node state via self."""

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


class ZoneMixin:
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

