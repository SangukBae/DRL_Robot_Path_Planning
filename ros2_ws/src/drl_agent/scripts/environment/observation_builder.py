"""Observation / environment-state assembly from LiDAR scan or point cloud.\n\nExtracted from environment.py. Builds environment_state (full 360 collision bins) and obs_state (front 180 policy input) from raw sensor messages, plus human-aware observation masking. Shares node sensor caches via self."""

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


class ObservationMixin:
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

