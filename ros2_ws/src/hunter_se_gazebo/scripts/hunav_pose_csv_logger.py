#!/usr/bin/env python3

import csv
import math
import os
import statistics
from typing import Dict, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import TransformStamped
from hunav_msgs.msg import Agents
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener


Point3 = Tuple[float, float, float]


def rotate_point(point: Point3, qx: float, qy: float, qz: float, qw: float) -> Point3:
    x, y, z = point

    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)

    rx = x + qw * tx + (qy * tz - qz * ty)
    ry = y + qw * ty + (qz * tx - qx * tz)
    rz = z + qw * tz + (qx * ty - qy * tx)
    return rx, ry, rz


def apply_transform(point: Point3, transform: TransformStamped) -> Point3:
    q = transform.transform.rotation
    tx = transform.transform.translation.x
    ty = transform.transform.translation.y
    tz = transform.transform.translation.z
    rx, ry, rz = rotate_point(point, q.x, q.y, q.z, q.w)
    return rx + tx, ry + ty, rz + tz


def median_point(points: List[Point3]) -> Point3:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return statistics.median(xs), statistics.median(ys), statistics.median(zs)


def percentile_value(values: List[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile_value() requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    q = max(0.0, min(1.0, q))
    idx = q * (len(ordered) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


class HunavPoseCsvLogger(Node):
    def __init__(self) -> None:
        super().__init__("hunav_pose_csv_logger")

        self.declare_parameter("csv_path", "/tmp/hunter_se_hunav_pose_log.csv")
        self.declare_parameter("human_states_topic", "/human_states")
        self.declare_parameter("pointcloud_topic", "/ouster/points")
        self.declare_parameter("target_frame", "map")
        self.declare_parameter("agent_names", ["agent1", "agent2"])
        self.declare_parameter("cluster_xy_radius", 0.55)
        self.declare_parameter("min_height_above_agent_base", 0.05)
        self.declare_parameter("max_height_above_agent_base", 1.60)
        self.declare_parameter("torso_min_height_above_agent_base", 0.35)
        self.declare_parameter("max_points_per_agent", 192)
        self.declare_parameter("min_cluster_points", 5)
        self.declare_parameter("xy_center_correction_radius", 0.30)
        self.csv_path = self.get_parameter("csv_path").get_parameter_value().string_value
        self.human_states_topic = self.get_parameter("human_states_topic").get_parameter_value().string_value
        self.pointcloud_topic = self.get_parameter("pointcloud_topic").get_parameter_value().string_value
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value
        self.agent_names = list(self.get_parameter("agent_names").get_parameter_value().string_array_value)
        self.cluster_xy_radius = float(self.get_parameter("cluster_xy_radius").value)
        self.min_height_above_agent_base = float(
            self.get_parameter("min_height_above_agent_base").value
        )
        self.max_height_above_agent_base = float(
            self.get_parameter("max_height_above_agent_base").value
        )
        self.torso_min_height_above_agent_base = float(
            self.get_parameter("torso_min_height_above_agent_base").value
        )
        self.max_points_per_agent = int(self.get_parameter("max_points_per_agent").value)
        self.min_cluster_points = int(self.get_parameter("min_cluster_points").value)
        self.xy_center_correction_radius = float(
            self.get_parameter("xy_center_correction_radius").value
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.latest_agents: Dict[str, Tuple[Point3, str]] = {}
        self.last_frame: Optional[str] = None
        self.last_tf_warn_ns = 0

        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "stamp_sec",
            "agent_name",
            "frame",
            "gazebo_x",
            "gazebo_y",
            "gazebo_z",
            "pred_cloud_x",
            "pred_cloud_y",
            "pred_cloud_z",
            "matched_points",
            "pointcloud_surface_x",
            "pointcloud_surface_y",
            "pointcloud_x",
            "pointcloud_y",
            "pointcloud_z",
            "pointcloud_torso_z",
            "error_xy",
            "error_z",
            "error_3d",
            "xy_points_used",
        ])
        self.csv_file.flush()

        self.create_subscription(Agents, self.human_states_topic, self.human_states_cb, 10)
        self.create_subscription(PointCloud2, self.pointcloud_topic, self.pointcloud_cb, 10)

        self.get_logger().info(f"Logging HuNav pose alignment CSV to {self.csv_path}")

    def destroy_node(self) -> bool:
        try:
            self.csv_file.close()
        except Exception:
            pass
        return super().destroy_node()

    def human_states_cb(self, msg: Agents) -> None:
        self.last_frame = msg.header.frame_id or self.target_frame
        current: Dict[str, Tuple[Point3, str]] = {}
        for agent in msg.agents:
            if agent.name in self.agent_names:
                pos = agent.position.position
                current[agent.name] = ((pos.x, pos.y, pos.z), self.last_frame)
        if current:
            self.latest_agents = current

    def pointcloud_cb(self, msg: PointCloud2) -> None:
        if not self.latest_agents:
            return

        frame = self.target_frame or self.last_frame
        if not frame:
            return

        try:
            # Use the latest available TF instead of the point cloud timestamp.
            # The ros_gz/RGL pipeline can stamp points ahead of TF publication,
            # and this logger only needs approximate alignment for comparison.
            latest_tf_time = Time()
            world_to_cloud = self.tf_buffer.lookup_transform(
                msg.header.frame_id,
                frame,
                latest_tf_time,
                timeout=Duration(seconds=0.02),
            )
            cloud_to_world = self.tf_buffer.lookup_transform(
                frame,
                msg.header.frame_id,
                latest_tf_time,
                timeout=Duration(seconds=0.02),
            )
        except TransformException as exc:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self.last_tf_warn_ns > 2_000_000_000:
                self.get_logger().warn(
                    f"Skipping CSV sample: transform {frame} <-> {msg.header.frame_id} unavailable: {exc}"
                )
                self.last_tf_warn_ns = now_ns
            return

        predicted_cloud_positions: Dict[str, Point3] = {}
        buckets: Dict[str, List[Tuple[float, Point3]]] = {
            name: [] for name in self.agent_names
        }
        sensor_world = (
            cloud_to_world.transform.translation.x,
            cloud_to_world.transform.translation.y,
            cloud_to_world.transform.translation.z,
        )

        for name in self.agent_names:
            if name not in self.latest_agents:
                continue
            gazebo_pos, _ = self.latest_agents[name]
            predicted_cloud_positions[name] = apply_transform(gazebo_pos, world_to_cloud)

        if not predicted_cloud_positions:
            return

        xy_radius_sq = self.cluster_xy_radius * self.cluster_xy_radius

        for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            px = float(point[0])
            py = float(point[1])
            pz = float(point[2])
            for name, predicted in predicted_cloud_positions.items():
                dx = px - predicted[0]
                dy = py - predicted[1]
                dist_sq = dx * dx + dy * dy
                if dist_sq > xy_radius_sq:
                    continue
                rel_z = pz - predicted[2]
                if rel_z < self.min_height_above_agent_base:
                    continue
                if rel_z > self.max_height_above_agent_base:
                    continue
                buckets[name].append((dist_sq, (px, py, pz)))

        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        for name in self.agent_names:
            if name not in self.latest_agents or name not in predicted_cloud_positions:
                continue

            gazebo_pos, agent_frame = self.latest_agents[name]
            predicted_cloud = predicted_cloud_positions[name]
            cluster_with_dist = buckets[name]
            cluster_with_dist.sort(key=lambda item: item[0])
            cluster = [
                point for _, point in cluster_with_dist[: self.max_points_per_agent]
            ]
            torso_cluster = [
                point
                for point in cluster
                if point[2] - predicted_cloud[2] >= self.torso_min_height_above_agent_base
            ]

            pc_world: Tuple[float, float, float]
            pc_surface_world: Tuple[float, float]
            pc_torso_z: float
            error_xy: float
            error_z: float
            error_3d: float
            if len(torso_cluster) >= self.min_cluster_points:
                xy_cluster = torso_cluster
            else:
                xy_cluster = cluster

            if len(cluster) >= self.min_cluster_points and len(xy_cluster) >= self.min_cluster_points:
                xy_world_points = [apply_transform(point, cloud_to_world) for point in xy_cluster]
                full_world_points = [apply_transform(point, cloud_to_world) for point in cluster]
                surface_x = statistics.median(p[0] for p in xy_world_points)
                surface_y = statistics.median(p[1] for p in xy_world_points)
                pc_surface_world = (surface_x, surface_y)
                view_dx = surface_x - sensor_world[0]
                view_dy = surface_y - sensor_world[1]
                view_norm = math.hypot(view_dx, view_dy)
                if view_norm > 1e-6:
                    corrected_x = surface_x + self.xy_center_correction_radius * (view_dx / view_norm)
                    corrected_y = surface_y + self.xy_center_correction_radius * (view_dy / view_norm)
                else:
                    corrected_x = surface_x
                    corrected_y = surface_y
                pc_world = (
                    corrected_x,
                    corrected_y,
                    percentile_value([p[2] for p in full_world_points], 0.10),
                )
                pc_torso_z = statistics.median(p[2] for p in xy_world_points)
                dx = pc_world[0] - gazebo_pos[0]
                dy = pc_world[1] - gazebo_pos[1]
                dz = pc_world[2] - gazebo_pos[2]
                error_xy = math.hypot(dx, dy)
                error_z = dz
                error_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
            else:
                xy_cluster = []
                pc_surface_world = (math.nan, math.nan)
                pc_world = (math.nan, math.nan, math.nan)
                pc_torso_z = math.nan
                error_xy = math.nan
                error_z = math.nan
                error_3d = math.nan

            self.csv_writer.writerow([
                f"{stamp_sec:.6f}",
                name,
                agent_frame,
                f"{gazebo_pos[0]:.6f}",
                f"{gazebo_pos[1]:.6f}",
                f"{gazebo_pos[2]:.6f}",
                f"{predicted_cloud[0]:.6f}",
                f"{predicted_cloud[1]:.6f}",
                f"{predicted_cloud[2]:.6f}",
                len(cluster),
                f"{pc_surface_world[0]:.6f}" if math.isfinite(pc_surface_world[0]) else "",
                f"{pc_surface_world[1]:.6f}" if math.isfinite(pc_surface_world[1]) else "",
                f"{pc_world[0]:.6f}" if math.isfinite(pc_world[0]) else "",
                f"{pc_world[1]:.6f}" if math.isfinite(pc_world[1]) else "",
                f"{pc_world[2]:.6f}" if math.isfinite(pc_world[2]) else "",
                f"{pc_torso_z:.6f}" if math.isfinite(pc_torso_z) else "",
                f"{error_xy:.6f}" if math.isfinite(error_xy) else "",
                f"{error_z:.6f}" if math.isfinite(error_z) else "",
                f"{error_3d:.6f}" if math.isfinite(error_3d) else "",
                len(xy_cluster),
            ])

        self.csv_file.flush()


def main() -> None:
    rclpy.init()
    node = HunavPoseCsvLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
