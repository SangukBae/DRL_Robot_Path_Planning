"""Gazebo entity I/O (spawn / delete / goal marker) over ros_gz service clients.\n\nExtracted from environment.py. Wraps the bounded-wait spawn/delete calls and the SDF builders. ROS/Gazebo client traffic is concentrated here; behaviour (bounded future wait, success/timeout semantics) is unchanged."""

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


# Pedestrians spawn as 8 separately-named part models that share ONE collision
# footprint, recorded in spawned_obstacle_records under the p_torso name only (and
# kept updated to the live body centre by the human-motion timer). The suffixes
# let _delete_spawned_obstacles map any surviving part back to its body footprint
# so a partial delete failure does not silently drop the human from the
# leftover-avoidance set used by non-pool start/goal sampling.
_HUMAN_PART_SUFFIXES = (
    "_v_torso", "_v_ll", "_v_rl", "_v_la", "_v_ra", "_p_torso", "_p_ll", "_p_rl",
)


def _human_group_prefix(name: str):
    """Return the shared prefix of a human part name (so '<pref>_p_torso' is its
    footprint record key), or None for a non-human (static) entity name."""
    for suf in _HUMAN_PART_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return None


class GazeboEntityMixin:
    def _await_future(self, future, timeout: float = 3.0, op: str = "service"):
        """Poll a service future until done or timed out.

        Safe to call from inside a service callback when the node runs under
        MultiThreadedExecutor: the other executor threads keep processing ROS
        callbacks (including the service response) while this thread sleeps.
        Returns the result, or None on timeout. ``op`` is a short label
        (pause/unpause/reset/set_pose/spawn/delete) so a timeout log says WHICH
        call's response never arrived — distinguishing a wait-stage stall (logged
        by ``_wait_for_srv``) from a response-stage stall (logged here).
        """
        deadline = time.time() + timeout
        while not future.done():
            if time.time() > deadline:
                self.get_logger().warn(
                    f"[gazebo] {op}: response future timed out after {timeout:.1f}s")
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
        """Delete all obstacles from the previous episode and wait for each confirmation.

        Record reconciliation is deferred until after every delete is attempted so
        a human's footprint record (keyed by its p_torso name, but representing the
        WHOLE body) is dropped only when ALL of that human's parts are confirmed
        gone. Popping per-name inline would drop the body footprint the instant the
        p_torso part deleted even if a proxy leg/arm delete failed — leaving a real,
        untracked leftover that non-pool start/goal sampling would not avoid.
        """
        if not self.spawned_obstacle_names:
            return
        remaining_names = []
        deleted_names = []
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
                    deleted_names.append(name)
            except Exception as e:
                self.get_logger().warn(f"DeleteEntity for {name}: {e}")
                remaining_names.append(name)
        self.spawned_obstacle_names = remaining_names
        self._reconcile_obstacle_records(deleted_names, remaining_names)

    def _reconcile_obstacle_records(self, deleted_names, remaining_names):
        """Drop spawned_obstacle_records entries for CONFIRMED-gone entities only.

        Static record key == entity name. A human's footprint key is
        '<pref>_p_torso' but represents the whole body, so it is kept while ANY of
        that human's 8 parts is still in ``remaining_names`` — otherwise a delete
        that removed the torso but left a proxy leg/arm would silently drop a real
        leftover from the non-pool start/goal leftover-avoidance set.
        """
        surviving_human_prefixes = {
            p for p in (_human_group_prefix(n) for n in remaining_names) if p
        }
        for name in deleted_names:
            pref = _human_group_prefix(name)
            if pref is None:
                self.spawned_obstacle_records.pop(name, None)        # static
            elif pref not in surviving_human_prefixes:
                self.spawned_obstacle_records.pop(pref + "_p_torso", None)
            # else: a part of this human is still present -> keep the footprint.

