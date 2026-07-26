"""Human (pedestrian) per-step kinematic motion + Gazebo pose publishing.\n\nExtracted from environment.py. Advances pedestrian state each tick (goal-driven motion + weak social avoidance, robot-non-reactive) and pushes the resulting model poses to Gazebo. Holds the per-step human update logic; spawn/goal sampling lives in human_spawn_sampler."""

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


class HumanMotionMixin:
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

    def _social_avoidance_offset(self, key, x, y, desired_yaw, human_pos, robot_xy=None):
        """Falcon-lite weak human-human avoidance: a small, capped heading nudge.

        Sums a repulsion vector from neighbouring HUMANS within
        human_social_avoid_radius (linear falloff: closer = stronger), blends it
        with the current goal-directed heading using human_social_avoid_strength,
        and returns the resulting heading change CLAMPED to
        human_social_avoid_max_heading_offset.  The robot is never considered by
        default, so humans stay non-reactive to it; the cap + the caller's
        yaw-rate limiter keep turns gentle so the constant-velocity aux labels
        stay valid. Returns 0.0 when disabled or no neighbour is close.

        PHASE1B ROBOT_REACTIVE (eval-only, default OFF): when
        ``human_robot_avoid_enabled`` and ``robot_xy`` is given, ALSO repels
        from the robot's ground-truth pose using its own radius/strength/cap
        (human_robot_avoid_*), independent of the human-human term above -- so
        this can be enabled even with human_social_avoid disabled. Training
        never sets human_robot_avoid_enabled, so Falcon-lite pedestrians stay
        non-reactive to the robot unless an eval preset turns it on.
        """
        human_on = self.human_social_avoid_enabled
        robot_on = robot_xy is not None and bool(
            self.get_parameter("human_robot_avoid_enabled").value)
        if not human_on and not robot_on:
            return 0.0
        rx = ry = 0.0
        if human_on and self.human_social_avoid_radius > 0.0:
            R = self.human_social_avoid_radius
            for k, (ox, oy) in human_pos.items():
                if k == key:
                    continue
                dx, dy = x - ox, y - oy
                d = math.hypot(dx, dy)
                if 1e-3 < d < R:
                    wgt = (1.0 - d / R) * self.human_social_avoid_strength
                    rx += (dx / d) * wgt
                    ry += (dy / d) * wgt
        if robot_on and self.human_robot_avoid_radius > 0.0:
            Rr = self.human_robot_avoid_radius
            dx, dy = x - robot_xy[0], y - robot_xy[1]
            d = math.hypot(dx, dy)
            if 1e-3 < d < Rr:
                wgt = (1.0 - d / Rr) * self.human_robot_avoid_strength
                rx += (dx / d) * wgt
                ry += (dy / d) * wgt
        if rx == 0.0 and ry == 0.0:
            return 0.0
        # Blend the unit goal heading with the (already-weighted) repulsion
        # vector, then cap the angular deviation so avoidance only ever bends
        # the path slightly.
        bx = math.cos(desired_yaw) + rx
        by = math.sin(desired_yaw) + ry
        delta = (math.atan2(by, bx) - desired_yaw + math.pi) % (2.0 * math.pi) - math.pi
        cap = max(
            self.human_social_avoid_max_heading_offset if human_on else 0.0,
            self.human_robot_avoid_max_heading_offset if robot_on else 0.0,
        )
        return float(np.clip(delta, -cap, cap))

    def _resolve_human_wall_collision(self, x, y, nx, ny, r):
        """Stop a kinematic pedestrian from walking THROUGH an internal wall.

        Humans are teleported via set_pose (no physics), so nothing stops them
        entering corridor / intersection / clutter internal walls. After the
        straight-line integration produced the candidate (nx, ny), reject it if
        the body (radius r) would overlap an internal wall, sliding along the wall
        axis-aligned when one component is still free so the human follows the wall
        instead of penetrating it. Returns (rx, ry, blocked) where ``blocked`` is
        True only when fully boxed in (no progress) so the caller can retarget.
        No-op on maps without internal walls (lobby / non-structured)."""
        if not self._point_in_walls(nx, ny, r):
            return nx, ny, False
        # Already inside a wall (shouldn't happen post-spawn) → let it move out.
        if self._point_in_walls(x, y, r):
            return nx, ny, False
        if not self._point_in_walls(nx, y, r):
            return nx, y, False          # slide along x (wall blocks y motion)
        if not self._point_in_walls(x, ny, r):
            return x, ny, False          # slide along y (wall blocks x motion)
        return x, y, True                # fully blocked → hold + retarget

    def _iter_static_obstacle_records_for_humans(self):
        """Yield active static-obstacle footprints for kinematic human avoidance.

        spawned_obstacle_records intentionally contains both static obstacles and
        the current proxy torso for each active pedestrian.  Human-vs-human
        spacing is handled separately by the social-avoidance term, so this helper
        filters out active human records and returns only static objects.
        """
        human_keys = set(getattr(self, "human_states", {}).keys())
        for name, rec in getattr(self, "spawned_obstacle_records", {}).items():
            if name in human_keys:
                continue
            try:
                ox, oy, radius = rec
            except Exception:
                continue
            yield float(ox), float(oy), float(radius)

    def _point_in_static_obstacle_for_human(self, x, y, r):
        margin = float(getattr(self, "human_static_obstacle_clearance", 0.35))
        for ox, oy, rr in self._iter_static_obstacle_records_for_humans():
            if math.hypot(x - ox, y - oy) < (float(r) + rr + margin):
                return True
        return False

    def _segment_hits_static_obstacle_for_human(self, x0, y0, x1, y1, r):
        margin = float(getattr(self, "human_static_obstacle_clearance", 0.35))
        vx, vy = x1 - x0, y1 - y0
        denom = vx * vx + vy * vy
        for ox, oy, rr in self._iter_static_obstacle_records_for_humans():
            if denom <= 1e-12:
                d = math.hypot(x0 - ox, y0 - oy)
            else:
                t = ((ox - x0) * vx + (oy - y0) * vy) / denom
                t = max(0.0, min(1.0, t))
                px, py = x0 + t * vx, y0 + t * vy
                d = math.hypot(px - ox, py - oy)
            if d < (float(r) + rr + margin):
                return True
        return False

    def _resolve_human_static_obstacle_collision(self, x, y, nx, ny, r):
        """Keep kinematic pedestrians from passing through static obstacles.

        Gazebo physics cannot stop these pedestrians because their parts are
        position-published.  We therefore apply the same cheap kinematic guard as
        internal walls: accept the candidate if clear, slide along one axis if
        possible, otherwise hold and let the caller retarget.
        """
        if not self._point_in_static_obstacle_for_human(nx, ny, r):
            return nx, ny, False
        # If a pose already starts overlapped due to a reset/catalog edge case,
        # allow motion that may take it out instead of trapping it forever.
        if self._point_in_static_obstacle_for_human(x, y, r):
            return nx, ny, False
        if (abs(nx - x) > 1e-9
                and not self._point_in_static_obstacle_for_human(nx, y, r)):
            return nx, y, False
        if (abs(ny - y) > 1e-9
                and not self._point_in_static_obstacle_for_human(x, ny, r)):
            return x, ny, False
        return x, y, True

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
            if not stopping and self._human_np_rng.rand() < stop_prob_tick:
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
                # Don't decelerate THROUGH an internal wall (slide / hold).
                new_x, new_y, _ = self._resolve_human_wall_collision(
                    x, y, new_x, new_y, state["radius"])
                # These pedestrians are kinematic pose updates, so Gazebo physics
                # will not stop them at static obstacles. Keep the same footprint
                # clear here as well.
                new_x, new_y, _ = self._resolve_human_static_obstacle_collision(
                    x, y, new_x, new_y, state["radius"])

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
                retarget_cooldown <= 0.0 and self._human_np_rng.rand() < retarget_prob_tick
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
                    state["heading_jitter"] = self._human_np_rng.uniform(
                        -self.human_heading_jitter, self.human_heading_jitter
                    )

            dx_t, dy_t = tx - x, ty - y
            desired_yaw_raw = math.atan2(dy_t, dx_t)
            if self.human_heading_jitter_on_retarget_only:
                desired_yaw_raw += float(state.get("heading_jitter", 0.0))
            else:
                desired_yaw_raw += self._human_np_rng.uniform(
                    -self.human_heading_jitter, self.human_heading_jitter
                )
            # Falcon-lite weak avoidance: nudge the desired heading away from
            # nearby humans by a small capped amount (then the rate limiter below
            # smooths it). Robot is considered ONLY when the eval-only
            # human_robot_avoid_enabled parameter is on (PHASE1B ROBOT_REACTIVE,
            # default OFF -- training never sets it).
            _robot_reactive = bool(self.get_parameter("human_robot_avoid_enabled").value)
            if self.human_social_avoid_enabled or _robot_reactive:
                desired_yaw_raw += self._social_avoidance_offset(
                    _key, x, y, desired_yaw_raw, human_pos,
                    robot_xy=(self.gt_x, self.gt_y) if _robot_reactive else None)
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
            # Keep kinematic humans from walking THROUGH an internal wall: slide
            # along it, or hold + retarget if fully boxed in (so they route around
            # instead of pressing into the wall).
            new_x, new_y, wall_blocked = self._resolve_human_wall_collision(
                x, y, new_x, new_y, state["radius"])
            new_x, new_y, static_blocked = self._resolve_human_static_obstacle_collision(
                x, y, new_x, new_y, state["radius"])
            if wall_blocked or static_blocked:
                # Force next-tick retarget (dist_to_target==0 < 0.5 triggers it),
                # which re-routes via the wall-aware waypoint/goal sampler.
                state["target_x"], state["target_y"] = new_x, new_y
                state["retarget_cooldown"] = 0.0

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
