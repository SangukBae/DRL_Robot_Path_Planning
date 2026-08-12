#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gazebo Ignition world control: pause/unpause/reset, entity teleport, and
physics-step advancement (legacy unpause/sleep/pause and the deterministic
multi_step path), plus the post-multi_step sensor-freshness wait these steps
rely on. Extracted unchanged from env/simulation/environment.py — see that
module's class docstring and docs/design/environment_design.md for the wider
Environment node this mixes into.
"""

import time

from ros_gz_interfaces.msg import Entity as GzEntity
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose

from drl_agent.env.humans.human_motion_manager import compute_human_tick_plan
from drl_agent.env.simulation.gazebo_service_wait import (
    GazeboServiceError, bounded_wait_for_service, compute_physics_step_count)


class GazeboRuntimeMixin:
    """Gazebo world control + physics-step advancement + sensor-freshness wait.

    Mixed into Environment (env/simulation/environment.py); every method here
    reads/writes Environment instance state via ``self`` exactly as it did
    before extraction (locks, callback groups, and ROS clients are all owned
    and initialised by Environment.__init__).
    """

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
        unlike the /clock target above.

        Plain time.sleep() poll, NOT rclpy.spin_once(self, ...): this node is
        already spinning on its own MultiThreadedExecutor (see module main()),
        so the scan/odom subscription callbacks that bump these counters keep
        firing on other executor worker threads while this thread sleeps --
        no manual pump needed. A bare rclpy.spin_once(self, ...) call with no
        executor= argument spins the process-global executor instead, and
        rclpy's Node.executor setter (node.py) detaches the node from
        whatever executor it was previously registered with as a side effect
        of that -- permanently removing it from THIS node's MultiThreadedExecutor
        (only removed, never re-added), so every subsequent service call
        (reset/step, even unrelated ones like get_parameters) times out
        forever. This previously looked like a GIL/executor-starvation issue
        because the bug only fires on the rare tick where the wait condition
        above is actually true (most ticks the background subscription
        callbacks already caught up) -- see memory cf_st_step_executor_hang.md."""
        t0 = time.time()
        while (
            self.scan_update_count <= prev_scan_updates
            or any(self._odom_role_count[r] <= prev_role_updates[r]
                   for r in ("gt", "loc", "proprio"))
        ) and (time.time() - t0 < self.gazebo_sensor_wait_timeout_sec):
            time.sleep(0.01)
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
