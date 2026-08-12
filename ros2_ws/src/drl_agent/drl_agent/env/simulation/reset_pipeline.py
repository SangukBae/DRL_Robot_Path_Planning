#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The /reset service pipeline: episode-state reinit, Gazebo world/entity
reset, start/goal sampling hookup, and the initial post-reset observation.
Extracted unchanged from env/simulation/environment.py — see that module's
class docstring and docs/design/environment_design.md for the wider
Environment node this mixes into.
"""

import math
import random
import time

import numpy as np
from squaternion import Quaternion

import drl_agent.common.seed_utils as seed_utils
from drl_agent.env.simulation.gazebo_service_wait import GazeboServiceError
from drl_agent.env.simulation.map_catalog import resolve_active_count


class ResetPipelineMixin:
    """The /reset service callback + its implementation + goal placement.

    Mixed into Environment (env/simulation/environment.py); every method here
    reads/writes Environment instance state via ``self`` exactly as it did
    before extraction.
    """

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
        # Plain time.sleep() poll, NOT rclpy.spin_once(self, ...) -- see the
        # comment on _wait_for_sensor_freshness above for why the latter
        # permanently detaches this node from its MultiThreadedExecutor.
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
            time.sleep(0.05)
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

