#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The /step service pipeline: action decode -> command publish -> Gazebo
physics advance -> observation/reward assembly -> CSV telemetry -> response.
Extracted unchanged from env/simulation/environment.py — see that module's
class docstring and docs/design/environment_design.md for the wider
Environment node this mixes into.
"""

import csv
import math

import numpy as np

import drl_agent.common.pure_pursuit as pure_pursuit
import drl_agent.env.rewards.reward_calculator as reward_calculator
from drl_agent.env.simulation.gazebo_service_wait import GazeboServiceError


class StepPipelineMixin:
    """The /step service callback + its implementation.

    Mixed into Environment (env/simulation/environment.py); every method here
    reads/writes Environment instance state via ``self`` exactly as it did
    before extraction.
    """

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

