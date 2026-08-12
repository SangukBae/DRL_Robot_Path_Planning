#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Counterfactual (CF) and spatiotemporal (ST) swept-path risk-target
computation, plus the aux-label wire assembly that carries those targets
(and the AUX_PRED future-risk label) onto /step and /reset responses.
Extracted unchanged from env/simulation/environment.py — see that module's
class docstring and docs/design/aux_prediction.md for the wider Environment
node this mixes into.
"""

import math

import numpy as np

import drl_agent.common.pure_pursuit as pure_pursuit
import drl_agent.env.observation.aux_prediction_labels as aux_labels


class RiskTargetsMixin:
    """Counterfactual/swept-path risk targets + aux-label wire assembly.

    Mixed into Environment (env/simulation/environment.py); every method here
    reads/writes Environment instance state via ``self`` exactly as it did
    before extraction.
    """

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
            target_v, target_steering = RiskTargetsMixin._decode_counterfactual_action(
                self, np.asarray(action, dtype=np.float32))
            rows.append([
                RiskTargetsMixin._swept_path_risk(
                    self, target_v, target_steering, horizon_sec=h,
                    humans=humans)[0]
                for h in self.counterfactual_risk_horizons
            ])
        executed_row = None
        if executed_v is not None:
            executed_row = np.asarray([
                RiskTargetsMixin._swept_path_risk(
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
            return RiskTargetsMixin._swept_path_risk(self, target_v, target_cmd_steering)
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
