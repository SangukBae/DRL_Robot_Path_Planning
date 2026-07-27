#!/usr/bin/env python3
"""Privileged per-episode dynamic-obstacle (pedestrian) avoidance telemetry.

Pure Python (``math``/stdlib only), ROS/torch-free and therefore unit-testable
in isolation. ``environment.py`` feeds it the *privileged* robot + pedestrian
ground truth once per ``/step``; it accumulates the human-interaction / clutter /
yield / near-event diagnostics that the curriculum trainer writes into the new
``dynamic_avoidance_metrics_<run_tag>.csv`` diagnostic log.

Why this lives in the ENV (not the trainer)
-------------------------------------------
The metrics below need information the agent never sees at run time:
  * exact robot<->pedestrian distances / closing speeds (TTC),
  * which pedestrian is nearest and its behaviour ``mode``,
  * whether a collision was against a human vs. a static obstacle,
  * whether the actual (per-stage-gated) YIELD command coincided with real risk.
The agent's aux label is a *prediction proxy* and is only present when
``aux_prediction.enabled`` on the env side; this accumulator instead uses the
simulator's privileged ``human_states``, so the diagnostics are available even
for an aux-OFF baseline as long as pedestrians are active. Nothing here is used
at deployment — it exists purely to DIAGNOSE dynamic-obstacle avoidance in sim.

NaN convention
--------------
A metric that is *undefined for this episode* (e.g. no pedestrian was ever
present, so there is no "min human distance") is reported as ``float('nan')`` for
numeric fields and ``""`` for string fields. Counts are always concrete ints.
Ratios over "human-observed steps" are NaN when that denominator is 0.
"""

import math


def _vel_xy(v, yaw):
    """World-frame (vx, vy) from signed speed + heading."""
    return v * math.cos(yaw), v * math.sin(yaw)


class DynamicAvoidanceEpisodeDiag:
    """Accumulate one episode's privileged dynamic-avoidance diagnostics.

    Call :meth:`reset` at episode start, :meth:`update` once per env step with
    the privileged robot/pedestrian ground truth, and :meth:`as_dict` at episode
    end to obtain the flat diagnostic record.
    """

    UNAVAILABLE = float("nan")

    def __init__(self, *, near_human_dist_m: float = 1.0,
                 interaction_radius_m: float = 2.0,
                 collision_attrib_radius_m: float = 0.7,
                 ttc_collision_radius_m: float = 0.5,
                 ttc_cap_s: float = 99.0,
                 static_clutter_lidar_m: float = 0.6):
        # A pedestrian within `near` counts as a "risk"/near step (also the
        # denominator threshold for the personal-space time ratio).
        self.near = float(near_human_dist_m)
        # A pedestrian within `interact` marks the episode as having a genuine
        # human interaction (separates human-driven from pure-clutter episodes).
        self.interact = float(interaction_radius_m)
        # At the collision step, a human centred within `coll_r` => the collision
        # is attributed to a human, else to a static obstacle.
        self.coll_r = float(collision_attrib_radius_m)
        self.ttc_r = float(ttc_collision_radius_m)
        self.ttc_cap = float(ttc_cap_s)
        # Min-LiDAR below this with NO human nearby => static clutter pressure.
        self.clutter_lidar = float(static_clutter_lidar_m)
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self):
        self.total_steps = 0
        self.observed_steps = 0          # steps with >= 1 pedestrian present
        self.min_dist = float("inf")     # closest robot<->human distance [m]
        self.min_ttc = float("inf")      # min time-to-contact (closing only) [s]
        self.below_near_steps = 0        # steps with a human within `near` (=risk)
        self.interaction = False
        self.static_clutter_steps = 0
        self.modes_present = set()
        self.nearest_mode = ""           # mode of the closest-ever pedestrian
        self.collision_type = "none"     # none | static | human
        self._collision_locked = False
        # Yield accounting (uses the env's ACTUAL, per-stage-gated yield decision).
        self.yield_available = 0         # yield axis active this stage (0/1)
        self.yield_steps = 0             # steps the robot was yielding
        self.yield_onsets = 0            # non-yield -> yield transitions (triggers)
        self.yield_in_risk_steps = 0     # yielding AND a human within `near`
        self.yield_no_risk_steps = 0     # yielding AND no human within `near`
        self._prev_yielding = False

    def update(self, robot_pose, robot_v, humans, min_lidar_m, collision,
               *, yielding=False, yield_available=False):
        """Fold one env step.

        Parameters
        ----------
        robot_pose : (rx, ry, ryaw)   ground-truth robot pose [m, m, rad]
        robot_v    : float            ground-truth signed speed [m/s]
        humans     : iterable of dicts with x, y and (v, yaw) or (vx, vy); an
                     optional ``mode`` string. Empty/None => no pedestrians.
        min_lidar_m: float            min 360deg collision-LiDAR range this step
        collision  : bool             True on the step a collision is registered
        yielding   : bool             the env's actual yield decision this step
        yield_available : bool        whether the yield axis is active this stage
        """
        self.total_steps += 1
        self.yield_available = int(bool(yield_available))
        rx, ry, ryaw = robot_pose
        rvx, rvy = _vel_xy(float(robot_v), float(ryaw))

        step_min_d = float("inf")
        step_min_mode = ""
        any_human = False
        for h in (humans or []):
            any_human = True
            hx, hy = float(h["x"]), float(h["y"])
            mode = str(h.get("mode", "") or "")
            if mode:
                self.modes_present.add(mode)
            px, py = hx - rx, hy - ry
            d = math.hypot(px, py)
            if d < step_min_d:
                step_min_d = d
                step_min_mode = mode
            # Time-to-contact from CURRENT relative motion; only closing humans
            # contribute (range-rate toward the robot > 0).
            if "vx" in h and "vy" in h:
                hvx, hvy = float(h["vx"]), float(h["vy"])
            else:
                hvx, hvy = _vel_xy(float(h.get("v", 0.0)), float(h.get("yaw", 0.0)))
            if d > 1e-6:
                closing = -(px * (hvx - rvx) + py * (hvy - rvy)) / d
                if closing > 1e-6:
                    gap = d - self.ttc_r
                    ttc = 0.0 if gap <= 0.0 else gap / closing
                    if ttc < self.min_ttc:
                        self.min_ttc = ttc

        if any_human:
            self.observed_steps += 1
            if step_min_d < self.min_dist:
                self.min_dist = step_min_d
                self.nearest_mode = step_min_mode
            if step_min_d < self.near:
                self.below_near_steps += 1
            if step_min_d < self.interact:
                self.interaction = True

        # Static-clutter pressure: geometry is close but the cause is NOT a human
        # (no pedestrian, or the nearest one is outside the collision-attrib band).
        if math.isfinite(min_lidar_m) and min_lidar_m < self.clutter_lidar:
            if (not any_human) or step_min_d > self.coll_r:
                self.static_clutter_steps += 1

        # Yield accounting from the env's real yield decision.
        is_risk = any_human and step_min_d < self.near
        if yielding:
            self.yield_steps += 1
            if not self._prev_yielding:
                self.yield_onsets += 1
            if is_risk:
                self.yield_in_risk_steps += 1
            else:
                self.yield_no_risk_steps += 1
        self._prev_yielding = bool(yielding)

        # Collision attribution — lock on the FIRST collision step of the episode.
        if collision and not self._collision_locked:
            self._collision_locked = True
            if any_human and step_min_d <= self.coll_r:
                self.collision_type = "human"
            else:
                self.collision_type = "static"

    # -- outputs -----------------------------------------------------------
    def state_key(self):
        """Cheap hashable signature of the OUTPUT-affecting state.

        Lets the caller publish the (relatively costly) ROS parameter only when
        the diagnostic content actually changed, instead of every env step —
        ``total_steps`` is deliberately excluded because it is not part of
        :meth:`as_dict`. Rounded floats keep sub-mm/us jitter from forcing a
        needless republish."""
        return (
            self.collision_type,
            self.observed_steps,
            self.below_near_steps,
            self.static_clutter_steps,
            self.interaction,
            self.yield_available,
            self.yield_steps,
            self.yield_onsets,
            self.yield_in_risk_steps,
            self.yield_no_risk_steps,
            self.nearest_mode,
            len(self.modes_present),
            None if not math.isfinite(self.min_dist) else round(self.min_dist, 4),
            None if not math.isfinite(self.min_ttc) else round(self.min_ttc, 4),
        )

    def _finite_or_nan(self, x):
        return float(x) if math.isfinite(x) else self.UNAVAILABLE

    def min_human_distance_m(self):
        return self._finite_or_nan(self.min_dist)

    def human_ttc_min(self):
        if not math.isfinite(self.min_ttc):
            return self.UNAVAILABLE
        return float(min(self.min_ttc, self.ttc_cap))

    def time_below_human_clearance_ratio(self):
        if self.observed_steps == 0:
            return self.UNAVAILABLE
        return self.below_near_steps / self.observed_steps

    def near_human_event(self):
        # 0 when pedestrians were present but never near; NaN only when no
        # pedestrian was ever observed (event is undefined).
        if self.observed_steps == 0:
            return self.UNAVAILABLE
        return int(self.below_near_steps > 0)

    def as_dict(self) -> dict:
        """Flat diagnostic record (all keys always present)."""
        return {
            "collision_object_type": self.collision_type,
            "min_human_distance_m": self.min_human_distance_m(),
            "human_ttc_min": self.human_ttc_min(),
            "time_below_human_clearance_ratio": self.time_below_human_clearance_ratio(),
            "near_human_event": self.near_human_event(),
            "has_human_interaction": int(self.interaction),
            "has_static_clutter_pressure": int(self.static_clutter_steps > 0),
            "static_clutter_steps": int(self.static_clutter_steps),
            "human_observed_steps": int(self.observed_steps),
            "nearest_human_mode": self.nearest_mode,
            "human_modes_present": ",".join(sorted(self.modes_present)),
            # Yield diagnostics (env's actual per-stage-gated decision).
            "yield_available": int(self.yield_available),
            "yield_used": int(self.yield_steps > 0),
            "yield_steps": int(self.yield_steps),
            "yield_trigger_count": int(self.yield_onsets),
            "yield_in_risk_steps": int(self.yield_in_risk_steps),
            "yield_no_risk_steps": int(self.yield_no_risk_steps),
            "risk_steps": int(self.below_near_steps),
        }
