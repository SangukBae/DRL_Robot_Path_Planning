#!/usr/bin/env python3
"""Shared Pure-Pursuit action→waypoint→command helpers.

Extracted from environment.py so the simulator (environment.py) and the
real-robot inference node (real_policy_runner.py) use *exactly* the same
action→cmd_vel convention. Pure functions only — no ROS, no state.

Convention (matches the trained policy):
  action[0] → waypoint distance r  ∈ [actions_low[0], actions_high[0]] m (forward)
  action[1] → waypoint angle theta ∈ [actions_low[1], actions_high[1]] rad
              (robot frame, positive = left / CCW)
  waypoint (robot frame): x_wp = r·cos(theta) (forward), y_wp = r·sin(theta) (left)
  command: center steering angle [rad] (clipped) + forward speed [m/s].
"""

import math
import numpy as np


def action_to_waypoint(action, actions_low, actions_high):
    """Map a normalized action in [-1, 1]^2 to (r, theta, x_wp, y_wp)."""
    a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
    low = np.asarray(actions_low, dtype=np.float32)
    high = np.asarray(actions_high, dtype=np.float32)
    cmd = 0.5 * (a + 1.0) * (high - low) + low  # [-1, 1] → [low, high]
    r = float(np.clip(cmd[0], low[0], high[0]))
    theta = float(np.clip(cmd[1], low[1], high[1]))
    return r, theta, r * math.cos(theta), r * math.sin(theta)


def waypoint_to_command(x_wp, y_wp, wheelbase_m, steering_limit_rad,
                        cruise_speed_mps, min_speed_mps, speed_steer_factor):
    """Pure Pursuit: robot-frame waypoint → (speed [m/s], steering [rad])."""
    L = math.hypot(x_wp, y_wp)
    if L < 1e-3:
        return 0.0, 0.0
    steering = math.atan2(2.0 * y_wp * wheelbase_m, L * L)
    steering = float(np.clip(steering, -steering_limit_rad, steering_limit_rad))
    steer_ratio = abs(steering) / max(steering_limit_rad, 1e-6)
    speed = cruise_speed_mps * (1.0 - speed_steer_factor * steer_ratio)
    speed = max(speed, min_speed_mps)
    return speed, steering
