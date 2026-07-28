"""STAGE 8: fixed physical-range observation normalization (ISOLATED, opt-in
experimental feature -- default OFF, NOT enabled for phase2/both).

Preferred over running (mean/std, e.g. Welford/BatchNorm-style) normalization
per the task spec: fixed physical ranges are deterministic, reproducible
across seeds/runs, and never depend on the specific trajectory a run happened
to see early in training.

Builds a single per-index (scale, offset) pair covering the FULL transported
state vector -- including the observation_time_context history frames, when
enabled -- and applies ``(x - offset) / scale`` element-wise BEFORE the state
reaches the encoder. Uses the SAME per-index scale vector for every frame in
the stack (current + history), satisfying "identical normalization applied
to encoder and temporal history inputs" by construction: the current frame
and every history frame's LiDAR block are literally the same slice pattern
tiled, so they always share one scale value per relative bin index.

Canonical 87-D current-frame layout (see CLAUDE.md "DRL State/Action Space"):
  [0:80]  LiDAR (front 180 deg, 80 bins)     -> lidar_scale     (physical range [m])
  [80]    goal distance                       -> goal_dist_scale (physical range [m])
  [81]    heading error to goal (theta)        -> heading_scale   (radians, symmetric)
  [82]    previous action r (waypoint dist)     -> prev_action_r_scale (already in
  [83]    previous action theta (waypoint ang)   prev_action_theta_scale  policy units)
  [84]    actual signed speed [m/s]             -> speed_scale
  [85]    actual yaw rate [rad/s]               -> yaw_rate_scale
  [86]    center steering angle [rad]           -> steering_scale

With observation_time_context enabled the transported state is
``current(87) + (history_len-1) * frame_len`` where frame_len is
``obs_dim(80)`` (obs-only history, the default) or ``obs_dim+agent_dim(87)``
(stack_agent_state=true) -- each history frame's leading ``obs_dim`` slice is
LiDAR and gets the SAME lidar_scale as the current frame's LiDAR block.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ObsNormalizationConfig:
    enabled: bool = False
    lidar_scale: float = 50.0
    goal_dist_scale: float = 20.0
    heading_scale: float = 3.14159265358979
    prev_action_r_scale: float = 1.0
    prev_action_theta_scale: float = 1.0
    speed_scale: float = 2.0
    yaw_rate_scale: float = 1.0
    steering_scale: float = 0.4363323129985824  # ~25 deg, overridden from config normally
    # Offsets: 0.0 for every block by default (all documented ranges are
    # already zero-centered or one-sided-from-zero; symmetric blocks like
    # heading/yaw_rate/steering use their scale as a symmetric divisor, not
    # an [0,1]-style min-max remap).
    offset: float = 0.0

    @classmethod
    def from_dict(cls, cfg: dict):
        cfg = dict(cfg or {})
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            lidar_scale=float(cfg.get("lidar_scale", 50.0)),
            goal_dist_scale=float(cfg.get("goal_dist_scale", 20.0)),
            heading_scale=float(cfg.get("heading_scale", 3.14159265358979)),
            prev_action_r_scale=float(cfg.get("prev_action_r_scale", 1.0)),
            prev_action_theta_scale=float(cfg.get("prev_action_theta_scale", 1.0)),
            speed_scale=float(cfg.get("speed_scale", 2.0)),
            yaw_rate_scale=float(cfg.get("yaw_rate_scale", 1.0)),
            steering_scale=float(cfg.get("steering_scale", 0.4363323129985824)),
            offset=float(cfg.get("offset", 0.0)),
        )

    def manifest_dict(self) -> dict:
        """STAGE 8: recorded verbatim into the checkpoint manifest so a
        checkpoint saved under a DIFFERENT normalization contract can never
        be silently loaded (see tqc_io's normalization-contract check)."""
        return {
            "enabled": bool(self.enabled),
            "lidar_scale": self.lidar_scale,
            "goal_dist_scale": self.goal_dist_scale,
            "heading_scale": self.heading_scale,
            "prev_action_r_scale": self.prev_action_r_scale,
            "prev_action_theta_scale": self.prev_action_theta_scale,
            "speed_scale": self.speed_scale,
            "yaw_rate_scale": self.yaw_rate_scale,
            "steering_scale": self.steering_scale,
            "offset": self.offset,
        }


def build_scale_vector(cfg: ObsNormalizationConfig, state_dim: int,
                        obs_dim: int = 80, agent_dim: int = 7,
                        history_len: int = 1, stack_agent_state: bool = False) -> np.ndarray:
    """Return a (state_dim,) float32 array: index i holds the scale to divide
    state[i] by. Tiles the current-frame's 7-slot agent scale pattern
    ([goal_dist, heading, prev_r, prev_theta, speed, yaw_rate, steering])
    once (current frame only -- history frames never carry agent slots
    unless stack_agent_state=True, matching TemporalFusionEncoder._split's
    OWN frame_len choice) and the lidar_scale for every obs_dim-wide LiDAR
    block, current AND every history frame alike.

    Fails fast (ValueError) if state_dim doesn't decompose evenly into
    current(obs_dim+agent_dim) + (history_len-1)*frame_len -- the same
    contract TemporalFusionEncoder.expected_state_dim() already enforces,
    checked independently here so a misconfigured normalizer can never
    silently apply a wrong-length scale vector.
    """
    current_dim = obs_dim + agent_dim
    frame_len = current_dim if stack_agent_state else obs_dim
    expected = current_dim + max(0, history_len - 1) * frame_len
    if expected != state_dim:
        raise ValueError(
            f"[obs_normalization] state_dim={state_dim} does not match "
            f"current({current_dim}) + (history_len-1)*frame_len "
            f"({max(0, history_len - 1)}*{frame_len}) = {expected}. Check "
            "obs_dim/agent_dim/history_len/stack_agent_state.")

    agent_scale = np.array([
        cfg.goal_dist_scale, cfg.heading_scale,
        cfg.prev_action_r_scale, cfg.prev_action_theta_scale,
        cfg.speed_scale, cfg.yaw_rate_scale, cfg.steering_scale,
    ], dtype=np.float32)
    if agent_scale.shape[0] != agent_dim:
        raise ValueError(
            f"[obs_normalization] agent_dim={agent_dim} but the fixed agent-"
            f"slot scale pattern has {agent_scale.shape[0]} entries -- the "
            "canonical 7-slot layout (goal_dist/heading/prev_r/prev_theta/"
            "speed/yaw_rate/steering) does not match this env's agent_dim.")

    lidar_block = np.full(obs_dim, cfg.lidar_scale, dtype=np.float32)
    current_frame = np.concatenate([lidar_block, agent_scale])
    history_frame = (
        np.concatenate([lidar_block, agent_scale]) if stack_agent_state
        else lidar_block
    )
    parts = [current_frame] + [history_frame] * max(0, history_len - 1)
    scale = np.concatenate(parts).astype(np.float32)
    scale[scale == 0.0] = 1.0  # never divide by zero (defensive; scales are configured > 0)
    return scale


class ObsNormalizer:
    """Stateless (no running stats) fixed-scale normalizer: y = (x - offset) / scale.
    Vectorized over a (state_dim,) or (batch, state_dim) numpy/torch input."""

    def __init__(self, cfg: ObsNormalizationConfig, state_dim: int, **layout_kwargs):
        self.cfg = cfg
        self.scale = build_scale_vector(cfg, state_dim, **layout_kwargs)
        self.offset = np.float32(cfg.offset)

    def normalize(self, state):
        if not self.cfg.enabled:
            return state
        if hasattr(state, "detach"):  # torch tensor
            import torch
            scale_t = torch.as_tensor(self.scale, dtype=state.dtype, device=state.device)
            return (state - self.offset) / scale_t
        return (np.asarray(state, dtype=np.float32) - self.offset) / self.scale
