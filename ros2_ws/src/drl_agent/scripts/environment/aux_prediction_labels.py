# AUX_PRED: privileged future-risk label generation for the auxiliary
# prediction network.  This module is the ONLY place that turns the
# simulator's privileged pedestrian ground-truth (environment.py
# self.human_states) into the fixed-size training labels consumed by the
# TQC auxiliary head.
#
# Design provenance (see docs/aux_prediction_tqc_design.md):
#   - DiPCAN (Monaci et al., RSS 2022): privileged information is used ONLY
#     to build training labels in simulation; nothing here is needed at
#     deployment / inference.
#   - Proximity-Aware (Cancelli et al., ICCV 2023): instead of regressing a
#     variable-length, ID-matched per-human trajectory, we predict a FIXED
#     size egocentric "risk" representation (Risk Estimation + Social Compass
#     collapsed into a per-sector risk map).  risk = clamp(1 - d/D_c, 0, 1).
#   - Falcon (From Cognition to Precognition, 2024/2025): the labels are a
#     future-aware auxiliary target; here we use several short horizons so the
#     shared encoder is pushed to encode where dynamic obstacles are GOING.
#
# The privileged pedestrians are propagated kinematically in environment.py;
# for the label we use a simple constant-velocity (CV) rollout per horizon,
# which is accurate over the short, deliberately-predictable horizons that
# environment_curriculum.py is designed to produce (non-reactive humans).
#
# Canonical label layout (1-D float32 vector), length = H*K + H:
#   [ risk_map (H * K) ][ min_dist_norm (H) ]
#     risk_map[h, k] in [0, 1]  -> closeness risk of nearest human in sector k
#     min_dist_norm[h] in [0, 1] -> clamp(min_distance / D_c, 0, 1) (1 = far)
# The min_dist block is ALWAYS emitted (cheap) so the env stays decoupled from
# which agent-side heads are enabled; the agent simply ignores it when its
# future-min-distance head is off.
#
# Wire format (what the env appends to the state, see wire_header / parse_aux_wire):
#   [ VERSION, K, H, h_0 .. h_{H-1} ][ risk_map (H*K) ][ min_dist_norm (H) ]
#     ^------------ geometry header ------------^^------------ label ------------^
# The header lets the consumer (EnvInterface) recover the EXACT geometry the env
# used and lets the trainer fail-fast on a STRUCTURAL mismatch (different
# num_sectors / horizons_sec), not merely a total-length mismatch.

import math

import numpy as np

DEFAULT_HORIZONS_SEC = [0.5, 1.0, 1.5]
DEFAULT_NUM_SECTORS = 16
DEFAULT_RISK_DISTANCE_SCALE = 3.0

# Bump when the wire layout below changes incompatibly.
AUX_WIRE_VERSION = 1
# Fixed-size part of the geometry header preceding the horizon list:
#   [VERSION, K, H]  (the H horizon values follow).
AUX_WIRE_HEADER_FIXED = 3


class AuxLabelConfig:
    """AUX_PRED: parsed config for label generation (env side)."""

    def __init__(self, cfg: dict = None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.horizons_sec = list(cfg.get("horizons_sec", DEFAULT_HORIZONS_SEC))
        self.num_sectors = int(cfg.get("num_sectors", DEFAULT_NUM_SECTORS))
        self.risk_distance_scale = float(
            cfg.get("risk_distance_scale", DEFAULT_RISK_DISTANCE_SCALE)
        )
        # Below D_c we start to count a human as "risk"; speeds below this are
        # treated as effectively stationary for the CV rollout.
        self.min_speed_for_motion = float(cfg.get("min_speed_for_motion", 0.05))

    @property
    def num_horizons(self) -> int:
        return len(self.horizons_sec)

    @property
    def label_dim(self) -> int:
        # H*K risk map + H min-distance scalars.
        return self.num_horizons * self.num_sectors + self.num_horizons

    def wire_header(self) -> list:
        """AUX_PRED: geometry header prepended to the label on the wire:
        [VERSION, K, H, h_0 .. h_{H-1}]."""
        return (
            [float(AUX_WIRE_VERSION), float(self.num_sectors), float(self.num_horizons)]
            + [float(h) for h in self.horizons_sec]
        )

    @property
    def wire_dim(self) -> int:
        """Total appended length = header (3 + H) + label (H*K + H)."""
        return AUX_WIRE_HEADER_FIXED + self.num_horizons + self.label_dim


def aux_label_dim(cfg: dict) -> int:
    """AUX_PRED: convenience helper returning the canonical label length."""
    return AuxLabelConfig(cfg).label_dim


def parse_aux_wire(tail):
    """AUX_PRED: split an appended wire tail into (meta, label).

    tail layout: [VERSION, K, H, h_0 .. h_{H-1}, <label of length H*K + H>].
    Returns
    -------
    (meta, label)
        meta : dict with version / num_sectors / num_horizons / horizons_sec,
               or None if the header is missing or malformed.
        label: float32 ndarray (the geometry header removed).  When the header
               cannot be parsed the whole tail is returned as the label so the
               caller can still decide what to do.
    """
    arr = np.asarray(tail, dtype=np.float32).ravel()
    if arr.shape[0] < AUX_WIRE_HEADER_FIXED:
        return None, arr.copy()
    version = int(round(float(arr[0])))
    num_sectors = int(round(float(arr[1])))
    num_horizons = int(round(float(arr[2])))
    hlen = AUX_WIRE_HEADER_FIXED + num_horizons
    if num_horizons <= 0 or num_sectors <= 0 or hlen > arr.shape[0]:
        return None, arr.copy()
    meta = {
        "version": version,
        "num_sectors": num_sectors,
        "num_horizons": num_horizons,
        "horizons_sec": [float(x) for x in arr[AUX_WIRE_HEADER_FIXED:hlen]],
    }
    label = arr[hlen:].copy()
    return meta, label


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compute_future_risk_labels(humans, robot_pose, cfg: AuxLabelConfig):
    """AUX_PRED: build the canonical label vector.

    Parameters
    ----------
    humans : iterable of dict-like
        Each item must expose x, y and a velocity estimate.  Accepted keys:
          - "x", "y"                 : current world position [m]
          - "vx", "vy" (optional)    : world-frame velocity [m/s]
          - "v", "yaw" (fallback)    : speed + heading -> velocity
    robot_pose : (rx, ry, ryaw)
        Current ground-truth robot pose in the same world frame.
    cfg : AuxLabelConfig

    Returns
    -------
    list[float] of length cfg.label_dim
    """
    H = cfg.num_horizons
    K = cfg.num_sectors
    Dc = max(cfg.risk_distance_scale, 1e-6)
    rx, ry, ryaw = robot_pose

    risk_map = [[0.0] * K for _ in range(H)]
    min_dist_norm = [1.0] * H  # 1.0 == no human within D_c

    humans = list(humans or [])
    if humans:
        # Pre-extract (x, y, vx, vy) once per human.
        parsed = []
        for s in humans:
            x = float(s["x"])
            y = float(s["y"])
            if "vx" in s and "vy" in s:
                vx = float(s["vx"])
                vy = float(s["vy"])
            else:
                v = float(s.get("v", 0.0))
                yaw = float(s.get("yaw", 0.0))
                if abs(v) < cfg.min_speed_for_motion:
                    vx = vy = 0.0
                else:
                    vx = v * math.cos(yaw)
                    vy = v * math.sin(yaw)
            parsed.append((x, y, vx, vy))

        for hi, h_sec in enumerate(cfg.horizons_sec):
            min_d = float("inf")
            row = risk_map[hi]
            for (x, y, vx, vy) in parsed:
                fx = x + vx * h_sec
                fy = y + vy * h_sec
                dx = fx - rx
                dy = fy - ry
                d = math.hypot(dx, dy)
                if d < min_d:
                    min_d = d
                ang = _wrap_pi(math.atan2(dy, dx) - ryaw)
                k = int((ang + math.pi) / (2.0 * math.pi) * K)
                if k < 0:
                    k = 0
                elif k >= K:
                    k = K - 1
                risk = 1.0 - d / Dc
                if risk < 0.0:
                    risk = 0.0
                elif risk > 1.0:
                    risk = 1.0
                if risk > row[k]:
                    row[k] = risk
            if math.isfinite(min_d):
                mn = min_d / Dc
                min_dist_norm[hi] = 0.0 if mn < 0.0 else (1.0 if mn > 1.0 else mn)

    out = []
    for hi in range(H):
        out.extend(risk_map[hi])
    out.extend(min_dist_norm)
    return out
