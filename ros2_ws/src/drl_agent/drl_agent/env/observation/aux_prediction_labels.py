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
# Canonical label layout (1-D float32 vector), append-only:
#   [ risk_map (H*K) ][ min_dist_norm (H) ][ ttc_norm (T) ][ hazard (S) ]
#     risk_map[h, k] in [0, 1]  -> closeness risk of nearest human in sector k
#     min_dist_norm[h] in [0, 1] -> clamp(min_distance / D_c, 0, 1) (1 = far)
#     ttc_norm (T in {0,1})  -> clamp(min time-to-contact / ttc_horizon, 0, 1),
#                               1 == safe (no human is CLOSING / beyond horizon),
#                               0 == contact imminent. Emitted iff ttc_head_enabled.
#     hazard (S)             -> per-sector {0,1} imminent-hazard flag (front-centred
#                               S bins; a CLOSING human within hazard_distance sets
#                               its sector). Emitted iff hazard_sector_head_enabled.
# risk_map + min_dist are ALWAYS emitted (cheap, the original v1 label) so the env
# stays decoupled from which agent heads are on; ttc / hazard are append-only and
# gated by env config. NOTE: the wire VERSION is always 2 now (the header always
# carries T/S, see below), so a disabled config is NOT byte-for-byte v1 on the
# wire — but the LABEL CONTENT (the risk_map + min_dist values) is identical to
# v1, and parse_aux_wire reads both v1 and v2 headers, so a v2 label with T=S=0
# is content-equivalent to the old v1 label.
#
# Wire format (what the env appends to the state, see wire_header / parse_aux_wire):
#   v2: [ VERSION, K, H, T, S, h_0 .. h_{H-1} ][ risk_map ][ min_dist ][ ttc ][ hazard ]
#   v1: [ VERSION, K, H, h_0 .. h_{H-1} ][ risk_map ][ min_dist ]   (legacy, T=S=0)
#     ^------------- geometry header -------------^^------------- label -------------^
# The header lets the consumer (EnvInterface) recover the EXACT geometry the env
# used and lets the trainer fail-fast on a STRUCTURAL mismatch (different
# num_sectors / horizons_sec / ttc / hazard sizes), not merely a total length.

import math

import numpy as np

DEFAULT_HORIZONS_SEC = [0.5, 1.0, 1.5]
DEFAULT_NUM_SECTORS = 16
DEFAULT_RISK_DISTANCE_SCALE = 3.0

# Bump when the wire layout below changes incompatibly. v2 adds the ttc/hazard
# header fields (T, S) and the append-only ttc/hazard label blocks.
AUX_WIRE_VERSION = 2
# Fixed-size part of the v2 geometry header preceding the horizon list:
#   [VERSION, K, H, T, S]  (the H horizon values follow). v1 used [VERSION, K, H].
AUX_WIRE_HEADER_FIXED = 5
AUX_WIRE_HEADER_FIXED_V1 = 3


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

        # ── Append-only direct dynamic-avoidance targets (v2, default OFF) ─────
        # TTC head: a single normalized minimum time-to-contact target. Only
        # CLOSING humans contribute; a non-closing / collision-free situation maps
        # to the safe sentinel (1.0). Gated: disabled -> no ttc block (label
        # content matches v1; the wire header is still v2 with ttc_len=0).
        self.ttc_head_enabled = bool(cfg.get("ttc_head_enabled", False))
        self.ttc_horizon_sec = float(cfg.get("ttc_horizon_sec", 3.0))
        # Contact distance for TTC / hazard (robot radius + human radius proxy).
        self.ttc_collision_radius = float(cfg.get("ttc_collision_radius", 0.5))
        # Hazard-sector head: low-dim front-centred imminent-hazard classification.
        self.hazard_sector_head_enabled = bool(
            cfg.get("hazard_sector_head_enabled", False))
        self.hazard_sector_bins = int(cfg.get("hazard_sector_bins", 3))
        # A human counts as a sector hazard only within this distance AND closing.
        self.hazard_distance = float(
            cfg.get("hazard_distance", self.risk_distance_scale))

    @property
    def num_horizons(self) -> int:
        return len(self.horizons_sec)

    @property
    def ttc_len(self) -> int:
        return 1 if self.ttc_head_enabled else 0

    @property
    def hazard_len(self) -> int:
        return self.hazard_sector_bins if self.hazard_sector_head_enabled else 0

    @property
    def label_dim(self) -> int:
        # H*K risk map + H min-distance scalars + optional ttc + optional hazard.
        return (self.num_horizons * self.num_sectors + self.num_horizons
                + self.ttc_len + self.hazard_len)

    def wire_header(self) -> list:
        """AUX_PRED: geometry header prepended to the label on the wire (v2):
        [VERSION, K, H, T, S, h_0 .. h_{H-1}]."""
        return (
            [float(AUX_WIRE_VERSION), float(self.num_sectors),
             float(self.num_horizons), float(self.ttc_len), float(self.hazard_len)]
            + [float(h) for h in self.horizons_sec]
        )

    @property
    def wire_dim(self) -> int:
        """Total appended length = header (5 + H) + label."""
        return AUX_WIRE_HEADER_FIXED + self.num_horizons + self.label_dim


def aux_label_dim(cfg: dict) -> int:
    """AUX_PRED: convenience helper returning the canonical label length."""
    return AuxLabelConfig(cfg).label_dim


# ── PHASE2: Action-Risk Head env-side supervision-target wire block ──────────
# A separate, independently-gated tiny block ([SENTINEL, risk_dir, min_dist_dir])
# environment.py may prepend ahead of the (optional) aux_prediction tail above —
# see environment.py::_append_aux_labels and environment_interface.py::
# EnvInterface._strip_action_risk_target. It is ACTION-CONDITIONED (computed for
# the exact theta just decoded from the selected action), unlike the aux label
# above (which is action-agnostic), so it needs no cur/next carry-over: it is
# attached directly to the response for the transition that just happened.
#
# ACTION_RISK_WIRE_SENTINEL can never collide with a legitimate wire value: the
# aux wire's own VERSION field is always a small positive int (1 or 2), and
# risk_dir/min_dist_dir are always in [0, 1].
ACTION_RISK_WIRE_SENTINEL = -999.0
ACTION_RISK_WIRE_LEN = 3   # [SENTINEL, risk_dir, min_dist_dir]


def strip_action_risk_wire(tail):
    """PHASE2: split an optional action-risk-head target off the FRONT of an
    appended wire tail.

    Returns
    -------
    (action_risk_target, remainder)
        action_risk_target : (risk_dir, min_dist_dir) floats, or None when the
                              block is absent (tail too short / no sentinel).
        remainder           : the tail with that block removed (the ORIGINAL
                              tail, unchanged, when the block is absent) — the
                              caller (EnvInterface) hands this on to
                              parse_aux_wire exactly as before.
    """
    arr = np.asarray(tail, dtype=np.float32).ravel()
    if arr.shape[0] >= ACTION_RISK_WIRE_LEN and float(arr[0]) == ACTION_RISK_WIRE_SENTINEL:
        return (float(arr[1]), float(arr[2])), arr[ACTION_RISK_WIRE_LEN:].copy()
    return None, arr.copy()


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
    if arr.shape[0] < AUX_WIRE_HEADER_FIXED_V1:
        return None, arr.copy()
    version = int(round(float(arr[0])))
    num_sectors = int(round(float(arr[1])))
    num_horizons = int(round(float(arr[2])))
    # v2 carries the ttc / hazard block sizes (T, S) in the fixed header; v1 did
    # not (they are 0). Branch on version so an old v1 wire still parses cleanly.
    if version >= 2:
        if arr.shape[0] < AUX_WIRE_HEADER_FIXED:
            return None, arr.copy()
        ttc_len = int(round(float(arr[3])))
        hazard_len = int(round(float(arr[4])))
        fixed = AUX_WIRE_HEADER_FIXED          # 5
    else:
        ttc_len = 0
        hazard_len = 0
        fixed = AUX_WIRE_HEADER_FIXED_V1        # 3
    hlen = fixed + num_horizons
    if num_horizons <= 0 or num_sectors <= 0 or hlen > arr.shape[0]:
        return None, arr.copy()
    meta = {
        "version": version,
        "num_sectors": num_sectors,
        "num_horizons": num_horizons,
        "ttc_len": ttc_len,
        "hazard_len": hazard_len,
        "horizons_sec": [float(x) for x in arr[fixed:hlen]],
    }
    label = arr[hlen:].copy()
    return meta, label


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def sector_index_for_theta(theta: float, num_sectors: int) -> int:
    """Angle (radians, robot frame) -> risk_map sector index.

    Canonical convention for this module's risk_map K-axis: straight ahead
    (theta=0) maps to the MIDDLE bin (num_sectors // 2), not bin 0. Also
    re-exported/duplicated (documented as matching this EXACTLY) by
    utils/risk_map_dump.py for eval-time dump records -- this is the single
    source of truth the formula must stay in sync with.
    """
    if num_sectors <= 0:
        return 0
    ang = _wrap_pi(theta)
    k = int((ang + math.pi) / (2.0 * math.pi) * num_sectors)
    return max(0, min(num_sectors - 1, k))


def _parse_humans(humans, min_speed_for_motion):
    """Pre-extract (x, y, vx, vy) once per human (empty list when no humans).
    Shared by compute_future_risk_labels and compute_directional_risk_map so
    the velocity-fallback convention (vx/vy if given, else v+yaw, else
    stationary below min_speed_for_motion) has ONE implementation."""
    parsed = []
    for s in (humans or []):
        x = float(s["x"])
        y = float(s["y"])
        if "vx" in s and "vy" in s:
            vx = float(s["vx"])
            vy = float(s["vy"])
        else:
            v = float(s.get("v", 0.0))
            yaw = float(s.get("yaw", 0.0))
            if abs(v) < min_speed_for_motion:
                vx = vy = 0.0
            else:
                vx = v * math.cos(yaw)
                vy = v * math.sin(yaw)
        parsed.append((x, y, vx, vy))
    return parsed


def compute_directional_risk_map(humans, robot_pose, cfg: AuxLabelConfig):
    """PHASE2: per-SECTOR (risk, min_dist) pair, using ONLY the first horizon
    (cfg.horizons_sec[0]) -- environment.py's _directional_risk_cfg is single-
    horizon by construction (see environment.py::_compute_directional_risk).

    UNLIKE compute_future_risk_labels's min_dist_norm -- which is, by design,
    the horizon's GLOBAL nearest-human distance over ALL humans/sectors, and
    MUST stay exactly that for AUX_PRED wire-format / checkpoint compatibility
    -- min_dist_row here is the nearest human WITHIN EACH SECTOR. That is what
    lets risk_dir/min_dist_dir (sliced at the SAME sector index in
    environment.py) both genuinely describe the SAME direction, instead of
    pairing a directional risk with a distance that could come from a human in
    a completely different direction.

    Returns (risk_row, min_dist_row), each length cfg.num_sectors, both in
    [0, 1] (min_dist_row[k]: 1 == no human in sector k within D_c).
    """
    K = cfg.num_sectors
    Dc = max(cfg.risk_distance_scale, 1e-6)
    rx, ry, ryaw = robot_pose
    h_sec = cfg.horizons_sec[0] if cfg.horizons_sec else 0.0

    risk_row = [0.0] * K
    min_dist_row = [1.0] * K  # 1.0 == no human within D_c in this sector

    for (x, y, vx, vy) in _parse_humans(humans, cfg.min_speed_for_motion):
        fx = x + vx * h_sec
        fy = y + vy * h_sec
        dx = fx - rx
        dy = fy - ry
        d = math.hypot(dx, dy)
        ang = math.atan2(dy, dx) - ryaw
        k = sector_index_for_theta(ang, K)

        risk = 1.0 - d / Dc
        if risk < 0.0:
            risk = 0.0
        elif risk > 1.0:
            risk = 1.0
        if risk > risk_row[k]:
            risk_row[k] = risk

        dn = d / Dc
        if dn < 0.0:
            dn = 0.0
        elif dn > 1.0:
            dn = 1.0
        if dn < min_dist_row[k]:
            min_dist_row[k] = dn

    return risk_row, min_dist_row


def compute_future_risk_labels(humans, robot_pose, cfg: AuxLabelConfig,
                               robot_vel=(0.0, 0.0)):
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
    robot_vel : (rvx, rvy)
        Current ground-truth robot world-frame velocity [m/s], used ONLY by the
        TTC / hazard targets (relative motion). Defaults to 0 (robot treated as
        stationary) so callers that do not pass it still get valid risk/min_dist.

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

    parsed = _parse_humans(humans, cfg.min_speed_for_motion)

    if parsed:
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
                ang = math.atan2(dy, dx) - ryaw
                k = sector_index_for_theta(ang, K)
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

    # ── Append-only direct targets (v2). Both use CURRENT relative motion. ─────
    if cfg.ttc_head_enabled:
        out.append(_compute_ttc_norm(parsed, rx, ry, robot_vel, cfg))
    if cfg.hazard_sector_head_enabled:
        out.extend(_compute_hazard_sectors(parsed, rx, ry, ryaw, robot_vel, cfg))
    return out


def _closing_speed(px, py, d, relvx, relvy):
    """Range-rate of the human TOWARD the robot: >0 closing, <=0 separating."""
    if d <= 1e-6:
        return 0.0
    return -(px * relvx + py * relvy) / d


def _compute_ttc_norm(parsed, rx, ry, robot_vel, cfg: AuxLabelConfig) -> float:
    """AUX_PRED (v2): normalized minimum time-to-contact in [0, 1].

    Only CLOSING humans contribute (range-rate > 0). TTC = (gap-to-contact) /
    closing_speed, where gap = max(0, d - collision_radius). No closing human (or
    no human) -> the SAFE SENTINEL 1.0; an already-overlapping/imminent contact
    -> 0.0. Clamped by ttc_horizon_sec so the target is bounded and dimensionless
    (1 = safe / >= horizon, 0 = contact now)."""
    rvx, rvy = robot_vel
    R = cfg.ttc_collision_radius
    Th = max(cfg.ttc_horizon_sec, 1e-6)
    min_ttc = float("inf")
    for (x, y, vx, vy) in parsed:
        px, py = x - rx, y - ry
        d = math.hypot(px, py)
        closing = _closing_speed(px, py, d, vx - rvx, vy - rvy)
        if closing <= 1e-6:
            continue
        gap = d - R
        ttc = 0.0 if gap <= 0.0 else gap / closing
        if ttc < min_ttc:
            min_ttc = ttc
    if not math.isfinite(min_ttc):
        return 1.0
    n = min_ttc / Th
    return 0.0 if n < 0.0 else (1.0 if n > 1.0 else n)


def _compute_hazard_sectors(parsed, rx, ry, ryaw, robot_vel, cfg: AuxLabelConfig):
    """AUX_PRED (v2): per-sector {0,1} imminent-hazard flags (front-centred bins).

    A human flags its sector iff it is within hazard_distance AND CLOSING. Bins
    are front-centred: bin 0 spans bearings around the robot heading. With no
    humans (or none qualifying) the result is all-zeros — a stable, well-defined
    'no hazard' target."""
    S = max(1, cfg.hazard_sector_bins)
    hz = [0.0] * S
    rvx, rvy = robot_vel
    Hd = cfg.hazard_distance
    bin_w = 2.0 * math.pi / S
    for (x, y, vx, vy) in parsed:
        px, py = x - rx, y - ry
        d = math.hypot(px, py)
        if d > Hd:
            continue
        closing = _closing_speed(px, py, d, vx - rvx, vy - rvy)
        if closing <= 1e-6:
            continue
        bearing = _wrap_pi(math.atan2(py, px) - ryaw)
        # front-centred: bin 0 = bearings in [-bin_w/2, bin_w/2).
        idx = int(round(bearing / bin_w)) % S
        hz[idx] = 1.0
    return hz
