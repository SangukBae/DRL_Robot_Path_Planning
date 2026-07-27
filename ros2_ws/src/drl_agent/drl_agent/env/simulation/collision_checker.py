"""Rectangular safety-region collision / proximity checking.

Extracted from ``environment.py`` (``_compute_rect_safety_hit``,
``_precompute_rect_safety_ranges``, ``_compute_rect_proximity``,
``check_collision``) into a single stateful, ROS-free object.

``RectSafetyChecker`` owns the inflated-footprint geometry and the per-bin
safety / warning ranges that ``environment.py`` previously precomputed once in
``__init__`` and cached on ``self._rect_safety_ranges`` / ``self._rect_warning_ranges``.
The numerics (ray–rectangle intersection, paper Algorithm 1 collision test,
reward-proximity deficit) are unchanged; ``environment.py`` builds one instance
and delegates ``check_collision`` / ``_compute_rect_proximity`` to it.
"""

import math

import numpy as np


class RectSafetyChecker:
    """Ray–rectangle safety geometry for the inflated robot footprint.

    Parameters mirror the values ``environment.py`` reads from
    ``environment.yaml`` (``safety_region`` block) plus the LiDAR bin layout.
    On construction it precomputes the hard (``safety``) and soft (``warning``)
    per-bin ranges, exactly as the original ``__init__`` did.
    """

    def __init__(
        self,
        *,
        d_front: float,
        d_rear: float,
        d_left: float,
        d_right: float,
        margin_front: float,
        margin_rear: float,
        margin_left: float,
        margin_right: float,
        bins,
        environment_dim: int,
        collision_threshold: float,
        lidar_max_range: float,
        warning_scale_front: float = 1.0,
        warning_scale_rear: float = 1.0,
        warning_scale_left: float = 1.0,
        warning_scale_right: float = 1.0,
    ):
        self.sr_d_front = float(d_front)
        self.sr_d_rear = float(d_rear)
        self.sr_d_left = float(d_left)
        self.sr_d_right = float(d_right)
        self.sr_margin_front = float(margin_front)
        self.sr_margin_rear = float(margin_rear)
        self.sr_margin_left = float(margin_left)
        self.sr_margin_right = float(margin_right)

        self.bins = bins
        self.environment_dim = int(environment_dim)
        self.collision_threshold = float(collision_threshold)
        self.lidar_max_range = float(lidar_max_range)

        # Precompute per-bin safety ranges (rectangular footprint, paper Algorithm 1)
        self._rect_safety_ranges = self.precompute_ranges()
        self._rect_warning_ranges = self.precompute_ranges(
            front_scale=warning_scale_front,
            rear_scale=warning_scale_rear,
            left_scale=warning_scale_left,
            right_scale=warning_scale_right,
        )

    # -- read-only accessors (parity with the old cached attributes) ---------
    @property
    def safety_ranges(self) -> np.ndarray:
        return self._rect_safety_ranges

    @property
    def warning_ranges(self) -> np.ndarray:
        return self._rect_warning_ranges

    def compute_safety_hit(
        self,
        angle: float,
        front_scale: float = 1.0,
        rear_scale: float = 1.0,
        left_scale: float = 1.0,
        right_scale: float = 1.0,
    ):
        """Ray–rectangle intersection distance for the inflated robot footprint.

        Returns the first hit distance and the face name. Optional per-face scales
        are applied to the returned distance so the same routine can be reused for
        both the hard boundary and the soft warning boundary.
        """
        d_f  = self.sr_d_front  + self.sr_margin_front
        d_r  = self.sr_d_rear   + self.sr_margin_rear
        d_l  = self.sr_d_left   + self.sr_margin_left
        d_ri = self.sr_d_right  + self.sr_margin_right
        ca, sa = math.cos(angle), math.sin(angle)
        candidates = []
        # Front face  x = +d_f
        if ca > 1e-9:
            t = d_f / ca
            if -d_ri - 1e-6 <= sa * t <= d_l + 1e-6:
                candidates.append((t, "front", front_scale))
        # Rear face   x = -d_r
        elif ca < -1e-9:
            t = d_r / (-ca)
            if -d_ri - 1e-6 <= sa * t <= d_l + 1e-6:
                candidates.append((t, "rear", rear_scale))
        # Left face   y = +d_l
        if sa > 1e-9:
            t = d_l / sa
            if -d_r - 1e-6 <= ca * t <= d_f + 1e-6:
                candidates.append((t, "left", left_scale))
        # Right face  y = -d_ri
        elif sa < -1e-9:
            t = d_ri / (-sa)
            if -d_r - 1e-6 <= ca * t <= d_f + 1e-6:
                candidates.append((t, "right", right_scale))
        if not candidates:
            fallback = max(d_f, d_r, d_l, d_ri)
            return fallback, "none"
        dist, face, scale = min(candidates, key=lambda item: item[0])
        return dist * scale, face

    def precompute_ranges(
        self,
        front_scale: float = 1.0,
        rear_scale: float = 1.0,
        left_scale: float = 1.0,
        right_scale: float = 1.0,
    ) -> np.ndarray:
        """Precompute V_range for every observation bin.

        The earlier phase-sampled version could leave central or boundary bins
        unselected depending on the face sampling phase, which caused missed
        collisions for head-on wall contact. Here every bin center gets its own
        ray-rectangle intersection distance, so the hard and soft boundaries are
        defined continuously across the full 360-degree collision bins (self.bins).
        """
        bin_centers = np.array([0.5 * (lo + hi) for lo, hi in self.bins], dtype=float)
        ranges = np.empty(self.environment_dim, dtype=float)
        for idx, angle in enumerate(bin_centers):
            ranges[idx], _ = self.compute_safety_hit(
                float(angle),
                front_scale=front_scale,
                rear_scale=rear_scale,
                left_scale=left_scale,
                right_scale=right_scale,
            )
        return ranges

    def compute_proximity(self, laser_data) -> float:
        """Proximity to the rectangular safety boundary for reward shaping.

        For each phase-selected bin (finite V_range), computes:
          deficit[i] = max(0, 1 - obs[i] / warning_range[i])
        where warning_range[i] is derived from the same rectangular geometry but
        with per-face warning scales.
        Returns the maximum deficit (0.0 = fully safe, 1.0 = at hard boundary).
        """
        if self._rect_warning_ranges is None or self._rect_safety_ranges is None:
            return 0.0
        obs      = np.asarray(laser_data, dtype=float)
        selected = np.isfinite(self._rect_safety_ranges)
        if not np.any(selected):
            return 0.0
        obs_sel     = obs[selected]
        warning_sel = self._rect_warning_ranges[selected]
        valid = (obs_sel > 0.0) & np.isfinite(obs_sel)
        if not np.any(valid):
            return 0.0
        deficits = np.maximum(0.0, 1.0 - obs_sel[valid] / np.maximum(warning_sel[valid], 1e-6))
        return float(np.max(deficits))

    def check_collision(self, laser_data):
        """Rectangular Safety Region collision detection (paper Algorithm 1).

        Collision is triggered when any LiDAR bin reads strictly less than the
        per-bin safety range V_range[i] (ray–rectangle intersection distance of
        the inflated robot footprint).  Zone infrastructure (_zone_mins etc.) is
        kept intact for the reward function's proximity penalty.

        Returns: (done, collision, min_laser_used)
        """
        if self._rect_safety_ranges is None:
            # Fallback: global-min rule (should not happen after __init__)
            min_laser = float(np.min(laser_data)) if len(laser_data) else float('inf')
            hit = min_laser < self.collision_threshold
            return hit, hit, min_laser

        obs = np.asarray(laser_data, dtype=float)
        # Only consider beams that returned a finite, positive, sub-max reading
        valid = (obs > 0.0) & np.isfinite(obs) & (obs < self.lidar_max_range)
        if not np.any(valid):
            return False, False, float('inf')

        # Phase-selected bins have a finite safety range; unselected bins carry
        # np.inf.  numpy evaluates (obs <= np.inf) as True for any finite obs,
        # so we must AND with the selected mask to avoid false collisions on
        # every unselected bin that receives a valid scan return.
        selected = np.isfinite(self._rect_safety_ranges)
        collision_mask = valid & selected & (obs <= self._rect_safety_ranges)
        if np.any(collision_mask):
            min_used = float(np.min(obs[collision_mask]))
            return True, True, min_used

        return False, False, float(np.min(obs[valid]))
