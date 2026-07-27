"""Per-episode observation-noise emulators (localization + proprioception).

Extracted from ``environment.py`` so the stateful noise models live in their
own ROS-free, unit-testable modules instead of being threaded through the
``Environment`` node as a dozen ``self._loc_*`` / ``self._pp_*`` attributes.

Two cohesive state owners:

* :class:`LocalizationNoiseModel` — the goal-observation corruption axis
  (per-episode registration bias + time-correlated OU measurement error +
  Brownian drift + rare relocalization jumps + optional yaw-flip stress test +
  latency buffer), with per-map-type / corridor-anisotropy multipliers.

* :class:`ProprioNoiseModel` — the proprioception axis (per-episode speed bias /
  scale + yaw-rate bias + per-step Gaussian on speed / yaw-rate / steering +
  latency buffer).

The numerics — and crucially the order of ``np.random`` draws — are unchanged,
so seeded runs reproduce the previous behaviour exactly. ``Environment`` builds
one of each, calls ``reset(...)`` at episode start and ``step(...)`` per RL step,
and keeps its own ``loc_est_*`` cache from the values these return. Both models
are pass-through (identity) when their config ``enabled`` flag is false.
"""

import math
from collections import deque

import numpy as np


class LocalizationNoiseModel:
    """Localization / goal-observation noise emulator.

    Owns all per-episode localization-noise state previously held directly on
    the ``Environment`` node. Construct with the resolved ``loc_noise`` config
    dict and the RL step period ``time_delta``.
    """

    def __init__(self, loc_noise: dict, time_delta: float):
        self.cfg = loc_noise
        self.dt = float(time_delta)
        self._current_map_type = ""
        # Per-episode localization-noise state (initialised in reset()).
        self._loc_bias_x = self._loc_bias_y = self._loc_bias_yaw = 0.0
        self._loc_rw_x = self._loc_rw_y = self._loc_rw_yaw = 0.0
        # OU (time-correlated) measurement-error state.
        self._loc_ou_x = self._loc_ou_y = self._loc_ou_yaw = 0.0
        # Effective per-episode params (base × map-type multiplier + anisotropy),
        # recomputed each reset; passthrough defaults keep step() safe pre-reset.
        self._loc_eff = {
            "sigma_x": 0.0, "sigma_y": 0.0, "sigma_yaw": 0.0,
            "drift_x": 0.0, "drift_y": 0.0, "drift_yaw": 0.0,
            "alpha_xy": 0.0, "alpha_yaw": 0.0, "jump_mult": 1.0,
        }
        self._loc_buf = deque([(0.0, 0.0, 0.0)], maxlen=1)

    def map_multiplier(self, map_type: str):
        """Return per-map-type localization multipliers for the given map_type
        from loc_noise['map_type_multipliers'].

        Returns (m_sigma_xy, m_sigma_yaw, m_drift, m_jump, along_axis, along_extra).
        Safe fallback (all 1.0, no anisotropy) for an empty / legacy / unmapped
        map_type, so non-structured runs are unaffected.

        `along_axis` ("x"/"y") + `along_extra` give anisotropy: the corridor runs
        along WORLD-X in _build_map_layouts (free band |y|<=w/2, full x), so a
        corridor entry uses along_axis: x to blow up the ALONG-corridor position
        (and drift) uncertainty — the real aperture-problem failure mode — while
        the cross-corridor axis keeps the base multiplier."""
        mt = map_type or ""
        table = self.cfg.get("map_type_multipliers", {}) or {}
        e = table.get(mt, {}) if isinstance(table, dict) else {}
        if not isinstance(e, dict):
            e = {}
        return (
            float(e.get("sigma_xy", 1.0)),
            float(e.get("sigma_yaw", 1.0)),
            float(e.get("drift", 1.0)),
            float(e.get("jump", 1.0)),
            str(e.get("along_axis", "") or ""),
            float(e.get("along_extra", 1.0)),
        )

    def reset(self, x0, y0, yaw0, current_map_type: str = ""):
        """Reset ALL per-episode localization-noise state and seed the latency
        buffer / estimate with the (bias-applied) initial pose.

        Returns the seeded estimate ``(x, y, yaw)`` so the caller can keep its
        ``loc_est_*`` cache in sync (the reset observation already matches the
        noise model, so the policy does not jump from a clean initial observation
        to a biased one on the first step). The zero-mean correlated (OU) error
        and the drift start from step 1, seeded at zero. The per-episode
        EFFECTIVE sigmas / drifts (base × current map-type multiplier + corridor
        anisotropy) and the OU memory factors are computed here, since the map
        type is known by reset time. Disabled → clean passthrough."""
        n = self.cfg
        self._current_map_type = current_map_type or ""
        x0, y0, yaw0 = float(x0), float(y0), float(yaw0)
        # Always reset the OU state so correlated error never leaks across episodes.
        self._loc_ou_x = self._loc_ou_y = self._loc_ou_yaw = 0.0
        if not n["enabled"]:
            self._loc_bias_x = self._loc_bias_y = self._loc_bias_yaw = 0.0
            self._loc_rw_x = self._loc_rw_y = self._loc_rw_yaw = 0.0
            self._loc_buf = deque([(x0, y0, yaw0)], maxlen=1)
            return x0, y0, yaw0
        # Per-episode bias (constant registration offset) — part of the goal-obs
        # corruption axis, so only sampled when that axis is enabled.
        _goal_on = n["noise_goal_enabled"]
        self._loc_bias_x = float(np.random.normal(0.0, n["bias_xy_m"]))  if (_goal_on and n["bias_xy_m"]  > 0.0) else 0.0
        self._loc_bias_y = float(np.random.normal(0.0, n["bias_xy_m"]))  if (_goal_on and n["bias_xy_m"]  > 0.0) else 0.0
        self._loc_bias_yaw = float(np.random.normal(0.0, n["bias_yaw_rad"])) if (_goal_on and n["bias_yaw_rad"] > 0.0) else 0.0
        self._loc_rw_x = self._loc_rw_y = self._loc_rw_yaw = 0.0

        # Effective per-axis sigma / drift = base × map-type multiplier (+ corridor
        # anisotropy on the along-corridor axis).
        m_sxy, m_syaw, m_drift, m_jump, along_axis, along_extra = self.map_multiplier(self._current_map_type)
        # Resolve the per-second drift intensity at READ time, preferring the
        # physical-name key `drift_*` when present (curriculum profiles set it as
        # a raw key via deep-merge, which bypasses the __init__ alias resolution)
        # and falling back to the legacy `random_walk_*` key otherwise.
        rate_xy  = float(n["drift_xy_mps"])    if "drift_xy_mps"    in n else float(n["random_walk_xy_mps"])
        rate_yaw = float(n["drift_yaw_radps"]) if "drift_yaw_radps" in n else float(n["random_walk_yaw_rps"])
        sx = n["sigma_xy_m"] * m_sxy
        sy = n["sigma_xy_m"] * m_sxy
        dx = rate_xy * m_drift
        dy = rate_xy * m_drift
        if along_axis == "x":
            sx *= along_extra; dx *= along_extra
        elif along_axis == "y":
            sy *= along_extra; dy *= along_extra
        dt = self.dt
        # OU memory factor alpha = exp(-dt/tau); tau=0 → alpha=0 → white noise
        # of std sigma (legacy behaviour). Stationary std of the OU process = sigma.
        ax   = math.exp(-dt / n["corr_time_xy_s"])  if n["corr_time_xy_s"]  > 0.0 else 0.0
        ayaw = math.exp(-dt / n["corr_time_yaw_s"]) if n["corr_time_yaw_s"] > 0.0 else 0.0
        self._loc_eff = {
            "sigma_x": sx, "sigma_y": sy, "sigma_yaw": n["sigma_yaw_rad"] * m_syaw,
            "drift_x": dx, "drift_y": dy, "drift_yaw": rate_yaw * m_drift,
            "alpha_xy": ax, "alpha_yaw": ayaw, "jump_mult": m_jump,
        }

        ix = x0 + self._loc_bias_x
        iy = y0 + self._loc_bias_y
        iyaw = (yaw0 + self._loc_bias_yaw + math.pi) % (2 * math.pi) - math.pi
        # Latency is its own ablation axis (noise_delay_enabled).
        delay = int(n["delay_steps"]) if n["noise_delay_enabled"] else 0
        maxlen = max(1, delay + 1)
        self._loc_buf = deque([(ix, iy, iyaw)] * maxlen, maxlen=maxlen)
        return ix, iy, iyaw

    def step(self, x, y, yaw):
        """Advance the localization-noise emulator ONE RL step and return the
        (delayed, noisy) estimated pose. No-op (passthrough) when disabled.

        Error = bias + accumulated drift + time-correlated OU measurement error,
        plus rare jumps. The OU term reduces to legacy white Gaussian when the
        correlation time is 0 (alpha=0). Effective per-axis magnitudes come from
        self._loc_eff (base × map-type multiplier + corridor anisotropy)."""
        n = self.cfg
        if not n["enabled"]:
            return x, y, yaw
        dt = self.dt
        eff = self._loc_eff
        sqrt_dt = math.sqrt(dt)

        ex, ey, eyaw = x, y, yaw

        # ── Goal-obs corruption axis (bias + OU + drift) ── gated by noise_goal_enabled.
        if n["noise_goal_enabled"]:
            # Time-correlated (OU) measurement error: e <- alpha*e + sqrt(1-alpha^2)*sigma*N.
            # alpha=0 ⇒ e = sigma*N(0,1) (identical to the previous per-step white noise).
            ax, ayaw = eff["alpha_xy"], eff["alpha_yaw"]
            bx = math.sqrt(max(0.0, 1.0 - ax * ax))
            byaw = math.sqrt(max(0.0, 1.0 - ayaw * ayaw))
            self._loc_ou_x   = ax * self._loc_ou_x   + bx   * eff["sigma_x"]   * float(np.random.normal())
            self._loc_ou_y   = ax * self._loc_ou_y   + bx   * eff["sigma_y"]   * float(np.random.normal())
            self._loc_ou_yaw = ayaw * self._loc_ou_yaw + byaw * eff["sigma_yaw"] * float(np.random.normal())

            # Random-walk drift — per-second intensity (Brownian √dt scaling so the
            # accumulated std ≈ rate·√t, matching the *_mps / *_rps unit names).
            self._loc_rw_x   += float(np.random.normal(0.0, eff["drift_x"]   * sqrt_dt))
            self._loc_rw_y   += float(np.random.normal(0.0, eff["drift_y"]   * sqrt_dt))
            self._loc_rw_yaw += float(np.random.normal(0.0, eff["drift_yaw"] * sqrt_dt))

            # Episode bias + accumulated drift + correlated measurement error.
            ex   += self._loc_bias_x   + self._loc_rw_x   + self._loc_ou_x
            ey   += self._loc_bias_y   + self._loc_rw_y   + self._loc_ou_y
            eyaw += self._loc_bias_yaw + self._loc_rw_yaw + self._loc_ou_yaw

        # ── Jump axis ── rare relocalization snaps, gated by noise_jump_enabled.
        # A large failure (big_jump) takes precedence over the small snap; both
        # scale with the map-type jump multiplier.
        if n["noise_jump_enabled"]:
            jm = eff["jump_mult"]
            r = np.random.rand()
            bp = n["big_jump_prob"]
            sp = n["jump_prob"]
            if bp > 0.0 and r < bp:
                ex   += float(np.random.uniform(-n["big_jump_xy_m"],  n["big_jump_xy_m"]))  * jm
                ey   += float(np.random.uniform(-n["big_jump_xy_m"],  n["big_jump_xy_m"]))  * jm
                eyaw += float(np.random.uniform(-n["big_jump_yaw_rad"], n["big_jump_yaw_rad"])) * jm
            elif sp > 0.0 and r < bp + sp:
                ex   += float(np.random.uniform(-n["jump_xy_m"],  n["jump_xy_m"]))  * jm
                ey   += float(np.random.uniform(-n["jump_xy_m"],  n["jump_xy_m"]))  * jm
                eyaw += float(np.random.uniform(-n["jump_yaw_rad"], n["jump_yaw_rad"])) * jm

        # ── Yaw-flip axis (STRESS-TEST) ── ±π mirror relocalization in symmetric
        # maps (corridor). OFF unless noise_flip_enabled — never in normal training.
        if (n["noise_flip_enabled"]
                and n["yaw_flip_prob"] > 0.0
                and (self._current_map_type or "") in n["yaw_flip_map_types"]
                and np.random.rand() < n["yaw_flip_prob"]):
            eyaw += math.pi

        eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi
        # Latency buffer: return the pose delay_steps ago.
        self._loc_buf.append((ex, ey, eyaw))
        return self._loc_buf[0]


class ProprioNoiseModel:
    """Proprioception observation noise emulator.

    Owns the per-episode proprio-noise state (speed bias / scale, yaw-rate bias,
    latency buffer). Affects ONLY the policy observation slots state[84:87] — the
    ground-truth caches used for reward / done / collision are untouched.
    """

    def __init__(self, proprio_noise: dict):
        self.cfg = proprio_noise
        self._pp_speed_bias = self._pp_yaw_bias = 0.0
        self._pp_speed_scale = 1.0
        self._pp_buf = deque([(0.0, 0.0, 0.0)], maxlen=1)

    def reset(self, speed0=0.0, yaw_rate0=0.0, steer0=0.0):
        """Reset per-episode proprio-noise state (bias + scale error sampled per
        episode) and SEED the latency buffer with a FULL initial noisy sample of
        the proprio, so the reset observation already reflects the noise model on
        EVERY axis — including steering, which has only a per-step Gaussian (no
        bias/scale). This removes the clean→noisy jump on the first step for all
        three proprio axes (speed / yaw-rate / steering). No-op when disabled."""
        p = self.cfg
        speed0, yaw_rate0, steer0 = float(speed0), float(yaw_rate0), float(steer0)
        if not p["enabled"]:
            self._pp_speed_bias = self._pp_yaw_bias = 0.0
            self._pp_speed_scale = 1.0
            self._pp_buf = deque([(speed0, yaw_rate0, steer0)], maxlen=1)
            return
        self._pp_speed_bias = float(np.random.normal(0.0, p["speed_bias_mps"]))    if p["speed_bias_mps"]   > 0.0 else 0.0
        self._pp_yaw_bias   = float(np.random.normal(0.0, p["yaw_rate_bias_radps"])) if p["yaw_rate_bias_radps"] > 0.0 else 0.0
        self._pp_speed_scale = 1.0 + (float(np.random.normal(0.0, p["speed_scale_sigma"])) if p["speed_scale_sigma"] > 0.0 else 0.0)
        # Seed = one full noise sample (bias + scale + per-step Gaussian) on each
        # axis, so steering (Gaussian-only) is seeded noisy too — not left clean.
        seed_speed = self._pp_speed_scale * speed0 + self._pp_speed_bias
        if p["speed_sigma_mps"] > 0.0:
            seed_speed += float(np.random.normal(0.0, p["speed_sigma_mps"]))
        seed_yaw = yaw_rate0 + self._pp_yaw_bias
        if p["yaw_rate_sigma_radps"] > 0.0:
            seed_yaw += float(np.random.normal(0.0, p["yaw_rate_sigma_radps"]))
        seed_steer = steer0
        if p["steer_sigma_rad"] > 0.0:
            seed_steer += float(np.random.normal(0.0, p["steer_sigma_rad"]))
        seed = (seed_speed, seed_yaw, seed_steer)
        maxlen = max(1, int(p["delay_steps"]) + 1)
        self._pp_buf = deque([seed] * maxlen, maxlen=maxlen)

    def peek(self):
        """Return the currently-emitted (delayed) proprio sample without advancing.

        Mirrors the env reading ``self._pp_buf[0]`` right after reset to seed the
        first observation."""
        return self._pp_buf[0]

    def step(self, speed, yaw_rate, steer):
        """Return the noisy (speed, yaw_rate, steer) proprio OBSERVATION. Affects
        ONLY the policy observation slots state[84:87] — the ground-truth caches
        used for reward / done / collision are untouched. No-op when disabled."""
        p = self.cfg
        if not p["enabled"]:
            return speed, yaw_rate, steer
        es = self._pp_speed_scale * float(speed) + self._pp_speed_bias
        if p["speed_sigma_mps"] > 0.0:
            es += float(np.random.normal(0.0, p["speed_sigma_mps"]))
        ew = float(yaw_rate) + self._pp_yaw_bias
        if p["yaw_rate_sigma_radps"] > 0.0:
            ew += float(np.random.normal(0.0, p["yaw_rate_sigma_radps"]))
        est = float(steer)
        if p["steer_sigma_rad"] > 0.0:
            est += float(np.random.normal(0.0, p["steer_sigma_rad"]))
        # Latency buffer (separate from the goal-obs delay).
        self._pp_buf.append((es, ew, est))
        return self._pp_buf[0]
