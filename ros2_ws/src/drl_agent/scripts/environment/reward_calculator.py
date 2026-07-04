"""Reward shaping for the DRL navigation environment.

Extracted verbatim from ``Environment.get_reward`` (environment.py) so the
reward function lives in one ROS-free, unit-testable place. The numerics are
unchanged: ``environment.py`` (and ``environment_360.py`` if it adopts this)
delegate to :func:`compute_reward`, and ``Environment.get_reward`` remains a
thin static wrapper for backward compatibility.

The reward decomposes into:
  progress + heading - curvature - obstacle - step - smoothing - wp_smoothing
with terminal rewards (+goal / -collision) short-circuiting everything else.
"""

import math

import numpy as np


def compute_human_risk_penalty(
    robot_x, robot_y, robot_yaw, robot_speed,
    humans,
    *,
    w_ps=0.4, d_ps=1.0,
    w_app=0.25, d_app=2.0, v_ref=1.0,
    w_ttc=0.3, ttc_safe=2.0,
    eps=1e-6,
):
    """Human-ONLY dynamic-collision-risk penalty (3 additive terms).

    Separate from the static obstacle-proximity penalty: it is computed from the
    privileged ground-truth robot + pedestrian kinematics (this is training-time
    shaping, not an observation), so it can reward direction-/closing-aware
    caution that a single nearest-distance reading cannot express.

    Parameters
    ----------
    robot_x, robot_y, robot_yaw : robot GT pose (m, m, rad).
    robot_speed                 : robot GT signed forward speed (m/s).
    humans                      : iterable of dicts with ``x``, ``y``, ``yaw``,
                                  ``v`` (GT pedestrian pose + speed). Empty / None
                                  → all terms 0 (so human-free stages cost nothing).

    Terms (each clamped, dimensionless, >= 0):
      * ``human_personal_space_penalty`` = w_ps * max(0, 1 - d/d_ps)^2, only
        INSIDE the personal-space radius d_ps (smoothly grows as the robot
        intrudes). Pure proximity — applies even to a standing person.
      * ``human_approach_rate_penalty``  = w_app * clamp(closing/v_ref, 0, 1),
        only while CLOSING (closing speed > 0) AND within d_app. Penalises
        actively driving at a person; 0 when not closing.
      * ``human_ttc_penalty``            = w_ttc * max(0, 1 - ttc/ttc_safe)^2,
        only when closing speed > 0 AND time-to-contact ttc < ttc_safe. Sharpens
        as a frontal/closing approach gets imminent.

    ``closing`` is the range-rate toward the human, derived from the RELATIVE
    velocity (human GT velocity minus robot GT velocity) projected on the
    robot→human unit vector; it is 0 when the pair is separating, so the
    approach/TTC terms vanish unless the gap is actually shrinking.

    Aggregation = the SINGLE most-dangerous human (max summed risk). This is the
    most numerically stable choice: the penalty cannot grow with crowd size (so
    the reward scale stays comparable across stages with different human counts),
    overlapping nearby people are not double-counted, and the three reported terms
    are self-consistent (all attributed to the same person). Returns a dict with
    the three keys above.
    """
    terms = {
        "human_personal_space_penalty": 0.0,
        "human_approach_rate_penalty": 0.0,
        "human_ttc_penalty": 0.0,
        # Raw closing-risk signals used for yield/stop GATING (not penalties).
        # Tracked over ALL humans (most-imminent), independent of the penalty
        # aggregation above, so a fast closer is not masked by a closer-but-
        # standing person. Defaults mean "no closing hazard".
        "human_min_ttc_s": float("inf"),     # min time-to-contact over closers
        "human_approach_norm": 0.0,          # max clamp(closing/v_ref, 0, 1)
        "human_min_dist_m": float("inf"),    # nearest human distance
    }
    if not humans:
        return terms

    rvx = robot_speed * math.cos(robot_yaw)
    rvy = robot_speed * math.sin(robot_yaw)
    best_total = -1.0
    min_ttc = float("inf")
    max_approach = 0.0
    min_dist = float("inf")
    for h in humans:
        dx = float(h["x"]) - robot_x
        dy = float(h["y"]) - robot_y
        d = math.hypot(dx, dy)
        if d < eps:
            d = eps
        hv = float(h.get("v", 0.0))
        hyaw = float(h.get("yaw", 0.0))
        # Relative velocity of the human w.r.t. the robot, then the closing speed
        # = -d(distance)/dt = -(p · v_rel)/|p|  (positive only while approaching).
        rel_vx = hv * math.cos(hyaw) - rvx
        rel_vy = hv * math.sin(hyaw) - rvy
        closing = -(dx * rel_vx + dy * rel_vy) / d
        if closing < 0.0:
            closing = 0.0

        # Raw gating signals (over all humans), independent of penalty selection.
        if d < min_dist:
            min_dist = d
        if closing > eps:
            _ttc = d / closing
            if _ttc < min_ttc:
                min_ttc = _ttc
            _appr = min(1.0, closing / max(v_ref, eps))
            if _appr > max_approach:
                max_approach = _appr

        pen_ps = (w_ps * (1.0 - d / max(d_ps, eps)) ** 2) if d < d_ps else 0.0
        # Clamp the closing ratio to [0, 1] so a fast approach cannot blow the
        # penalty past w_app (keeps the reward bounded / stable).
        pen_app = (
            w_app * min(1.0, closing / max(v_ref, eps))
            if (closing > 0.0 and d < d_app) else 0.0
        )
        pen_ttc = 0.0
        if closing > eps:
            ttc = d / closing
            if ttc < ttc_safe:
                pen_ttc = w_ttc * (1.0 - ttc / max(ttc_safe, eps)) ** 2

        total = pen_ps + pen_app + pen_ttc
        if total > best_total:
            best_total = total
            terms["human_personal_space_penalty"] = float(pen_ps)
            terms["human_approach_rate_penalty"] = float(pen_app)
            terms["human_ttc_penalty"] = float(pen_ttc)
    terms["human_min_ttc_s"] = float(min_ttc)
    terms["human_approach_norm"] = float(max_approach)
    terms["human_min_dist_m"] = float(min_dist)
    return terms


def compute_reward(
    target, collision,
    v, w,                                  # m/s, rad/s (Pure Pursuit 출력)
    prev_goal_dist, curr_goal_dist,
    theta_err=None,
    rect_proximity=None,
    zmins=None, zthrs=None,
    min_laser=None,
    v_max=1.5, w_max=6.0,

    # ---- 튜닝 파라미터 ----
    k_p=0.5,                 # 진행 보상 게인 (누적 양수 보상 과대 억제)
    progress_clip=0.25,

    # 곡률 페널티 (waypoint RL에서 Pure Pursuit가 처리하므로 기본 0)
    lambda_k=0.0,

    # 장애물 근접 (존 기반)
    z_weights=(0.6, 0.85, 1.0, 0.85, 0.6),
    safety_margin=1.5,
    w_obs=1.5,

    # 장애물 근접 (폴백)
    d_safe_base=0.55,
    d_safe_speed=0.30,

    # 헤딩/시간/스무딩
    k_h=0.03,
    step_pen=0.05,
    k_smooth=0.0,
    prev_v=None, prev_w=None,

    # 웨이포인트 스무딩 (급격한 방향 전환 억제)
    waypoint_theta=0.0,
    prev_waypoint_theta=0.0,
    k_smooth_wp=0.05,

    # 사람 전용 동적 위험 페널티 (environment.py에서 GT 사람/로봇 상태로 미리 계산).
    # None/빈 dict → 세 항 모두 0 (사람이 없는 stage·timestep에서는 영향 없음).
    human_risk_terms=None,

    # ---- Stop/Yield shaping (opt-in; default OFF → byte-identical reward) ----
    # Lets the policy be REWARDED for slowing/stopping ONLY while a human
    # collision is imminent, and lightly PENALISED for idling when it is safe to
    # move. Gated by the raw closing-risk signals in `human_risk_terms`
    # (`human_min_ttc_s`, `human_approach_norm`), so it is exactly 0 in
    # human-free stages / timesteps. All terms are bounded.
    yield_enabled=False,
    yield_risk_gate_ttc_s=2.0,      # risk-active when min TTC < this
    yield_risk_gate_approach=0.15,  # OR when normalised closing rate > this
    yield_w_bonus=0.05,             # gain on the slow-down bonus
    yield_max_bonus=0.1,            # hard cap on the slow-down bonus
    yield_idle_w=0.03,              # gain on the idle (loitering) penalty
    yield_idle_speed_mps=0.1,       # "idle" = |speed| below this
    yield_step_relief_scale=1.0,    # multiply step_pen while risk-active (1=none, 0=full relief)
    cruise_speed_mps=None,          # speed normaliser for the slow factor (defaults to v_max)

    # ---- Anti-freeze penalty (opt-in; default OFF → byte-identical reward) ----
    # Penalise ONLY a SUSTAINED, no-progress, low-speed hold while a human hazard
    # is active (risk_active). Short yields are untouched: the penalty starts only
    # after `antifreeze_min_streak` consecutive freeze steps. The streak counter is
    # owned by the caller (per episode) — passed in via freeze_streak_in and
    # returned in terms["freeze_streak"].
    antifreeze_enabled=False,
    antifreeze_speed_mps=0.12,      # "frozen" = |speed| below this [m/s]
    antifreeze_min_streak=12,       # consecutive freeze steps before penalty starts
    antifreeze_w=0.02,              # per-step penalty once the streak is exceeded
    freeze_streak_in=0,             # consecutive freeze steps BEFORE this step (caller-owned)

    # ---- Explicit 3D-action yield shaping (opt-in; default OFF → unchanged) ----
    # Keyed on the COMMANDED yield (action[2]), not on inferred speed. Penalises
    # yielding while it is SAFE (risk_low) and escalates with yield duration. No
    # positive "valid yield" bonus — a legitimate (risk_active) yield is handled
    # ONLY by the step-penalty relief above, so stopping is never itself rewarded.
    yield_action=False,             # did the policy command yield this step
    yield_w_bad=0.0,                # penalty for yielding while SAFE (risk_low)
    yield_streak_in=0,              # consecutive commanded-yield steps BEFORE this (caller-owned)
    yield_streak_grace=8,           # yields shorter than this are not escalated
    yield_streak_grow_w=0.0,        # per-step growth once the grace streak is exceeded

    # ---- Generic low-progress (stall) penalty (opt-in; NOT human-gated) --------
    # Fires when it is SAFE (no human hazard AND low static proximity) yet the
    # robot makes no progress at low speed. Covers human-free stalls that the
    # human-gated anti-freeze above cannot. Bounded, small, constant per step.
    stall_enabled=False,
    stall_w=0.02,                   # per-step penalty while safely stalled
    stall_progress_eps=0.0,         # "no progress" = progress <= this
    stall_speed_mps=0.15,           # "low speed" = |speed| below this [m/s]
    stall_static_risk_thr=0.2,      # "low static risk" = rect_proximity below this
    return_terms=False,
):
    terms = {
        "delta_d": 0.0,
        "progress": 0.0,
        "heading": 0.0,
        "curv_pen": 0.0,
        "obstacle": 0.0,
        "step_pen": 0.0,
        "smooth": 0.0,
        "wp_smooth": 0.0,
        # Human-only dynamic-risk terms (always present so the CSV schema is
        # stable even on terminal steps / human-free stages).
        "human_personal_space_penalty": 0.0,
        "human_approach_rate_penalty": 0.0,
        "human_ttc_penalty": 0.0,
        # Raw closing-risk gating signals (passthrough for logging/diagnostics).
        "human_min_ttc_s": float("inf"),
        "human_approach_norm": 0.0,
        # Stop/yield shaping (0 unless yield_enabled and gated risk fires).
        "yield_bonus": 0.0,
        "idle_pen": 0.0,
        # Anti-freeze (0 unless antifreeze_enabled and a sustained freeze fires).
        "freeze_pen": 0.0,
        "freeze_streak": 0.0,
        # Explicit 3D-action yield shaping (0 unless yield_action fires).
        "bad_yield_pen": 0.0,
        "yield_streak": 0.0,
        # Generic low-progress stall penalty (0 unless stall_enabled + safe stall).
        "stall_pen": 0.0,
        "terminal": 0.0,
    }
    # 터미널
    if target:
        terms["terminal"] = 20.0
        return (20.0, terms) if return_terms else 20.0
    if collision:
        terms["terminal"] = -30.0
        return (-30.0, terms) if return_terms else -30.0

    # 정규화
    v_n = v / max(v_max, 1e-6)
    w_n = w / max(w_max, 1e-6)

    # 1) 진행 보상
    delta_d  = np.clip(prev_goal_dist - curr_goal_dist, -progress_clip, progress_clip)
    progress = k_p * delta_d
    terms["delta_d"] = float(delta_d)
    terms["progress"] = float(progress)

    # 2) 곡률 페널티 (lambda_k=0 → disabled for waypoint RL)
    kappa    = abs(w_n) / (abs(v_n) + 1e-3)
    curv_pen = lambda_k * kappa
    terms["curv_pen"] = float(curv_pen)

    # 2b) 웨이포인트 스무딩 (연속 step 간 waypoint 각도 변화 억제)
    dtheta = abs(waypoint_theta - prev_waypoint_theta)
    if dtheta > math.pi:
        dtheta = 2.0 * math.pi - dtheta
    wp_smooth = k_smooth_wp * dtheta / math.pi
    terms["wp_smooth"] = float(wp_smooth)

    # 3) 장애물 근접 페널티 (직사각형 우선 → 레거시 zone → 글로벌 min 폴백)
    obstacle = 0.0
    if rect_proximity is not None:
        # 충돌/보상 기하 통일: _compute_rect_proximity() 값 직접 사용
        obstacle = w_obs * float(rect_proximity)   # 0 ~ w_obs
    elif zmins is not None and zthrs is not None and len(zmins) == 5 and len(zthrs) == 5:
        # 레거시 zone 경로 (호환성 유지, 현재는 use_zone_collision=false로 미사용)
        deficits = []
        for i in range(5):
            thr_expanded = max(1e-6, safety_margin * float(zthrs[i]))
            zmin = float(zmins[i])
            d = max(0.0, 1.0 - (zmin / thr_expanded))
            deficits.append(d)
        wsum = sum(z_weights)
        weighted = sum(wi * di for wi, di in zip(z_weights, deficits)) / max(wsum, 1e-6)
        obstacle = w_obs * weighted
    else:
        # 폴백: 글로벌 min_laser 기반 (속도 의존 안전거리)
        if min_laser is not None and np.isfinite(min_laser):
            d_safe = d_safe_base + d_safe_speed * abs(v)
            if min_laser < d_safe:
                obstacle = w_obs * (1.0 - min_laser / max(d_safe, 1e-6))
    terms["obstacle"] = float(obstacle)

    # 4) 헤딩 보너스 — goal에 가까워지는 step에서만 부여 (reward hacking 방지)
    heading = (k_h * max(0.0, math.cos(theta_err))
               if (theta_err is not None and delta_d > 0.0)
               else 0.0)
    terms["heading"] = float(heading)

    # 5) 스무딩(선택)
    smooth = 0.0
    if k_smooth > 0.0 and prev_v is not None and prev_w is not None:
        dv = abs(v - prev_v) / max(v_max, 1e-6)
        dw = abs(w - prev_w) / max(w_max, 1e-6)
        smooth = k_smooth * 0.5 * (dv + dw)
    terms["smooth"] = float(smooth)
    terms["step_pen"] = float(step_pen)

    # 5b) 사람 전용 동적 위험 페널티 (정적 장애물 페널티와 별개로 합산).
    # 미리 계산된 값만 더하므로 사람이 없으면 0이고 기존 신호를 깨지 않는다.
    human_pen = 0.0
    if human_risk_terms:
        for _k in ("human_personal_space_penalty",
                   "human_approach_rate_penalty",
                   "human_ttc_penalty"):
            _v = float(human_risk_terms.get(_k, 0.0) or 0.0)
            terms[_k] = _v
            human_pen += _v
        # Passthrough raw gating signals for logging.
        terms["human_min_ttc_s"] = float(
            human_risk_terms.get("human_min_ttc_s", float("inf")))
        terms["human_approach_norm"] = float(
            human_risk_terms.get("human_approach_norm", 0.0) or 0.0)

    # 5c) Stop/yield shaping (opt-in). Reward slowing/stopping only while a
    # human collision is imminent; lightly penalise idling when it is safe to
    # move. Both bounded; both exactly 0 when yield_enabled is False (the final
    # sum is then byte-identical to the original reward).
    #
    # HARD-GATED on "a human is actually present this step" (finite
    # human_min_dist_m). This keeps human-FREE stages/timesteps (e.g. curriculum
    # stages 0-2) at exactly 0 yield shaping — INCLUDING the idle penalty, which
    # would otherwise punish low speed even with no people around. There the only
    # pressure to keep moving stays the (un-relieved) step penalty, exactly as
    # before. The bonus/relief were already human-gated via risk_active; this
    # extends the same gate to the idle penalty.
    yield_bonus = 0.0
    idle_pen = 0.0
    freeze_pen = 0.0
    freeze_streak = 0
    speed_abs = abs(v)
    humans_present = bool(
        human_risk_terms
        and math.isfinite(float(human_risk_terms.get("human_min_dist_m", float("inf"))))
    )
    # Closing-risk gate, defined ONCE and shared by yield + anti-freeze so both
    # agree on what "a human hazard is active" means. False when no human present.
    _ttc = float(terms["human_min_ttc_s"])
    _appr = float(terms["human_approach_norm"])
    risk_active = humans_present and (
        (_ttc < yield_risk_gate_ttc_s) or (_appr > yield_risk_gate_approach))
    if yield_enabled and humans_present:
        if risk_active:
            # (A) relieve the step penalty so waiting out a hazard is not punished
            step_pen = step_pen * yield_step_relief_scale
            # (B) reward slowing/stopping, scaled by how slow AND how risky
            _cruise = (cruise_speed_mps if (cruise_speed_mps and cruise_speed_mps > 0.0)
                       else max(v_max, 1e-6))
            slow_factor = max(0.0, min(1.0, 1.0 - speed_abs / _cruise))
            ttc_risk = (max(0.0, min(1.0, 1.0 - _ttc / max(yield_risk_gate_ttc_s, 1e-6)))
                        if math.isfinite(_ttc) else 0.0)
            risk_level = max(ttc_risk, min(1.0, _appr))
            yield_bonus = min(yield_max_bonus, yield_w_bonus * slow_factor * risk_level)
        elif speed_abs < yield_idle_speed_mps:
            # (C) discourage idling when it is safe to move (no imminent hazard)
            idle_pen = yield_idle_w * (1.0 - speed_abs / max(yield_idle_speed_mps, 1e-6))
    # (D) Anti-freeze: penalise ONLY a SUSTAINED, no-progress, low-speed hold while
    # a human hazard is active. A brief yield survives (the streak must first
    # exceed antifreeze_min_streak); normal driving / avoidance / human-free steps
    # never trigger it (risk_active is False there → streak resets to 0).
    if (antifreeze_enabled and risk_active
            and speed_abs < antifreeze_speed_mps and progress <= 0.0):
        freeze_streak = int(freeze_streak_in) + 1
        if freeze_streak > antifreeze_min_streak:
            freeze_pen = antifreeze_w
    terms["step_pen"] = float(step_pen)
    terms["yield_bonus"] = float(yield_bonus)
    terms["idle_pen"] = float(idle_pen)
    terms["freeze_pen"] = float(freeze_pen)
    terms["freeze_streak"] = float(freeze_streak)

    # 5d) Explicit 3D-action yield shaping (keyed on COMMANDED yield, not speed).
    # "risk_low" = no human hazard active AND low static proximity, so it does not
    # depend on humans being present. A yield here is INAPPROPRIATE (safe to move)
    # → penalise; a yield while risk_active is left to the step-relief above (no
    # extra bonus). Prolonged yields escalate once past the grace streak.
    static_risk = float(rect_proximity) if rect_proximity is not None else 0.0
    risk_low = (not risk_active) and (static_risk < stall_static_risk_thr)
    bad_yield_pen = 0.0
    yield_streak = 0
    if yield_action:
        yield_streak = int(yield_streak_in) + 1
        if risk_active:
            # Legitimate yield: only a duration cost once it drags on.
            if yield_streak > yield_streak_grace:
                bad_yield_pen += yield_streak_grow_w * (yield_streak - yield_streak_grace)
        else:
            # Yielding while safe → strong penalty + duration growth.
            bad_yield_pen += yield_w_bad + yield_streak_grow_w * yield_streak
    terms["bad_yield_pen"] = float(bad_yield_pen)
    terms["yield_streak"] = float(yield_streak)

    # 5e) Generic low-progress (stall) penalty — NOT human-gated. Safe + no
    # progress + low speed → a small constant time cost. This is the mid-episode
    # anti-freeze pressure for the whole state space (incl. human-free stages),
    # complementing the terminal timeout penalty rather than replacing it.
    stall_pen = 0.0
    if (stall_enabled and risk_low
            and progress <= stall_progress_eps
            and speed_abs < stall_speed_mps):
        stall_pen = stall_w
    terms["stall_pen"] = float(stall_pen)

    # 6) 시간 페널티 및 합산
    reward = (progress + heading + yield_bonus
              - curv_pen - obstacle - step_pen - smooth - wp_smooth - human_pen - idle_pen
              - freeze_pen - bad_yield_pen - stall_pen)

    return (float(reward), terms) if return_terms else float(reward)
