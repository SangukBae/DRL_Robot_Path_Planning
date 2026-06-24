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
    }
    if not humans:
        return terms

    rvx = robot_speed * math.cos(robot_yaw)
    rvy = robot_speed * math.sin(robot_yaw)
    best_total = -1.0
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

    # 6) 시간 페널티 및 합산
    reward = (progress + heading
              - curv_pen - obstacle - step_pen - smooth - wp_smooth - human_pen)

    return (float(reward), terms) if return_terms else float(reward)
