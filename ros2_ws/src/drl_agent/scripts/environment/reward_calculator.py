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

    # 6) 시간 페널티 및 합산
    reward = progress + heading - curv_pen - obstacle - step_pen - smooth - wp_smooth

    return (float(reward), terms) if return_terms else float(reward)
