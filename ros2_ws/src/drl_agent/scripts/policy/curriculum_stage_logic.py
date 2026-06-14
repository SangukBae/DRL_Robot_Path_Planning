"""Curriculum stage-advancement decision logic (ROS-free, pure).

Extracted from ``TrainTQCCurriculum._check_stage_advance`` so the promotion-gate
decision is a single pure function that can be unit-tested without ROS / torch /
a running environment. The trainer keeps a thin ``_check_stage_advance`` wrapper
that gathers its current state + config, calls :func:`should_advance_stage`, and
logs the returned reasons.
"""


def per_stage_threshold(lst, stage_idx, default):
    """Per-stage threshold lookup.

    Empty / falsy list → non-blocking ``default``; a shorter list reuses its last
    entry (so a 4-entry list still covers stage 4+).
    """
    if not lst:
        return default
    return float(lst[min(stage_idx, len(lst) - 1)])


def should_advance_stage(
    *,
    enabled: bool,
    curriculum_stage: int,
    num_stages: int,
    global_t: int,
    timesteps_before_training: int,
    stage_start_step: int,
    min_stage_steps: int,
    total_episodes: int,
    stage_start_ep: int,
    min_stage_eps: int,
    pass_sr,
    pass_cr,
    pass_spl,
    pass_clear,
    metrics: dict,
):
    """Decide whether an eval pass should count toward stage promotion.

    Returns ``(advance, reasons)``: ``advance`` is True only when every gate is
    satisfied; ``reasons`` lists the human-readable gate failures (empty when
    advancing, or when a hard precondition such as warmup / min-time blocks
    promotion without a "gate not met" message).

    The numerics mirror the original ``_check_stage_advance`` exactly: the SR /
    CR gates always apply, while the SPL / clearance gates apply only when their
    configured threshold is > 0 (so legacy success-only configs are unchanged).
    """
    if not enabled:
        return False, []
    if curriculum_stage >= num_stages - 1:
        return False, []   # already at the final stage
    # No promotion during warmup
    if global_t <= timesteps_before_training:
        return False, []
    # Minimum time / episode count in the current stage
    if global_t - stage_start_step < min_stage_steps:
        return False, []
    if total_episodes - stage_start_ep < min_stage_eps:
        return False, []

    stage_idx = curriculum_stage

    req_sr    = per_stage_threshold(pass_sr,    stage_idx, 0.0)
    req_cr    = per_stage_threshold(pass_cr,    stage_idx, 1.0)
    req_spl   = per_stage_threshold(pass_spl,   stage_idx, 0.0)   # 0.0 → SPL gate disabled
    req_clear = per_stage_threshold(pass_clear, stage_idx, 0.0)   # 0.0 → clearance gate disabled

    sr    = float(metrics.get("success_rate",   0.0))
    cr    = float(metrics.get("collision_rate", 1.0))
    spl   = float(metrics.get("spl", 0.0) or 0.0)
    clear = float(metrics.get("lidar_clearance_rate", 0.0) or 0.0)

    # Existing gates always apply; the quality gates apply only when configured
    # (>0), keeping success-only behaviour intact for legacy configs.
    reasons = []
    if sr < req_sr:
        reasons.append(f"success {sr*100:.1f}%<{req_sr*100:.0f}%")
    if cr > req_cr:
        reasons.append(f"collision {cr*100:.1f}%>{req_cr*100:.0f}%")
    if req_spl > 0.0 and spl < req_spl:
        reasons.append(f"SPL {spl:.3f}<{req_spl:.2f}")
    if req_clear > 0.0 and clear < req_clear:
        reasons.append(f"clearance {clear:.3f}<{req_clear:.2f}")

    if reasons:
        return False, reasons
    return True, []
