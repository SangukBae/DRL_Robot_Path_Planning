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
    pass_psc=None,
    pass_h_coll=None,
    pass_per_map_sr=None,
    pass_per_map_cr=None,
):
    """Decide whether an eval pass should count toward stage promotion.

    Returns ``(advance, reasons)``: ``advance`` is True only when every gate is
    satisfied; ``reasons`` lists the human-readable gate failures (empty when
    advancing, or when a hard precondition such as warmup / min-time blocks
    promotion without a "gate not met" message).

    The numerics mirror the original ``_check_stage_advance`` exactly: the SR /
    CR gates always apply, while the SPL / clearance gates apply only when their
    configured threshold is > 0 (so legacy success-only configs are unchanged).

    Additional gates (all default-disabled, so omitting them reproduces the
    legacy behaviour):
      * ``pass_psc``        — min Personal-Space-Compliance fraction. Active only
                              when its per-stage threshold is > 0 AND a PSC value
                              exists (``metrics["psc"]`` is not None); a missing
                              value (labels off) never blocks.
      * ``pass_h_coll``     — max human-collision episode rate. Active only when
                              its per-stage threshold is < 1.0 AND a value exists
                              (``metrics["h_coll_rate"]`` is not None).
      * ``pass_per_map_sr`` /
        ``pass_per_map_cr`` — per-map_type success floor / collision cap. EVERY
                              map in ``metrics["per_map"]`` must pass. Active only
                              when configured; a fallback eval (the per-map
                              breakdown is the training mix, ``eval_map_applied``
                              False) BLOCKS rather than trusting bad data.
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

    # ── Human-safety HARD gates (only when configured AND the metric exists) ──
    # PSC / H-Coll are label-derived: when the env emits no labels they are None,
    # and a None value must NEVER block promotion (a label-free run keeps the
    # legacy success/collision behaviour).
    req_psc = per_stage_threshold(pass_psc, stage_idx, 0.0)        # 0.0 → disabled
    if req_psc > 0.0:
        psc_val = metrics.get("psc")
        if psc_val is not None and float(psc_val) < req_psc:
            reasons.append(f"PSC {float(psc_val):.3f}<{req_psc:.2f}")

    req_hc = per_stage_threshold(pass_h_coll, stage_idx, 1.0)      # 1.0 → disabled
    if req_hc < 1.0:
        hc_val = metrics.get("h_coll_rate")
        if hc_val is not None and float(hc_val) > req_hc:
            reasons.append(f"H-Coll {float(hc_val)*100:.1f}%>{req_hc*100:.1f}%")

    # ── Per-map FAIL-FAST gates: EVERY eval map_type must clear its floor/cap ──
    req_pm_sr = per_stage_threshold(pass_per_map_sr, stage_idx, 0.0)   # 0.0 → off
    req_pm_cr = per_stage_threshold(pass_per_map_cr, stage_idx, 1.0)   # 1.0 → off
    per_map_active = (req_pm_sr > 0.0 or req_pm_cr < 1.0)
    if per_map_active:
        # A fallback eval (per-map breakdown is the training mix, not
        # eval_map_types) cannot be trusted for a per-map gate → block.
        if not bool(metrics.get("eval_map_applied", True)):
            reasons.append(
                "per-map gate blocked: eval ran on the training map mix "
                "(eval_map_applied=false), per-map breakdown not trustworthy")
        else:
            per_map = metrics.get("per_map") or {}
            for mt in sorted(per_map):
                d = per_map[mt] or {}
                # NB: avoid `x or default` — a legitimate 0.0 collision rate (the
                # SAFEST case) must not fall back to the disabled default.
                _sr = d.get("success_rate")
                _cr = d.get("collision_rate")
                m_sr = float(_sr) if _sr is not None else 0.0
                m_cr = float(_cr) if _cr is not None else 1.0
                if req_pm_sr > 0.0 and m_sr < req_pm_sr:
                    reasons.append(
                        f"map[{mt}] success {m_sr*100:.1f}%<{req_pm_sr*100:.0f}%")
                if req_pm_cr < 1.0 and m_cr > req_pm_cr:
                    reasons.append(
                        f"map[{mt}] collision {m_cr*100:.1f}%>{req_pm_cr*100:.0f}%")

    if reasons:
        return False, reasons
    return True, []
