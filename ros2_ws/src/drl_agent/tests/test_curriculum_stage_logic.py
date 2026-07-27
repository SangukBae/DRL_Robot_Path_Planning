"""ROS-free unit tests for curriculum_stage_logic.should_advance_stage."""

import drl_agent.training.curriculum.stage_logic as csl


def _base(**over):
    kw = dict(
        enabled=True,
        curriculum_stage=0,
        num_stages=5,
        global_t=100_000,
        timesteps_before_training=25_000,
        stage_start_step=25_000,
        min_stage_steps=10_000,
        total_episodes=200,
        stage_start_ep=0,
        min_stage_eps=50,
        pass_sr=[0.8],
        pass_cr=[0.1],
        pass_spl=[0.0],
        pass_clear=[0.0],
        metrics={"success_rate": 0.9, "collision_rate": 0.05},
    )
    kw.update(over)
    return kw


def test_per_stage_threshold_helper():
    assert csl.per_stage_threshold([], 3, 0.5) == 0.5          # empty → default
    assert csl.per_stage_threshold([0.1, 0.2], 0, 0.0) == 0.1  # indexed
    assert csl.per_stage_threshold([0.1, 0.2], 9, 0.0) == 0.2  # clamp to last


def test_advance_when_all_gates_met():
    advance, reasons = csl.should_advance_stage(**_base())
    assert advance is True
    assert reasons == []


def test_blocked_when_disabled():
    advance, reasons = csl.should_advance_stage(**_base(enabled=False))
    assert advance is False and reasons == []


def test_blocked_at_final_stage():
    advance, _ = csl.should_advance_stage(**_base(curriculum_stage=4, num_stages=5))
    assert advance is False


def test_blocked_during_warmup():
    advance, _ = csl.should_advance_stage(**_base(global_t=20_000))
    assert advance is False


def test_blocked_by_min_stage_steps():
    advance, _ = csl.should_advance_stage(
        **_base(stage_start_step=95_000)  # only 5k steps in stage < 10k min
    )
    assert advance is False


def test_blocked_by_min_stage_eps():
    advance, _ = csl.should_advance_stage(
        **_base(total_episodes=210, stage_start_ep=200)  # 10 eps < 50 min
    )
    assert advance is False


def test_gate_failure_reasons_reported():
    advance, reasons = csl.should_advance_stage(
        **_base(metrics={"success_rate": 0.5, "collision_rate": 0.3})
    )
    assert advance is False
    assert any("success" in r for r in reasons)
    assert any("collision" in r for r in reasons)


def test_optional_gates_only_apply_when_configured():
    # SPL gate disabled (0.0) → low spl does not block.
    advance, reasons = csl.should_advance_stage(
        **_base(pass_spl=[0.0], metrics={"success_rate": 0.9, "collision_rate": 0.0, "spl": 0.01})
    )
    assert advance is True and reasons == []
    # SPL gate enabled and unmet → blocks with reason.
    advance, reasons = csl.should_advance_stage(
        **_base(pass_spl=[0.6], metrics={"success_rate": 0.9, "collision_rate": 0.0, "spl": 0.01})
    )
    assert advance is False
    assert any("SPL" in r for r in reasons)


# ── Human-safety HARD gates (PSC / H-Coll) ──────────────────────────────────

def test_psc_gate_blocks_when_below_threshold():
    advance, reasons = csl.should_advance_stage(
        **_base(pass_psc=[0.99],
                metrics={"success_rate": 0.95, "collision_rate": 0.0, "psc": 0.90})
    )
    assert advance is False
    assert any("PSC" in r for r in reasons)


def test_psc_gate_passes_when_met():
    advance, reasons = csl.should_advance_stage(
        **_base(pass_psc=[0.99],
                metrics={"success_rate": 0.95, "collision_rate": 0.0, "psc": 0.995})
    )
    assert advance is True and reasons == []


def test_psc_gate_skips_when_value_missing():
    # psc absent / None (labels off) must NOT block, even with a strict gate.
    advance, reasons = csl.should_advance_stage(
        **_base(pass_psc=[0.99],
                metrics={"success_rate": 0.95, "collision_rate": 0.0, "psc": None})
    )
    assert advance is True and reasons == []
    advance2, _ = csl.should_advance_stage(
        **_base(pass_psc=[0.99],
                metrics={"success_rate": 0.95, "collision_rate": 0.0})  # no psc key
    )
    assert advance2 is True


def test_h_coll_gate_blocks_on_any_human_collision():
    advance, reasons = csl.should_advance_stage(
        **_base(pass_h_coll=[0.0],
                metrics={"success_rate": 0.95, "collision_rate": 0.0, "h_coll_rate": 0.05})
    )
    assert advance is False
    assert any("H-Coll" in r for r in reasons)


def test_h_coll_gate_passes_at_zero_and_skips_when_missing():
    advance, reasons = csl.should_advance_stage(
        **_base(pass_h_coll=[0.0],
                metrics={"success_rate": 0.95, "collision_rate": 0.0, "h_coll_rate": 0.0})
    )
    assert advance is True and reasons == []
    # Missing value → gate self-disables (no block).
    advance2, _ = csl.should_advance_stage(
        **_base(pass_h_coll=[0.0],
                metrics={"success_rate": 0.95, "collision_rate": 0.0, "h_coll_rate": None})
    )
    assert advance2 is True


# ── Per-map FAIL-FAST gates ─────────────────────────────────────────────────

def _per_map_metrics(per_map, eval_map_applied=True, **extra):
    m = {"success_rate": 0.95, "collision_rate": 0.0,
         "eval_map_applied": eval_map_applied, "per_map": per_map}
    m.update(extra)
    return m


def test_per_map_gate_blocks_single_collapsing_map_despite_good_average():
    # Global average is great, but ONE map (clutter) collapsed → must block.
    per_map = {
        "corridor": {"success_rate": 0.98, "collision_rate": 0.0},
        "clutter":  {"success_rate": 0.40, "collision_rate": 0.30},
    }
    advance, reasons = csl.should_advance_stage(
        **_base(pass_per_map_sr=[0.70], pass_per_map_cr=[0.20],
                metrics=_per_map_metrics(per_map))
    )
    assert advance is False
    assert any("map[clutter]" in r and "success" in r for r in reasons)
    assert any("map[clutter]" in r and "collision" in r for r in reasons)
    # The healthy map is not blamed.
    assert not any("map[corridor]" in r for r in reasons)


def test_per_map_gate_passes_when_every_map_clears_floor():
    per_map = {
        "corridor": {"success_rate": 0.85, "collision_rate": 0.05},
        "clutter":  {"success_rate": 0.75, "collision_rate": 0.10},
    }
    advance, reasons = csl.should_advance_stage(
        **_base(pass_per_map_sr=[0.70], pass_per_map_cr=[0.20],
                metrics=_per_map_metrics(per_map))
    )
    assert advance is True and reasons == []


def test_per_map_gate_blocks_on_fallback_eval():
    # eval_map_applied false → per-map breakdown is the training mix, untrusted.
    per_map = {"corridor": {"success_rate": 0.99, "collision_rate": 0.0}}
    advance, reasons = csl.should_advance_stage(
        **_base(pass_per_map_sr=[0.70],
                metrics=_per_map_metrics(per_map, eval_map_applied=False))
    )
    assert advance is False
    assert any("eval_map_applied=false" in r for r in reasons)


def test_per_map_gate_inactive_when_unconfigured():
    # No per-map thresholds → fallback eval / collapsing map do not block.
    per_map = {"clutter": {"success_rate": 0.10, "collision_rate": 0.50}}
    advance, _ = csl.should_advance_stage(
        **_base(metrics=_per_map_metrics(per_map, eval_map_applied=False))
    )
    assert advance is True
