"""ROS-free unit tests for curriculum_stage_logic.should_advance_stage."""

import curriculum_stage_logic as csl


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
