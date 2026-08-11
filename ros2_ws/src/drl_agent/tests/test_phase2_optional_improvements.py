"""Regression tests for the two independently switchable phase2 upgrades
(spatiotemporal_lidar, counterfactual_multi_horizon_risk) and the CF
stability/generalization improvements built on top of the latter:
actor-penalty warm-up/ramp, the executed-action multi-horizon target,
weighted_mean horizon aggregation, and OOD/calibration logging."""

import os

import numpy as np
import pytest
import torch

from drl_agent.env.observation.aux_prediction_labels import (
    counterfactual_risk_wire, strip_counterfactual_risk_wire,
    executed_action_risk_wire, strip_executed_action_risk_wire,
    validate_executed_action_risk_target,
    read_and_validate_counterfactual_risk_targets)
from drl_agent.rl.algorithms.tqc.agent import Agent, _frozen_params
from drl_agent.rl.networks.action_risk_head import (
    CounterfactualRiskConfig, CounterfactualMultiHorizonRiskHead)
from drl_agent.rl.networks.aux_temporal import ScanTemporalEncoder
from drl_agent.rl.replay.buffer import LAP


OBS, AGENT, HISTORY = 80, 7, 4
STATE_DIM = OBS + AGENT + (HISTORY - 1) * OBS
ACTION_DIM = 3
CANDIDATES = [
    [-0.5, -1.0, -1.0], [-0.5, 0.0, -1.0], [-0.5, 1.0, -1.0],
    [0.5, -1.0, -1.0], [0.5, 0.0, -1.0], [0.5, 1.0, -1.0],
    [0.0, 0.0, 1.0],
]


def test_counterfactual_wire_roundtrip_leaves_following_tail_untouched():
    target = np.arange(21, dtype=np.float32).reshape(7, 3) / 21.0
    wire = counterfactual_risk_wire(target, [0.5, 1.0, 2.0]) + [-999.0, 0.2, 0.8]
    restored, meta, remainder = strip_counterfactual_risk_wire(wire)
    assert np.allclose(restored, target)
    assert meta == {"num_candidates": 7, "num_horizons": 3,
                    "horizons_sec": [0.5, 1.0, 2.0]}
    assert np.allclose(remainder, [-999.0, 0.2, 0.8])


def test_spatiotemporal_encoder_retains_angular_order_and_range_rate():
    torch.manual_seed(3)
    enc = ScanTemporalEncoder(
        HISTORY, OBS, feature_dim=16, encoder_type="spatiotemporal",
        angular_tokens=8, use_range_rate=True).eval()
    scan = torch.zeros(1, HISTORY, OBS)
    scan[:, :, 5:10] = torch.tensor([4.0, 3.0, 2.0, 1.0]).view(1, 4, 1)
    mirrored = torch.flip(scan, dims=[-1])
    with torch.no_grad():
        left = enc(scan)
        right = enc(mirrored)
    assert left.shape == (1, 16)
    assert not torch.allclose(left, right), "left/right angular position was pooled away"


def test_counterfactual_head_has_action_gradient_while_weights_are_frozen():
    cfg = CounterfactualRiskConfig({
        "enabled": True, "horizons_sec": [0.5, 1.0],
        "candidate_actions": CANDIDATES})
    head = CounterfactualMultiHorizonRiskHead(12, ACTION_DIM, cfg)
    z = torch.randn(5, 12)
    action = torch.randn(5, ACTION_DIM, requires_grad=True)
    with _frozen_params(head):
        loss = head(z, action).max(dim=1).values.mean()
    loss.backward()
    assert action.grad is not None and action.grad.abs().sum() > 0
    assert all(p.grad is None for p in head.parameters())


# --------------------------------------------------------------------------- #
#  Executed-action multi-horizon target: wire round-trip
# --------------------------------------------------------------------------- #
def test_executed_action_wire_roundtrip_leaves_following_tail_untouched():
    target = np.array([0.1, 0.4, 0.7, 0.9], dtype=np.float32)
    wire = executed_action_risk_wire(target, [0.5, 1.0, 1.5, 2.0]) + [-999.0, 0.2, 0.8]
    restored, remainder = strip_executed_action_risk_wire(wire)
    assert np.allclose(restored, target)
    assert np.allclose(remainder, [-999.0, 0.2, 0.8])


def test_executed_action_wire_absent_passes_through_unchanged():
    tail = [-999.0, 0.3, 0.6]
    restored, remainder = strip_executed_action_risk_wire(tail)
    assert restored is None
    assert np.allclose(remainder, tail)


# --------------------------------------------------------------------------- #
#  validate_executed_action_risk_target: the low-level fail-fast used by the
#  shared trainer-side counterfactual-contract helper.
# --------------------------------------------------------------------------- #
def test_validate_executed_action_target_length_mismatch_raises():
    with pytest.raises(RuntimeError):
        validate_executed_action_risk_target([0.1, 0.2, 0.3], expected_num_horizons=4)


def test_validate_executed_action_target_nan_raises():
    with pytest.raises(RuntimeError):
        validate_executed_action_risk_target(
            [0.1, np.nan, 0.3, 0.4], expected_num_horizons=4)


def test_validate_executed_action_target_inf_raises():
    with pytest.raises(RuntimeError):
        validate_executed_action_risk_target(
            [0.1, np.inf, 0.3, 0.4], expected_num_horizons=4)


def test_validate_executed_action_target_valid_passes_through():
    out = validate_executed_action_risk_target(
        [0.1, 0.2, 0.3, 0.4], expected_num_horizons=4)
    assert out.shape == (4,)
    assert out.dtype == np.float32
    assert np.allclose(out, [0.1, 0.2, 0.3, 0.4])


def _contract_trainer(*, enabled=True):
    cfg = type("Cfg", (), {
        "num_candidates": 7,
        "num_horizons": 4,
        "horizons_sec": [0.5, 1.0, 1.5, 2.0],
    })()
    agent = type("AgentStub", (), {
        "counterfactual_risk_enabled": enabled,
        "counterfactual_risk_cfg": cfg,
    })()
    return type("TrainerStub", (), {
        "rl_agent": agent,
        "last_counterfactual_risk_target": np.zeros((7, 4), dtype=np.float32),
        "last_counterfactual_executed_target": np.zeros(4, dtype=np.float32),
        "last_counterfactual_risk_meta": {
            "num_candidates": 7,
            "num_horizons": 4,
            "horizons_sec": [0.5, 1.0, 1.5, 2.0],
        },
    })()


def test_shared_counterfactual_contract_helper_returns_validated_targets():
    trainer = _contract_trainer()
    fixed, executed = read_and_validate_counterfactual_risk_targets(trainer)
    assert fixed is trainer.last_counterfactual_risk_target
    assert executed.shape == (4,)
    assert executed.dtype == np.float32


def test_shared_counterfactual_contract_helper_rejects_metadata_drift():
    trainer = _contract_trainer()
    trainer.last_counterfactual_risk_meta["num_horizons"] = 3
    with pytest.raises(RuntimeError, match="contract mismatch"):
        read_and_validate_counterfactual_risk_targets(trainer)


# --------------------------------------------------------------------------- #
#  Replay buffer: executed-action target save/load + alignment with `action`
# --------------------------------------------------------------------------- #
def test_executed_action_target_stays_aligned_with_stored_action():
    buf = LAP(state_dim=4, action_dim=2, device=torch.device("cpu"), max_size=16,
              batch_size=4, prioritized=False, counterfactual_risk_dim=6,
              executed_action_risk_dim=2)
    n = 8
    for i in range(n):
        buf.add(
            np.full(4, float(i), dtype=np.float32),
            np.array([float(i), float(-i)], dtype=np.float32),
            np.zeros(4, dtype=np.float32), 0.0, 0.0,
            counterfactual_risk_target=np.zeros(6, dtype=np.float32),
            executed_action_risk_target=np.array([float(i), float(i)], dtype=np.float32))
    idx = np.arange(n)
    _, action, _, _, _ = buf.get_batch_by_indices(idx)
    executed = buf.get_last_executed_action_risk(idx)
    assert executed is not None
    # executed_action_risk_target[i] == [i, i]; action[i, 0] == i / max_action(=1).
    assert torch.allclose(action[:, 0].cpu(), executed[:, 0].cpu())


def test_replay_buffer_executed_action_target_save_load_roundtrip(tmp_path):
    buf = LAP(state_dim=4, action_dim=2, device=torch.device("cpu"), max_size=16,
              batch_size=4, prioritized=False, counterfactual_risk_dim=6,
              executed_action_risk_dim=2)
    rng = np.random.default_rng(1)
    for _ in range(6):
        buf.add(
            rng.normal(size=4).astype(np.float32),
            rng.normal(size=2).astype(np.float32),
            rng.normal(size=4).astype(np.float32), 0.5, 0.0,
            counterfactual_risk_target=rng.normal(size=6).astype(np.float32),
            executed_action_risk_target=rng.normal(size=2).astype(np.float32))
    prefix = str(tmp_path / "rb")
    buf.save(prefix)

    buf2 = LAP(state_dim=4, action_dim=2, device=torch.device("cpu"), max_size=16,
               batch_size=4, prioritized=False, counterfactual_risk_dim=6,
               executed_action_risk_dim=2)
    assert buf2.load(prefix) is True
    np.testing.assert_allclose(
        buf2.executed_action_risk_target[:buf2.size],
        buf.executed_action_risk_target[:buf.size])

    # A checkpoint saved WITHOUT the executed-action target fails fast rather
    # than silently zero-filling it (matches the fixed-candidate contract).
    buf_legacy = LAP(state_dim=4, action_dim=2, device=torch.device("cpu"),
                     max_size=16, batch_size=4, prioritized=False,
                     counterfactual_risk_dim=6, executed_action_risk_dim=0)
    for _ in range(6):
        buf_legacy.add(
            rng.normal(size=4).astype(np.float32),
            rng.normal(size=2).astype(np.float32),
            rng.normal(size=4).astype(np.float32), 0.5, 0.0,
            counterfactual_risk_target=rng.normal(size=6).astype(np.float32))
    legacy_prefix = str(tmp_path / "rb_legacy")
    buf_legacy.save(legacy_prefix)
    buf3 = LAP(state_dim=4, action_dim=2, device=torch.device("cpu"), max_size=16,
               batch_size=4, prioritized=False, counterfactual_risk_dim=6,
               executed_action_risk_dim=2)
    with pytest.raises(RuntimeError):
        buf3.load(legacy_prefix)


# --------------------------------------------------------------------------- #
#  Actor-penalty warm-up/ramp: pure config-level boundary tests
# --------------------------------------------------------------------------- #
def _cf_cfg_with_ramp(warmup, ramp, weight=0.5, horizons=(0.5, 1.0)):
    return CounterfactualRiskConfig({
        "enabled": True, "horizons_sec": list(horizons),
        "candidate_actions": CANDIDATES,
        "actor_penalty_weight": weight,
        "actor_penalty_warmup_updates": warmup,
        "actor_penalty_ramp_updates": ramp,
    })


def test_actor_penalty_warmup_start_end_and_ramp_midpoint():
    cfg = _cf_cfg_with_ramp(warmup=10, ramp=20, weight=0.5)
    assert cfg.effective_actor_penalty_weight(0) == 0.0
    assert cfg.effective_actor_penalty_weight(1) == 0.0
    assert cfg.effective_actor_penalty_weight(10) == 0.0        # last warmup update
    assert cfg.effective_actor_penalty_weight(11) == pytest.approx(0.5 * 1 / 20)
    assert cfg.effective_actor_penalty_weight(20) == pytest.approx(0.5 * 10 / 20)  # ramp midpoint
    assert cfg.effective_actor_penalty_weight(30) == pytest.approx(0.5)  # ramp complete
    assert cfg.effective_actor_penalty_weight(31) == pytest.approx(0.5)  # past ramp


def test_actor_penalty_defaults_to_full_weight_immediately():
    """warmup=ramp=0 (the pre-existing default) is byte-identical to the
    original "full weight from the first supervised update" behaviour."""
    cfg = _cf_cfg_with_ramp(warmup=0, ramp=0, weight=0.2)
    assert cfg.effective_actor_penalty_weight(1) == pytest.approx(0.2)
    assert cfg.effective_actor_penalty_weight(1000) == pytest.approx(0.2)


def test_actor_penalty_warmup_only_no_ramp_segment():
    cfg = _cf_cfg_with_ramp(warmup=5, ramp=0, weight=0.3)
    assert cfg.effective_actor_penalty_weight(5) == 0.0
    assert cfg.effective_actor_penalty_weight(6) == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
#  weighted_mean horizon aggregation: validation + normalization
# --------------------------------------------------------------------------- #
def test_weighted_mean_horizon_weights_normalize_to_sum_one():
    cfg = CounterfactualRiskConfig({
        "enabled": True, "horizons_sec": [0.5, 1.0, 1.5, 2.0],
        "candidate_actions": CANDIDATES,
        "actor_risk_aggregation": "weighted_mean",
        "horizon_weights": [4.0, 3.0, 2.0, 1.0],
    })
    assert cfg.horizon_weights == pytest.approx([0.4, 0.3, 0.2, 0.1])


def test_weighted_mean_requires_horizon_weights():
    with pytest.raises(ValueError):
        CounterfactualRiskConfig({
            "enabled": True, "horizons_sec": [0.5, 1.0],
            "candidate_actions": CANDIDATES,
            "actor_risk_aggregation": "weighted_mean",
        })


def test_horizon_weights_length_mismatch_raises():
    with pytest.raises(ValueError):
        CounterfactualRiskConfig({
            "enabled": True, "horizons_sec": [0.5, 1.0],
            "candidate_actions": CANDIDATES,
            "actor_risk_aggregation": "weighted_mean",
            "horizon_weights": [1.0, 1.0, 1.0],
        })


def test_horizon_weights_negative_entry_raises():
    with pytest.raises(ValueError):
        CounterfactualRiskConfig({
            "enabled": True, "horizons_sec": [0.5, 1.0],
            "candidate_actions": CANDIDATES,
            "actor_risk_aggregation": "weighted_mean",
            "horizon_weights": [1.0, -0.5],
        })


def test_horizon_weights_zero_sum_raises():
    with pytest.raises(ValueError):
        CounterfactualRiskConfig({
            "enabled": True, "horizons_sec": [0.5, 1.0],
            "candidate_actions": CANDIDATES,
            "actor_risk_aggregation": "weighted_mean",
            "horizon_weights": [0.0, 0.0],
        })


def test_max_and_mean_aggregation_unaffected_by_new_validation():
    """max/mean stay valid with no horizon_weights at all (backward compat)."""
    for agg in ("max", "mean"):
        cfg = CounterfactualRiskConfig({
            "enabled": True, "horizons_sec": [0.5, 1.0],
            "candidate_actions": CANDIDATES, "actor_risk_aggregation": agg})
        assert cfg.horizon_weights is None


# --------------------------------------------------------------------------- #
#  Agent-level fixtures/helpers
# --------------------------------------------------------------------------- #
def _agent_hp(spatiotemporal, counterfactual, **cf_overrides):
    cf_cfg = {
        "enabled": counterfactual, "horizons_sec": [0.5, 1.0],
        "candidate_actions": CANDIDATES, "hidden_dim": 16,
        "loss_weight": 0.1, "actor_penalty_weight": 0.2,
        "use_temporal_context": True, "enable_from_stage": 0,
    }
    cf_cfg.update(cf_overrides)
    return {
        "batch_size": 4, "buffer_size": 64, "n_critics": 2,
        "n_quantiles": 5,
        "temporal_actor_context": {
            "enabled": True, "history_len": HISTORY,
            "temporal_feature_dim": 16, "encoder_type": "conv1d",
            "stack_agent_state": False, "stage_enable_from": 0,
        },
        "spatiotemporal_lidar": {
            "enabled": spatiotemporal, "angular_tokens": 8,
            "use_range_rate": True,
        },
        "counterfactual_multi_horizon_risk": cf_cfg,
    }


def _fill_buffer(agent, n=16, counterfactual=True):
    for _ in range(n):
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action = np.random.uniform(-1, 1, ACTION_DIM).astype(np.float32)
        kwargs = {}
        if counterfactual:
            kwargs["counterfactual_risk_target"] = np.random.rand(
                len(CANDIDATES), 2).astype(np.float32)
            kwargs["executed_action_risk_target"] = np.random.rand(
                2).astype(np.float32)
        agent.replay_buffer.add(state, action, state, 0.1, 0.0, **kwargs)


@pytest.mark.parametrize("spatiotemporal,counterfactual", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_all_four_toggle_combinations_train_one_step(
        tmp_path, spatiotemporal, counterfactual):
    agent = Agent(
        STATE_DIM, ACTION_DIM, 1.0,
        _agent_hp(spatiotemporal, counterfactual),
        log_dir=str(tmp_path), env_obs_dim=OBS, env_agent_dim=AGENT)
    _fill_buffer(agent, n=16, counterfactual=counterfactual)
    before = ([p.detach().clone() for p in agent.counterfactual_risk_head.parameters()]
              if counterfactual else None)
    agent.train()
    if counterfactual:
        assert any(not torch.equal(a, b) for a, b in zip(
            before, agent.counterfactual_risk_head.parameters()))
        assert agent._cf_supervised_updates == 1


# --------------------------------------------------------------------------- #
#  Item 2: the warm-up counter only advances on an ACTUAL supervised update
# --------------------------------------------------------------------------- #
def test_supervised_counter_does_not_advance_without_a_target(tmp_path):
    agent = Agent(
        STATE_DIM, ACTION_DIM, 1.0,
        _agent_hp(False, True, enable_from_stage=5),
        log_dir=str(tmp_path), env_obs_dim=OBS, env_agent_dim=AGENT)
    _fill_buffer(agent, n=16, counterfactual=True)

    # Stage 0 < enable_from_stage=5 -> _counterfactual_risk_active is False,
    # so the supervised block never runs and the counter must stay at 0.
    for _ in range(3):
        agent.train()
    assert agent._cf_supervised_updates == 0
    assert agent._cf_effective_actor_penalty_weight == 0.0

    agent.set_curriculum_stage(5)
    agent.train()
    assert agent._cf_supervised_updates == 1


# --------------------------------------------------------------------------- #
#  Item 3: warm-up counter survives a checkpoint save/load round-trip
# --------------------------------------------------------------------------- #
def test_cf_warmup_counter_checkpoint_roundtrip(tmp_path):
    hp = _agent_hp(False, True)
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0, hp,
                  log_dir=str(tmp_path / "log1"), env_obs_dim=OBS, env_agent_dim=AGENT)
    _fill_buffer(agent, n=16, counterfactual=True)
    for _ in range(3):
        agent.train()
    assert agent._cf_supervised_updates == 3

    ckpt_dir = str(tmp_path / "ckpt")
    agent.save(ckpt_dir, "run")

    agent2 = Agent(STATE_DIM, ACTION_DIM, 1.0, hp,
                   log_dir=str(tmp_path / "log2"), env_obs_dim=OBS, env_agent_dim=AGENT)
    assert agent2._cf_supervised_updates == 0
    agent2.load(ckpt_dir, "run")
    assert agent2._cf_supervised_updates == 3


def test_cf_warmup_counter_defaults_to_zero_on_legacy_checkpoint(tmp_path):
    """A checkpoint saved before this feature (no *_cf_state.json) resumes
    with the counter at 0 (ramp restarts) instead of erroring."""
    hp = _agent_hp(False, True)
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0, hp,
                  log_dir=str(tmp_path / "log1"), env_obs_dim=OBS, env_agent_dim=AGENT)
    _fill_buffer(agent, n=16, counterfactual=True)
    agent.train()
    ckpt_dir = str(tmp_path / "ckpt")
    agent.save(ckpt_dir, "run")
    os.remove(f"{ckpt_dir}/run_cf_state.json")

    agent2 = Agent(STATE_DIM, ACTION_DIM, 1.0, hp,
                   log_dir=str(tmp_path / "log2"), env_obs_dim=OBS, env_agent_dim=AGENT)
    agent2.load(ckpt_dir, "run")
    assert agent2._cf_supervised_updates == 0


# --------------------------------------------------------------------------- #
#  Item 8: actor gradient flows into the action, never into the head's params
# --------------------------------------------------------------------------- #
def test_actor_optimizer_never_owns_counterfactual_head_params(tmp_path):
    agent = Agent(
        STATE_DIM, ACTION_DIM, 1.0, _agent_hp(False, True),
        log_dir=str(tmp_path), env_obs_dim=OBS, env_agent_dim=AGENT)
    actor_param_ids = {id(p) for g in agent.actor_optimizer.param_groups
                       for p in g["params"]}
    cf_param_ids = {id(p) for p in agent.counterfactual_risk_head.parameters()}
    assert actor_param_ids.isdisjoint(cf_param_ids)


def test_cf_head_params_unchanged_by_actor_loss_when_supervised_inactive(tmp_path):
    """With the stage gate closed, the supervised block never runs and (per
    the new cf_supervised_ran gate) neither does the actor penalty -- so a
    train() call must leave the head parameters byte-identical, and must
    still move the actor's own parameters (the critic/entropy path is
    unaffected by any of this)."""
    agent = Agent(
        STATE_DIM, ACTION_DIM, 1.0,
        _agent_hp(False, True, enable_from_stage=5),
        log_dir=str(tmp_path), env_obs_dim=OBS, env_agent_dim=AGENT)
    _fill_buffer(agent, n=16, counterfactual=True)
    head_before = [p.detach().clone() for p in agent.counterfactual_risk_head.parameters()]
    actor_before = [p.detach().clone() for p in agent.actor.parameters()]
    agent.train()
    assert all(torch.equal(a, b) for a, b in zip(
        head_before, agent.counterfactual_risk_head.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(
        actor_before, agent.actor.parameters()))


# --------------------------------------------------------------------------- #
#  Item 9: CF OFF -> replay buffer / head stay exactly as before this feature
# --------------------------------------------------------------------------- #
def test_counterfactual_disabled_keeps_buffer_and_head_unchanged(tmp_path):
    agent = Agent(
        STATE_DIM, ACTION_DIM, 1.0, _agent_hp(False, False),
        log_dir=str(tmp_path), env_obs_dim=OBS, env_agent_dim=AGENT)
    assert agent.replay_buffer.counterfactual_risk_dim == 0
    assert agent.replay_buffer.executed_action_risk_dim == 0
    assert agent.replay_buffer.get_last_counterfactual_risk() is None
    assert agent.replay_buffer.get_last_executed_action_risk() is None
    assert agent.counterfactual_risk_head is None
    _fill_buffer(agent, n=16, counterfactual=False)
    agent.train()  # must not raise with CF fully off


# --------------------------------------------------------------------------- #
#  Item 4 (agent-level): weighted_mean actually changes the actor penalty
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#  Action-contract-change reset (stage 5 unseals the yield channel in
#  phase2/both_trajrisk_rbs(_cf_st) -- reset_buffer_on_promote_to: [5]):
#  the CF actor-penalty warm-up/ramp must re-arm alongside the replay-buffer
#  reset, WITHOUT throwing away the head's learned weights.
# --------------------------------------------------------------------------- #
def test_cf_schedule_resets_on_action_contract_change(tmp_path):
    agent = Agent(
        STATE_DIM, ACTION_DIM, 1.0, _agent_hp(False, True),
        log_dir=str(tmp_path), env_obs_dim=OBS, env_agent_dim=AGENT)
    agent._cf_supervised_updates = 20000
    agent._cf_effective_actor_penalty_weight = 0.1

    head_before = [
        p.detach().clone()
        for p in agent.counterfactual_risk_head.parameters()
    ]

    agent.reset_counterfactual_penalty_schedule()

    assert agent._cf_supervised_updates == 0
    assert agent._cf_effective_actor_penalty_weight == 0.0

    # Head knowledge (the learned dynamic-risk representation) is preserved --
    # only the actor-penalty ramp state resets, not the head itself.
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            head_before,
            agent.counterfactual_risk_head.parameters(),
        )
    )


def test_reset_penalty_schedule_is_noop_when_cf_disabled(tmp_path):
    agent = Agent(
        STATE_DIM, ACTION_DIM, 1.0, _agent_hp(False, False),
        log_dir=str(tmp_path), env_obs_dim=OBS, env_agent_dim=AGENT)
    agent.reset_counterfactual_penalty_schedule()  # must not raise
    assert agent.counterfactual_risk_head is None


def test_supervised_counter_advances_at_stage_3(tmp_path):
    agent = Agent(
        STATE_DIM, ACTION_DIM, 1.0, _agent_hp(False, True, enable_from_stage=3),
        log_dir=str(tmp_path), env_obs_dim=OBS, env_agent_dim=AGENT)
    agent.set_curriculum_stage(3)
    _fill_buffer(agent, n=16, counterfactual=True)
    agent.train()
    assert agent._cf_supervised_updates == 1


def test_promotion_sequence_resets_buffer_and_penalty_schedule(tmp_path):
    """Mirror the shared reset path after a stage-5 push succeeds."""
    agent = Agent(
        STATE_DIM, ACTION_DIM, 1.0, _agent_hp(False, True, enable_from_stage=3),
        log_dir=str(tmp_path), env_obs_dim=OBS, env_agent_dim=AGENT)
    agent.set_curriculum_stage(3)
    _fill_buffer(agent, n=16, counterfactual=True)
    for _ in range(5):
        agent.train()
    assert agent._cf_supervised_updates == 5
    assert agent.replay_buffer.size == 16

    # The trainer first pushes the stage and aborts without mutation on failure;
    # only a successful push calls the agent's atomic replay/schedule reset.
    agent.set_curriculum_stage(5)
    agent.reset_replay_for_action_contract_change()

    assert agent.replay_buffer.size == 0
    assert agent._cf_supervised_updates == 0
    assert agent._cf_effective_actor_penalty_weight == 0.0
    # A supervised update right after promotion is stage-5's FIRST, not a
    # continuation of the pre-reset count.
    _fill_buffer(agent, n=16, counterfactual=True)
    agent.train()
    assert agent._cf_supervised_updates == 1


def test_real_profile_warmup_ramp_boundaries():
    """Exercises the exact actor_penalty_warmup_updates=5000 / ramp_updates=
    10000 values shipped in both_trajrisk_rbs_cf_st's hyperparameters_tqc.yaml
    (see test_phase2_cf_st_profile.py for the profile-file-level check of
    these same numbers)."""
    cfg = CounterfactualRiskConfig({
        "enabled": True, "horizons_sec": [0.5, 1.0, 1.5, 2.0],
        "candidate_actions": CANDIDATES,
        "actor_penalty_weight": 0.1,
        "actor_penalty_warmup_updates": 5000,
        "actor_penalty_ramp_updates": 10000,
    })
    assert cfg.effective_actor_penalty_weight(5000) == 0.0
    assert cfg.effective_actor_penalty_weight(5001) == pytest.approx(0.1 * 1 / 10000)
    assert cfg.effective_actor_penalty_weight(10000) == pytest.approx(0.1 * 5000 / 10000)
    assert cfg.effective_actor_penalty_weight(15000) == pytest.approx(0.1)


def test_weighted_mean_aggregation_trains_without_error(tmp_path):
    hp = _agent_hp(
        False, True,
        actor_risk_aggregation="weighted_mean", horizon_weights=[0.7, 0.3],
        actor_penalty_warmup_updates=0, actor_penalty_ramp_updates=0)
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0, hp, log_dir=str(tmp_path),
                  env_obs_dim=OBS, env_agent_dim=AGENT)
    assert agent._cf_horizon_weights is not None
    assert torch.allclose(agent._cf_horizon_weights.cpu(),
                          torch.tensor([0.7, 0.3]))
    _fill_buffer(agent, n=16, counterfactual=True)
    agent.train()
    assert agent._cf_supervised_updates == 1
