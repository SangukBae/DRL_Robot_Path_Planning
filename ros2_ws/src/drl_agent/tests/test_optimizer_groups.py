"""Unit tests for Stage-8 (ISOLATED experimental feature, default OFF, NOT
enabled for phase2/both) optimizer parameter-group separation: separate LRs
for critic/encoder/aux_head/action_risk_head within ONE Adam instance (same
.zero_grad()/.step() call sites, unchanged).
"""

import pytest

try:
    import numpy as np
    import torch

    from drl_agent.rl.algorithms.tqc.agent import Agent
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

pytestmark = pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")

STATE_DIM, ACTION_DIM = 87, 2


def _hp(**over):
    hp = dict(batch_size=8, buffer_size=200, n_critics=2, n_quantiles=5, critic_lr=3e-4)
    hp.update(over)
    return hp


def _fill(agent, n=16, aux_dim=0, action_risk_dim=0):
    rng = np.random.default_rng(0)
    for _ in range(n):
        agent.replay_buffer.add(
            rng.normal(size=STATE_DIM).astype(np.float32),
            rng.uniform(-1, 1, ACTION_DIM).astype(np.float32),
            rng.normal(size=STATE_DIM).astype(np.float32), 0.1, 0.0,
            aux_target=(rng.normal(size=aux_dim).astype(np.float32) if aux_dim else None),
            action_risk_target=(rng.normal(size=action_risk_dim).astype(np.float32)
                                 if action_risk_dim else None),
        )


def test_disabled_by_default_single_effective_lr(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0, _hp(critic_lr=1e-3), log_dir=str(tmp_path))
    for g in agent.critic_optimizer.param_groups:
        assert g["lr"] == pytest.approx(1e-3)


def test_enabled_without_overrides_is_behaviorally_equivalent(tmp_path):
    # optimizer_groups.enabled=true but no per-component LR set -> every
    # group falls back to critic_lr -> same effective LR everywhere.
    hp = _hp(critic_lr=1e-3, action_risk_head={"enabled": True, "hidden_dim": 16},
              optimizer_groups={"enabled": True})
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0, hp, log_dir=str(tmp_path))
    assert len(agent.critic_optimizer.param_groups) >= 2  # critic + action_risk_head
    for g in agent.critic_optimizer.param_groups:
        assert g["lr"] == pytest.approx(1e-3)


def test_enabled_with_encoder_lr_override_creates_a_distinct_group(tmp_path):
    hp = _hp(critic_lr=1e-3, aux_prediction={"enabled": True, "latent_dim": 32,
                                              "encoder_hidden_dim": 64},
              optimizer_groups={"enabled": True, "encoder_lr": 1e-5})
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0, hp, log_dir=str(tmp_path))
    lrs = {g["lr"] for g in agent.critic_optimizer.param_groups}
    assert 1e-3 in (pytest.approx(v) for v in lrs) or any(
        abs(v - 1e-3) < 1e-9 for v in lrs)
    assert any(abs(v - 1e-5) < 1e-12 for v in lrs), \
        f"expected an encoder-only param group at lr=1e-5, got groups {lrs}"


def test_action_risk_head_gets_its_own_configured_lr(tmp_path):
    hp = _hp(critic_lr=1e-3,
              action_risk_head={"enabled": True, "hidden_dim": 16},
              optimizer_groups={"enabled": True, "action_risk_head_lr": 5e-4})
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0, hp, log_dir=str(tmp_path))
    lrs = [g["lr"] for g in agent.critic_optimizer.param_groups]
    assert any(abs(v - 5e-4) < 1e-12 for v in lrs)
    assert any(abs(v - 1e-3) < 1e-12 for v in lrs)  # critic's own group unaffected


def test_grouped_optimizer_still_trains_end_to_end(tmp_path):
    hp = _hp(action_risk_head={"enabled": True, "hidden_dim": 16},
              optimizer_groups={"enabled": True, "encoder_lr": 1e-4,
                                 "action_risk_head_lr": 2e-4})
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0, hp, log_dir=str(tmp_path))
    _fill(agent, action_risk_dim=2)
    critic_before = [p.detach().clone() for p in agent.critic.parameters()]
    for _ in range(5):
        agent.train()
    critic_after = list(agent.critic.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(critic_before, critic_after))


def test_smaller_encoder_lr_updates_encoder_less_than_critic_lr_would(tmp_path):
    """The actual point of the feature: a smaller encoder_lr must produce a
    SMALLER encoder parameter change than using critic_lr for everything,
    all else (init, batch, seed) held equal."""
    torch.manual_seed(0)
    np.random.seed(0)
    hp_common = dict(critic_lr=1e-2,
                      aux_prediction={"enabled": True, "latent_dim": 16,
                                       "encoder_hidden_dim": 32})

    torch.manual_seed(1)
    agent_flat = Agent(STATE_DIM, ACTION_DIM, 1.0, _hp(**hp_common), log_dir=str(tmp_path / "flat"))
    torch.manual_seed(1)
    agent_split = Agent(STATE_DIM, ACTION_DIM, 1.0,
                         _hp(**hp_common, optimizer_groups={"enabled": True, "encoder_lr": 1e-5}),
                         log_dir=str(tmp_path / "split"))
    # Same initial encoder weights (same seed at construction).
    for p1, p2 in zip(agent_flat.encoder.parameters(), agent_split.encoder.parameters()):
        assert torch.allclose(p1, p2)

    rng = np.random.default_rng(2)
    batch = [(rng.normal(size=STATE_DIM).astype(np.float32),
              rng.uniform(-1, 1, ACTION_DIM).astype(np.float32),
              rng.normal(size=STATE_DIM).astype(np.float32),
              float(rng.normal()),
              rng.normal(size=16).astype(np.float32)) for _ in range(16)]
    for agent in (agent_flat, agent_split):
        for s, a, ns, r, aux in batch:
            agent.replay_buffer.add(s, a, ns, r, 0.0, aux_target=aux)

    enc_before = [p.detach().clone() for p in agent_flat.encoder.parameters()]
    agent_flat.train()
    agent_split.train()

    flat_delta = sum(
        (a - b).abs().sum().item()
        for a, b in zip(agent_flat.encoder.parameters(), enc_before)
    )
    split_delta = sum(
        (a - b).abs().sum().item()
        for a, b in zip(agent_split.encoder.parameters(), enc_before)
    )
    assert split_delta < flat_delta, (
        f"encoder_lr=1e-5 should move the encoder LESS than critic_lr=1e-2 "
        f"would; got split_delta={split_delta} flat_delta={flat_delta}")
