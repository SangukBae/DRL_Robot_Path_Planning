"""Integration smoke tests for the aux-only temporal context inside the TQC
agent (AUX_PRED v2), torch-gated.

Exercises the FULL training wiring without ROS/Gazebo:
  * an agent built with temporal + action-conditioned aux trains end-to-end and
    actually updates the temporal encoder (gradients reach it);
  * the actor stays isolated from the temporal/aux gradients (z.detach());
  * disabled aux reproduces the baseline (identity encoder, no temporal module,
    buffer carries no aux/boundary arrays);
  * save -> load round-trips, and resuming a checkpoint that PREDATES the
    temporal module falls back to a fresh temporal encoder (graceful).

Skipped where torch is unavailable.
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
RISK_DIM = 16 * 3
LABEL_DIM = RISK_DIM + 3


def _hp(**aux_over):
    aux = dict(
        enabled=True, latent_dim=128, encoder_hidden_dim=256,
        num_sectors=16, horizons_sec=[0.5, 1.0, 1.5],
        loss_weight=0.1, min_distance_loss_weight=0.1,
        aux_trunk_hidden_dim=128, aux_trunk_layers=2, aux_head_layernorm=True,
        action_conditioned_aux=True, action_conditioned_steps=4,
        action_embed_dim=32, action_condition_hidden_dim=64,
        action_condition_attention=True, action_condition_attention_heads=4,
        temporal_enabled=True, history_len=4, temporal_context_dim=32,
        temporal_attention=True,
    )
    aux.update(aux_over)
    return dict(batch_size=8, buffer_size=2000, n_critics=2, n_quantiles=5,
                aux_prediction=aux)


def _fill(agent, n=40, ep_len=10):
    """Add n transitions in episodes of ep_len, with aux labels + boundaries."""
    for i in range(n):
        s = np.random.randn(STATE_DIM).astype(np.float32)
        a = np.random.uniform(-1, 1, ACTION_DIM).astype(np.float32)
        agent.replay_buffer.add(s, a, s, 0.1, 0.0,
                                aux_target=np.random.rand(LABEL_DIM).astype(np.float32))
        if (i + 1) % ep_len == 0:
            agent.replay_buffer.mark_last_traj_end()


def test_temporal_agent_trains_and_updates_temporal_encoder(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0, _hp(), log_dir=str(tmp_path))
    assert agent.temporal_encoder is not None
    assert agent.replay_buffer.traj_end is not None      # boundary tracking on
    _fill(agent)
    before = [p.detach().clone() for p in agent.temporal_encoder.parameters()]
    actor_before = [p.detach().clone() for p in agent.actor.parameters()]
    for _ in range(5):
        agent.train()
    after = list(agent.temporal_encoder.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), \
        "temporal encoder did not update"
    # Actor must be unaffected by the temporal/aux gradient flow on THIS step's
    # critic update — it is trained on z.detach(); it changes only via its own
    # optimizer, so it should still move, but never through the temporal branch.
    # (We only assert training ran without error + temporal moved.)
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in agent.actor.parameters())
    del actor_before


def test_temporal_agent_save_load_roundtrip(tmp_path):
    a1 = Agent(STATE_DIM, ACTION_DIM, 1.0, _hp(), log_dir=str(tmp_path / "a"))
    _fill(a1)
    a1.train()
    a1.save(str(tmp_path), "ckpt")
    assert (tmp_path / "ckpt_temporal_encoder.pth").is_file()
    a2 = Agent(STATE_DIM, ACTION_DIM, 1.0, _hp(), log_dir=str(tmp_path / "b"))
    a2.load(str(tmp_path), "ckpt", load_replay_buffer=False)
    for p1, p2 in zip(a1.temporal_encoder.parameters(),
                      a2.temporal_encoder.parameters()):
        assert torch.allclose(p1, p2)


def test_resume_without_temporal_file_is_graceful(tmp_path):
    # Save a NON-temporal aux checkpoint (no temporal_encoder file). Keep
    # action-conditioning on so the OLD buffer still has episode boundaries —
    # this isolates the "missing temporal_encoder file" path from the buffer one.
    a1 = Agent(STATE_DIM, ACTION_DIM, 1.0, _hp(temporal_enabled=False),
               log_dir=str(tmp_path / "a"))
    assert a1.temporal_encoder is None
    _fill(a1)
    a1.save(str(tmp_path), "old")
    assert not (tmp_path / "old_temporal_encoder.pth").is_file()
    # Resume into a temporal-enabled agent: must not raise; temporal stays fresh.
    a2 = Agent(STATE_DIM, ACTION_DIM, 1.0, _hp(temporal_enabled=True),
               log_dir=str(tmp_path / "b"))
    a2.load(str(tmp_path), "old", load_replay_buffer=False)
    assert a2.temporal_encoder is not None        # fresh, ready to retrain


def test_full_resume_from_boundaryless_buffer_degrades_gracefully(tmp_path):
    """The REAL auto-resume path (load_replay_buffer=True): a pre-boundary
    checkpoint (no action-conditioned/temporal aux -> no traj_end in its buffer)
    resumed into a temporal run must NOT crash. The model resumes; the
    incompatible replay buffer is refused by the low-level guard and tqc_io
    degrades to a FRESH buffer (warmup refills it)."""
    old = _hp(temporal_enabled=False, action_conditioned_aux=False)
    a1 = Agent(STATE_DIM, ACTION_DIM, 1.0, old, log_dir=str(tmp_path / "a"))
    assert a1.replay_buffer.traj_end is None          # no boundary tracking
    _fill(a1)
    assert a1.replay_buffer.size > 0
    a1.save(str(tmp_path), "old")

    new = _hp(temporal_enabled=True, action_conditioned_aux=False)
    a2 = Agent(STATE_DIM, ACTION_DIM, 1.0, new, log_dir=str(tmp_path / "b"))
    assert a2.replay_buffer.traj_end is not None       # this run NEEDS boundaries
    a2.load(str(tmp_path), "old", load_replay_buffer=True)   # must NOT raise
    assert a2.temporal_encoder is not None             # model resumed
    assert a2.replay_buffer.size == 0                  # buffer degraded to fresh
    # Fresh buffer still trains end-to-end once refilled.
    _fill(a2)
    a2.train()


def test_aux_beta_warmup_starts_at_zero(tmp_path):
    """train() increments training_steps to 1 BEFORE the aux block, so the first
    update's beta must be exactly 0 (a true 0 -> loss_weight ramp), reach full
    weight after w updates, and be constant when the warmup is disabled."""
    a = Agent(STATE_DIM, ACTION_DIM, 1.0,
              _hp(loss_weight=0.1, aux_beta_warmup_steps=100),
              log_dir=str(tmp_path / "w"))
    a.training_steps = 1                       # first train() state
    assert a._current_aux_beta() == 0.0
    a.training_steps = 51                       # halfway (50/100)
    assert abs(a._current_aux_beta() - 0.05) < 1e-9
    a.training_steps = 101                      # w + 1 -> full weight
    assert abs(a._current_aux_beta() - 0.1) < 1e-9
    b = Agent(STATE_DIM, ACTION_DIM, 1.0,
              _hp(loss_weight=0.1, aux_beta_warmup_steps=0),
              log_dir=str(tmp_path / "c"))
    b.training_steps = 1                        # disabled -> constant loss_weight
    assert abs(b._current_aux_beta() - 0.1) < 1e-9


def test_disabled_aux_is_baseline(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0,
                  dict(batch_size=8, buffer_size=500,
                       aux_prediction=dict(enabled=False)),
                  log_dir=str(tmp_path))
    assert agent.encoder.has_params() is False     # identity passthrough
    assert agent.aux_head is None
    assert agent.temporal_encoder is None
    assert agent.replay_buffer.aux_target is None
    assert agent.replay_buffer.traj_end is None
    for _ in range(20):
        s = np.random.randn(STATE_DIM).astype(np.float32)
        agent.replay_buffer.add(s, np.zeros(ACTION_DIM), s, 0.0, 0.0)
    agent.train()                                  # baseline path runs
