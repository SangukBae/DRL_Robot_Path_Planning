"""Unit tests for tqc_networks (Actor / Critic / quantile_huber_loss).

torch-gated: skipped automatically where torch is not installed (e.g. the bare
CI image), and run on the training box where torch is available.
"""

import pytest

# torch-gated without a module-level Skipped (which can abort directory-wide
# collection under this pytest + ament plugin stack). Import defensively and gate
# the tests with skipif instead.
try:
    import torch
    import tqc_networks as net
    _HAVE_TORCH = True
except Exception:  # pragma: no cover - exercised only where torch is absent
    _HAVE_TORCH = False

pytestmark = pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")


STATE_DIM = 12
ACTION_DIM = 2
BATCH = 8


def test_actor_action_in_tanh_range():
    actor = net.Actor(STATE_DIM, ACTION_DIM, hdim=32)
    s = torch.randn(BATCH, STATE_DIM)
    a = actor(s)
    assert a.shape == (BATCH, ACTION_DIM)
    assert torch.all(a <= 1.0) and torch.all(a >= -1.0)


def test_actor_deterministic_mode_is_repeatable():
    actor = net.Actor(STATE_DIM, ACTION_DIM, hdim=32)
    actor.eval()
    s = torch.randn(BATCH, STATE_DIM)
    a1 = actor(s, deterministic=True)
    a2 = actor(s, deterministic=True)
    assert torch.allclose(a1, a2)


def test_actor_action_log_prob_shapes():
    actor = net.Actor(STATE_DIM, ACTION_DIM, hdim=32)
    s = torch.randn(BATCH, STATE_DIM)
    a, logp = actor.action_log_prob(s)
    assert a.shape == (BATCH, ACTION_DIM)
    assert logp.shape == (BATCH, 1)


def test_critic_output_shape():
    n_critics, n_quantiles = 3, 25
    critic = net.Critic(STATE_DIM, ACTION_DIM, hdim=32,
                        n_quantiles=n_quantiles, n_critics=n_critics)
    s = torch.randn(BATCH, STATE_DIM)
    a = torch.randn(BATCH, ACTION_DIM)
    q = critic(s, a)
    assert q.shape == (BATCH, n_critics, n_quantiles)


def test_quantile_huber_loss_nonnegative_and_zero_at_match():
    cur = torch.randn(BATCH, 2, 25)
    tgt = torch.randn(BATCH, 1, 25)
    loss = net.quantile_huber_loss(cur, tgt)
    assert loss.item() >= 0.0
    # Perfectly matching quantiles → zero loss.
    same = torch.randn(BATCH, 1, 25)
    loss0 = net.quantile_huber_loss(same.expand(BATCH, 2, 25).contiguous(), same)
    assert loss0.item() == pytest.approx(0.0, abs=1e-6)
