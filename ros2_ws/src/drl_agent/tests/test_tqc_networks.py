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
    import drl_agent.rl.networks.tqc as net
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


def test_quantile_huber_loss_nonnegative():
    cur = torch.randn(BATCH, 2, 25)
    tgt = torch.randn(BATCH, 1, 25)
    assert net.quantile_huber_loss(cur, tgt).item() >= 0.0


def test_quantile_huber_loss_zero_for_degenerate_equal_distribution():
    # quantile_huber_loss compares EVERY current-quantile index against EVERY
    # target-quantile index (td_errors = target.unsqueeze(2) - current.unsqueeze(-1)
    # is the full (n_quantiles, n_target_quantiles) cross product, not an
    # index-matched diff) -- the standard QR-DQN/TQC pairwise objective, used
    # because a network's quantile atoms have no a-priori correspondence to the
    # target's. So "current == target" only forces the DIAGONAL pairs (i == j)
    # to zero; loss is fully zero only in the degenerate case where the
    # off-diagonal pairs are ALSO zero, i.e. every quantile value is identical
    # (a constant distribution) -- not merely "the same 25 arbitrary values".
    same = torch.full((BATCH, 1, 25), 0.7)
    cur = same.expand(BATCH, 2, 25).contiguous()
    assert net.quantile_huber_loss(cur, same).item() == pytest.approx(0.0, abs=1e-6)


def test_quantile_huber_loss_zero_for_single_quantile_match():
    # With n_quantiles == n_target_quantiles == 1 there is only the (0, 0)
    # pair -- no off-diagonal terms exist, so a match IS exactly zero
    # regardless of the (arbitrary, non-constant) quantile value.
    same = torch.randn(BATCH, 1, 1)
    cur = same.expand(BATCH, 2, 1).contiguous()
    assert net.quantile_huber_loss(cur, same).item() == pytest.approx(0.0, abs=1e-6)


def test_quantile_huber_loss_matches_manual_pairwise_computation():
    # Independent (non-vectorised, spec-derived) re-implementation on a small
    # hand-sized case, to pin down the exact pairwise cross-product + Huber +
    # asymmetric quantile-weight formula the implementation must follow.
    kappa = 1.0
    current = torch.tensor([[[0.0, 1.0]]])   # (batch=1, n_critics=1, n_quantiles=2)
    target = torch.tensor([[[0.5, -1.0]]])   # (batch=1, 1, n_target_quantiles=2)

    n_quantiles = current.shape[-1]
    taus = [(i + 0.5) / n_quantiles for i in range(n_quantiles)]
    total = 0.0
    count = 0
    for i, tau in enumerate(taus):
        for j in range(target.shape[-1]):
            u = target[0, 0, j].item() - current[0, 0, i].item()
            huber = 0.5 * u ** 2 if abs(u) <= kappa else kappa * (abs(u) - 0.5 * kappa)
            total += abs(tau - float(u < 0)) * huber
            count += 1
    expected = total / count  # quantile_huber_loss(..., sum_over_quantiles=False) == .mean()

    loss = net.quantile_huber_loss(current, target, sum_over_quantiles=False, kappa=kappa)
    assert loss.item() == pytest.approx(expected, abs=1e-6)
