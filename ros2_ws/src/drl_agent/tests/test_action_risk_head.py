"""Integration / unit tests for PHASE2 candidate 2: the Critic-connected
Action-Risk Head (action_risk_head.py + tqc_networks.Critic(extra_dim) +
tqc_agent.Agent wiring), torch-gated.

Covers:
  * ActionRiskConfig defaults (disabled) and ActionRiskHead forward shape.
  * Critic(extra_dim=0) is unaffected (baseline forward, no `extra` needed).
  * Critic(extra_dim=2) requires `extra` and produces the same output shape.
  * Agent fail-fast: critic_risk_input.enabled=true requires
    action_risk_head.enabled=true.
  * Agent smoke test, action_risk_head ONLY (critic_risk_input off): builds,
    the buffer accepts action_risk_target, train() runs end-to-end and the
    head's parameters actually receive gradient / update.
  * Agent smoke test, BOTH enabled: Critic built with extra_dim=2, train() runs
    end-to-end (forward/backward/update through all 3 critic call sites).
  * Default config (both disabled) is exactly the pre-PHASE2 baseline: no head,
    Critic input width unchanged, buffer carries no action_risk_target array.

Skipped where torch is unavailable.
"""

import pytest

try:
    import numpy as np
    import torch

    from action_risk_head import ActionRiskConfig, ActionRiskHead
    from tqc_networks import Critic
    from tqc_agent import Agent
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

pytestmark = pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")

STATE_DIM, ACTION_DIM = 87, 2


# --------------------------------------------------------------------------- #
#  ActionRiskConfig / ActionRiskHead (module-level, no Agent)
# --------------------------------------------------------------------------- #
def test_action_risk_config_defaults_disabled():
    cfg = ActionRiskConfig({})
    assert cfg.enabled is False
    assert cfg.hidden_dim == 64
    assert cfg.loss_weight == 0.1


def test_action_risk_head_forward_shape():
    cfg = ActionRiskConfig(dict(enabled=True, hidden_dim=32))
    head = ActionRiskHead(latent_dim=128, action_dim=ACTION_DIM, cfg=cfg)
    z = torch.randn(8, 128)
    a = torch.randn(8, ACTION_DIM)
    out = head(z, a)
    assert out.shape == (8, 2)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)  # sigmoid outputs


def test_frozen_head_forward_keeps_input_gradient_but_not_param_gradient():
    """Regression for the reviewed actor-update bug: tqc_agent.py's actor block
    must NOT fully detach the head's output for `actions_pi` -- that would sever
    d(extra)/d(actions_pi), the exact pathway that lets the critic's learned
    "penalise risk-raising actions" signal reach the actor. The fix freezes the
    head's OWN parameters (requires_grad_(False) for the forward call only) so
    gradient still flows to the ACTION input while the head's parameters get
    none. This test isolates that mechanism directly (independent of the full
    Agent/critic machinery in the end-to-end tests below)."""
    cfg = ActionRiskConfig(dict(enabled=True, hidden_dim=16))
    head = ActionRiskHead(latent_dim=8, action_dim=2, cfg=cfg)
    z = torch.randn(4, 8)
    action = torch.randn(4, 2, requires_grad=True)

    for p in head.parameters():
        p.requires_grad_(False)
    out = head(z, action)
    for p in head.parameters():
        p.requires_grad_(True)

    out.sum().backward()

    assert action.grad is not None and torch.any(action.grad != 0.0), \
        "gradient must still reach the action input (the actor's live sample)"
    assert all(p.grad is None for p in head.parameters()), \
        "the head's own parameters must receive NO gradient from this path"


# --------------------------------------------------------------------------- #
#  Critic extra_dim
# --------------------------------------------------------------------------- #
def test_critic_extra_dim_zero_is_baseline():
    c = Critic(STATE_DIM, ACTION_DIM, hdim=32, n_quantiles=5, n_critics=2)
    s = torch.randn(4, STATE_DIM)
    a = torch.randn(4, ACTION_DIM)
    out = c(s, a)
    assert out.shape == (4, 2, 5)


def test_critic_extra_dim_two_requires_extra():
    c = Critic(STATE_DIM, ACTION_DIM, hdim=32, n_quantiles=5, n_critics=2, extra_dim=2)
    s = torch.randn(4, STATE_DIM)
    a = torch.randn(4, ACTION_DIM)
    with pytest.raises(ValueError):
        c(s, a)  # extra missing
    extra = torch.rand(4, 2)
    out = c(s, a, extra=extra)
    assert out.shape == (4, 2, 5)


# --------------------------------------------------------------------------- #
#  Agent-level wiring
# --------------------------------------------------------------------------- #
def _hp(action_risk=None, critic_risk_input=None, **over):
    hp = dict(batch_size=8, buffer_size=500, n_critics=2, n_quantiles=5)
    if action_risk is not None:
        hp["action_risk_head"] = action_risk
    if critic_risk_input is not None:
        hp["critic_risk_input"] = critic_risk_input
    hp.update(over)
    return hp


def _fill(agent, n=32, action_risk_target=False):
    for _ in range(n):
        s = np.random.randn(STATE_DIM).astype(np.float32)
        a = np.random.uniform(-1, 1, ACTION_DIM).astype(np.float32)
        kwargs = {}
        if action_risk_target:
            kwargs["action_risk_target"] = np.random.rand(2).astype(np.float32)
        agent.replay_buffer.add(s, a, s, 0.1, 0.0, **kwargs)


def test_critic_risk_input_requires_action_risk_head(tmp_path):
    with pytest.raises(RuntimeError):
        Agent(STATE_DIM, ACTION_DIM, 1.0,
              _hp(action_risk={"enabled": False},
                  critic_risk_input={"enabled": True}),
              log_dir=str(tmp_path))


def test_default_config_is_unchanged_baseline(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0, _hp(), log_dir=str(tmp_path))
    assert agent.action_risk_enabled is False
    assert agent.action_risk_head is None
    assert agent.critic_risk_input_enabled is False
    assert agent.critic.extra_dim == 0
    assert agent.replay_buffer.action_risk_target is None
    _fill(agent)
    agent.train()  # baseline path runs unchanged


def test_action_risk_head_only_trains_end_to_end(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0,
                  _hp(action_risk={"enabled": True, "hidden_dim": 32}),
                  log_dir=str(tmp_path))
    assert agent.action_risk_head is not None
    assert agent.critic_risk_input_enabled is False
    assert agent.critic.extra_dim == 0     # critic shape unchanged (head-only)
    assert agent.replay_buffer.action_risk_dim == 2
    _fill(agent, action_risk_target=True)

    before = [p.detach().clone() for p in agent.action_risk_head.parameters()]
    for _ in range(5):
        agent.train()
    after = list(agent.action_risk_head.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), \
        "action_risk_head did not update from its supervised loss"


def test_both_enabled_trains_end_to_end_with_extended_critic(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0,
                  _hp(action_risk={"enabled": True, "hidden_dim": 32},
                      critic_risk_input={"enabled": True}),
                  log_dir=str(tmp_path))
    assert agent.critic_risk_input_enabled is True
    assert agent.critic.extra_dim == 2
    assert agent.critic_target.extra_dim == 2
    assert agent.action_risk_head_target is not None
    _fill(agent, action_risk_target=True)

    critic_before = [p.detach().clone() for p in agent.critic.parameters()]
    for _ in range(5):
        agent.train()  # exercises all 3 critic call sites with extra=
    critic_after = list(agent.critic.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(critic_before, critic_after))

    # select_action / inference path is untouched by critic_risk_input (the
    # actor never consumes the risk prediction directly).
    s = np.random.randn(STATE_DIM).astype(np.float32)
    action = agent.select_action(s, use_exploration=False)
    assert action.shape == (ACTION_DIM,)


def test_save_load_roundtrip_with_both_enabled(tmp_path):
    a1 = Agent(STATE_DIM, ACTION_DIM, 1.0,
               _hp(action_risk={"enabled": True, "hidden_dim": 32},
                   critic_risk_input={"enabled": True}),
               log_dir=str(tmp_path / "a"))
    _fill(a1, action_risk_target=True)
    a1.train()
    a1.save(str(tmp_path), "ckpt")
    assert (tmp_path / "ckpt_action_risk_head.pth").is_file()

    a2 = Agent(STATE_DIM, ACTION_DIM, 1.0,
               _hp(action_risk={"enabled": True, "hidden_dim": 32},
                   critic_risk_input={"enabled": True}),
               log_dir=str(tmp_path / "b"))
    a2.load(str(tmp_path), "ckpt", load_replay_buffer=False)
    for p1, p2 in zip(a1.action_risk_head.parameters(),
                      a2.action_risk_head.parameters()):
        assert torch.allclose(p1, p2)
