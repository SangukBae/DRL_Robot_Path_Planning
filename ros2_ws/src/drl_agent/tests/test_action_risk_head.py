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

    from drl_agent.rl.networks.action_risk_head import ActionRiskConfig, ActionRiskHead
    from drl_agent.rl.networks.tqc import Critic
    from drl_agent.rl.algorithms.tqc.agent import Agent
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


# --------------------------------------------------------------------------- #
#  PHASE2 temporal context: ActionRiskConfig defaults + ActionRiskHead shapes
# --------------------------------------------------------------------------- #
def test_action_risk_config_temporal_defaults_off():
    cfg = ActionRiskConfig({})
    assert cfg.use_temporal_context is False
    assert cfg.temporal_context_source == "actor"


def test_temporal_dim_zero_is_byte_identical_to_original_head():
    """use_temporal_context=false (temporal_dim=0, the default) must build the
    ORIGINAL [z, action] -> hidden Linear width -- no size change, no new
    required argument."""
    cfg = ActionRiskConfig(dict(enabled=True, hidden_dim=32))
    head = ActionRiskHead(latent_dim=128, action_dim=ACTION_DIM, cfg=cfg)
    assert head.temporal_dim == 0
    assert head.l1.in_features == 128 + ACTION_DIM
    z = torch.randn(8, 128)
    a = torch.randn(8, ACTION_DIM)
    out = head(z, a)  # no temporal_feature kwarg -- must still work
    assert out.shape == (8, 2)


def test_temporal_dim_positive_widens_input_and_requires_temporal_feature():
    cfg = ActionRiskConfig(dict(enabled=True, hidden_dim=32, use_temporal_context=True))
    head = ActionRiskHead(latent_dim=128, action_dim=ACTION_DIM, cfg=cfg, temporal_dim=32)
    assert head.temporal_dim == 32
    assert head.l1.in_features == 128 + 32 + ACTION_DIM

    z = torch.randn(8, 128)
    a = torch.randn(8, ACTION_DIM)
    temporal = torch.randn(8, 32)

    out = head(z, a, temporal_feature=temporal)
    assert out.shape == (8, 2)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)

    with pytest.raises(RuntimeError):
        head(z, a)  # temporal_dim>0 but no temporal_feature -> fail loud


def test_temporal_head_forward_keeps_action_gradient_and_freezes_params():
    """Same freeze-mechanism regression as
    test_frozen_head_forward_keeps_input_gradient_but_not_param_gradient, with
    a non-empty temporal_feature concatenated in: the extra input branch must
    not interfere with the action-direction gradient or the parameter freeze."""
    cfg = ActionRiskConfig(dict(enabled=True, hidden_dim=16, use_temporal_context=True))
    head = ActionRiskHead(latent_dim=8, action_dim=2, cfg=cfg, temporal_dim=5)
    z = torch.randn(4, 8)
    temporal = torch.randn(4, 5)
    action = torch.randn(4, 2, requires_grad=True)

    for p in head.parameters():
        p.requires_grad_(False)
    out = head(z, action, temporal_feature=temporal)
    for p in head.parameters():
        p.requires_grad_(True)

    out.sum().backward()

    assert action.grad is not None and torch.any(action.grad != 0.0), \
        "gradient must still reach the action input"
    assert all(p.grad is None for p in head.parameters()), \
        "the head's own parameters must receive NO gradient from this path"


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


# --------------------------------------------------------------------------- #
#  STAGE 3: curriculum-stage-gated forward pass (enable_from_stage)
# --------------------------------------------------------------------------- #
def test_enable_from_stage_defaults_to_zero_always_active():
    cfg = ActionRiskConfig(dict(enabled=True))
    assert cfg.enable_from_stage == 0


def test_head_only_below_threshold_skips_forward_call_entirely(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0,
                  _hp(action_risk={"enabled": True, "hidden_dim": 32,
                                    "enable_from_stage": 3}),
                  log_dir=str(tmp_path))
    _fill(agent, action_risk_target=True)
    agent.set_curriculum_stage(0)
    assert agent._action_risk_active is False

    calls = []
    real_forward = agent.action_risk_head.forward
    agent.action_risk_head.forward = lambda *a, **kw: (calls.append(1) or real_forward(*a, **kw))
    agent.train()
    assert calls == [], "action_risk_head forward ran below enable_from_stage"

    before = [p.detach().clone() for p in agent.action_risk_head.parameters()]
    for _ in range(5):
        agent.train()
    after = list(agent.action_risk_head.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after)), \
        "action_risk_head must not update below enable_from_stage (no loss term)"


def test_head_only_above_threshold_runs_forward_and_trains(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0,
                  _hp(action_risk={"enabled": True, "hidden_dim": 32,
                                    "enable_from_stage": 3}),
                  log_dir=str(tmp_path))
    _fill(agent, action_risk_target=True)
    agent.set_curriculum_stage(3)
    assert agent._action_risk_active is True

    calls = []
    real_forward = agent.action_risk_head.forward
    agent.action_risk_head.forward = lambda *a, **kw: (calls.append(1) or real_forward(*a, **kw))
    agent.train()
    assert calls, "action_risk_head forward must run at/above enable_from_stage"

    before = [p.detach().clone() for p in agent.action_risk_head.parameters()]
    for _ in range(5):
        agent.train()
    after = list(agent.action_risk_head.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_critic_risk_input_below_threshold_feeds_fixed_zero_extra(tmp_path):
    """critic_risk_input requires action_risk_head, and its extra_dim=2 input
    width must NEVER change with stage -- below enable_from_stage the head's
    forward is skipped but the critic still gets a (batch, 2) all-zero extra
    tensor, not None (which would mismatch Critic(extra_dim=2)'s contract)."""
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0,
                  _hp(action_risk={"enabled": True, "hidden_dim": 32,
                                    "enable_from_stage": 3},
                      critic_risk_input={"enabled": True}),
                  log_dir=str(tmp_path))
    assert agent.critic.extra_dim == 2
    _fill(agent, action_risk_target=True)
    agent.set_curriculum_stage(0)
    assert agent._action_risk_active is False

    calls = []
    real_forward = agent.action_risk_head.forward
    agent.action_risk_head.forward = lambda *a, **kw: (calls.append(1) or real_forward(*a, **kw))
    real_target_forward = agent.action_risk_head_target.forward
    agent.action_risk_head_target.forward = (
        lambda *a, **kw: (calls.append(1) or real_target_forward(*a, **kw)))
    # train() must run to completion (critic forward requires a correctly
    # shaped extra tensor at all 3 call sites) without ever invoking either
    # action_risk_head's forward.
    agent.train()
    assert calls == [], "action_risk_head(_target) forward ran below enable_from_stage"


def test_gate_recomputed_on_set_curriculum_stage_transition(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0,
                  _hp(action_risk={"enabled": True, "hidden_dim": 32,
                                    "enable_from_stage": 3}),
                  log_dir=str(tmp_path))
    agent.set_curriculum_stage(2)
    assert agent._action_risk_active is False
    agent.set_curriculum_stage(3)
    assert agent._action_risk_active is True
    agent.set_curriculum_stage(2)
    assert agent._action_risk_active is False, \
        "gate must re-derive from the CURRENT stage, not latch true forever"


def test_gate_is_false_at_init_before_any_set_curriculum_stage_call_when_threshold_positive(tmp_path):
    # Agent.current_stage defaults to 0; a positive enable_from_stage must
    # leave the gate inactive even if set_curriculum_stage is never called
    # (e.g. a non-curriculum trainer that never invokes it).
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0,
                  _hp(action_risk={"enabled": True, "hidden_dim": 32,
                                    "enable_from_stage": 3}),
                  log_dir=str(tmp_path))
    assert agent._action_risk_active is False


def test_gate_true_at_init_when_threshold_zero_default(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0,
                  _hp(action_risk={"enabled": True, "hidden_dim": 32}),
                  log_dir=str(tmp_path))
    assert agent._action_risk_active is True


# --------------------------------------------------------------------------- #
#  PHASE2 temporal context: Agent-level fail-fast + end-to-end wiring
# --------------------------------------------------------------------------- #
OBS, AGENT_DIM, HIST = 80, 7, 4
CUR = OBS + AGENT_DIM                     # 87
TEMPORAL_STATE_DIM = CUR + (HIST - 1) * OBS  # 327 (stacked obs history)


def _tac(**over):
    tac = dict(enabled=True, history_len=HIST, temporal_feature_dim=32,
               encoder_type="conv1d", stack_agent_state=False, stage_enable_from=0)
    tac.update(over)
    return tac


def test_use_temporal_context_requires_action_risk_head_enabled(tmp_path):
    with pytest.raises(RuntimeError):
        Agent(STATE_DIM, ACTION_DIM, 1.0,
              _hp(action_risk={"enabled": False, "use_temporal_context": True}),
              log_dir=str(tmp_path))


def test_use_temporal_context_requires_temporal_actor_context_enabled(tmp_path):
    """The fail-fast this task specifically requires: use_temporal_context=true
    with temporal_actor_context OFF (the default) must error immediately, not
    silently fall back to the un-augmented [z, action] head."""
    with pytest.raises(RuntimeError):
        Agent(STATE_DIM, ACTION_DIM, 1.0,
              _hp(action_risk={"enabled": True, "use_temporal_context": True}),
              log_dir=str(tmp_path))
    # temporal_actor_context explicitly disabled -> same failure.
    with pytest.raises(RuntimeError):
        Agent(STATE_DIM, ACTION_DIM, 1.0,
              _hp(action_risk={"enabled": True, "use_temporal_context": True},
                  temporal_actor_context={"enabled": False}),
              log_dir=str(tmp_path))


def test_unsupported_temporal_context_source_fails_fast(tmp_path):
    with pytest.raises(RuntimeError):
        Agent(TEMPORAL_STATE_DIM, ACTION_DIM, 1.0,
              _hp(action_risk={"enabled": True, "use_temporal_context": True,
                                "temporal_context_source": "aux"},
                  temporal_actor_context=_tac()),
              log_dir=str(tmp_path),
              env_obs_dim=OBS, env_agent_dim=AGENT_DIM)


def test_temporal_context_off_by_default_even_with_temporal_actor_context_on(tmp_path):
    """action_risk_head.use_temporal_context defaults false -> enabling
    temporal_actor_context alone must NOT change the head's input width."""
    agent = Agent(TEMPORAL_STATE_DIM, ACTION_DIM, 1.0,
                  _hp(action_risk={"enabled": True, "hidden_dim": 32},
                      temporal_actor_context=_tac()),
                  log_dir=str(tmp_path),
                  env_obs_dim=OBS, env_agent_dim=AGENT_DIM)
    assert agent.action_risk_temporal_enabled is False
    assert agent.action_risk_head.temporal_dim == 0
    assert agent.action_risk_head.l1.in_features == agent.encoder.out_dim + ACTION_DIM


def _fill_temporal(agent, n=40, ep_len=10):
    for i in range(n):
        s = np.random.randn(TEMPORAL_STATE_DIM).astype(np.float32)
        a = np.random.uniform(-1, 1, ACTION_DIM).astype(np.float32)
        agent.replay_buffer.add(s, a, s, 0.1, 0.0,
                                 action_risk_target=np.random.rand(2).astype(np.float32))
        if (i + 1) % ep_len == 0:
            agent.replay_buffer.mark_last_traj_end()


def test_temporal_context_on_trains_end_to_end(tmp_path):
    agent = Agent(TEMPORAL_STATE_DIM, ACTION_DIM, 1.0,
                  _hp(action_risk={"enabled": True, "hidden_dim": 32,
                                    "use_temporal_context": True},
                      critic_risk_input={"enabled": True},
                      temporal_actor_context=_tac()),
                  log_dir=str(tmp_path),
                  env_obs_dim=OBS, env_agent_dim=AGENT_DIM)
    assert agent.action_risk_temporal_enabled is True
    assert agent.action_risk_head.temporal_dim == 32
    assert agent.action_risk_head.l1.in_features == agent.encoder.out_dim + 32 + ACTION_DIM
    assert agent.critic.extra_dim == 2   # critic_risk_input width unaffected by temporal_dim

    _fill_temporal(agent, n=40)
    before = [p.detach().clone() for p in agent.action_risk_head.parameters()]
    for _ in range(5):
        agent.train()
    after = list(agent.action_risk_head.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), \
        "temporal action_risk_head did not update from its supervised loss"

    # select_action still takes the stacked 327-D state and returns a 2-D action
    # (the actor's own input width is untouched by the risk-head temporal option).
    a = agent.select_action(np.random.randn(TEMPORAL_STATE_DIM).astype(np.float32))
    assert a.shape == (ACTION_DIM,)


def test_actor_update_gradient_isolation_with_temporal_context(tmp_path):
    """Replicates tqc_agent.py's actor-update block EXACTLY (same real Agent
    objects, same detach/freeze calls) in isolation, so the actor_loss
    gradient's effect on each module can be checked directly without also
    running the critic-trunk backward first (which legitimately DOES update
    the temporal encoder, and would otherwise mask a leak from the actor
    path specifically):
      * agent.actor.parameters()            MUST receive gradient (qf_pi path)
      * agent.action_risk_head.parameters() MUST receive NO gradient (frozen)
      * agent.encoder.temporal.parameters() MUST receive NO gradient (the
        temporal feature fed to action_risk_head in the actor block is
        .detach()-ed, mirroring why z_actor itself is detached from z)
    """
    agent = Agent(TEMPORAL_STATE_DIM, ACTION_DIM, 1.0,
                  _hp(action_risk={"enabled": True, "hidden_dim": 32,
                                    "use_temporal_context": True},
                      critic_risk_input={"enabled": True},
                      temporal_actor_context=_tac()),
                  log_dir=str(tmp_path),
                  env_obs_dim=OBS, env_agent_dim=AGENT_DIM)
    _fill_temporal(agent, n=40)

    agent.critic_optimizer.zero_grad()
    agent.actor_optimizer.zero_grad()

    state, action, next_state, reward, not_done = agent.replay_buffer.sample()
    z = agent.encoder(state)
    z_actor = z.detach()
    actions_pi, log_prob = agent.actor.action_log_prob(z_actor)

    ar_temporal_pi = agent.encoder.temporal_feature(state).detach()
    for p in agent.action_risk_head.parameters():
        p.requires_grad_(False)
    extra_pi = agent.action_risk_head(z_actor, actions_pi, temporal_feature=ar_temporal_pi)
    for p in agent.action_risk_head.parameters():
        p.requires_grad_(True)

    qf_pi = agent.critic(z_actor, actions_pi, extra=extra_pi)
    qf_pi = qf_pi.mean(dim=2).mean(dim=1, keepdim=True)
    actor_loss = (0.2 * log_prob - qf_pi).mean()
    actor_loss.backward()

    assert any(p.grad is not None and torch.any(p.grad != 0.0)
               for p in agent.actor.parameters()), \
        "actor_loss must reach the actor's own parameters"
    assert all(p.grad is None for p in agent.action_risk_head.parameters()), \
        "action_risk_head params must receive NO gradient from actor_loss"
    assert all(p.grad is None for p in agent.encoder.temporal.parameters()), \
        "temporal encoder must receive NO gradient from actor_loss (leaked " \
        "through the un-detached temporal_feature side-input)"


def test_temporal_checkpoint_incompatibility_falls_back_gracefully(tmp_path):
    """'기존 checkpoint와 최대한 호환되게 하되, 입력 차원이 바뀌는 경우
    fresh-run-only 경고를 명확히 남긴다': a checkpoint saved WITHOUT temporal
    context must NOT crash a resume into a run WITH it enabled -- tqc_io.load's
    existing graceful-mismatch handling (shape mismatch -> fresh-init just the
    head + rebuild the critic optimizer) must cover this case too."""
    a1 = Agent(TEMPORAL_STATE_DIM, ACTION_DIM, 1.0,
               _hp(action_risk={"enabled": True, "hidden_dim": 32},
                   temporal_actor_context=_tac()),
               log_dir=str(tmp_path / "a"),
               env_obs_dim=OBS, env_agent_dim=AGENT_DIM)
    _fill_temporal(a1, n=16)
    a1.train()
    a1.save(str(tmp_path), "ckpt")
    assert a1.action_risk_head.temporal_dim == 0

    a2 = Agent(TEMPORAL_STATE_DIM, ACTION_DIM, 1.0,
               _hp(action_risk={"enabled": True, "hidden_dim": 32,
                                 "use_temporal_context": True},
                   temporal_actor_context=_tac()),
               log_dir=str(tmp_path / "b"),
               env_obs_dim=OBS, env_agent_dim=AGENT_DIM)
    assert a2.action_risk_head.temporal_dim == 32
    fresh_before = [p.detach().clone() for p in a2.action_risk_head.parameters()]
    fresh_target_before = [p.detach().clone()
                            for p in a2.action_risk_head_target.parameters()]

    a2.load(str(tmp_path), "ckpt", load_replay_buffer=False)  # must NOT raise

    # tqc_io.load() snapshots the pre-load state and restores it verbatim on a
    # RuntimeError, so a shape mismatch (e.g. l1.weight: 89 vs 121 in_features)
    # must leave EVERY parameter -- not just the resized one -- exactly as this
    # run's own fresh init. Without that snapshot/restore, torch's strict
    # load_state_dict copies matching-shape tensors (l1.bias/out.*) in place
    # BEFORE raising, silently hybridising "fresh" head with a1's incompatible
    # checkpoint values -- this is the regression this test locks.
    fresh_after = list(a2.action_risk_head.parameters())
    assert all(torch.equal(b, a) for b, a in zip(fresh_before, fresh_after)), \
        "a shape-mismatched checkpoint must leave the head FULLY fresh-init, " \
        "not a hybrid of fresh + partially-loaded incompatible weights"
    target_after = list(a2.action_risk_head_target.parameters())
    assert all(torch.equal(b, a) for b, a in zip(fresh_target_before, target_after)), \
        "the TARGET head must also stay fully fresh-init on a mismatch"
    assert a2.action_risk_head.temporal_dim == 32  # architecture stays as configured

    # The rest of the run is still usable (actor/critic/encoder resumed).
    _fill_temporal(a2, n=16)
    a2.train()
