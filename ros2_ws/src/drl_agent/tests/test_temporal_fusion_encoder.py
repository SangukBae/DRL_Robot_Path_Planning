"""Unit tests for the actor/critic-facing temporal path (TEMPORAL_ACTOR), torch-gated.

Locks the contract that the raw stacked state is SPLIT (current 87-D on the main
path + scan history compressed by a small encoder) and fused back to latent_dim,
so the actor/critic never ingest the raw 327-D stack, and the temporal strength is
gated per stage WITHOUT changing any tensor shape.

torch imported defensively (a module-level Skipped can abort directory collection
under the ament pytest plugin stack), matching tests/test_aux_prediction.py.
"""

import pytest

try:
    import numpy as np
    import torch
    import torch.nn.functional as F

    import drl_agent.rl.networks.aux_prediction as ap
    import drl_agent.rl.networks.aux_temporal as apt
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

pytestmark = pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")

OBS, AGENT, N = 80, 7, 4
CUR = OBS + AGENT                      # 87
STATE_DIM = CUR + (N - 1) * OBS        # 327
LATENT = 128


def _aux_cfg():
    return ap.AuxPredConfig(dict(enabled=True, latent_dim=LATENT, encoder_hidden_dim=256))


def _enc(encoder_type="conv1d", feature_dim=32):
    return apt.TemporalFusionEncoder(
        STATE_DIM, OBS, AGENT, history_len=N, stack_agent_state=False,
        aux_cfg=_aux_cfg(), feature_dim=feature_dim, encoder_type=encoder_type)


def _make_state(b=3):
    """Stacked state with a DISTINCT constant per frame so the split is checkable:
    obs_t=1.0, agent_t=2.0, obs_{t-1}=3.0, obs_{t-2}=4.0, obs_{t-3}=5.0."""
    s = torch.zeros(b, STATE_DIM)
    s[:, :OBS] = 1.0                       # obs_t
    s[:, OBS:CUR] = 2.0                    # agent_t
    s[:, CUR:CUR + OBS] = 3.0             # obs_{t-1}
    s[:, CUR + OBS:CUR + 2 * OBS] = 4.0  # obs_{t-2}
    s[:, CUR + 2 * OBS:CUR + 3 * OBS] = 5.0  # obs_{t-3}
    return s


def test_expected_state_dim_and_out_dim():
    enc = _enc()
    assert enc.expected_state_dim() == STATE_DIM
    assert enc.out_dim == LATENT            # actor/critic width unchanged
    assert enc.has_params() is True


def test_split_routes_current_and_scan_history_chronologically():
    enc = _enc()
    cur, scan = enc._split(_make_state(b=2))
    assert cur.shape == (2, CUR)
    assert scan.shape == (2, N, OBS)
    # current = [obs_t, agent_t]
    assert torch.all(cur[:, :OBS] == 1.0) and torch.all(cur[:, OBS:] == 2.0)
    # scan is CHRONOLOGICAL: oldest (obs_{t-3}=5) ... newest (obs_t=1)
    assert torch.all(scan[:, 0] == 5.0)
    assert torch.all(scan[:, 1] == 4.0)
    assert torch.all(scan[:, 2] == 3.0)
    assert torch.all(scan[:, 3] == 1.0)


def test_forward_shape_is_latent_dim():
    for et in ("mlp", "conv1d", "gru"):
        out = _enc(et)(_make_state(b=5))
        assert out.shape == (5, LATENT), et


def test_scan_temporal_encoder_feature_dim():
    for et in ("mlp", "conv1d", "gru"):
        te = apt.ScanTemporalEncoder(N, OBS, feature_dim=32, encoder_type=et)
        out = te(torch.randn(6, N, OBS))
        assert out.shape == (6, 32), et


def test_gain_zero_makes_history_irrelevant():
    enc = _enc().eval()
    enc.set_temporal_gain(0.0)
    base = _make_state(b=4)
    other = base.clone()
    other[:, CUR:] = torch.randn(4, STATE_DIM - CUR)   # perturb ONLY history
    with torch.no_grad():
        o1 = enc(base)
        o2 = enc(other)
    assert torch.allclose(o1, o2, atol=1e-6), "history leaked at temporal_gain=0"
    # The CURRENT state still drives the output even at gain 0 (main path active).
    cur_changed = base.clone()
    cur_changed[:, :CUR] = torch.randn(4, CUR)
    with torch.no_grad():
        o3 = enc(cur_changed)
    assert not torch.allclose(o1, o3, atol=1e-6)


def test_gain_one_uses_history():
    enc = _enc().eval()
    enc.set_temporal_gain(1.0)
    base = _make_state(b=4)
    other = base.clone()
    other[:, CUR:] = base[:, CUR:] + 1.0               # change history only
    with torch.no_grad():
        o1, o2 = enc(base), enc(other)
    assert not torch.allclose(o1, o2, atol=1e-6), "history ignored at temporal_gain=1"


def test_gain_zero_skips_the_temporal_encoder_forward_call():
    # STAGE 3: gain==0 must SKIP self.temporal(scan) entirely (not compute it
    # and multiply by 0) -- verified by spying on the actual forward call, not
    # just checking the output value.
    enc = _enc().eval()
    enc.set_temporal_gain(0.0)
    calls = []
    real_forward = enc.temporal.forward
    enc.temporal.forward = lambda *a, **kw: (calls.append(1) or real_forward(*a, **kw))
    with torch.no_grad():
        enc(_make_state(b=2))
    assert calls == [], "temporal encoder forward was called despite gain==0"


def test_gain_one_does_not_skip_the_temporal_encoder_forward_call():
    enc = _enc().eval()
    enc.set_temporal_gain(1.0)
    calls = []
    real_forward = enc.temporal.forward
    enc.temporal.forward = lambda *a, **kw: (calls.append(1) or real_forward(*a, **kw))
    with torch.no_grad():
        enc(_make_state(b=2))
    assert calls == [1], "temporal encoder forward must run when gain==1"


def test_skip_and_compute_then_zero_are_numerically_equivalent():
    # Output/gradient-contract equivalence: an encoder using the skip path
    # (gain==0) must produce IDENTICAL forward output to one that computes
    # self.temporal(scan) and multiplies by a gain frozen at exactly 0.0 --
    # proving the optimization changes only WHETHER the compute runs, never
    # the resulting values seen downstream by the fusion/actor/critic.
    torch.manual_seed(0)
    enc_skip = _enc().eval()
    enc_skip.set_temporal_gain(0.0)
    enc_ref = _enc().eval()
    enc_ref.load_state_dict(enc_skip.state_dict())
    state = _make_state(b=4)
    with torch.no_grad():
        out_skip = enc_skip(state)
        # Reference: force the OLD always-compute-then-scale path explicitly.
        cur, scan = enc_ref._split(state)
        z_cur = enc_ref.main(cur)
        z_temporal_ref = enc_ref.temporal(scan) * 0.0
        out_ref = F.elu(enc_ref.fusion(torch.cat([z_cur, z_temporal_ref], dim=-1)))
    assert torch.allclose(out_skip, out_ref, atol=1e-6)


def test_temporal_feature_also_skips_forward_call_at_gain_zero():
    enc = _enc().eval()
    enc.set_temporal_gain(0.0)
    calls = []
    real_forward = enc.temporal.forward
    enc.temporal.forward = lambda *a, **kw: (calls.append(1) or real_forward(*a, **kw))
    with torch.no_grad():
        out = enc.temporal_feature(_make_state(b=2))
    assert calls == []
    assert torch.equal(out, torch.zeros(2, enc.temporal.out_dim))


def test_gain_zero_gives_temporal_encoder_zero_gradient():
    enc = _enc()
    enc.set_temporal_gain(0.0)
    out = enc(_make_state(b=3))
    out.sum().backward()
    for p in enc.temporal.parameters():
        if p.grad is not None:
            assert torch.allclose(p.grad, torch.zeros_like(p.grad)), \
                "temporal encoder must get zero gradient when gain==0"
    # main path still receives gradient.
    main_grads = [p.grad for p in enc.main.parameters() if p.grad is not None]
    assert main_grads and any(g.abs().sum() > 0 for g in main_grads)


def test_state_dict_roundtrip():
    a = _enc()
    b = _enc()
    b.load_state_dict(a.state_dict())     # same structure -> strict load OK
    a.eval(); b.eval()
    s = _make_state(b=2)
    with torch.no_grad():
        assert torch.allclose(a(s), b(s), atol=1e-6)


# ── End-to-end: Agent wired with temporal_actor_context ──────────────────────
def _agent_hp():
    aux = dict(enabled=True, latent_dim=LATENT, encoder_hidden_dim=256,
               num_sectors=16, horizons_sec=[0.5, 1.0, 1.5],
               loss_weight=0.1, min_distance_loss_weight=0.1,
               stagewise_loss_schedule=[0.0, 0.0, 0.1, 0.3, 0.5],
               action_conditioned_aux=False, temporal_enabled=False)
    tac = dict(enabled=True, history_len=N, temporal_feature_dim=32,
               encoder_type="conv1d", stack_agent_state=False, stage_enable_from=2)
    return dict(batch_size=8, buffer_size=2000, n_critics=2, n_quantiles=5,
                aux_prediction=aux, temporal_actor_context=tac)


def _fill(agent, n=40, ep_len=10):
    for i in range(n):
        s = np.random.randn(STATE_DIM).astype(np.float32)
        a = np.random.uniform(-1, 1, 2).astype(np.float32)
        agent.replay_buffer.add(s, a, s, 0.1, 0.0,
                                aux_target=np.random.rand(16 * 3 + 3).astype(np.float32))
        if (i + 1) % ep_len == 0:
            agent.replay_buffer.mark_last_traj_end()


def test_agent_builds_fusion_encoder_and_actor_width_unchanged():
    from drl_agent.rl.algorithms.tqc.agent import Agent
    agent = Agent(STATE_DIM, 2, 1.0, _agent_hp(), env_obs_dim=OBS, env_agent_dim=AGENT)
    assert isinstance(agent.encoder, apt.TemporalFusionEncoder)
    assert agent.encoder.out_dim == LATENT
    # actor/critic consume the fused latent (width unchanged), NOT raw 327-D.
    assert agent.actor.l1.in_features == LATENT
    # select_action takes the stacked 327-D state and returns a 2-D action.
    a = agent.select_action(np.random.randn(STATE_DIM).astype(np.float32))
    assert a.shape == (2,)


def test_agent_stage_gating_controls_temporal_gradient():
    from drl_agent.rl.algorithms.tqc.agent import Agent
    agent = Agent(STATE_DIM, 2, 1.0, _agent_hp(), env_obs_dim=OBS, env_agent_dim=AGENT)
    _fill(agent)

    # Stage 0: temporal gain 0 -> temporal encoder must receive NO gradient.
    agent.set_curriculum_stage(0)
    assert agent.encoder.temporal_gain == 0.0
    t_before = [p.detach().clone() for p in agent.encoder.temporal.parameters()]
    for _ in range(5):
        agent.train()
    t_after = list(agent.encoder.temporal.parameters())
    assert all(torch.equal(b, a) for b, a in zip(t_before, t_after)), \
        "temporal encoder changed at stage 0 (gain should be 0)"

    # Stage 2: gain 1 -> temporal encoder now learns; aux beta follows schedule.
    agent.set_curriculum_stage(2)
    assert agent.encoder.temporal_gain == 1.0
    assert agent._current_aux_beta() == 0.1
    t_before2 = [p.detach().clone() for p in agent.encoder.temporal.parameters()]
    for _ in range(5):
        agent.train()
    t_after2 = list(agent.encoder.temporal.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(t_before2, t_after2)), \
        "temporal encoder did not update at stage 2 (gain 1)"


# --------------------------------------------------------------------------- #
#  STAGE 3: aux forward pass skipped entirely when the effective beta is 0
#  (not just its loss contribution zeroed) -- _agent_hp()'s
#  stagewise_loss_schedule=[0.0, 0.0, 0.1, 0.3, 0.5] makes stage 0/1 beta==0.
# --------------------------------------------------------------------------- #
def test_aux_head_forward_skipped_when_stagewise_beta_is_zero():
    from drl_agent.rl.algorithms.tqc.agent import Agent
    agent = Agent(STATE_DIM, 2, 1.0, _agent_hp(), env_obs_dim=OBS, env_agent_dim=AGENT)
    _fill(agent)
    agent.set_curriculum_stage(0)
    assert agent._current_aux_beta() == 0.0

    calls = []
    real_forward = agent.aux_head.forward
    agent.aux_head.forward = lambda *a, **kw: (calls.append(1) or real_forward(*a, **kw))
    agent.train()
    assert calls == [], "aux_head forward ran despite beta==0 for this stage"


def test_aux_head_forward_runs_when_stagewise_beta_is_nonzero():
    from drl_agent.rl.algorithms.tqc.agent import Agent
    agent = Agent(STATE_DIM, 2, 1.0, _agent_hp(), env_obs_dim=OBS, env_agent_dim=AGENT)
    _fill(agent)
    agent.set_curriculum_stage(2)
    assert agent._current_aux_beta() == 0.1

    calls = []
    real_forward = agent.aux_head.forward
    agent.aux_head.forward = lambda *a, **kw: (calls.append(1) or real_forward(*a, **kw))
    agent.train()
    assert calls, "aux_head forward must run when beta != 0"


def test_aux_head_params_do_not_update_while_beta_is_zero():
    from drl_agent.rl.algorithms.tqc.agent import Agent
    agent = Agent(STATE_DIM, 2, 1.0, _agent_hp(), env_obs_dim=OBS, env_agent_dim=AGENT)
    _fill(agent)
    agent.set_curriculum_stage(0)
    before = [p.detach().clone() for p in agent.aux_head.parameters()]
    for _ in range(5):
        agent.train()
    after = list(agent.aux_head.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after)), \
        "aux_head must not update while beta==0 (no loss term reaches it)"


def test_agent_state_dim_mismatch_fails_fast():
    from drl_agent.rl.algorithms.tqc.agent import Agent
    with pytest.raises(RuntimeError, match="state_dim"):
        Agent(STATE_DIM + 5, 2, 1.0, _agent_hp(), env_obs_dim=OBS, env_agent_dim=AGENT)


def test_agent_save_load_roundtrip(tmp_path):
    from drl_agent.rl.algorithms.tqc.agent import Agent
    a = Agent(STATE_DIM, 2, 1.0, _agent_hp(), env_obs_dim=OBS, env_agent_dim=AGENT)
    _fill(a)
    a.set_curriculum_stage(2)
    a.train()
    a.save(str(tmp_path), "ckpt")
    b = Agent(STATE_DIM, 2, 1.0, _agent_hp(), env_obs_dim=OBS, env_agent_dim=AGENT)
    b.load(str(tmp_path), "ckpt")
    b.set_curriculum_stage(2)
    s = np.random.randn(STATE_DIM).astype(np.float32)
    assert b.select_action(s, use_exploration=False).shape == (2,)


def test_agent_temporal_only_ablation_without_aux():
    """Decoupled ablation: temporal_actor_context ON with aux_prediction OFF. The
    fusion encoder still builds (main = identity 87 -> fused width 87, no aux head)
    and the temporal+fusion params train from the CRITIC loss alone."""
    from drl_agent.rl.algorithms.tqc.agent import Agent
    hp = _agent_hp()
    hp["aux_prediction"] = dict(enabled=False)      # aux supervision OFF
    agent = Agent(STATE_DIM, 2, 1.0, hp, env_obs_dim=OBS, env_agent_dim=AGENT)
    assert isinstance(agent.encoder, apt.TemporalFusionEncoder)
    assert agent.aux_head is None                    # no aux supervision
    assert agent.encoder.out_dim == CUR             # identity main -> fused 87
    assert agent.actor.l1.in_features == CUR
    # Fill WITHOUT aux labels (aux_dim == 0).
    for i in range(40):
        s = np.random.randn(STATE_DIM).astype(np.float32)
        a = np.random.uniform(-1, 1, 2).astype(np.float32)
        agent.replay_buffer.add(s, a, s, 0.1, 0.0)
    agent.set_curriculum_stage(2)                    # temporal gain 1
    t_before = [p.detach().clone() for p in agent.encoder.temporal.parameters()]
    for _ in range(5):
        agent.train()
    t_after = list(agent.encoder.temporal.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(t_before, t_after)), \
        "temporal encoder must train from the critic loss even without aux"
    assert agent.select_action(np.random.randn(STATE_DIM).astype(np.float32)).shape == (2,)
