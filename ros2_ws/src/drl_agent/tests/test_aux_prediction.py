"""Unit tests for the auxiliary-prediction networks (AUX_PRED), torch-gated.

Covers the strengthened aux branch:
  * AuxPredConfig parses the new aux-branch keys (attention + deeper trunk).
  * SharedEncoder identity vs enabled output dims (actor/critic contract).
  * AuxiliaryHead / ActionConditionedAuxHead forward shapes incl. the optional
    min-distance + distributional heads.
  * compute_aux_loss over the canonical label, with every head enabled.
  * BOUNDARY-SAFETY CONTRACT: actions past ``valid_len`` never influence the
    action-conditioned prediction — verified for the GRU-only path AND the new
    attention path (zero gradient + output invariance to padded actions).
  * Gradient isolation surrogate: the aux head + encoder receive aux gradients.
  * state_dict save/load round-trip for the attention head.

torch is imported defensively (a module-level Skipped can abort directory
collection under the ament pytest plugin stack), matching tests/test_tqc_io.py.
"""

import pytest

try:
    import numpy as np
    import torch

    import aux_prediction as ap
    import aux_prediction_losses as apl
    import aux_prediction_temporal as apt
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

pytestmark = pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _cfg(**over):
    """Curriculum-like aux config with the strengthened aux branch on."""
    base = dict(
        enabled=True,
        latent_dim=128,
        encoder_hidden_dim=256,
        num_sectors=16,
        horizons_sec=[0.5, 1.0, 1.5],
        loss_weight=0.1,
        min_distance_loss_weight=0.1,
        use_distributional_aux=False,
        aux_trunk_hidden_dim=256,
        aux_trunk_layers=2,
        aux_head_layernorm=True,
        action_conditioned_aux=True,
        action_conditioned_steps=4,
        action_embed_dim=64,
        action_condition_hidden_dim=256,
        action_condition_attention=True,
        action_condition_attention_heads=4,
    )
    base.update(over)
    return ap.AuxPredConfig(base)


STATE_DIM = 87
ACTION_DIM = 2


# --------------------------------------------------------------------------- #
#  config
# --------------------------------------------------------------------------- #
def test_config_parses_new_aux_branch_keys():
    c = _cfg()
    assert c.action_embed_dim == 64
    assert c.action_condition_hidden_dim == 256
    assert c.action_condition_attention is True
    assert c.action_condition_attention_heads == 4
    assert c.aux_trunk_hidden_dim == 256
    assert c.aux_trunk_layers == 2
    assert c.aux_head_layernorm is True
    assert c.min_distance_enabled is True
    assert c.risk_dim == 16 * 3
    assert c.label_dim == 16 * 3 + 3


def test_config_defaults_preserve_legacy_trunk():
    """An old-style config (no new keys) reproduces the original 1-layer,
    no-LayerNorm, hidden=max(latent,128) trunk and no attention."""
    c = ap.AuxPredConfig(dict(enabled=True, latent_dim=128))
    assert c.aux_trunk_layers == 1
    assert c.aux_head_layernorm is False
    assert c.aux_trunk_hidden_dim == 128
    assert c.action_condition_attention is False


# --------------------------------------------------------------------------- #
#  encoder
# --------------------------------------------------------------------------- #
def test_shared_encoder_identity_when_disabled():
    c = ap.AuxPredConfig(dict(enabled=False))
    enc = ap.SharedEncoder(STATE_DIM, c)
    assert enc.has_params() is False
    assert enc.out_dim == STATE_DIM
    x = torch.randn(5, STATE_DIM)
    assert torch.equal(enc(x), x)            # byte-for-byte passthrough


def test_shared_encoder_latent_dims():
    enc = ap.SharedEncoder(STATE_DIM, _cfg())
    assert enc.has_params() is True
    assert enc.out_dim == 128
    z = enc(torch.randn(5, STATE_DIM))
    assert z.shape == (5, 128)


# --------------------------------------------------------------------------- #
#  heads forward shapes
# --------------------------------------------------------------------------- #
def test_plain_aux_head_emits_all_enabled_outputs():
    c = _cfg(use_distributional_aux=True)
    head = ap.AuxiliaryHead(c.latent_dim, c)
    out = head(torch.randn(7, c.latent_dim))
    assert out["risk_map"].shape == (7, c.risk_dim)
    assert out["min_dist"].shape == (7, c.num_horizons)
    assert out["risk_quant"].shape == (7, c.risk_dim, c.num_quantiles)
    assert torch.all((out["risk_map"] >= 0) & (out["risk_map"] <= 1))


@pytest.mark.parametrize("attention", [False, True])
def test_action_conditioned_head_forward(attention):
    c = _cfg(action_condition_attention=attention)
    head = ap.ActionConditionedAuxHead(c.latent_dim, ACTION_DIM, c)
    b, k = 7, c.action_conditioned_steps
    z = torch.randn(b, c.latent_dim)
    fa = torch.randn(b, k, ACTION_DIM)
    vl = torch.randint(1, k + 1, (b,))
    out = head(z, fa, vl)
    assert out["risk_map"].shape == (b, c.risk_dim)
    assert out["min_dist"].shape == (b, c.num_horizons)
    assert (head.attention is not None) == attention


# --------------------------------------------------------------------------- #
#  loss
# --------------------------------------------------------------------------- #
def test_compute_aux_loss_with_all_terms():
    c = _cfg(use_distributional_aux=True)
    head = ap.ActionConditionedAuxHead(c.latent_dim, ACTION_DIM, c)
    b, k = 6, c.action_conditioned_steps
    z = torch.randn(b, c.latent_dim)
    fa = torch.randn(b, k, ACTION_DIM)
    vl = torch.full((b,), k, dtype=torch.long)
    pred = head(z, fa, vl)
    label = torch.rand(b, c.label_dim)
    loss, logs = apl.compute_aux_loss(pred, label, c, torch.device("cpu"))
    assert loss.requires_grad
    assert "aux/risk_mse" in logs
    assert "aux/min_dist_mse" in logs        # min_distance_loss_weight > 0
    assert "aux/risk_quantile" in logs       # use_distributional_aux
    assert np.isfinite(logs["aux/loss"])


# --------------------------------------------------------------------------- #
#  BOUNDARY SAFETY (the buffer contract): padded actions never affect output
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("attention", [False, True])
def test_padded_actions_do_not_change_output(attention):
    c = _cfg(action_condition_attention=attention)
    head = ap.ActionConditionedAuxHead(c.latent_dim, ACTION_DIM, c).eval()
    b, k = 4, c.action_conditioned_steps
    z = torch.randn(b, c.latent_dim)
    vl = torch.full((b,), 2, dtype=torch.long)   # only first 2 actions valid
    fa = torch.randn(b, k, ACTION_DIM)
    fa2 = fa.clone()
    fa2[:, 2:, :] = torch.randn(b, k - 2, ACTION_DIM)  # perturb ONLY padded steps
    with torch.no_grad():
        o1 = head(z, fa, vl)["risk_map"]
        o2 = head(z, fa2, vl)["risk_map"]
    assert torch.allclose(o1, o2, atol=1e-6), "out-of-episode actions leaked into output"


@pytest.mark.parametrize("attention", [False, True])
def test_padded_actions_have_zero_gradient(attention):
    c = _cfg(action_condition_attention=attention)
    head = ap.ActionConditionedAuxHead(c.latent_dim, ACTION_DIM, c)
    b, k = 4, c.action_conditioned_steps
    z = torch.randn(b, c.latent_dim)
    vl = torch.full((b,), 2, dtype=torch.long)
    fa = torch.randn(b, k, ACTION_DIM, requires_grad=True)
    head(z, fa, vl)["risk_map"].sum().backward()
    g = fa.grad
    assert g is not None
    assert g[:, :2, :].abs().sum() > 0            # valid steps DO carry gradient
    assert torch.count_nonzero(g[:, 2:, :]) == 0  # padded steps carry NONE


# --------------------------------------------------------------------------- #
#  gradient flow into aux head + encoder (trunk-update contract)
# --------------------------------------------------------------------------- #
def test_aux_loss_updates_encoder_and_head_not_actor_path():
    c = _cfg()
    enc = ap.SharedEncoder(STATE_DIM, c)
    head = ap.ActionConditionedAuxHead(c.latent_dim, ACTION_DIM, c)
    b, k = 5, c.action_conditioned_steps
    s = torch.randn(b, STATE_DIM)
    fa = torch.randn(b, k, ACTION_DIM)
    vl = torch.full((b,), k, dtype=torch.long)
    label = torch.rand(b, c.label_dim)

    z = enc(s)
    # actor would read z.detach(); aux reads the graphed z (tqc_agent contract).
    pred = head(z, fa, vl)
    loss, _ = apl.compute_aux_loss(pred, label, c, torch.device("cpu"))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    # z.detach() severs the encoder from any downstream (actor) consumer.
    assert z.detach().requires_grad is False


# --------------------------------------------------------------------------- #
#  save / load round-trip (attention head)
# --------------------------------------------------------------------------- #
def test_attention_head_state_dict_roundtrip():
    c = _cfg()
    h1 = ap.ActionConditionedAuxHead(c.latent_dim, ACTION_DIM, c).eval()
    h2 = ap.ActionConditionedAuxHead(c.latent_dim, ACTION_DIM, c).eval()
    h2.load_state_dict(h1.state_dict())
    b, k = 3, c.action_conditioned_steps
    z = torch.randn(b, c.latent_dim)
    fa = torch.randn(b, k, ACTION_DIM)
    vl = torch.full((b,), k, dtype=torch.long)
    with torch.no_grad():
        assert torch.allclose(h1(z, fa, vl)["risk_map"], h2(z, fa, vl)["risk_map"])


def test_param_count_summary(capsys):
    """Not an assertion of an exact number — emits the block sizes so the
    capacity change is visible in -s runs and guards against an empty head."""
    c = _cfg()
    enc = ap.SharedEncoder(STATE_DIM, c)
    head = ap.ActionConditionedAuxHead(c.latent_dim, ACTION_DIM, c)
    n_enc = sum(p.numel() for p in enc.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    print(f"\n[aux params] encoder={n_enc}  action_cond_head={n_head}")
    assert n_enc > 0 and n_head > 0


# --------------------------------------------------------------------------- #
#  v2 aux-only TEMPORAL context (GRU over recent state history)
# --------------------------------------------------------------------------- #
def _tcfg(**over):
    """Temporal-enabled config (attention on by default for the harder path)."""
    base = dict(
        temporal_enabled=True,
        history_len=4,
        temporal_context_dim=64,
        temporal_attention=True,
        action_condition_attention_heads=4,
    )
    base.update(over)
    return _cfg(**base)


def test_config_parses_temporal_and_loss_keys():
    c = _tcfg()
    assert c.temporal_enabled is True
    assert c.history_len == 4
    assert c.temporal_context_dim == 64
    assert c.temporal_attention is True
    assert c.aux_beta_warmup_steps == 0          # default off
    assert c.distributional_loss_weight == 1.0   # default keeps prior semantics
    assert c.encoder_layernorm is False
    c2 = _cfg(aux_beta_warmup_steps=5000, distributional_loss_weight=0.5)
    assert c2.aux_beta_warmup_steps == 5000
    assert c2.distributional_loss_weight == 0.5


def test_temporal_encoder_disabled_is_zero_width():
    c = _cfg(temporal_enabled=False)
    te = apt.TemporalContextEncoder(c.latent_dim, c)
    assert te.out_dim == 0


def test_temporal_encoder_forward_shape():
    c = _tcfg()
    te = apt.TemporalContextEncoder(c.latent_dim, c)
    assert te.out_dim == 64
    b, n = 5, c.history_len
    z_seq = torch.randn(b, n, c.latent_dim)
    vl = torch.randint(1, n + 1, (b,))
    ctx = te(z_seq, vl)
    assert ctx.shape == (b, 64)


@pytest.mark.parametrize("attention", [False, True])
def test_temporal_padded_history_does_not_change_output(attention):
    """Boundary-safety: history steps past valid_len never affect the context."""
    c = _tcfg(temporal_attention=attention)
    te = apt.TemporalContextEncoder(c.latent_dim, c).eval()
    b, n = 4, c.history_len
    vl = torch.full((b,), 2, dtype=torch.long)   # only first 2 history steps valid
    z = torch.randn(b, n, c.latent_dim)
    z2 = z.clone()
    z2[:, 2:, :] = torch.randn(b, n - 2, c.latent_dim)   # perturb ONLY padded steps
    with torch.no_grad():
        assert torch.allclose(te(z, vl), te(z2, vl), atol=1e-6)


@pytest.mark.parametrize("attention", [False, True])
def test_temporal_padded_history_zero_gradient(attention):
    c = _tcfg(temporal_attention=attention)
    te = apt.TemporalContextEncoder(c.latent_dim, c)
    b, n = 4, c.history_len
    vl = torch.full((b,), 2, dtype=torch.long)
    z = torch.randn(b, n, c.latent_dim, requires_grad=True)
    # pow(2).sum(), NOT sum(): with temporal_attention the context exits a
    # LayerNorm (zero-mean rows), so a plain sum() cancels to ~0 and would give a
    # degenerate (near-zero) gradient even for valid steps. A quadratic probe
    # gives every valid input a real gradient while padded inputs still get none.
    te(z, vl).pow(2).sum().backward()
    g = z.grad
    assert g[:, :2, :].abs().sum() > 0            # valid steps carry gradient
    assert torch.count_nonzero(g[:, 2:, :]) == 0  # padded steps carry NONE


@pytest.mark.parametrize("action_cond", [False, True])
def test_head_accepts_temporal_context(action_cond):
    """Both heads concatenate the temporal context and keep the output shape."""
    c = _tcfg()
    te = apt.TemporalContextEncoder(c.latent_dim, c)
    b, n, k = 6, c.history_len, c.action_conditioned_steps
    z = torch.randn(b, c.latent_dim)
    z_seq = torch.randn(b, n, c.latent_dim)
    vl_hist = torch.full((b,), n, dtype=torch.long)
    tctx = te(z_seq, vl_hist)
    if action_cond:
        head = ap.ActionConditionedAuxHead(
            c.latent_dim, ACTION_DIM, c, temporal_dim=te.out_dim)
        fa = torch.randn(b, k, ACTION_DIM)
        vl = torch.full((b,), k, dtype=torch.long)
        out = head(z, fa, vl, temporal_ctx=tctx)
    else:
        head = ap.AuxiliaryHead(c.latent_dim, c, temporal_dim=te.out_dim)
        out = head(z, temporal_ctx=tctx)
    assert out["risk_map"].shape == (b, c.risk_dim)


def test_temporal_branch_grad_flows_into_encoder_and_temporal():
    """The temporal aux path shapes BOTH the shared encoder and the temporal GRU
    (it shares the encoder graph), but never the actor (z.detach() contract)."""
    c = _tcfg()
    enc = ap.SharedEncoder(STATE_DIM, c)
    te = apt.TemporalContextEncoder(c.latent_dim, c)
    head = ap.AuxiliaryHead(c.latent_dim, c, temporal_dim=te.out_dim)
    b, n = 5, c.history_len
    hist = torch.randn(b, n, STATE_DIM)
    vl = torch.full((b,), n, dtype=torch.long)
    z_seq = enc(hist.reshape(b * n, STATE_DIM)).reshape(b, n, -1)
    z = z_seq[:, 0]                       # current-state latent (graphed)
    tctx = te(z_seq, vl)
    label = torch.rand(b, c.label_dim)
    loss, _ = apl.compute_aux_loss(head(z, temporal_ctx=tctx), label,
                                   c, torch.device("cpu"))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in te.parameters())


def test_temporal_encoder_state_dict_roundtrip():
    c = _tcfg()
    t1 = apt.TemporalContextEncoder(c.latent_dim, c).eval()
    t2 = apt.TemporalContextEncoder(c.latent_dim, c).eval()
    t2.load_state_dict(t1.state_dict())
    b, n = 3, c.history_len
    z_seq = torch.randn(b, n, c.latent_dim)
    vl = torch.full((b,), n, dtype=torch.long)
    with torch.no_grad():
        assert torch.allclose(t1(z_seq, vl), t2(z_seq, vl))


# --------------------------------------------------------------------------- #
#  shared encoder: optional LayerNorm keeps the actor/critic width contract
# --------------------------------------------------------------------------- #
def test_encoder_layernorm_keeps_out_dim_and_default_is_byte_identical():
    # LayerNorm ON -> still out_dim == latent_dim (actor/critic input width fixed),
    # but more parameters than the plain encoder.
    c_ln = _cfg(encoder_layernorm=True)
    enc_ln = ap.SharedEncoder(STATE_DIM, c_ln)
    assert enc_ln.out_dim == c_ln.latent_dim
    c_plain = _cfg(encoder_layernorm=False)
    enc_plain = ap.SharedEncoder(STATE_DIM, c_plain)
    assert sum(p.numel() for p in enc_ln.parameters()) > \
        sum(p.numel() for p in enc_plain.parameters())
    # Disabled aux -> identity passthrough is unaffected by the new flag.
    enc_id = ap.SharedEncoder(STATE_DIM, ap.AuxPredConfig(
        dict(enabled=False, encoder_layernorm=True)))
    x = torch.randn(4, STATE_DIM)
    assert torch.equal(enc_id(x), x)


# --------------------------------------------------------------------------- #
#  loss balancing: distributional term is actually weighted
# --------------------------------------------------------------------------- #
def test_distributional_loss_weight_scales_term():
    c = _cfg(use_distributional_aux=True, min_distance_loss_weight=0.0,
             distributional_loss_weight=0.0)
    head = ap.AuxiliaryHead(c.latent_dim, c)
    b = 6
    z = torch.randn(b, c.latent_dim)
    pred = head(z)
    label = torch.rand(b, c.label_dim)
    # weight 0 -> total loss == risk MSE alone (term present but contributes 0).
    total, logs = apl.compute_aux_loss(pred, label, c, torch.device("cpu"))
    assert "aux/risk_quantile" in logs
    assert abs(float(total.detach()) - logs["aux/risk_mse"]) < 1e-6
