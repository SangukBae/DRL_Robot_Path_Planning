# AUX_PRED: shared encoder + auxiliary prediction head for TQC.
#
# This module isolates every network piece that the auxiliary prediction
# feature adds on top of the baseline TQC agent (tqc_agent.py).  It is built so
# that disabling it (AuxPredConfig.enabled == False) reproduces the original
# raw-state actor/critic behaviour exactly: in that case SharedEncoder is an
# identity passthrough with NO parameters, so actor/critic keep consuming the
# 87-D state directly and no auxiliary optimizer is created.
#
# Design provenance (docs/aux_prediction_tqc_design.md):
#   - Falcon: a shared latent feeds a future-prediction auxiliary head; the
#     auxiliary loss back-propagates into the shared representation only.
#   - Proximity-Aware: the auxiliary target is a fixed-size egocentric risk /
#     compass map (per-sector risk over multiple horizons), not a per-human
#     ID-matched trajectory -> no masking / data association needed.
#   - DiPCAN: the auxiliary head is a TRAINING-ONLY branch; the deployed policy
#     is just encoder + actor, no privileged information required.
#
# Gradient rule (enforced by tqc_agent.py, documented here):
#   - encoder is updated by  critic_loss + beta_aux * aux_loss
#   - actor reads  z = encoder(state).detach()  -> actor loss never touches
#     the encoder.

import torch
import torch.nn as nn
import torch.nn.functional as F

# Defaults mirror environment/aux_prediction_labels.py so the agent-side head
# and the env-side label generation stay layout-compatible.
DEFAULT_HORIZONS_SEC = [0.5, 1.0, 1.5]
DEFAULT_NUM_SECTORS = 16
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]


class AuxPredConfig:
    """AUX_PRED: parsed auxiliary-prediction config (agent side)."""

    def __init__(self, cfg: dict = None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.version = int(cfg.get("version", 1))

        # Shared encoder.
        self.encoder_hidden_dim = int(cfg.get("encoder_hidden_dim", 256))
        self.latent_dim = int(cfg.get("latent_dim", 128))
        # Optional LayerNorm inside the shared encoder (default OFF -> the encoder
        # is byte-for-byte the original 2-layer ELU MLP). Kept opt-in because the
        # encoder feeds the actor/critic: changing its architecture invalidates a
        # strict encoder-state load on resume (tqc_io handles this with a logged
        # graceful fallback). out_dim (== latent_dim) is unchanged either way, so
        # actor/critic input width never moves.
        self.encoder_layernorm = bool(cfg.get("encoder_layernorm", False))

        # Label geometry (must match env-side AuxLabelConfig).
        self.num_sectors = int(cfg.get("num_sectors", DEFAULT_NUM_SECTORS))
        self.horizons_sec = list(cfg.get("horizons_sec", DEFAULT_HORIZONS_SEC))

        # Loss weights.
        self.loss_weight = float(cfg.get("loss_weight", 0.1))
        self.min_distance_loss_weight = float(cfg.get("min_distance_loss_weight", 0.0))
        # Optional beta warmup: ramp the trunk-level aux weight (beta_aux) linearly
        # from 0 to loss_weight over this many gradient steps so a noisy, freshly-
        # initialised aux head does not perturb early critic learning. 0 disables
        # the schedule (constant beta), keeping the previous behaviour.
        self.aux_beta_warmup_steps = int(cfg.get("aux_beta_warmup_steps", 0))
        # Optional per-curriculum-stage aux weight schedule (overrides loss_weight
        # + warmup when non-empty). List indexed by stage, clamped to the last
        # entry. Lets aux supervision ramp in with the curriculum, e.g.
        # [0.0, 0.0, 0.1, 0.3, 0.5] -> off for the easy stages, growing as the
        # dynamic task arrives. Empty -> legacy constant loss_weight (+ warmup).
        self.stagewise_loss_schedule = [
            float(x) for x in (cfg.get("stagewise_loss_schedule", []) or [])
        ]

        # Optional distributional auxiliary (v2, default off).
        self.use_distributional_aux = bool(cfg.get("use_distributional_aux", False))
        self.distributional_quantiles = list(
            cfg.get("distributional_quantiles", DEFAULT_QUANTILES)
        )
        # Weight on the distributional (pinball) risk term. Previously this term
        # was added with an implicit weight of 1.0; expose it so the quantile
        # objective can be balanced against the risk-map MSE without code edits.
        self.distributional_loss_weight = float(
            cfg.get("distributional_loss_weight", 1.0))

        # Aux-ONLY temporal context (v2, WIRED, opt-in). A small GRU summarises
        # the shared-encoder latents of the last `history_len` in-episode states
        # and its output is concatenated onto the AUX-HEAD input only; the
        # actor/critic path stays non-recurrent. temporal_mode/state_stack_len are
        # retained for backward config compatibility but no longer gate the path
        # (temporal_enabled is the single switch). See aux_prediction_temporal.py.
        self.temporal_enabled = bool(cfg.get("temporal_enabled", False))
        self.temporal_mode = str(cfg.get("temporal_mode", "none"))
        self.history_len = int(cfg.get("history_len", 4))
        self.state_stack_len = int(cfg.get("state_stack_len", 1))
        self.temporal_context_dim = int(cfg.get("temporal_context_dim", 64))
        # Falcon-style masked self-attention over the temporal GRU sequence
        # (reuses action_condition_attention_heads). Aux branch only.
        self.temporal_attention = bool(cfg.get("temporal_attention", False))

        # Action-conditioned auxiliary (Proximity-Aware style): predict the same
        # future risk map from z_t AND the upcoming action sequence
        # [a_t, .., a_{t+K-1}].  Only the aux branch changes; actor/critic paths
        # are untouched.  Requires enabled == True.
        self.action_conditioned_aux = bool(cfg.get("action_conditioned_aux", False))
        self.action_conditioned_steps = int(cfg.get("action_conditioned_steps", 4))
        self.action_embed_dim = int(cfg.get("action_embed_dim", 32))
        self.action_condition_hidden_dim = int(cfg.get("action_condition_hidden_dim", 128))
        # Falcon-style self-attention over the action-GRU output sequence. ONE
        # MultiheadAttention block (residual + LayerNorm) applied to the per-step
        # GRU outputs BEFORE the boundary-safe gather, so the context can weigh the
        # whole in-episode action window instead of relying on the GRU's last
        # hidden alone. Out-of-episode steps (k >= valid_len) are key-padding
        # masked, so they still never influence the prediction. Aux-branch only;
        # actor/critic are untouched. Default OFF -> head reduces to the GRU path.
        self.action_condition_attention = bool(cfg.get("action_condition_attention", False))
        self.action_condition_attention_heads = int(
            cfg.get("action_condition_attention_heads", 4))
        # A4 (aux fusion): how the action context is fused with the shared latent
        # z_t in the action-conditioned head.
        #   "concat" (default): fused = [z_t, action_ctx]  (original behaviour,
        #                       byte-for-byte unchanged).
        #   "film":  ctx -> (gamma, beta); z_mod = (1+gamma)*z_t + beta; then
        #            fused = [z_mod, action_ctx]. Identity-initialised so at start
        #            z_mod == z_t (output == concat head), learning to use the
        #            action context gradually. FiLM adds a small generator ->
        #            FRESH-RUN aux head (state_dict changes). Only affects the
        #            action-conditioned head; AuxiliaryHead is untouched.
        self.fusion_type = str(cfg.get("fusion_type", "concat")).lower()
        if self.fusion_type not in ("concat", "film"):
            raise ValueError(
                f"aux_prediction.fusion_type must be 'concat' or 'film' "
                f"(got {self.fusion_type!r}).")

        # Aux-head trunk capacity (shared by AuxiliaryHead + ActionConditionedAuxHead).
        # Defaults reproduce the original single ``Linear -> ELU`` trunk so existing
        # / disabled configs are byte-for-byte unchanged; the curriculum config opts
        # into a deeper LayerNorm trunk. These touch ONLY the aux branch.
        self.aux_trunk_hidden_dim = int(
            cfg.get("aux_trunk_hidden_dim", max(self.latent_dim, 128)))
        self.aux_trunk_layers = int(cfg.get("aux_trunk_layers", 1))
        self.aux_head_layernorm = bool(cfg.get("aux_head_layernorm", False))

        # ── Direct dynamic-avoidance heads (append-only, default OFF) ──────────
        # These add SMALL output heads on the SAME aux trunk (training-only; the
        # actor inference graph is untouched). They give the shared encoder a more
        # direct supervisory signal for closing dynamic obstacles than the risk
        # map alone. Keys / sizes MUST mirror the env AuxLabelConfig (verified via
        # the wire header). risk_map / min_dist heads are unchanged & retained.
        self.ttc_head_enabled = bool(cfg.get("ttc_head_enabled", False))
        self.ttc_horizon_sec = float(cfg.get("ttc_horizon_sec", 3.0))
        self.ttc_loss_weight = float(cfg.get("ttc_loss_weight", 1.0))
        self.hazard_sector_head_enabled = bool(
            cfg.get("hazard_sector_head_enabled", False))
        self.hazard_sector_bins = int(cfg.get("hazard_sector_bins", 3))
        self.hazard_sector_loss_weight = float(
            cfg.get("hazard_sector_loss_weight", 1.0))

        # STAGE 4: sparse/degenerate positive-event handling for imbalanced aux
        # targets (most timesteps -- even with a human present -- carry no near-
        # risk event, so plain MSE/BCE can be dominated by the safe majority).
        # Both default to the ORIGINAL unweighted loss (byte-identical) unless
        # explicitly configured.
        #   hazard_pos_weight: passed straight to F.binary_cross_entropy_with_
        #   logits' pos_weight (None -> vanilla BCE, the current default).
        #   Capped at hazard_pos_weight_cap to avoid an extreme, destabilising
        #   imbalance ratio if computed/configured too aggressively.
        _hpw = cfg.get("hazard_pos_weight", None)
        self.hazard_pos_weight_cap = float(cfg.get("hazard_pos_weight_cap", 20.0))
        self.hazard_pos_weight = (
            None if _hpw is None else min(float(_hpw), self.hazard_pos_weight_cap))
        # risk_map_positive_weight: extra multiplicative weight (>= 1.0) applied
        # to risk-map loss terms for cells whose target exceeds
        # risk_map_positive_threshold (nonzero/positive risk). 1.0 -> uniform
        # weighting, i.e. the original plain loss. RISK_BALANCE: capped (same
        # rationale as hazard_pos_weight_cap above) so a raw data imbalance
        # ratio is never applied unbounded.
        self.risk_map_positive_weight_cap = float(cfg.get("risk_map_positive_weight_cap", 10.0))
        _rmw = float(cfg.get("risk_map_positive_weight", 1.0))
        self.risk_map_positive_weight = (
            min(_rmw, self.risk_map_positive_weight_cap) if _rmw > 0 else 1.0)
        self.risk_map_positive_threshold = float(
            cfg.get("risk_map_positive_threshold", 0.0))
        # RISK_BALANCE: base loss for the risk-map term -- "mse" (default,
        # byte-identical to before) or "smooth_l1" (more stable once
        # risk_map_positive_weight up-weights rare positive cells).
        self.risk_map_loss_type = str(cfg.get("risk_map_loss_type", "mse")).strip().lower()
        if self.risk_map_loss_type not in ("mse", "smooth_l1"):
            raise ValueError(
                f"aux_prediction.risk_map_loss_type={self.risk_map_loss_type!r} "
                "must be 'mse' or 'smooth_l1'."
            )

    # --- derived geometry -------------------------------------------------
    @property
    def num_horizons(self) -> int:
        return len(self.horizons_sec)

    @property
    def num_quantiles(self) -> int:
        return len(self.distributional_quantiles)

    @property
    def min_distance_enabled(self) -> bool:
        # v2 head: on when it carries any loss weight.
        return self.min_distance_loss_weight > 0.0

    @property
    def risk_dim(self) -> int:
        return self.num_horizons * self.num_sectors

    @property
    def ttc_len(self) -> int:
        return 1 if self.ttc_head_enabled else 0

    @property
    def hazard_len(self) -> int:
        return self.hazard_sector_bins if self.hazard_sector_head_enabled else 0

    @property
    def label_dim(self) -> int:
        # Canonical env label = H*K risk + H min-dist (+ optional ttc + hazard).
        # MUST equal the env AuxLabelConfig.label_dim (checked via the wire header).
        return (self.risk_dim + self.num_horizons
                + self.ttc_len + self.hazard_len)


class SharedEncoder(nn.Module):
    """AUX_PRED: 87 -> hidden -> latent (ELU) shared trunk.

    When ``enabled`` is False this is a parameter-free identity passthrough so
    the baseline TQC path is byte-for-byte unchanged and ``out_dim ==
    state_dim``.
    """

    def __init__(self, state_dim: int, cfg: AuxPredConfig):
        super().__init__()
        self.enabled = cfg.enabled
        if not self.enabled:
            self.out_dim = state_dim
            self._identity = True
            return

        self._identity = False
        hdim = cfg.encoder_hidden_dim
        ldim = cfg.latent_dim
        self.l1 = nn.Linear(state_dim, hdim)
        self.l2 = nn.Linear(hdim, ldim)
        self.out_dim = ldim
        # Optional LayerNorm (opt-in). When OFF these are None and forward() is
        # the original `elu(l2(elu(l1(x))))`, so the parameter set + numerics are
        # byte-for-byte identical to baseline. When ON, a LayerNorm is applied
        # after each hidden pre-activation (stabilises the latent that feeds the
        # actor/critic) WITHOUT changing out_dim.
        self._use_ln = bool(cfg.encoder_layernorm)
        self.ln1 = nn.LayerNorm(hdim) if self._use_ln else None
        self.ln2 = nn.LayerNorm(ldim) if self._use_ln else None

    def forward(self, state):
        if self._identity:
            return state
        h = self.l1(state)
        if self.ln1 is not None:
            h = self.ln1(h)
        z = F.elu(h)
        h2 = self.l2(z)
        if self.ln2 is not None:
            h2 = self.ln2(h2)
        return F.elu(h2)

    def has_params(self) -> bool:
        return not self._identity


def _build_aux_trunk(in_dim: int, cfg: AuxPredConfig):
    """AUX_PRED: build the aux-head trunk and return (module, out_dim).

    ``aux_trunk_layers`` Linear blocks of width ``aux_trunk_hidden_dim``; each is
    ``Linear [-> LayerNorm] -> ELU`` (LayerNorm gated by ``aux_head_layernorm``).
    With the defaults (1 layer, no LayerNorm, hidden = max(latent_dim, 128)) this
    is exactly the original single ``Linear -> ELU`` trunk, so heads built for a
    config that does not set the new keys are unchanged. Aux branch only.
    """
    hidden = cfg.aux_trunk_hidden_dim
    layers = []
    d = in_dim
    for _ in range(max(1, cfg.aux_trunk_layers)):
        layers.append(nn.Linear(d, hidden))
        if cfg.aux_head_layernorm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.ELU())
        d = hidden
    return nn.Sequential(*layers), hidden


def _maybe_cat_temporal(z, temporal_ctx, temporal_dim: int):
    """AUX_PRED: concatenate the aux-only temporal context onto the head input.

    Concatenation (not cross-attention) is the deliberate fusion choice: it is
    the SAME stable pattern already used to fuse z_t with the action-GRU context,
    and avoids the extra instability/compute of cross-attending three
    heterogeneous vectors. When the head was built with temporal_dim > 0 but no
    context is supplied (e.g. an eval call without history), a zero pad keeps the
    trunk input width valid.
    """
    if temporal_dim <= 0:
        return z
    if temporal_ctx is None:
        temporal_ctx = z.new_zeros((z.shape[0], temporal_dim))
    return torch.cat([z, temporal_ctx], dim=-1)


def _build_output_heads(module: nn.Module, hidden: int, cfg: AuxPredConfig):
    """AUX_PRED: attach the risk-map (+ optional min-dist / quantile) output
    Linears to ``module`` as direct attributes.

    Assigning ``module.risk_head = nn.Linear(...)`` keeps the SAME state_dict keys
    (``risk_head.*`` / ``min_dist_head.*`` / ``risk_quant_head.*``) the heads used
    before this refactor, so an unchanged-architecture checkpoint still loads
    strictly (no spurious fresh-init on resume). Shared by AuxiliaryHead and
    ActionConditionedAuxHead so both emit the identical output dict.
    """
    module.risk_head = nn.Linear(hidden, cfg.risk_dim)
    module.min_dist_head = (
        nn.Linear(hidden, cfg.num_horizons) if cfg.min_distance_enabled else None
    )
    module.risk_quant_head = (
        nn.Linear(hidden, cfg.risk_dim * cfg.num_quantiles)
        if cfg.use_distributional_aux
        else None
    )
    # Direct dynamic-avoidance heads (append-only, gated). Small Linears on the
    # SAME trunk; absent (None) when disabled so the state_dict keys / param set
    # are unchanged for a baseline / v1 config (strict checkpoint load still works).
    module.ttc_head = (
        nn.Linear(hidden, 1) if cfg.ttc_head_enabled else None
    )
    module.hazard_head = (
        nn.Linear(hidden, cfg.hazard_sector_bins)
        if cfg.hazard_sector_head_enabled else None
    )


def _apply_output_heads(module: nn.Module, feat, batch: int):
    """AUX_PRED: run the heads built by _build_output_heads on a trunk feature."""
    out = {"risk_map": torch.sigmoid(module.risk_head(feat))}
    if module.min_dist_head is not None:
        out["min_dist"] = torch.sigmoid(module.min_dist_head(feat))
    if module.risk_quant_head is not None:
        out["risk_quant"] = module.risk_quant_head(feat).view(
            batch, module.cfg.risk_dim, module.cfg.num_quantiles
        )
    # ttc: scalar in [0, 1] (sigmoid; 1 = safe). hazard: per-sector LOGITS (the
    # loss uses BCE-with-logits), shape (B, S).
    if getattr(module, "ttc_head", None) is not None:
        out["ttc"] = torch.sigmoid(module.ttc_head(feat))
    if getattr(module, "hazard_head", None) is not None:
        out["hazard"] = module.hazard_head(feat)
    return out


class AuxiliaryHead(nn.Module):
    """AUX_PRED: future risk-map head (+ optional min-distance / distributional).

    Inputs the shared latent z and predicts:
      - risk_map: (B, H*K) in [0, 1]                          (v1, always)
      - min_dist: (B, H)   in [0, 1]   if cfg.min_distance_enabled   (v2)
      - risk_quant: (B, H*K, Q)        if cfg.use_distributional_aux (v2)
    """

    def __init__(self, latent_dim: int, cfg: AuxPredConfig, temporal_dim: int = 0):
        super().__init__()
        self.cfg = cfg
        self.temporal_dim = int(temporal_dim)
        self.trunk, hidden = _build_aux_trunk(latent_dim + self.temporal_dim, cfg)
        _build_output_heads(self, hidden, cfg)

    def forward(self, z, temporal_ctx=None):
        feat = self.trunk(_maybe_cat_temporal(z, temporal_ctx, self.temporal_dim))
        return _apply_output_heads(self, feat, z.shape[0])


class ActionConditionedAuxHead(nn.Module):
    """AUX_PRED: action-conditioned future risk-map head (Proximity-Aware style).

    Predicts the SAME future risk map / min-dist / distributional targets as
    AuxiliaryHead, but conditions on the upcoming action sequence in addition to
    the shared latent z_t:

        a_seq = [a_t, a_{t+1}, ..., a_{t+K-1}]   (actions taken from s_t onward,
                                                  within the same episode)
        e_k   = Linear(a_k)                      (action embedding)
        ctx   = GRU(e_0 .. e_{L-1}) last hidden  (L = valid_len, boundary-safe)
        feat  = trunk([z_t, ctx])  ->  risk / min_dist / risk_quant heads

    Only the FIRST ``valid_len`` actions are consumed (the rest are zero-pad past
    an episode boundary / the buffer write head), so out-of-episode actions never
    influence the prediction.  Output dict keys match AuxiliaryHead so
    aux_prediction_losses.compute_aux_loss is reused unchanged.

    Provenance: ProximitySocialNav social_auxiliary_tasks.RiskEstimation uses a
    Linear action embedder + GRU seeded by the belief to predict the proximity
    feature k steps ahead; here the latent z_t plays the role of the belief and
    is concatenated with the GRU context.
    """

    def __init__(self, latent_dim: int, action_dim: int, cfg: AuxPredConfig,
                 temporal_dim: int = 0):
        super().__init__()
        self.cfg = cfg
        self.action_dim = action_dim
        self.gru_hidden = cfg.action_condition_hidden_dim
        self.temporal_dim = int(temporal_dim)

        self.action_embed = nn.Linear(action_dim, cfg.action_embed_dim)
        self.gru = nn.GRU(cfg.action_embed_dim, self.gru_hidden, batch_first=True)

        # Optional Falcon-style self-attention over the GRU output sequence.
        # heads must divide gru_hidden; fall back to 1 head otherwise.
        self.attention = None
        if cfg.action_condition_attention:
            heads = cfg.action_condition_attention_heads
            if self.gru_hidden % heads != 0:
                heads = 1
            self.attention = nn.MultiheadAttention(
                self.gru_hidden, heads, batch_first=True)
            self.attn_norm = nn.LayerNorm(self.gru_hidden)

        # Fusion = concat(z_t [or FiLM-modulated z_t], action_ctx, temporal_ctx).
        # Concatenation is the deliberate, stable fusion (same as the existing
        # z + action_ctx merge); the temporal context is just one more block when
        # temporal is enabled. A4: fusion_type="film" modulates z_t by the action
        # context BEFORE the concat, but keeps the direct action_ctx block — so
        # the trunk input width (combined) is IDENTICAL for concat and film; only
        # a small FiLM generator is added.
        self._latent_dim = latent_dim
        self.fusion_type = str(getattr(cfg, "fusion_type", "concat"))

        # Build the trunk + output heads FIRST so that every shared submodule
        # (action_embed, gru, attention, trunk, heads) consumes the RNG in the
        # SAME order for concat and film. With a fixed seed the film head's
        # trunk/heads then initialise IDENTICALLY to the concat head's — A4 is a
        # PURE fusion ablation (only the extra film_gen differs), not a run whose
        # trunk init is perturbed by an earlier film_gen draw.
        combined = latent_dim + self.gru_hidden + self.temporal_dim
        self.trunk, hidden = _build_aux_trunk(combined, cfg)
        _build_output_heads(self, hidden, cfg)

        # A4: FiLM generator built LAST (after all shared modules) and
        # identity-initialised. action_ctx (gru_hidden) -> (gamma, beta) each of
        # size latent_dim; zero weight + zero bias => gamma=0, beta=0 at start ->
        # z_mod = (1+0)*z + 0 = z, so the film head's initial OUTPUT equals the
        # concat head's too. It LEARNS to deviate. None for the default concat.
        self.film_gen = None
        if self.fusion_type == "film":
            self.film_gen = nn.Linear(self.gru_hidden, 2 * latent_dim)
            nn.init.zeros_(self.film_gen.weight)
            nn.init.zeros_(self.film_gen.bias)

    def forward(self, z, future_actions, valid_len, temporal_ctx=None):
        """z: (B, latent_dim); future_actions: (B, K, action_dim);
        valid_len: (B,) long in [1, K] = number of leading in-episode actions;
        temporal_ctx: (B, temporal_dim) or None (aux-only recent-state context)."""
        b = z.shape[0]
        emb = self.action_embed(future_actions)        # (B, K, E)
        out_seq, _ = self.gru(emb)                      # (B, K, H_gru)
        if self.attention is not None:
            # Mask out-of-episode steps (k >= valid_len) so attention — and hence
            # the gathered context — never mixes in padded / cross-boundary
            # actions. key_padding_mask True == ignore that key.
            k = out_seq.shape[1]
            ar = torch.arange(k, device=out_seq.device).unsqueeze(0)   # (1, K)
            key_padding_mask = ar >= valid_len.unsqueeze(1)            # (B, K)
            attn_out, _ = self.attention(
                out_seq, out_seq, out_seq,
                key_padding_mask=key_padding_mask, need_weights=False)
            out_seq = self.attn_norm(out_seq + attn_out)  # residual + LayerNorm
        # Take the (optionally attended) sequence output AFTER consuming exactly
        # the valid actions. valid_len >= 1, so index 0 is always in-episode and
        # never masked, keeping the gathered position finite.
        idx = (valid_len - 1).clamp(min=0)              # (B,)
        ctx = out_seq[torch.arange(b, device=out_seq.device), idx]  # (B, H_gru)

        # A4: optional FiLM modulation of z_t by the action context (identity at
        # init). No-op (film_gen is None) for the default concat fusion.
        z_fused = z
        if self.film_gen is not None:
            gamma, beta = self.film_gen(ctx).split(self._latent_dim, dim=-1)
            z_fused = (1.0 + gamma) * z + beta

        fused = _maybe_cat_temporal(torch.cat([z_fused, ctx], dim=-1),
                                    temporal_ctx, self.temporal_dim)
        feat = self.trunk(fused)
        return _apply_output_heads(self, feat, b)
