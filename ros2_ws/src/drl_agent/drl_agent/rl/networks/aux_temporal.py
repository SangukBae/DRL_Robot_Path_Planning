# AUX_PRED (v2): aux-ONLY temporal context over recent state history.
#
# STATUS: WIRED (opt-in via AuxPredConfig.temporal_enabled).
#   tqc_agent.py constructs TemporalContextEncoder when temporal is enabled, the
#   replay buffer samples a boundary-safe backward state window
#   (LAP.get_last_state_history), and the encoded context is concatenated onto
#   the AUX-HEAD input ONLY.  The actor/critic path is NOT recurrent and never
#   sees this context, so off-policy i.i.d. stability is preserved.
#
# REDUNDANCY NOTE (observation time-context / frame stacking):
#   When the env's observation_time_context (actor-visible obs_state frame
#   stacking) is ON, each STORED state already contains the recent obs history,
#   so this aux-only temporal GRU then summarises a window of ALREADY-stacked
#   states (a history-of-histories) — its marginal value is small and largely
#   redundant with the actor's own time context. It is kept ON-by-config purely
#   for ablation; the trainer logs a warning when both are enabled. Prefer the
#   actor-visible stacking (the primary fix); disable temporal_enabled once it is
#   on, unless you specifically want to ablate the aux temporal branch.
#
# Why a GRU over latents (not a temporal-conv, not a recurrent encoder):
#   - The action-conditioned aux head already uses a GRU + masked self-attention
#     with a leading-valid boundary mask.  Reusing that exact pattern keeps ONE
#     well-tested masking contract in the codebase instead of inventing a second.
#   - History length is short (N≈4) and VARIABLE (an episode boundary / the
#     buffer seam truncates it).  A GRU with a per-row valid_len gather handles a
#     variable window naturally; a fixed-kernel temporal conv would need extra
#     padding-mask bookkeeping for the same guarantee.
#   - We deliberately do NOT make the shared encoder recurrent: that would change
#     the actor/critic input path and hurt off-policy replay stability.  Instead
#     the SAME shared encoder is applied per-step to the history states and a
#     SEPARATE small GRU summarises the latent sequence for the aux head only.
#
# Provenance: Falcon uses an LSTM over the rollout to forecast future human
# motion.  Here the temporal cue is backward-looking (recent state history) and
# confined to the aux branch, so it enriches the shared representation without
# turning the policy into an RNN.

import torch
import torch.nn as nn
import torch.nn.functional as F

from drl_agent.rl.networks.aux_prediction import SharedEncoder


class ScanTemporalEncoder(nn.Module):
    """Compress a short SCAN history into a small temporal feature.

    This is the actor/critic-facing temporal path: the recent LiDAR scans (NOT
    the raw point cloud, NOT the agent_state) are summarised into a compact
    ``feature_dim`` vector so the policy sees motion cues WITHOUT the shared MLP
    having to ingest the raw ``history_len * obs_dim`` concatenation. It is
    deliberately light and NON-recurrent by default (the actor/critic bodies must
    stay non-recurrent for off-policy replay stability); a tiny GRU is offered
    only as an ablation ``encoder_type``.

    Input : scan_hist (B, history_len, obs_dim), CHRONOLOGICAL (oldest -> newest).
    Output: (B, feature_dim).
    """

    def __init__(self, history_len: int, obs_dim: int, feature_dim: int = 32,
                 encoder_type: str = "conv1d", hidden: int = 128):
        super().__init__()
        self.history_len = int(history_len)
        self.obs_dim = int(obs_dim)
        self.feature_dim = int(feature_dim)
        self.encoder_type = str(encoder_type)
        self.out_dim = self.feature_dim

        if self.encoder_type == "mlp":
            # Flatten the (small) history and run a 2-layer MLP.
            self.net = nn.Sequential(
                nn.Linear(self.history_len * self.obs_dim, hidden), nn.ELU(),
                nn.Linear(hidden, self.feature_dim), nn.ELU(),
            )
        elif self.encoder_type == "conv1d":
            # Treat the time frames as input CHANNELS and convolve over the scan
            # bins (local angular structure), then global-pool over bins.
            c = max(16, self.feature_dim)
            self.conv = nn.Sequential(
                nn.Conv1d(self.history_len, c, kernel_size=5, stride=2, padding=2),
                nn.ELU(),
                nn.Conv1d(c, c, kernel_size=5, stride=2, padding=2), nn.ELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.proj = nn.Linear(c, self.feature_dim)
        elif self.encoder_type == "gru":
            h = max(self.feature_dim, 32)
            self.gru = nn.GRU(self.obs_dim, h, batch_first=True)
            self.proj = nn.Linear(h, self.feature_dim)
        else:
            raise ValueError(
                f"temporal_actor_context.encoder_type must be "
                f"mlp|conv1d|gru (got {self.encoder_type!r})")

    def forward(self, scan_hist):
        if self.encoder_type == "mlp":
            return self.net(scan_hist.reshape(scan_hist.shape[0], -1))
        if self.encoder_type == "conv1d":
            x = self.conv(scan_hist)                 # (B, c, 1)
            return F.elu(self.proj(x.squeeze(-1)))
        out, _ = self.gru(scan_hist)                 # (B, T, h)
        return F.elu(self.proj(out[:, -1]))          # last (newest) step


class TemporalFusionEncoder(nn.Module):
    """Shared encoder that keeps the CURRENT state on the main path and feeds the
    actor/critic a COMPRESSED temporal feature instead of the raw frame stack.

    The env transports the stacked state ``[obs_t(O), agent_t(A), obs_{t-1}(O),
    ..., obs_{t-(N-1)}(O)]`` (current frame first; see obs_time_context.py). This
    module SPLITS it back into:
      * current_state = [obs_t, agent_t]  (O+A, e.g. 87)  -> main SharedEncoder
      * scan_history  = [obs_{t-(N-1)}, .., obs_t]  (N x O) -> ScanTemporalEncoder
    and fuses ``z_cur`` (latent_dim) with ``z_temporal`` (feature_dim) through a
    small Linear back to ``latent_dim``, so the actor/critic input width is
    UNCHANGED. The raw ``N*O`` stack therefore never reaches the actor/critic main
    path directly; only the light current-state MLP + the small temporal feature do.

    ``temporal_gain`` (set per curriculum stage) scales z_temporal: 0 at early
    stages (the policy effectively sees only the current state, and the temporal
    encoder receives no gradient) ramping to 1 once temporal learning is enabled.
    Tensor shapes are FIXED regardless of the gain, so nothing reshapes per stage.

    Inference-friendly: a single forward(state) -> fused latent; no privileged
    info and no recurrence on the actor/critic bodies.
    """

    def __init__(self, state_dim: int, obs_dim: int, agent_dim: int,
                 history_len: int, stack_agent_state: bool, aux_cfg,
                 feature_dim: int = 32, encoder_type: str = "conv1d"):
        super().__init__()
        self.state_dim = int(state_dim)
        self.obs_dim = int(obs_dim)
        self.agent_dim = int(agent_dim)
        self.history_len = int(history_len)
        self.stack_agent_state = bool(stack_agent_state)
        self.current_dim = self.obs_dim + self.agent_dim
        self.frame_len = self.current_dim if self.stack_agent_state else self.obs_dim

        # Main path over the CURRENT 87-D state only. Reuses the aux SharedEncoder
        # so its hidden/latent/LayerNorm config is shared with the rest of AUX.
        # INDEPENDENT of aux supervision: when aux_prediction.enabled=false this is
        # a parameter-free identity (out_dim == current_dim == 87), so the fused
        # latent is [current(87) + temporal] -> 87 and the temporal/fusion still
        # train from the critic loss with NO aux head (temporal-only ablation).
        # When aux is enabled it is a real 87 -> latent_dim MLP.
        self.main = SharedEncoder(self.current_dim, aux_cfg)
        ldim = self.main.out_dim
        self.temporal = ScanTemporalEncoder(
            self.history_len, self.obs_dim, feature_dim, encoder_type)
        self.fusion = nn.Linear(ldim + self.temporal.out_dim, ldim)
        self.out_dim = ldim
        self.temporal_gain = 1.0
        self._identity = False

    # --- uniform encoder surface (matches SharedEncoder) -------------------
    def has_params(self) -> bool:
        return True

    def set_temporal_gain(self, gain: float):
        self.temporal_gain = float(gain)

    def expected_state_dim(self) -> int:
        return self.current_dim + (self.history_len - 1) * self.frame_len

    def _split(self, state):
        """Return (current_state (B, O+A), scan_history (B, N, O) chronological)."""
        cur = state[:, : self.current_dim]
        frames = [state[:, : self.obs_dim]]                 # obs_t (newest)
        for k in range(1, self.history_len):
            base = self.current_dim + (k - 1) * self.frame_len
            frames.append(state[:, base: base + self.obs_dim])  # obs_{t-k}
        frames.reverse()                                    # oldest -> newest
        scan = torch.stack(frames, dim=1)                   # (B, N, O)
        return cur, scan

    def forward(self, state):
        cur, scan = self._split(state)
        z_cur = self.main(cur)
        z_temporal = self.temporal(scan) * self.temporal_gain
        return F.elu(self.fusion(torch.cat([z_cur, z_temporal], dim=-1)))

    def temporal_feature(self, state):
        """PHASE2: the RAW (pre-fusion) compressed temporal feature z_temporal
        for ``state``, scaled by the same ``temporal_gain`` as ``forward()``.

        ``forward()`` only returns the FUSED latent (z_temporal is squashed
        back down to ``latent_dim`` through ``self.fusion``, a bottleneck
        shared by the critic/actor/aux paths). A consumer that wants a
        DEDICATED view of the temporal signal itself (e.g. action_risk_head's
        optional ``use_temporal_context``) calls this instead -- on the SAME
        ``state`` batch already passed to ``forward()``, so the split/gain
        logic is guaranteed identical, at the cost of recomputing the (cheap,
        ``feature_dim``-wide) ScanTemporalEncoder pass a second time."""
        _, scan = self._split(state)
        return self.temporal(scan) * self.temporal_gain


class TemporalContextEncoder(nn.Module):
    """AUX_PRED (v2): GRU over the shared-encoder latents of the last N states.

    Input
    -----
    z_seq      : (B, N, latent_dim)  REVERSE-time latent window: index 0 is the
                 CURRENT state s_t (always valid), index k is s_{t-k}; steps past
                 the episode start / buffer seam are zero-padded.
    valid_len  : (B,) long in [1, N]  number of LEADING valid (in-episode) steps.

    Output
    ------
    (B, out_dim) temporal context, or (B, 0) when disabled (so an unconditional
    ``cat`` is a no-op and the v1 path is byte-for-byte unchanged).

    Boundary safety mirrors ActionConditionedAuxHead exactly: out-of-episode
    steps (k >= valid_len) are key-padding masked in the optional self-attention
    and excluded by the valid_len-1 gather, so padded / cross-boundary states can
    never influence the context (locked by tests).
    """

    def __init__(self, latent_dim: int, cfg, context_dim: int = None):
        super().__init__()
        self.enabled = bool(getattr(cfg, "temporal_enabled", False))
        self.history_len = max(1, int(getattr(cfg, "history_len", 4)))
        if not self.enabled:
            self.out_dim = 0
            return

        self.context_dim = int(context_dim or cfg.temporal_context_dim)
        self.gru = nn.GRU(latent_dim, self.context_dim, batch_first=True)

        # Optional Falcon-style masked self-attention over the GRU output
        # sequence (same block as the action path). heads must divide
        # context_dim; fall back to 1 head otherwise.
        self.attention = None
        if bool(getattr(cfg, "temporal_attention", False)):
            heads = int(getattr(cfg, "action_condition_attention_heads", 4))
            if self.context_dim % heads != 0:
                heads = 1
            self.attention = nn.MultiheadAttention(
                self.context_dim, heads, batch_first=True)
            self.attn_norm = nn.LayerNorm(self.context_dim)
        self.out_dim = self.context_dim

    def forward(self, z_seq, valid_len):
        b = z_seq.shape[0]
        out_seq, _ = self.gru(z_seq)                    # (B, N, context_dim)
        if self.attention is not None:
            n = out_seq.shape[1]
            ar = torch.arange(n, device=out_seq.device).unsqueeze(0)   # (1, N)
            key_padding_mask = ar >= valid_len.unsqueeze(1)            # (B, N)
            attn_out, _ = self.attention(
                out_seq, out_seq, out_seq,
                key_padding_mask=key_padding_mask, need_weights=False)
            out_seq = self.attn_norm(out_seq + attn_out)
        # Gather the output AFTER consuming exactly the valid (in-episode) steps.
        # valid_len >= 1, so index 0 (the current state) is always in-episode and
        # never masked, keeping the gathered position finite.
        idx = (valid_len - 1).clamp(min=0)              # (B,)
        return out_seq[torch.arange(b, device=out_seq.device), idx]  # (B, ctx)
