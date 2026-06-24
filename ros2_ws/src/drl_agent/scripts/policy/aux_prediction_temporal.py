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
