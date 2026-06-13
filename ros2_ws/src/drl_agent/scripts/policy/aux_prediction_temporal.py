# AUX_PRED (v2): optional temporal / history extension for the auxiliary head.
#
# STATUS: SCAFFOLD ONLY -- NOT WIRED INTO TRAINING OR INFERENCE.
#   tqc_agent.py never constructs TemporalContextEncoder, the replay buffer does
#   not sample stacked states / history, and the aux head does not receive a
#   temporal context.  AuxPredConfig parses temporal_enabled / temporal_mode /
#   history_len / state_stack_len, but setting them has NO effect today.  This
#   file only fixes the intended interface so the feature can be finished later
#   (needs: buffer stacked-state sampling + concatenating the context onto the
#   aux-head input only, leaving the v1 actor/critic path unchanged).
#
# This is a SECOND-STAGE, opt-in module.  The primary single-step path in
# tqc_agent.py + aux_prediction.py does NOT import or depend on anything here;
# everything is gated behind AuxPredConfig.temporal_enabled so the default
# (v1) training path is untouched.
#
# Rather than forcing the shared encoder to become a recurrent network (which
# would change the actor/critic input path and hurt off-policy stability), the
# temporal signal is provided as a small, separate branch:
#
#   - "state_stack": the caller stacks the last `state_stack_len` observations
#     and feeds the concatenation to TemporalContextEncoder, producing a small
#     context vector that is concatenated onto the shared latent ONLY for the
#     auxiliary head (not for actor/critic).  This keeps the v1 actor/critic
#     path byte-for-byte identical.
#
# Provenance: Falcon uses an LSTM over the rollout to forecast future human
# trajectories; here we approximate that temporal cue with a light MLP over a
# short stack of recent states, which is compatible with i.i.d. off-policy
# replay sampling when the buffer also stores the stacked observation.

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalContextEncoder(nn.Module):
    """AUX_PRED (v2): small MLP over a stack of recent states -> context vec.

    Input : (B, state_stack_len * state_dim)
    Output: (B, context_dim)
    """

    def __init__(self, state_dim: int, cfg, context_dim: int = 64):
        super().__init__()
        self.enabled = bool(cfg.temporal_enabled)
        self.mode = cfg.temporal_mode
        self.stack_len = max(1, int(cfg.state_stack_len))
        self.context_dim = context_dim
        if not self.enabled:
            self.out_dim = 0
            return
        in_dim = state_dim * self.stack_len
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ELU(),
            nn.Linear(128, context_dim),
            nn.ELU(),
        )
        self.out_dim = context_dim

    def forward(self, stacked_state):
        if not self.enabled:
            # Caller should not use the output, but return an empty tensor with
            # a valid batch dim so accidental concatenation is a no-op.
            b = stacked_state.shape[0]
            return stacked_state.new_zeros((b, 0))
        return self.net(stacked_state)
