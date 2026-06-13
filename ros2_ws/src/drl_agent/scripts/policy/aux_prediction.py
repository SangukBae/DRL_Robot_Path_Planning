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

        # Label geometry (must match env-side AuxLabelConfig).
        self.num_sectors = int(cfg.get("num_sectors", DEFAULT_NUM_SECTORS))
        self.horizons_sec = list(cfg.get("horizons_sec", DEFAULT_HORIZONS_SEC))

        # Loss weights.
        self.loss_weight = float(cfg.get("loss_weight", 0.1))
        self.min_distance_loss_weight = float(cfg.get("min_distance_loss_weight", 0.0))

        # Optional distributional auxiliary (v2, default off).
        self.use_distributional_aux = bool(cfg.get("use_distributional_aux", False))
        self.distributional_quantiles = list(
            cfg.get("distributional_quantiles", DEFAULT_QUANTILES)
        )

        # Optional temporal extension (v2): SCAFFOLD ONLY -- NOT WIRED.  These
        # flags are parsed but currently have no effect; tqc_agent.py never
        # constructs the temporal module (aux_prediction_temporal.py).  See
        # docs/aux_prediction_tqc_design.md.
        self.temporal_enabled = bool(cfg.get("temporal_enabled", False))
        self.temporal_mode = str(cfg.get("temporal_mode", "none"))
        self.history_len = int(cfg.get("history_len", 4))
        self.state_stack_len = int(cfg.get("state_stack_len", 1))

        # Action-conditioned auxiliary (Proximity-Aware style): predict the same
        # future risk map from z_t AND the upcoming action sequence
        # [a_t, .., a_{t+K-1}].  Only the aux branch changes; actor/critic paths
        # are untouched.  Requires enabled == True.
        self.action_conditioned_aux = bool(cfg.get("action_conditioned_aux", False))
        self.action_conditioned_steps = int(cfg.get("action_conditioned_steps", 4))
        self.action_embed_dim = int(cfg.get("action_embed_dim", 32))
        self.action_condition_hidden_dim = int(cfg.get("action_condition_hidden_dim", 128))

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
    def label_dim(self) -> int:
        # Canonical env label = H*K risk + H min-dist (see aux_prediction_labels).
        return self.risk_dim + self.num_horizons


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

    def forward(self, state):
        if self._identity:
            return state
        z = F.elu(self.l1(state))
        z = F.elu(self.l2(z))
        return z

    def has_params(self) -> bool:
        return not self._identity


class AuxiliaryHead(nn.Module):
    """AUX_PRED: future risk-map head (+ optional min-distance / distributional).

    Inputs the shared latent z and predicts:
      - risk_map: (B, H*K) in [0, 1]                          (v1, always)
      - min_dist: (B, H)   in [0, 1]   if cfg.min_distance_enabled   (v2)
      - risk_quant: (B, H*K, Q)        if cfg.use_distributional_aux (v2)
    """

    def __init__(self, latent_dim: int, cfg: AuxPredConfig):
        super().__init__()
        self.cfg = cfg
        hidden = max(cfg.latent_dim, 128)

        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ELU(),
        )
        # v1: per-sector, per-horizon risk map.
        self.risk_head = nn.Linear(hidden, cfg.risk_dim)

        # v2 (optional): future min-distance per horizon.
        self.min_dist_head = (
            nn.Linear(hidden, cfg.num_horizons) if cfg.min_distance_enabled else None
        )
        # v2 (optional): distributional risk (predict Q quantiles per risk cell).
        self.risk_quant_head = (
            nn.Linear(hidden, cfg.risk_dim * cfg.num_quantiles)
            if cfg.use_distributional_aux
            else None
        )

    def forward(self, z):
        feat = self.trunk(z)
        out = {"risk_map": torch.sigmoid(self.risk_head(feat))}
        if self.min_dist_head is not None:
            out["min_dist"] = torch.sigmoid(self.min_dist_head(feat))
        if self.risk_quant_head is not None:
            b = z.shape[0]
            q = self.risk_quant_head(feat).view(
                b, self.cfg.risk_dim, self.cfg.num_quantiles
            )
            out["risk_quant"] = q
        return out


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

    def __init__(self, latent_dim: int, action_dim: int, cfg: AuxPredConfig):
        super().__init__()
        self.cfg = cfg
        self.action_dim = action_dim
        self.gru_hidden = cfg.action_condition_hidden_dim

        self.action_embed = nn.Linear(action_dim, cfg.action_embed_dim)
        self.gru = nn.GRU(cfg.action_embed_dim, self.gru_hidden, batch_first=True)

        combined = latent_dim + self.gru_hidden
        hidden = max(cfg.latent_dim, 128)
        self.trunk = nn.Sequential(
            nn.Linear(combined, hidden),
            nn.ELU(),
        )
        self.risk_head = nn.Linear(hidden, cfg.risk_dim)
        self.min_dist_head = (
            nn.Linear(hidden, cfg.num_horizons) if cfg.min_distance_enabled else None
        )
        self.risk_quant_head = (
            nn.Linear(hidden, cfg.risk_dim * cfg.num_quantiles)
            if cfg.use_distributional_aux
            else None
        )

    def forward(self, z, future_actions, valid_len):
        """z: (B, latent_dim); future_actions: (B, K, action_dim);
        valid_len: (B,) long in [1, K] = number of leading in-episode actions."""
        b = z.shape[0]
        emb = self.action_embed(future_actions)        # (B, K, E)
        out_seq, _ = self.gru(emb)                      # (B, K, H_gru)
        # Take the GRU output AFTER consuming exactly the valid actions.
        idx = (valid_len - 1).clamp(min=0)              # (B,)
        ctx = out_seq[torch.arange(b, device=out_seq.device), idx]  # (B, H_gru)

        feat = self.trunk(torch.cat([z, ctx], dim=-1))
        out = {"risk_map": torch.sigmoid(self.risk_head(feat))}
        if self.min_dist_head is not None:
            out["min_dist"] = torch.sigmoid(self.min_dist_head(feat))
        if self.risk_quant_head is not None:
            q = self.risk_quant_head(feat).view(
                b, self.cfg.risk_dim, self.cfg.num_quantiles
            )
            out["risk_quant"] = q
        return out
