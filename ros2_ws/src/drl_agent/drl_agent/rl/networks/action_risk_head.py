# PHASE2: Critic-connected Action-Risk Head.
#
# A small supervised head that predicts the SELECTED action's directional risk
# from [z, action] (or [z, temporal_feature, action], see below): risk_dir
# (closeness risk in the action's waypoint-theta sector) and min_dist_dir
# (nearest-human distance in that sector), both in [0, 1] -- same
# convention/target as the env-side directional_risk block (see
# environment/aux_prediction_labels.py, environment.py::_compute_directional_
# risk). Trained via supervised MSE against the PRIVILEGED GT target stored in
# the replay buffer (buffer.py's action_risk_target, aligned with the stored
# `action`), added into the trunk loss like aux_prediction's AuxiliaryHead.
#
# Optional temporal context (action_risk_head.use_temporal_context, default
# false -> byte-identical [z, action] head): a moving pedestrian's future risk
# direction depends on recent motion, which a single fused z only sees
# indirectly (through temporal_actor_context's shared fusion bottleneck).
# When enabled, the caller (tqc_agent.py) concatenates the SAME compressed
# temporal feature the actor already reuses from temporal_actor_context
# (TemporalFusionEncoder.temporal_feature()), giving this head a DEDICATED
# view of it: [z, temporal_feature, action]. Fail-fasts at Agent construction
# if temporal_actor_context is not enabled -- there is no fallback source,
# see tqc_agent.py.
#
# Gradient rule (enforced by tqc_agent.py, documented here):
#   - the head's OWN PARAMETERS are updated ONLY by its dedicated supervised
#     loss (added into critic_loss + beta_aux*aux_loss + action_risk_beta*
#     action_risk_loss). critic_loss / actor_loss must never move this head's
#     weights -- that would let Q-optimisation pressure corrupt the risk
#     estimate away from ground truth.
#   - HOW that's enforced differs by call site, because the actor update needs
#     an input-side gradient the other two don't:
#       * critic-loss call (stored batch `action`) and the TD-target call
#         (under torch.no_grad() already): the prediction is fully detached /
#         no-grad -- nothing downstream needs d(loss)/d(action) for a fixed
#         past action.
#       * actor-loss call (`actions_pi`, the actor's own live sample): the
#         prediction is NOT detached -- d(extra)/d(actions_pi) must stay
#         connected, or the critic's learned "penalise risk-raising actions"
#         signal can never reach the actor (the whole point of critic_risk_
#         input). Instead the head's PARAMETERS are frozen (requires_grad_
#         (False) for that one forward call) so actor_loss.backward() cannot
#         update them, while the input-side gradient into actions_pi (and
#         hence into the actor) still flows. The actor still never sees the
#         risk prediction directly -- only through the critic's Q.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionRiskConfig:
    """PHASE2: parsed action_risk_head config (agent side)."""

    def __init__(self, cfg: dict = None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.hidden_dim = int(cfg.get("hidden_dim", 64))
        self.loss_weight = float(cfg.get("loss_weight", 0.1))
        # RISK_BALANCE: optional up-weighting of positive-risk elements in the
        # supervised loss (target > positive_threshold), same convention as
        # aux_prediction's risk_map_positive_weight. Capped so a raw imbalance
        # ratio from the data is never used unbounded (a wildly rare-event
        # ratio would otherwise destabilise the trunk loss). Default 1.0 ->
        # uniform weight -> byte-identical to the original plain F.mse_loss.
        self.pos_weight_cap = float(cfg.get("pos_weight_cap", 10.0))
        _pw = float(cfg.get("pos_weight", 1.0))
        self.pos_weight = min(_pw, self.pos_weight_cap) if _pw > 0 else 1.0
        self.positive_threshold = float(cfg.get("positive_threshold", 0.5))
        # "mse" (default, byte-identical to before) or "smooth_l1" (more
        # stable under the up-weighted rare-event regime -- less sensitive to
        # the occasional large-error outlier than a squared term).
        self.loss_type = str(cfg.get("loss_type", "mse")).strip().lower()
        if self.loss_type not in ("mse", "smooth_l1"):
            raise ValueError(
                f"action_risk_head.loss_type={self.loss_type!r} must be "
                "'mse' or 'smooth_l1'."
            )
        # Default OFF -> the head's input width (and therefore its state_dict)
        # is unchanged from the original [z, action] contract; only tqc_agent.py
        # setting this true (which it only does when temporal_actor_context is
        # also enabled -- see the fail-fast there) changes anything.
        self.use_temporal_context = bool(cfg.get("use_temporal_context", False))
        # Only "actor" (reuse temporal_actor_context's feature) is implemented;
        # tqc_agent.py fail-fasts on any other value instead of guessing.
        self.temporal_context_source = str(cfg.get("temporal_context_source", "actor"))
        # STAGE 3: curriculum stage at/after which the head actually runs its
        # forward pass (both for its own supervised loss AND for feeding the
        # critic via critic_risk_input). Default 0 -> active from the start,
        # identical to the pre-Stage-3 behaviour. Below this stage the agent
        # skips the forward pass entirely (not just its loss weight) and, when
        # critic_risk_input is enabled, feeds the critic a fixed all-zero
        # (batch, 2) tensor instead so the critic's input shape never changes
        # with stage (see Agent.train()'s action-risk gating).
        self.enable_from_stage = int(cfg.get("enable_from_stage", 0))


class ActionRiskHead(nn.Module):
    """PHASE2: Linear(latent_dim[+temporal_dim]+action_dim -> hidden) -> ELU ->
    Linear(hidden -> 2), sigmoid outputs (risk_dir, min_dist_dir), both in [0, 1].

    ``temporal_dim`` is 0 (the default) unless the caller explicitly enables
    ``action_risk_head.use_temporal_context`` AND passes the actor's temporal
    feature width -- in which case ``forward()`` REQUIRES a matching
    ``temporal_feature`` tensor on every call (no silent zero-fill fallback,
    so a wiring bug fails loudly instead of training on a degraded input)."""

    def __init__(self, latent_dim: int, action_dim: int, cfg: ActionRiskConfig,
                 temporal_dim: int = 0):
        super().__init__()
        hidden = cfg.hidden_dim
        self.temporal_dim = int(temporal_dim)
        self.l1 = nn.Linear(latent_dim + self.temporal_dim + action_dim, hidden)
        self.act = nn.ELU()
        self.out = nn.Linear(hidden, 2)

    def forward(self, z, action, temporal_feature=None):
        if self.temporal_dim > 0:
            if temporal_feature is None:
                raise RuntimeError(
                    "ActionRiskHead was built with temporal_dim="
                    f"{self.temporal_dim} (use_temporal_context=true) but "
                    "forward() was called without temporal_feature."
                )
            h = self.act(self.l1(torch.cat([z, temporal_feature, action], dim=-1)))
        else:
            h = self.act(self.l1(torch.cat([z, action], dim=-1)))
        pred = torch.sigmoid(self.out(h))
        return pred  # (B, 2): [:, 0] = risk_dir, [:, 1] = min_dist_dir


def weighted_action_risk_loss(pred, target, cfg: ActionRiskConfig):
    """RISK_BALANCE: supervised loss for the Action-Risk Head's (risk_dir,
    min_dist_dir) prediction, with optional up-weighting of positive-risk
    target elements (cfg.pos_weight, already capped at construction) and a
    configurable base loss (cfg.loss_type). cfg.pos_weight == 1.0 (default)
    -> byte-identical to the previous plain F.mse_loss.

    Also returns a log dict with the positive/safe loss breakdown and a
    positive-class recall/F1 -- computed on risk_dir (column 0) ONLY, since
    min_dist_dir (column 1) has the OPPOSITE polarity (low == risky) and
    thresholding it the same way as risk_dir does not mean "predicted risky"
    -- so an "all-safe collapse" -- great overall error, ~0 recall on the
    rare positive-risk elements -- is visible instead of hidden behind a good
    aggregate RMSE.
    """
    elementwise = (
        F.smooth_l1_loss(pred, target, reduction="none")
        if cfg.loss_type == "smooth_l1"
        else F.mse_loss(pred, target, reduction="none")
    )
    # RISK_BALANCE fix: risk_dir (col 0) is the danger signal (higher = more
    # risk); min_dist_dir (col 1) is a DISTANCE where 1 == far/safe. Masking
    # elementwise on `target > threshold` up-weighted min_dist_dir==1 rows --
    # i.e. up-weighted the SAFE, no-human-present case -- exactly backwards
    # from the rare-risky-event intent. Base the positive mask on risk_dir
    # ONLY and broadcast it across the row so both outputs of a genuinely
    # risky transition share the up-weight.
    pos_mask = (target[:, 0:1] > cfg.positive_threshold).expand_as(target)
    if cfg.pos_weight != 1.0:
        weight = torch.where(
            pos_mask, torch.full_like(target, cfg.pos_weight), torch.ones_like(target))
        loss = (weight * elementwise).mean()
    else:
        loss = elementwise.mean()

    logs = {"action_risk/pos_weight_applied": float(cfg.pos_weight)}
    with torch.no_grad():
        n_pos = int(pos_mask.sum().item())
        n_tot = int(pos_mask.numel())
        if n_pos > 0:
            logs["action_risk/positive_loss"] = float(elementwise[pos_mask].mean().item())
        if n_pos < n_tot:
            logs["action_risk/safe_loss"] = float(elementwise[~pos_mask].mean().item())
        # RISK_BALANCE fix: recall/F1 must compare risk_dir ONLY (column 0).
        # pos_mask's two columns are identical (broadcast from risk_dir), so
        # pos_mask[:, 0] is exactly "target risk_dir > threshold" -- applying
        # the same test to pred[:, 1] (min_dist_dir) does not mean "predicted
        # risky" (opposite polarity), and mixing both columns' TP/FP counts
        # produced a nonsensical recall (e.g. a perfectly correct
        # [risk=0.9, min_dist=0.1] prediction scored recall=0.5, and a safe
        # [0.1, 0.9] prediction generated a spurious false positive).
        risk_dir_pos = pos_mask[:, 0]
        n_risk_pos = int(risk_dir_pos.sum().item())
        pred_risk_pos = pred[:, 0] > cfg.positive_threshold
        tp = int((pred_risk_pos & risk_dir_pos).sum().item())
        fp = int((pred_risk_pos & ~risk_dir_pos).sum().item())
        if n_risk_pos > 0:
            recall = tp / n_risk_pos
            precision = tp / max(tp + fp, 1)
            logs["action_risk/positive_recall"] = float(recall)
            logs["action_risk/positive_f1"] = (
                float(2 * precision * recall / (precision + recall))
                if (precision + recall) > 0 else 0.0
            )
    return loss, logs
