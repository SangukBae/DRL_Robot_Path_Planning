# PHASE2: Critic-connected Action-Risk Head.
#
# A small supervised head that predicts the SELECTED action's directional risk
# from [z, action]: risk_dir (closeness risk in the action's waypoint-theta
# sector) and min_dist_dir (nearest-human distance in that sector), both in
# [0, 1] -- same convention/target as the env-side directional_risk block (see
# environment/aux_prediction_labels.py, environment.py::_compute_directional_
# risk). Trained via supervised MSE against the PRIVILEGED GT target stored in
# the replay buffer (buffer.py's action_risk_target, aligned with the stored
# `action`), added into the trunk loss like aux_prediction's AuxiliaryHead.
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


class ActionRiskConfig:
    """PHASE2: parsed action_risk_head config (agent side)."""

    def __init__(self, cfg: dict = None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.hidden_dim = int(cfg.get("hidden_dim", 64))
        self.loss_weight = float(cfg.get("loss_weight", 0.1))


class ActionRiskHead(nn.Module):
    """PHASE2: Linear(latent_dim+action_dim -> hidden) -> ELU -> Linear(hidden -> 2),
    sigmoid outputs (risk_dir, min_dist_dir), both in [0, 1]."""

    def __init__(self, latent_dim: int, action_dim: int, cfg: ActionRiskConfig):
        super().__init__()
        hidden = cfg.hidden_dim
        self.l1 = nn.Linear(latent_dim + action_dim, hidden)
        self.act = nn.ELU()
        self.out = nn.Linear(hidden, 2)

    def forward(self, z, action):
        h = self.act(self.l1(torch.cat([z, action], dim=-1)))
        pred = torch.sigmoid(self.out(h))
        return pred  # (B, 2): [:, 0] = risk_dir, [:, 1] = min_dist_dir
