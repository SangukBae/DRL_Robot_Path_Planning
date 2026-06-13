# AUX_PRED: auxiliary-prediction loss helpers.
#
# Kept separate from tqc_agent.py so the auxiliary objective can be tuned /
# removed without touching the core TQC update.  All losses operate on the
# canonical label vector produced by environment/aux_prediction_labels.py:
#
#   label = [ risk_map (H*K) ][ min_dist_norm (H) ]
#
# v1 (default): plain MSE on the risk map (Proximity-Aware style regression).
# v2 (optional): + MSE on future min-distance, and/or a distributional risk
#                term using a pinball / quantile-Huber loss (TQC-flavoured).

import torch
import torch.nn.functional as F


def _split_label(label, cfg):
    """AUX_PRED: slice the canonical label into (risk, min_dist) blocks."""
    risk = label[:, : cfg.risk_dim]
    min_dist = label[:, cfg.risk_dim : cfg.risk_dim + cfg.num_horizons]
    return risk, min_dist


def quantile_pinball_loss(pred_quant, target, quantiles, kappa=1.0):
    """AUX_PRED: quantile-Huber (pinball) loss for distributional risk.

    pred_quant : (B, N, Q)   predicted quantiles per cell
    target     : (B, N)      scalar regression target broadcast over Q
    quantiles  : 1-D tensor (Q,) of tau values in (0, 1)
    """
    target = target.unsqueeze(-1)  # (B, N, 1)
    td = target - pred_quant       # (B, N, Q)
    huber = torch.where(
        td.abs() <= kappa,
        0.5 * td.pow(2),
        kappa * (td.abs() - 0.5 * kappa),
    )
    tau = quantiles.view(1, 1, -1)
    loss = (tau - (td < 0).float()).abs() * huber
    return loss.mean()


def compute_aux_loss(pred, label, cfg, device):
    """AUX_PRED: combine the enabled auxiliary terms into one scalar loss.

    Returns (loss_tensor, log_dict).  log_dict holds detached float metrics.
    """
    risk_target, min_dist_target = _split_label(label, cfg)

    # --- v1: risk-map regression (always on) ---
    risk_loss = F.mse_loss(pred["risk_map"], risk_target)
    total = risk_loss
    logs = {"aux/risk_mse": float(risk_loss.detach().item())}

    # --- v2: future min-distance regression (optional) ---
    if "min_dist" in pred and cfg.min_distance_loss_weight > 0.0:
        md_loss = F.mse_loss(pred["min_dist"], min_dist_target)
        total = total + cfg.min_distance_loss_weight * md_loss
        logs["aux/min_dist_mse"] = float(md_loss.detach().item())

    # --- v2: distributional risk (optional) ---
    if "risk_quant" in pred and cfg.use_distributional_aux:
        tau = torch.as_tensor(
            cfg.distributional_quantiles, dtype=torch.float32, device=device
        )
        dist_loss = quantile_pinball_loss(pred["risk_quant"], risk_target, tau)
        total = total + dist_loss
        logs["aux/risk_quantile"] = float(dist_loss.detach().item())

    logs["aux/loss"] = float(total.detach().item())
    return total, logs
