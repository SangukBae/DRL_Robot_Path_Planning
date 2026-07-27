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
    """AUX_PRED: slice the canonical label into its blocks (append-only layout
    [risk (H*K)][min_dist (H)][ttc (T)][hazard (S)]).

    Returns (risk, min_dist, ttc, hazard); ttc / hazard are None when the
    corresponding head is disabled (their blocks are absent from the label)."""
    o = 0
    risk = label[:, o : o + cfg.risk_dim]; o += cfg.risk_dim
    min_dist = label[:, o : o + cfg.num_horizons]; o += cfg.num_horizons
    ttc = None
    if cfg.ttc_len:
        ttc = label[:, o : o + cfg.ttc_len]; o += cfg.ttc_len
    hazard = None
    if cfg.hazard_len:
        hazard = label[:, o : o + cfg.hazard_len]; o += cfg.hazard_len
    return risk, min_dist, ttc, hazard


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
    risk_target, min_dist_target, ttc_target, hazard_target = _split_label(label, cfg)

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
    # Weighted by distributional_loss_weight (was an implicit 1.0) so the pinball
    # term can be balanced against the risk-map MSE. The risk target lives in
    # [0, 1] and the pinball uses kappa=1.0, so |td| <= 1 keeps it in the
    # quadratic Huber regime — i.e. it largely echoes the MSE signal with added
    # quantile spread. That is exactly why it stays OFF by default (it buys
    # little for the cost) yet remains a balanced, tested option when wanted.
    if "risk_quant" in pred and cfg.use_distributional_aux:
        tau = torch.as_tensor(
            cfg.distributional_quantiles, dtype=torch.float32, device=device
        )
        dist_loss = quantile_pinball_loss(pred["risk_quant"], risk_target, tau)
        total = total + cfg.distributional_loss_weight * dist_loss
        logs["aux/risk_quantile"] = float(dist_loss.detach().item())

    # --- v2: direct TTC regression (optional) ---
    # MSE on the normalized min time-to-contact (both in [0, 1], 1 = safe). A
    # direct, low-variance closing-risk signal for the shared encoder.
    if "ttc" in pred and ttc_target is not None and cfg.ttc_loss_weight > 0.0:
        ttc_loss = F.mse_loss(pred["ttc"], ttc_target)
        total = total + cfg.ttc_loss_weight * ttc_loss
        logs["aux/ttc_mse"] = float(ttc_loss.detach().item())

    # --- v2: hazard-sector classification (optional) ---
    # Per-sector binary imminent-hazard target -> BCE-with-logits (pred["hazard"]
    # is raw logits). Multi-label (independent sectors), stable when all-zero
    # (no humans / no hazard).
    if "hazard" in pred and hazard_target is not None and cfg.hazard_sector_loss_weight > 0.0:
        hz_loss = F.binary_cross_entropy_with_logits(pred["hazard"], hazard_target)
        total = total + cfg.hazard_sector_loss_weight * hz_loss
        logs["aux/hazard_bce"] = float(hz_loss.detach().item())

    logs["aux/loss"] = float(total.detach().item())
    return total, logs
