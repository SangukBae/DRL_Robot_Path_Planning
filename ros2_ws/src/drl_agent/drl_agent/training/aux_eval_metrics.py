# AUX_PRED: formal evaluation metrics for the auxiliary-prediction head.
#
# These are the *paper* aux metrics (computed in the trainer's evaluation loop),
# kept SEPARATE from the training-time monitoring losses (aux/loss, aux/risk_mse,
# ... in aux_prediction_losses.py).  Pure NumPy, no torch / ROS dependency, so it
# is unit-testable offline.
#
# Canonical label layout (matches environment/aux_prediction_labels.py):
#   label = [ risk_map (H*K) ][ min_dist_norm (H) ]
#     risk in [0,1] (1 = a human is right on top of the robot in that sector)
#     min_dist_norm in [0,1] = clamp(min_distance / D_c, 0, 1) (1 = far / none)
#
# Metrics (all averaged over (sample, horizon[, sector]) as noted):
#   aux_risk_rmse        RMSE over every risk-map cell (predicted vs GT).
#   aux_min_dist_mae_m   MAE of the future min-distance, in METRES (normalized
#                        error * D_c).  When the head has no min-dist output the
#                        prediction is derived from the risk map:
#                        d_norm = clamp(1 - max_sector_risk, 0, 1).
#   aux_peak_sector_acc  Fraction of (sample, horizon) where argmax-sector of the
#                        predicted risk equals argmax-sector of the GT risk.
#                        Tie rule: np.argmax -> lowest index, applied to BOTH
#                        pred and GT.  Horizons whose GT row is all-zero (no human
#                        within D_c, so "peak sector" is undefined) are EXCLUDED.
#   aux_near_event_f1    Binary near-event = future min-distance < threshold [m].
#                        precision/recall/F1 over all (sample, horizon); every
#                        ratio is zero-division-safe (0.0 when the denominator 0).

import numpy as np

# Default positive-event threshold (metres). The trainer overrides this from a
# ROS param / manifest so it is never silently hard-coded for a real run.
DEFAULT_NEAR_EVENT_THRESHOLD_M = 0.5


def split_label(label_2d, num_horizons, num_sectors):
    """Split a [N, H*K + H] label batch into (risk[N,H,K], min_dist_norm[N,H])."""
    arr = np.asarray(label_2d, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    H, K = int(num_horizons), int(num_sectors)
    risk_dim = H * K
    risk = arr[:, :risk_dim].reshape(-1, H, K)
    md = arr[:, risk_dim:risk_dim + H]
    return risk, md


def min_dist_norm_from_risk(risk_NHK):
    """Derive a normalized future min-distance per horizon from a risk map.

    risk = 1 - d/D_c  =>  d_norm = clamp(1 - peak_sector_risk, 0, 1).
    Used when the aux head has no dedicated min-distance output.
    """
    peak = np.asarray(risk_NHK, dtype=np.float64).max(axis=-1)  # [N, H]
    return np.clip(1.0 - peak, 0.0, 1.0)


def _safe_ratio(num, den):
    return float(num) / float(den) if den > 0 else 0.0


def compute_aux_eval_metrics(risk_pred, risk_gt, md_pred, md_gt, *,
                             num_horizons, num_sectors, risk_distance_scale,
                             near_event_threshold_m=DEFAULT_NEAR_EVENT_THRESHOLD_M):
    """Compute the four formal aux metrics for a stacked batch.

    Parameters
    ----------
    risk_pred, risk_gt : [N, H*K] (or [N, H, K]) predicted / GT risk maps in [0,1].
    md_pred            : [N, H] predicted normalized min-dist, or None (derive
                         from risk_pred).
    md_gt              : [N, H] GT normalized min-dist.
    risk_distance_scale: D_c [m] used to de-normalize distances.
    near_event_threshold_m: positive-event distance threshold [m].

    Returns a dict (empty if N == 0).
    """
    risk_pred = np.asarray(risk_pred, dtype=np.float64).reshape(-1, num_horizons, num_sectors)
    risk_gt = np.asarray(risk_gt, dtype=np.float64).reshape(-1, num_horizons, num_sectors)
    N = risk_pred.shape[0]
    if N == 0:
        return {}
    Dc = max(float(risk_distance_scale), 1e-6)

    md_gt = np.asarray(md_gt, dtype=np.float64).reshape(-1, num_horizons)
    if md_pred is None:
        md_pred = min_dist_norm_from_risk(risk_pred)
    else:
        md_pred = np.asarray(md_pred, dtype=np.float64).reshape(-1, num_horizons)

    # --- aux_risk_rmse over every cell ---
    risk_rmse = float(np.sqrt(np.mean((risk_pred - risk_gt) ** 2)))

    # --- aux_min_dist_mae_m (metres) ---
    min_dist_mae_m = float(np.mean(np.abs(md_pred - md_gt)) * Dc)

    # --- aux_peak_sector_acc (exclude all-zero GT rows) ---
    gt_peak = np.argmax(risk_gt, axis=-1)        # [N, H] lowest-index tie
    pred_peak = np.argmax(risk_pred, axis=-1)    # [N, H]
    has_peak = risk_gt.max(axis=-1) > 1e-9       # [N, H] GT has a real peak
    counted = int(has_peak.sum())
    if counted > 0:
        peak_acc = float(((pred_peak == gt_peak) & has_peak).sum()) / counted
    else:
        peak_acc = float("nan")

    # --- aux_near_event_f1 ---
    thr = float(near_event_threshold_m)
    gt_pos = (md_gt * Dc) < thr
    pred_pos = (md_pred * Dc) < thr
    tp = int((gt_pos & pred_pos).sum())
    fp = int((~gt_pos & pred_pos).sum())
    fn = int((gt_pos & ~pred_pos).sum())
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)

    # STAGE 4: dynamic-sample / positive-event counts alongside RMSE. A batch
    # dominated by constant-zero (no-human) GT rows makes aux_risk_rmse
    # trivially low WITHOUT the head having learned anything predictive -- these
    # counts let a reader (or aggregate_results.py) tell that apart from a
    # genuinely accurate prediction on real dynamic data. "Dynamic" = the GT
    # risk map has ANY nonzero cell anywhere across its horizons/sectors for
    # that sample (a human/obstacle was within risk_distance_scale at some
    # point); "positive event" = the same near-event threshold already used
    # for precision/recall/F1 above (gt_pos, per (sample, horizon)).
    dynamic_mask = risk_gt.reshape(N, -1).max(axis=-1) > 1e-9
    dynamic_count = int(dynamic_mask.sum())

    return {
        "aux_risk_rmse": round(risk_rmse, 6),
        "aux_min_dist_mae_m": round(min_dist_mae_m, 6),
        "aux_peak_sector_acc": (round(peak_acc, 6) if peak_acc == peak_acc else float("nan")),
        "aux_near_event_precision": round(precision, 6),
        "aux_near_event_recall": round(recall, 6),
        "aux_near_event_f1": round(f1, 6),
        "aux_eval_samples": N,
        "aux_near_event_threshold_m": thr,
        "aux_dynamic_sample_count": dynamic_count,
        "aux_dynamic_sample_frac": round(dynamic_count / N, 6),
        "aux_positive_event_count": int(gt_pos.sum()),
    }


class AuxEvalAccumulator:
    """Collects predicted/GT risk + min-dist over an evaluation and finalizes the
    formal aux metrics — globally and (optionally) per map_type.

    Usage (per eval episode, after the episode so action-conditioned predictions
    are boundary-safe):
        acc.add_batch(risk_pred, risk_gt, md_pred, md_gt, map_type=mt)
    then once:
        metrics = acc.finalize()            # global dict
        per_map = acc.finalize_per_map()    # {map_type: dict}

    Extension point: ESR / AD / ALV / encounter-count style metrics can be added
    by collecting extra per-episode arrays alongside the risk/min-dist buffers
    here, then extending finalize() — no trainer-loop change needed.
    """

    def __init__(self, *, num_horizons, num_sectors, risk_distance_scale,
                 near_event_threshold_m=DEFAULT_NEAR_EVENT_THRESHOLD_M):
        self.H = int(num_horizons)
        self.K = int(num_sectors)
        self.Dc = float(risk_distance_scale)
        self.thr = float(near_event_threshold_m)
        self.reset()

    def reset(self):
        self._risk_pred, self._risk_gt = [], []
        self._md_pred, self._md_gt = [], []
        self._has_md_pred = True          # becomes False if any batch lacks md_pred
        self._by_map = {}                 # map_type -> list of (rp, rg, mp, mg)

    def add_batch(self, risk_pred, risk_gt, md_pred, md_gt, map_type=None):
        """Add one episode's per-step stacked predictions/GT. md_pred may be None
        (then derived from risk at finalize time)."""
        rp = np.asarray(risk_pred, dtype=np.float64).reshape(-1, self.H * self.K)
        rg = np.asarray(risk_gt, dtype=np.float64).reshape(-1, self.H * self.K)
        if rp.shape[0] == 0:
            return
        mg = np.asarray(md_gt, dtype=np.float64).reshape(-1, self.H)
        if md_pred is None:
            self._has_md_pred = False
            mp = min_dist_norm_from_risk(rp.reshape(-1, self.H, self.K))
        else:
            mp = np.asarray(md_pred, dtype=np.float64).reshape(-1, self.H)
        self._risk_pred.append(rp); self._risk_gt.append(rg)
        self._md_pred.append(mp); self._md_gt.append(mg)
        if map_type is not None:
            self._by_map.setdefault(map_type, []).append((rp, rg, mp, mg))

    def _finalize_lists(self, rp_list, rg_list, mp_list, mg_list):
        if not rp_list:
            return {}
        rp = np.concatenate(rp_list, axis=0)
        rg = np.concatenate(rg_list, axis=0)
        mp = np.concatenate(mp_list, axis=0) if self._has_md_pred else None
        mg = np.concatenate(mg_list, axis=0)
        return compute_aux_eval_metrics(
            rp, rg, mp, mg,
            num_horizons=self.H, num_sectors=self.K,
            risk_distance_scale=self.Dc, near_event_threshold_m=self.thr,
        )

    def finalize(self) -> dict:
        return self._finalize_lists(self._risk_pred, self._risk_gt,
                                    self._md_pred, self._md_gt)

    def finalize_per_map(self) -> dict:
        out = {}
        for mt, items in self._by_map.items():
            rp = [a for (a, _, _, _) in items]
            rg = [b for (_, b, _, _) in items]
            mp = [c for (_, _, c, _) in items]
            mg = [d for (_, _, _, d) in items]
            out[mt] = self._finalize_lists(rp, rg, mp, mg)
        return out

    def has_data(self) -> bool:
        return len(self._risk_pred) > 0
