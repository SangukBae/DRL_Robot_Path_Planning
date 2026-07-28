"""Unit tests for drl_agent.training.aux_eval_metrics (pure NumPy, no ROS/torch).

Previously untested at the functional level (only referenced by the canonical-
import check in test_package_migration.py). Covers the existing RMSE/peak-
accuracy/near-event-F1 contract plus STAGE 4's new dynamic-sample and
positive-event-count reporting: a formal aux evaluation report must let a
reader judge whether a low RMSE is meaningful or just an artifact of mostly-
constant-zero (no-human) regions in the eval data.
"""

import numpy as np
import pytest

import drl_agent.training.aux_eval_metrics as aem

H, K = 2, 4  # 2 horizons, 4 sectors


def _zeros(n):
    # "No human present": risk map all-zero AND min_dist_norm == 1.0 (far/
    # safe) -- min_dist_norm == 0.0 would mean "touching" (max danger), the
    # OPPOSITE of no-human, per the canonical label convention (1 = far/none).
    return np.zeros((n, H, K)), np.ones((n, H))


def test_empty_batch_returns_empty_dict():
    rp, rg = np.zeros((0, H, K)), np.zeros((0, H, K))
    mg = np.zeros((0, H))
    out = aem.compute_aux_eval_metrics(
        rp, rg, None, mg, num_horizons=H, num_sectors=K, risk_distance_scale=5.0)
    assert out == {}


def test_perfect_prediction_gives_zero_rmse():
    rng = np.random.default_rng(0)
    risk_gt = rng.random((10, H, K))
    md_gt = rng.random((10, H))
    out = aem.compute_aux_eval_metrics(
        risk_gt, risk_gt, md_gt, md_gt,
        num_horizons=H, num_sectors=K, risk_distance_scale=5.0)
    assert out["aux_risk_rmse"] == 0.0
    assert out["aux_min_dist_mae_m"] == 0.0


def test_all_zero_gt_gives_nan_peak_sector_acc():
    risk_gt, md_gt = _zeros(5)
    risk_pred = np.random.default_rng(1).random((5, H, K))
    out = aem.compute_aux_eval_metrics(
        risk_pred, risk_gt, None, md_gt,
        num_horizons=H, num_sectors=K, risk_distance_scale=5.0)
    assert out["aux_peak_sector_acc"] != out["aux_peak_sector_acc"]  # NaN


# --------------------------------------------------------------------------- #
#  STAGE 4: dynamic-sample / positive-event counts
# --------------------------------------------------------------------------- #
def test_all_constant_zero_batch_reports_zero_dynamic_fraction():
    # A batch where NO sample has any nonzero risk cell anywhere (equivalent
    # to an eval window with no humans at all, or a stage-0/1/2-only slice) --
    # the RMSE would trivially be near-perfect (predicting ~0 everywhere is
    # "easy"), so the report must make that explicit via a 0 dynamic fraction/
    # count, not let a reader mistake the low RMSE for real predictive skill.
    risk_gt, md_gt = _zeros(8)
    risk_pred = np.zeros((8, H, K))
    out = aem.compute_aux_eval_metrics(
        risk_pred, risk_gt, None, md_gt,
        num_horizons=H, num_sectors=K, risk_distance_scale=5.0)
    assert out["aux_dynamic_sample_count"] == 0
    assert out["aux_dynamic_sample_frac"] == 0.0
    assert out["aux_positive_event_count"] == 0


def test_mixed_batch_counts_only_nonzero_samples_as_dynamic():
    risk_gt = np.zeros((10, H, K))
    risk_gt[3, 0, 1] = 0.8   # sample 3 has a nonzero risk cell
    risk_gt[7, 1, 2] = 0.5   # sample 7 has a nonzero risk cell
    md_gt = np.ones((10, H))  # far / safe by default
    risk_pred = np.zeros((10, H, K))
    out = aem.compute_aux_eval_metrics(
        risk_pred, risk_gt, None, md_gt,
        num_horizons=H, num_sectors=K, risk_distance_scale=5.0)
    assert out["aux_dynamic_sample_count"] == 2
    assert out["aux_dynamic_sample_frac"] == 0.2
    assert out["aux_eval_samples"] == 10


def test_positive_event_count_matches_near_event_ground_truth():
    risk_gt, _ = _zeros(6)
    md_gt = np.full((6, H), 0.9)  # far by default (normalized, * Dc = 4.5m)
    md_gt[2, 0] = 0.01  # close: 0.01 * 5.0 = 0.05m < threshold
    md_gt[4, 1] = 0.02  # close: 0.02 * 5.0 = 0.10m < threshold
    md_pred = md_gt.copy()  # perfect prediction -> irrelevant to GT count
    out = aem.compute_aux_eval_metrics(
        np.zeros((6, H, K)), risk_gt, md_pred, md_gt,
        num_horizons=H, num_sectors=K, risk_distance_scale=5.0,
        near_event_threshold_m=0.5)
    assert out["aux_positive_event_count"] == 2


def test_accumulator_finalize_includes_dynamic_and_positive_counts():
    acc = aem.AuxEvalAccumulator(num_horizons=H, num_sectors=K, risk_distance_scale=5.0)
    risk_gt_ep1, md_gt_ep1 = _zeros(4)                     # fully static episode
    risk_gt_ep2, md_gt_ep2 = _zeros(3)
    risk_gt_ep2[1, 0, 0] = 0.7                             # one dynamic sample
    acc.add_batch(np.zeros((4, H, K)), risk_gt_ep1, None, md_gt_ep1, map_type="corridor")
    acc.add_batch(np.zeros((3, H, K)), risk_gt_ep2, None, md_gt_ep2, map_type="corridor")
    out = acc.finalize()
    assert out["aux_eval_samples"] == 7
    assert out["aux_dynamic_sample_count"] == 1
    assert out["aux_dynamic_sample_frac"] == pytest.approx(1 / 7)


def test_accumulator_has_data_and_reset():
    acc = aem.AuxEvalAccumulator(num_horizons=H, num_sectors=K, risk_distance_scale=5.0)
    assert acc.has_data() is False
    risk_gt, md_gt = _zeros(2)
    acc.add_batch(np.zeros((2, H, K)), risk_gt, None, md_gt)
    assert acc.has_data() is True
    acc.reset()
    assert acc.has_data() is False
    assert acc.finalize() == {}
