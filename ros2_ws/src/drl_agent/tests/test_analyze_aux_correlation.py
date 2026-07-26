"""Unit tests for scripts/utils/analyze_aux_correlation.py (Phase 1a).

Pure-function / file-I/O tests, no ROS -- follows the existing convention
(direct import of the module, no fixtures beyond pytest's tmp_path).
"""

import csv
import json
import math
import warnings

import analyze_aux_correlation as aac


def _write_eval_summary_csv(path, rows):
    fieldnames = [
        "seed", "aux_enabled", "aux_version", "eval_global_t", "curriculum_stage",
        "eval_eps", "success_rate", "collision_rate", "timeout_rate", "mean_reward",
        "mean_final_goal_dist", "spl", "cte", "jerk", "stl", "psc", "h_coll_rate",
        "lidar_clearance_rate", "aux_risk_rmse", "aux_min_dist_mae_m",
        "aux_peak_sector_acc", "aux_near_event_f1",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _row(**kw):
    base = dict(
        seed=0, aux_enabled=1, aux_version=1, eval_global_t=12000, curriculum_stage=0,
        eval_eps=10, success_rate=0.5, collision_rate=0.3, timeout_rate=0.2,
        mean_reward=1.0, mean_final_goal_dist=1.0, spl=0.5, cte=1.0, jerk=0.1,
        stl=0.5, psc=0.8, h_coll_rate=0.1, lidar_clearance_rate=0.9,
        aux_risk_rmse=0.1, aux_min_dist_mae_m=0.2, aux_peak_sector_acc=0.7,
        aux_near_event_f1=0.6,
    )
    base.update(kw)
    return base


def test_pearson_r_perfect_positive_correlation():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 4.0, 6.0, 8.0, 10.0]
    r, n = aac.pearson_r(xs, ys)
    assert n == 5
    assert math.isclose(r, 1.0, rel_tol=1e-6)


def test_pearson_r_perfect_negative_correlation():
    xs = [1.0, 2.0, 3.0]
    ys = [3.0, 2.0, 1.0]
    r, n = aac.pearson_r(xs, ys)
    assert n == 3
    assert math.isclose(r, -1.0, rel_tol=1e-6)


def test_pearson_r_drops_nan_pairs():
    xs = [1.0, 2.0, float("nan"), 4.0, 5.0]
    ys = [2.0, 4.0, 6.0, float("nan"), 10.0]
    r, n = aac.pearson_r(xs, ys)
    # only (1,2),(2,4),(5,10) survive -> still perfectly correlated
    assert n == 3
    assert math.isclose(r, 1.0, rel_tol=1e-6)


def test_pearson_r_too_few_samples_is_nan_not_crash():
    r, n = aac.pearson_r([1.0], [2.0])
    assert n == 1
    assert math.isnan(r)


def test_pearson_r_constant_series_is_nan_not_crash():
    r, n = aac.pearson_r([1.0, 1.0, 1.0], [2.0, 3.0, 4.0])
    assert n == 3
    assert math.isnan(r)


def test_load_rows_missing_file_warns_not_raises(tmp_path):
    missing = tmp_path / "nope.csv"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rows = aac.load_rows([str(missing)])
    assert rows == []
    assert any("not found" in str(x.message) for x in w)


def test_load_rows_missing_columns_warns(tmp_path):
    p = tmp_path / "partial.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "success_rate"])
        w.writerow([0, 0.5])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rows = aac.load_rows([str(p)])
    assert len(rows) == 1
    assert any("missing columns" in str(x.message) for x in w)


def test_compute_correlations_covers_all_pairs():
    rows = [_row(aux_risk_rmse=v, h_coll_rate=1.0 - v) for v in (0.1, 0.2, 0.3, 0.4)]
    corr = aac.compute_correlations(rows)
    assert len(corr) == len(aac.AUX_METRICS) * len(aac.AVOIDANCE_METRICS)
    r, n = corr[("aux_risk_rmse", "h_coll_rate")]["r"], corr[("aux_risk_rmse", "h_coll_rate")]["n"]
    assert n == 4
    assert math.isclose(r, -1.0, rel_tol=1e-6)


def test_main_end_to_end_writes_outputs_no_plots(tmp_path):
    csv_path = tmp_path / "eval_summary_20260101_000000.csv"
    _write_eval_summary_csv(csv_path, [
        _row(eval_global_t=12000 * i, aux_risk_rmse=0.1 * i, h_coll_rate=0.5 - 0.05 * i)
        for i in range(1, 6)
    ])
    out_dir = tmp_path / "out"
    rc = aac.main(["--csv", str(csv_path), "--out", str(out_dir), "--no-plots"])
    assert rc == 0
    assert (out_dir / "aux_correlation_summary.csv").exists()
    summary_json = json.loads((out_dir / "aux_correlation_summary.json").read_text())
    assert "aux_risk_rmse__vs__h_coll_rate" in summary_json
    assert summary_json["aux_risk_rmse__vs__h_coll_rate"]["n_samples"] == 5


def test_main_no_input_does_not_crash(tmp_path):
    out_dir = tmp_path / "out_empty"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        rc = aac.main(["--out", str(out_dir), "--no-plots"])
    assert rc == 0
    assert (out_dir / "aux_correlation_summary.json").exists()
