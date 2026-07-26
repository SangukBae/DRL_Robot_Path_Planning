"""Unit tests for scripts/utils/analyze_yield_freezing.py (Phase 1a).

Pure-function / file-I/O tests, no ROS -- follows the existing convention
(direct import of the module, no fixtures beyond pytest's tmp_path).
"""

import csv
import json
import math
import warnings

import analyze_yield_freezing as ayf


def _write_dyn_avoid_csv(path, rows):
    fieldnames = [
        "episode", "global_t", "curriculum_stage", "map_type", "seed",
        "aux_enabled", "aux_version", "success", "collision", "timeout",
        "total_reward", "steps", "yield_available", "yield_used",
        "yield_trigger_count", "yield_steps", "yield_in_risk_steps",
        "yield_no_risk_steps", "risk_steps", "low_obs_speed_frac",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _row(**kw):
    base = dict(
        episode=1, global_t=100, curriculum_stage=5, map_type="corridor", seed=0,
        aux_enabled=1, aux_version=1, success=0, collision=0, timeout=0,
        total_reward=1.0, steps=100, yield_available=1, yield_used=1,
        yield_trigger_count=2, yield_steps=10, yield_in_risk_steps=7,
        yield_no_risk_steps=3, risk_steps=12, low_obs_speed_frac=0.1,
    )
    base.update(kw)
    return base


def test_derive_episode_metrics_basic_ratios():
    row = _row()
    out = ayf.derive_episode_metrics(row)
    assert out["yield_precision"] == 7 / 10
    assert out["yield_recall"] == 7 / 12
    assert out["bad_yield_rate"] == 3 / 10
    assert out["yield_mean_streak_steps"] == 10 / 2
    assert out["yield_step_frac"] == 10 / 100


def test_derive_episode_metrics_zero_denominator_is_nan_not_crash():
    row = _row(yield_steps=0, yield_trigger_count=0, risk_steps=0)
    out = ayf.derive_episode_metrics(row)
    assert math.isnan(out["yield_precision"])
    assert math.isnan(out["yield_recall"])
    assert math.isnan(out["bad_yield_rate"])
    assert math.isnan(out["yield_mean_streak_steps"])


def test_derive_episode_metrics_handles_nan_string_and_empty():
    row = _row(yield_steps="nan", risk_steps="")
    out = ayf.derive_episode_metrics(row)
    assert math.isnan(out["yield_precision"])
    assert math.isnan(out["yield_recall"])


def test_load_episode_rows_missing_file_warns_not_raises(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rows = ayf.load_episode_rows([str(missing)])
    assert rows == []
    assert any("not found" in str(x.message) for x in w)


def test_load_episode_rows_missing_columns_warns_and_fills_nan(tmp_path):
    p = tmp_path / "partial.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "steps", "timeout"])
        w.writerow([1, 50, 0])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rows = ayf.load_episode_rows([str(p)])
    assert len(rows) == 1
    assert any("missing columns" in str(x.message) for x in w)
    derived = ayf.derive_episode_metrics(rows[0])
    assert math.isnan(derived["yield_precision"])


def test_summarize_group_by_map_type(tmp_path):
    rows = [
        ayf.derive_episode_metrics(_row(map_type="corridor", timeout=0)),
        ayf.derive_episode_metrics(_row(map_type="corridor", timeout=1,
                                         yield_steps=5, yield_used=1)),
        ayf.derive_episode_metrics(_row(map_type="intersection", timeout=0)),
    ]
    summary = ayf.summarize(rows, group_by="map_type")
    assert set(summary.keys()) == {"corridor", "intersection"}
    assert summary["corridor"]["n_episodes"] == 2
    assert summary["corridor"]["timeout_n"] == 1
    assert summary["intersection"]["n_episodes"] == 1


def test_main_end_to_end_writes_outputs(tmp_path):
    csv_path = tmp_path / "dynamic_avoidance_metrics_20260101_000000.csv"
    _write_dyn_avoid_csv(csv_path, [_row(episode=i) for i in range(1, 4)])
    out_dir = tmp_path / "out"
    rc = ayf.main(["--csv", str(csv_path), "--out", str(out_dir)])
    assert rc == 0
    assert (out_dir / "per_episode_yield_freezing.csv").exists()
    assert (out_dir / "yield_freezing_summary.csv").exists()
    summary_json = json.loads((out_dir / "yield_freezing_summary.json").read_text())
    assert summary_json["n_episodes"] == 3
    assert summary_json["groups"]["all"]["n_episodes"] == 3


def test_main_no_input_does_not_crash(tmp_path):
    out_dir = tmp_path / "out_empty"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        rc = ayf.main(["--out", str(out_dir)])
    assert rc == 0
    assert (out_dir / "yield_freezing_summary.json").exists()
