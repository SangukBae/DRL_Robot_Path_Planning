"""ROS-free unit tests for scripts/utils/aux_ablation_summary.py's CSV/manifest
discovery -- specifically the RUN_LAYOUT dual-layout glob change (matching both
the legacy ``eval_summary_<tag>.csv`` and the new plain ``eval_summary.csv``
under ``runtime/experiments/<run_id>/logs/``).
"""

import json

import aux_ablation_summary as aas


def test_find_csvs_matches_legacy_timestamped_filename(tmp_path):
    logs = tmp_path / "tqc" / "seed_0" / "logs"
    logs.mkdir(parents=True)
    csv_path = logs / "eval_summary_20260701_000000.csv"
    csv_path.write_text("seed\n0\n")

    found = aas._find_csvs([str(tmp_path)])
    assert found == [str(csv_path)]


def test_find_csvs_matches_new_run_layout_plain_filename(tmp_path):
    logs = tmp_path / "experiments" / "20260726_101200_tqc_both_seed0" / "logs"
    logs.mkdir(parents=True)
    csv_path = logs / "eval_summary.csv"
    csv_path.write_text("seed\n0\n")

    found = aas._find_csvs([str(tmp_path)])
    assert found == [str(csv_path)]


def test_find_csvs_deduplicates_and_accepts_explicit_file(tmp_path):
    logs = tmp_path / "experiments" / "run" / "logs"
    logs.mkdir(parents=True)
    csv_path = logs / "eval_summary.csv"
    csv_path.write_text("seed\n0\n")

    # Passing the dir AND the explicit file must not double-count it.
    found = aas._find_csvs([str(tmp_path), str(csv_path)])
    assert len(found) == 1


def test_run_tag_of_is_empty_for_plain_filename():
    # A plain "eval_summary.csv" (new layout) carries no run tag.
    assert aas._run_tag_of("/x/logs/eval_summary.csv") == ""


def test_run_tag_of_extracts_legacy_tag():
    assert aas._run_tag_of("/x/logs/eval_summary_20260701_000000.csv") == "20260701_000000"


def test_load_manifest_falls_back_to_plain_run_manifest_for_new_layout(tmp_path):
    logs = tmp_path / "experiments" / "run" / "logs"
    logs.mkdir(parents=True)
    csv_path = logs / "eval_summary.csv"
    csv_path.write_text("seed\n0\n")
    manifest = {"run_id": "20260726_101200_tqc_both_seed0", "seed": 0}
    (logs / "run_manifest.json").write_text(json.dumps(manifest))

    loaded = aas._load_manifest(str(csv_path))
    assert loaded == manifest
