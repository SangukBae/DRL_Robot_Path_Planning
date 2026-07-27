"""Unit tests for drl_agent.evaluation.risk_map_dump (Phase 1b).

Pure-function / file-I/O tests, no ROS/torch -- follows the existing test
convention (direct import, plain fake dicts, pytest's tmp_path fixture).
"""

import math

import drl_agent.evaluation.risk_map_dump as rmd


def test_sector_index_for_theta_matches_aux_prediction_labels_convention():
    # Straight ahead (theta=0) -> the MIDDLE bin, not bin 0 (this is the
    # risk_map convention, NOT the front-centered hazard-sector convention --
    # see aux_prediction_labels.py:248-253).
    K = 16
    assert rmd.sector_index_for_theta(0.0, K) == K // 2
    # Directly behind (+/- pi) -> bin 0 or K-1 (wrap boundary).
    assert rmd.sector_index_for_theta(math.pi, K) in (0, K - 1)
    assert rmd.sector_index_for_theta(-math.pi, K) in (0, K - 1)


def test_sector_index_for_theta_clamped_within_range():
    K = 8
    for theta in (-10.0, -math.pi - 0.01, 0.0, 1.0, math.pi + 0.01, 10.0):
        idx = rmd.sector_index_for_theta(theta, K)
        assert 0 <= idx < K


def test_sector_index_for_theta_zero_sectors_does_not_crash():
    assert rmd.sector_index_for_theta(0.5, 0) == 0


def test_human_state_summary_from_dict_of_dicts():
    human_states = {
        "h1": {"x": 1.0, "y": 2.0, "yaw": 0.5, "v": 0.3, "mode": "crossing"},
        "h2": {"x": -1.0, "y": 0.0},  # missing yaw/v/mode -> defaults
    }
    out = rmd.human_state_summary(human_states)
    assert len(out) == 2
    assert out[0] == {"x": 1.0, "y": 2.0, "yaw": 0.5, "v": 0.3, "mode": "crossing"}
    assert out[1] == {"x": -1.0, "y": 0.0, "yaw": 0.0, "v": 0.0, "mode": ""}


def test_human_state_summary_none_returns_empty_list():
    assert rmd.human_state_summary(None) == []


def test_writer_round_trip_with_full_aux_data(tmp_path):
    path = tmp_path / "dump.jsonl"
    with rmd.RiskMapDumpWriter(str(path)) as w:
        w.write_step(
            episode=1, step=0,
            robot_pose=(0.0, 0.0, 0.0),
            action_r=1.5, action_theta=0.1,
            num_sectors=16,
            gt_risk_map=[0.1] * 48, gt_min_dist=[0.9, 0.8, 0.7],
            pred_risk_map=[0.2] * 48, pred_min_dist=[0.85, 0.75, 0.65],
            humans=[{"x": 1.0, "y": 1.0, "yaw": 0.0, "v": 0.5, "mode": "waiting"}],
        )
    records = rmd.read_dump(str(path))
    assert len(records) == 1
    r = records[0]
    assert r["episode"] == 1 and r["step"] == 0
    assert r["robot_x"] == 0.0 and r["robot_yaw"] == 0.0
    assert len(r["gt_risk_map"]) == 48
    assert len(r["pred_risk_map"]) == 48
    assert r["sector_index"] == 8  # theta=0.1 (near-zero) -> near-middle bin
    assert r["humans"][0]["mode"] == "waiting"


def test_writer_handles_aux_disabled_gracefully(tmp_path):
    """aux off at the env AND the policy -> every aux-related field is None,
    never a crash, and the record is still a valid JSON line."""
    path = tmp_path / "dump_aux_off.jsonl"
    with rmd.RiskMapDumpWriter(str(path)) as w:
        record = w.write_step(
            episode=2, step=5,
            robot_pose=(1.0, 2.0, 0.3),
            action_r=1.0, action_theta=0.0,
            num_sectors=None,   # unknown geometry when aux is fully off
            gt_risk_map=None, gt_min_dist=None,
            pred_risk_map=None, pred_min_dist=None,
            humans=None,
        )
    assert record["gt_risk_map"] is None
    assert record["pred_risk_map"] is None
    assert record["sector_index"] is None
    assert record["humans"] == []
    records = rmd.read_dump(str(path))
    assert records[0]["gt_risk_map"] is None


def test_writer_multiple_steps_append_in_order(tmp_path):
    path = tmp_path / "multi.jsonl"
    with rmd.RiskMapDumpWriter(str(path)) as w:
        for step in range(3):
            w.write_step(episode=1, step=step)
    records = rmd.read_dump(str(path))
    assert [r["step"] for r in records] == [0, 1, 2]


def test_writer_extra_fields_are_merged(tmp_path):
    path = tmp_path / "extra.jsonl"
    with rmd.RiskMapDumpWriter(str(path)) as w:
        w.write_step(episode=1, step=0, extra={"map_type": "corridor"})
    records = rmd.read_dump(str(path))
    assert records[0]["map_type"] == "corridor"


def test_writer_creates_missing_parent_directories(tmp_path):
    """-p dump_path:=some/new/dir/file.jsonl must not crash (finding: the
    writer previously assumed the parent directory already existed)."""
    path = tmp_path / "new" / "nested" / "dir" / "dump.jsonl"
    assert not path.parent.exists()
    with rmd.RiskMapDumpWriter(str(path)) as w:
        w.write_step(episode=1, step=0)
    assert path.exists()
    assert rmd.read_dump(str(path))[0]["step"] == 0
