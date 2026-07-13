#!/usr/bin/env python3
"""Unit tests for the DYN_AVOID dynamic-obstacle avoidance telemetry.

Covers the pure privileged accumulator (dynamic_avoidance_telemetry) and the
consolidated CSV writer (dynamic_avoidance_log): header==row, NaN conventions,
collision attribution, yield accounting, nearest-mode tracking, and the
static-clutter vs. human-interaction split.
"""

import csv
import math
import os
import sys

# tests/ -> package root; add scripts/environment + scripts/utils to the path,
# mirroring how the trainer imports these modules at runtime.
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PKG, "scripts", "environment"))
sys.path.insert(0, os.path.join(_PKG, "scripts", "utils"))

from dynamic_avoidance_telemetry import DynamicAvoidanceEpisodeDiag  # noqa: E402
from dynamic_avoidance_log import (  # noqa: E402
    DynamicAvoidanceCSV, DYN_AVOID_HEADER, driving_mean, low_obs_speed_frac,
)


def _robot_still():
    return (0.0, 0.0, 0.0), 0.0  # pose, speed


# ── accumulator ──────────────────────────────────────────────────────────
def test_no_humans_reports_unavailable():
    d = DynamicAvoidanceEpisodeDiag()
    pose, v = _robot_still()
    for _ in range(5):
        d.update(pose, v, [], min_lidar_m=3.0, collision=False)
    out = d.as_dict()
    assert out["collision_object_type"] == "none"
    assert math.isnan(out["min_human_distance_m"])
    assert math.isnan(out["human_ttc_min"])
    assert math.isnan(out["time_below_human_clearance_ratio"])
    assert math.isnan(out["near_human_event"])   # undefined: no human ever seen
    assert out["has_human_interaction"] == 0
    assert out["human_observed_steps"] == 0
    assert out["nearest_human_mode"] == ""
    assert out["human_modes_present"] == ""


def test_min_distance_and_near_ratio_and_mode():
    d = DynamicAvoidanceEpisodeDiag(near_human_dist_m=1.0, interaction_radius_m=2.0)
    pose, v = _robot_still()
    # step 1: human at 1.5 m (within interaction, outside near)
    d.update(pose, v, [{"x": 1.5, "y": 0.0, "v": 0.0, "yaw": 0.0, "mode": "crossing"}],
             min_lidar_m=1.5, collision=False)
    # step 2: human at 0.8 m (within near) -> risk step; closest ever
    d.update(pose, v, [{"x": 0.8, "y": 0.0, "v": 0.0, "yaw": 0.0, "mode": "crossing"}],
             min_lidar_m=0.8, collision=False)
    out = d.as_dict()
    assert abs(out["min_human_distance_m"] - 0.8) < 1e-6
    assert out["human_observed_steps"] == 2
    assert abs(out["time_below_human_clearance_ratio"] - 0.5) < 1e-6  # 1 of 2 steps
    assert out["near_human_event"] == 1
    assert out["has_human_interaction"] == 1
    assert out["nearest_human_mode"] == "crossing"
    assert out["human_modes_present"] == "crossing"
    assert out["risk_steps"] == 1


def test_modes_present_sorted_union():
    d = DynamicAvoidanceEpisodeDiag()
    pose, v = _robot_still()
    d.update(pose, v, [
        {"x": 3.0, "y": 0.0, "mode": "waiting"},
        {"x": 1.0, "y": 0.0, "mode": "crossing"},
    ], min_lidar_m=1.0, collision=False)
    out = d.as_dict()
    assert out["human_modes_present"] == "crossing,waiting"
    assert out["nearest_human_mode"] == "crossing"   # nearest = 1.0 m human


def test_ttc_only_closing_humans():
    d = DynamicAvoidanceEpisodeDiag(ttc_collision_radius_m=0.5)
    pose, v = _robot_still()
    # human 3 m ahead, moving toward robot at 1 m/s (yaw=pi -> -x direction)
    d.update(pose, v, [{"x": 3.0, "y": 0.0, "v": 1.0, "yaw": math.pi}],
             min_lidar_m=3.0, collision=False)
    out = d.as_dict()
    # gap = 3 - 0.5 = 2.5 m, closing 1 m/s -> 2.5 s
    assert abs(out["human_ttc_min"] - 2.5) < 1e-3
    # separating human contributes no TTC
    d2 = DynamicAvoidanceEpisodeDiag()
    d2.update(pose, v, [{"x": 3.0, "y": 0.0, "v": 1.0, "yaw": 0.0}],
              min_lidar_m=3.0, collision=False)
    assert math.isnan(d2.as_dict()["human_ttc_min"])


def test_collision_attribution_human_vs_static():
    # Human close at the collision step -> "human".
    d = DynamicAvoidanceEpisodeDiag(collision_attrib_radius_m=0.7)
    pose, v = _robot_still()
    d.update(pose, v, [{"x": 0.5, "y": 0.0, "mode": "crossing"}],
             min_lidar_m=0.4, collision=True)
    assert d.as_dict()["collision_object_type"] == "human"
    # Collision with only a far human -> "static".
    d2 = DynamicAvoidanceEpisodeDiag(collision_attrib_radius_m=0.7)
    d2.update(pose, v, [{"x": 3.0, "y": 0.0, "mode": "crossing"}],
              min_lidar_m=0.3, collision=True)
    assert d2.as_dict()["collision_object_type"] == "static"
    # Collision with no humans at all -> "static".
    d3 = DynamicAvoidanceEpisodeDiag()
    d3.update(pose, v, [], min_lidar_m=0.3, collision=True)
    assert d3.as_dict()["collision_object_type"] == "static"


def test_collision_locks_on_first_step():
    d = DynamicAvoidanceEpisodeDiag(collision_attrib_radius_m=0.7)
    pose, v = _robot_still()
    d.update(pose, v, [{"x": 0.5, "y": 0.0}], min_lidar_m=0.4, collision=True)
    # a later (hypothetical) static-only collision must NOT overwrite the type
    d.update(pose, v, [{"x": 3.0, "y": 0.0}], min_lidar_m=0.3, collision=True)
    assert d.as_dict()["collision_object_type"] == "human"


def test_static_clutter_pressure_split():
    d = DynamicAvoidanceEpisodeDiag(collision_attrib_radius_m=0.7,
                                    static_clutter_lidar_m=0.6)
    pose, v = _robot_still()
    # close geometry, no human nearby -> static clutter
    d.update(pose, v, [{"x": 3.0, "y": 0.0}], min_lidar_m=0.4, collision=False)
    # close geometry AND a human right there -> NOT static clutter
    d.update(pose, v, [{"x": 0.4, "y": 0.0}], min_lidar_m=0.4, collision=False)
    out = d.as_dict()
    assert out["static_clutter_steps"] == 1
    assert out["has_static_clutter_pressure"] == 1


def test_yield_accounting_onsets_and_risk():
    d = DynamicAvoidanceEpisodeDiag(near_human_dist_m=1.0)
    pose, v = _robot_still()
    far = [{"x": 5.0, "y": 0.0}]
    near = [{"x": 0.5, "y": 0.0}]
    # not yielding
    d.update(pose, v, far, 5.0, False, yielding=False, yield_available=True)
    # yield onset, no risk (human far)
    d.update(pose, v, far, 5.0, False, yielding=True, yield_available=True)
    # still yielding, now with a near human -> in-risk
    d.update(pose, v, near, 0.5, False, yielding=True, yield_available=True)
    # stop yielding
    d.update(pose, v, near, 0.5, False, yielding=False, yield_available=True)
    # yield again -> second onset
    d.update(pose, v, near, 0.5, False, yielding=True, yield_available=True)
    out = d.as_dict()
    assert out["yield_available"] == 1
    assert out["yield_used"] == 1
    assert out["yield_steps"] == 3
    assert out["yield_trigger_count"] == 2           # two onsets
    assert out["yield_in_risk_steps"] == 2           # steps 3 and 5
    assert out["yield_no_risk_steps"] == 1           # step 2
    assert out["risk_steps"] == 3                     # steps 3,4,5 had near human


def test_reset_clears_state():
    d = DynamicAvoidanceEpisodeDiag()
    pose, v = _robot_still()
    d.update(pose, v, [{"x": 0.4, "y": 0.0, "mode": "crossing"}], 0.3, True,
             yielding=True, yield_available=True)
    d.reset()
    out = d.as_dict()
    assert out["collision_object_type"] == "none"
    assert out["yield_steps"] == 0
    assert out["human_modes_present"] == ""
    assert math.isnan(out["min_human_distance_m"])


# ── state_key gating (per-step publish is skipped on unchanged content) ────
def test_state_key_stable_when_no_new_info():
    d = DynamicAvoidanceEpisodeDiag()
    pose, v = _robot_still()
    d.update(pose, v, [], min_lidar_m=3.0, collision=False)
    k1 = d.state_key()
    # another human-free, clutter-free step adds no diagnostic content
    d.update(pose, v, [], min_lidar_m=3.0, collision=False)
    assert d.state_key() == k1        # -> env would skip the ROS publish
    # a human appearing IS new content -> key changes
    d.update(pose, v, [{"x": 0.9, "y": 0.0, "mode": "crossing"}], 0.9, False)
    assert d.state_key() != k1


def test_state_key_excludes_total_steps():
    # total_steps advances every step but is NOT part of as_dict, so 3 human-free
    # steps must leave the key identical to a fresh accumulator's key.
    d = DynamicAvoidanceEpisodeDiag()
    pose, v = _robot_still()
    for _ in range(3):
        d.update(pose, v, [], min_lidar_m=5.0, collision=False)
    assert d.total_steps == 3
    assert d.state_key() == DynamicAvoidanceEpisodeDiag().state_key()
    # collision changes content -> key must change
    before = d.state_key()
    d.update(pose, v, [], min_lidar_m=0.2, collision=True)
    assert d.state_key() != before


# ── episode_driving re-exposure parity (abs signed speed, all-steps denom) ─
def test_low_obs_speed_frac_matches_base_definition():
    thr = 0.12
    # reverse (negative) speeds below the magnitude threshold count as low.
    buf = [-0.05, 0.05, 0.5, -0.3]
    # base: mean(abs(sv) < thr) = mean([1,1,0,0]) = 0.5
    assert abs(low_obs_speed_frac(buf, thr) - 0.5) < 1e-9
    # a naive `v < thr` (no abs) would wrongly count -0.3 and -0.05 -> 0.75; guard
    assert low_obs_speed_frac(buf, thr) != 0.75
    # NaN samples count as NOT low; denominator is ALL steps.
    nan = float("nan")
    assert abs(low_obs_speed_frac([nan, 0.05, 0.5], thr) - (1.0 / 3.0)) < 1e-9
    # empty buffer -> NaN.
    assert math.isnan(low_obs_speed_frac([], thr))


def test_driving_mean_matches_base():
    assert abs(driving_mean([0.2, 0.4, 0.6]) - 0.4) < 1e-9
    assert math.isnan(driving_mean([]))


# ── CSV writer ───────────────────────────────────────────────────────────
def _read_rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_csv_header_equals_row_length(tmp_path):
    w = DynamicAvoidanceCSV(str(tmp_path), "unit")
    assert os.path.basename(w.path) == "dynamic_avoidance_metrics_unit.csv"
    d = DynamicAvoidanceEpisodeDiag()
    pose, v = _robot_still()
    d.update(pose, v, [{"x": 0.5, "y": 0.0, "mode": "crossing"}], 0.4, False,
             yielding=True, yield_available=True)
    ep_metrics = {"near_collision_count": 2, "mean_speed_mps": 0.7,
                  "lidar_clearance_rate": 0.9}
    w.write_episode(
        episode=1, global_t=100, stage=3, map_type="corridor",
        seed=0, aux_enabled=1, aux_version=2,
        success=True, collision=False, timeout=False,
        total_reward=12.5, steps=200,
        ep_metrics=ep_metrics, final_goal_dist_m=0.2, mean_gazebo_rtf=0.8,
        mean_cmd_v_mps=0.6, mean_cmd_steering_rad=0.05, low_obs_speed_frac=0.1,
        diag=d.as_dict())
    rows = _read_rows(w.path)
    assert rows[0] == DYN_AVOID_HEADER
    assert len(rows) == 2
    assert len(rows[1]) == len(DYN_AVOID_HEADER)


def test_csv_near_event_success_and_missing_diag(tmp_path):
    w = DynamicAvoidanceCSV(str(tmp_path), "miss")
    ep_metrics = {"near_collision_count": 0, "mean_speed_mps": 0.5,
                  "lidar_clearance_rate": 1.0}
    # No diag at all + a collision -> collision_object_type must be "unknown".
    w.write_episode(
        episode=1, global_t=1, stage=0, map_type="", seed=1,
        aux_enabled=0, aux_version=0,
        success=False, collision=True, timeout=False,
        total_reward=-5.0, steps=50, ep_metrics=ep_metrics,
        final_goal_dist_m=1.0, mean_gazebo_rtf=float("nan"),
        mean_cmd_v_mps=float("nan"), mean_cmd_steering_rad=float("nan"),
        low_obs_speed_frac=float("nan"), diag={})
    rows = _read_rows(w.path)
    header, row = rows[0], rows[1]
    rec = dict(zip(header, row))
    assert rec["collision_object_type"] == "unknown"
    assert rec["min_human_distance_m"] == "nan"
    assert rec["near_human_event"] == "nan"
    assert rec["near_human_event_success"] == "nan"


def test_csv_near_event_success_true(tmp_path):
    w = DynamicAvoidanceCSV(str(tmp_path), "succ")
    d = DynamicAvoidanceEpisodeDiag(near_human_dist_m=1.0)
    pose, v = _robot_still()
    d.update(pose, v, [{"x": 0.5, "y": 0.0, "mode": "crossing"}], 0.5, False)
    ep_metrics = {"near_collision_count": 1, "mean_speed_mps": 0.6,
                  "lidar_clearance_rate": 0.8}
    w.write_episode(
        episode=2, global_t=2, stage=1, map_type="lobby", seed=0,
        aux_enabled=1, aux_version=2, success=True, collision=False,
        timeout=False, total_reward=9.0, steps=120, ep_metrics=ep_metrics,
        final_goal_dist_m=0.1, mean_gazebo_rtf=0.9, mean_cmd_v_mps=0.5,
        mean_cmd_steering_rad=0.0, low_obs_speed_frac=0.0, diag=d.as_dict())
    rec = dict(zip(*(_read_rows(w.path))))
    assert rec["near_human_event"] == "1"
    assert rec["near_human_event_success"] == "1"
    assert rec["nearest_human_mode"] == "crossing"
