"""ROS-free unit tests for the env-side aux label generation (AUX_PRED v2).

Covers the append-only TTC + hazard-sector targets and the v2 wire format:
  * label_dim / wire_header grow only when the heads are enabled (backward compat)
  * parse_aux_wire round-trips the v2 header (incl. ttc_len / hazard_len) and
    still parses a legacy v1 header
  * TTC target is the safe sentinel (1.0) when NOT closing, and shrinks toward 0
    as a frontal closing approach gets imminent
  * hazard sectors are all-zero with no humans, and a closing frontal human flags
    the FRONT sector (front-centred bins)
"""

import math

import numpy as np

import aux_prediction_labels as al


def _cfg(**over):
    base = dict(enabled=True, num_sectors=16, horizons_sec=[0.5, 1.0, 1.5],
                risk_distance_scale=3.0)
    base.update(over)
    return al.AuxLabelConfig(base)


# ── label_dim / wire header (append-only, backward compatible) ───────────────
def test_label_dim_backward_compatible_when_disabled():
    c = _cfg()  # ttc/hazard off by default
    assert c.ttc_len == 0 and c.hazard_len == 0
    assert c.label_dim == 3 * 16 + 3            # exact v1 length


def test_label_dim_grows_with_heads():
    c = _cfg(ttc_head_enabled=True, hazard_sector_head_enabled=True,
             hazard_sector_bins=3)
    assert c.label_dim == 3 * 16 + 3 + 1 + 3


def test_wire_roundtrip_v2_carries_ttc_hazard_sizes():
    c = _cfg(ttc_head_enabled=True, hazard_sector_head_enabled=True,
             hazard_sector_bins=3)
    label = [0.0] * c.label_dim
    wire = c.wire_header() + label
    meta, parsed = al.parse_aux_wire(wire)
    assert meta["version"] == al.AUX_WIRE_VERSION == 2
    assert meta["num_sectors"] == 16 and meta["num_horizons"] == 3
    assert meta["ttc_len"] == 1 and meta["hazard_len"] == 3
    assert len(parsed) == c.label_dim


def test_wire_parses_legacy_v1_header():
    # A hand-built v1 wire: [VERSION=1, K, H, h0,h1,h2, <label>]
    K, H = 16, 3
    label = [0.0] * (K * H + H)
    wire = [1.0, float(K), float(H), 0.5, 1.0, 1.5] + label
    meta, parsed = al.parse_aux_wire(wire)
    assert meta["version"] == 1
    assert meta["ttc_len"] == 0 and meta["hazard_len"] == 0
    assert len(parsed) == K * H + H


# ── TTC target ───────────────────────────────────────────────────────────────
def _ttc_value(humans, robot_pose, robot_vel, **cfgkw):
    c = _cfg(ttc_head_enabled=True, hazard_sector_head_enabled=False, **cfgkw)
    lab = al.compute_future_risk_labels(humans, robot_pose, c, robot_vel=robot_vel)
    # ttc is the single value right after [risk (H*K)][min_dist (H)]
    return lab[c.num_sectors * c.num_horizons + c.num_horizons]


def test_ttc_sentinel_when_not_closing():
    # Robot stationary, human standing still 1.5 m ahead -> no closing -> 1.0.
    v = _ttc_value([{"x": 1.5, "y": 0.0, "yaw": 0.0, "v": 0.0}],
                   (0.0, 0.0, 0.0), (0.0, 0.0))
    assert v == 1.0
    # Human moving AWAY (separating) -> still sentinel.
    v2 = _ttc_value([{"x": 1.5, "y": 0.0, "yaw": 0.0, "v": 1.0}],  # walking +x
                    (0.0, 0.0, 0.0), (0.0, 0.0))
    assert v2 == 1.0


def test_ttc_shrinks_as_frontal_closing_gets_imminent():
    # Robot drives +x at 1 m/s toward a standing human -> closing 1 m/s.
    R, Th = 0.5, 3.0
    far = _ttc_value([{"x": 3.0, "y": 0.0, "yaw": 0.0, "v": 0.0}],
                     (0.0, 0.0, 0.0), (1.0, 0.0),
                     ttc_collision_radius=R, ttc_horizon_sec=Th)
    near = _ttc_value([{"x": 1.0, "y": 0.0, "yaw": 0.0, "v": 0.0}],
                      (0.0, 0.0, 0.0), (1.0, 0.0),
                      ttc_collision_radius=R, ttc_horizon_sec=Th)
    # gap/closing: far=(3-0.5)/1=2.5 -> /3=0.833 ; near=(1-0.5)/1=0.5 -> /3=0.167
    assert near < far < 1.0
    assert far == __import__("pytest").approx(2.5 / 3.0, abs=1e-6)
    assert near == __import__("pytest").approx(0.5 / 3.0, abs=1e-6)


def test_ttc_zero_on_contact():
    # Human already within the collision radius and closing -> 0.0.
    v = _ttc_value([{"x": 0.3, "y": 0.0, "yaw": 0.0, "v": 0.0}],
                   (0.0, 0.0, 0.0), (1.0, 0.0), ttc_collision_radius=0.5)
    assert v == 0.0


def test_ttc_safe_when_no_humans():
    v = _ttc_value([], (0.0, 0.0, 0.0), (1.0, 0.0))
    assert v == 1.0


# ── Hazard sectors ───────────────────────────────────────────────────────────
def _hazard_vec(humans, robot_pose, robot_vel, bins=3, **cfgkw):
    c = _cfg(ttc_head_enabled=False, hazard_sector_head_enabled=True,
             hazard_sector_bins=bins, **cfgkw)
    lab = al.compute_future_risk_labels(humans, robot_pose, c, robot_vel=robot_vel)
    return list(lab[-bins:]), c


def test_hazard_all_zero_with_no_humans():
    hz, c = _hazard_vec([], (0.0, 0.0, 0.0), (1.0, 0.0))
    assert hz == [0.0, 0.0, 0.0]


def test_hazard_all_zero_when_not_closing():
    # Standing human within range but robot stationary -> not closing -> no flag.
    hz, _ = _hazard_vec([{"x": 1.0, "y": 0.0, "yaw": 0.0, "v": 0.0}],
                        (0.0, 0.0, 0.0), (0.0, 0.0), hazard_distance=3.0)
    assert hz == [0.0, 0.0, 0.0]


def test_hazard_flags_front_sector_on_frontal_closing():
    # Robot heading +x at 1 m/s, closing human dead ahead within range.
    hz, _ = _hazard_vec([{"x": 1.0, "y": 0.0, "yaw": 0.0, "v": 0.0}],
                        (0.0, 0.0, 0.0), (1.0, 0.0), hazard_distance=3.0)
    assert hz[0] == 1.0                      # front bin (bin 0 = front-centred)
    assert hz[1] == 0.0 and hz[2] == 0.0


def test_hazard_left_vs_right_sectors():
    # Robot at origin heading +x (stationary). For S=3 front-centred bins, the
    # front bin spans bearing [-60, 60]; left = (60, 180], right = [-180, -60).
    # Place each human well inside its bin and walking TOWARD the robot (closing).
    # LEFT: human at (0.5, 1.5) -> bearing ~71.6deg (left bin), moving to origin.
    yaw_to_origin = lambda px, py: math.atan2(-py, -px)
    left, _ = _hazard_vec(
        [{"x": 0.5, "y": 1.5, "yaw": yaw_to_origin(0.5, 1.5), "v": 1.0}],
        (0.0, 0.0, 0.0), (0.0, 0.0), hazard_distance=3.0)
    assert left[1] == 1.0 and left[0] == 0.0 and left[2] == 0.0
    # RIGHT: human at (0.5, -1.5) -> bearing ~-71.6deg (right bin).
    right, _ = _hazard_vec(
        [{"x": 0.5, "y": -1.5, "yaw": yaw_to_origin(0.5, -1.5), "v": 1.0}],
        (0.0, 0.0, 0.0), (0.0, 0.0), hazard_distance=3.0)
    assert right[2] == 1.0 and right[0] == 0.0 and right[1] == 0.0


def test_hazard_ignores_far_humans():
    hz, _ = _hazard_vec([{"x": 5.0, "y": 0.0, "yaw": 0.0, "v": 0.0}],
                        (0.0, 0.0, 0.0), (1.0, 0.0), hazard_distance=3.0)
    assert hz == [0.0, 0.0, 0.0]


# ── Env (AuxLabelConfig) vs agent (AuxPredConfig) label-geometry contract ────
def test_env_agent_aux_label_geometry_agree():
    """The env emits the label and the agent slices/heads it; the two configs
    live in separate files and MUST agree on every geometry field the runtime
    fail-fast (curriculum_aux_eval._check_aux_label_contract) checks. Verify the
    shipped configs statically. AuxPredConfig lives in a torch module, so import
    it lazily and skip if torch is unavailable."""
    import pytest
    from pathlib import Path
    import yaml
    try:
        import aux_prediction as ap                  # agent side (torch)
    except Exception:
        pytest.skip("aux_prediction unavailable (torch not installed)")

    cfg_dir = Path(__file__).resolve().parent.parent / "config"
    load = lambda n: yaml.safe_load(open(cfg_dir / n))
    env_aux = load("environment_curriculum.yaml")["environment"]["aux_prediction"]
    agent_aux = load("hyperparameters_tqc.yaml")["hyperparameters"]["aux_prediction"]
    e = al.AuxLabelConfig(env_aux)
    a = ap.AuxPredConfig(agent_aux)

    assert e.num_sectors == a.num_sectors
    assert e.num_horizons == a.num_horizons
    assert list(e.horizons_sec) == list(a.horizons_sec)
    assert e.ttc_len == a.ttc_len
    assert e.hazard_len == a.hazard_len
    # The single most important invariant: identical total label length.
    assert e.label_dim == a.label_dim
    # And the wire header the agent validates round-trips to the agent's geometry.
    meta, label = al.parse_aux_wire(e.wire_header() + [0.0] * e.label_dim)
    assert meta["version"] == al.AUX_WIRE_VERSION
    assert meta["num_sectors"] == a.num_sectors
    assert meta["num_horizons"] == a.num_horizons
    assert meta["ttc_len"] == a.ttc_len
    assert meta["hazard_len"] == a.hazard_len
    assert len(label) == a.label_dim
