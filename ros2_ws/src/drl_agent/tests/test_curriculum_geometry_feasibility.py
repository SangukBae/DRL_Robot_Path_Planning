"""Regression tests for structured-map geometry vs. the active curriculum.

These tests guard against a silent config contradiction where the reserved
passage, wall-clearance margin and obstacle radii leave *no* legal off-lane
placement area in corridor/intersection maps. That failure mode produces the
runtime warnings seen in Gazebo resets ("Pool: no free pose for static") and
episodes collapse to empty lanes.
"""

from pathlib import Path

import yaml

import drl_agent.env.simulation.map_catalog as mc
import drl_agent.env.simulation.map_layout_registry as reg


ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = ROOT / "config" / "environment_curriculum.yaml"
ASSET_CFG = ROOT.parent / "drl_obstacle_assets" / "config" / "obstacle_catalog.yaml"


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _lane_fit(radius: float, lane_width: float, passage_width: float,
              wall_clearance: float, passage_safety_margin: float) -> bool:
    return reg.structured_lane_footprint_fits(
        radius=radius,
        lane_width=lane_width,
        passage_width=passage_width,
        wall_clearance=wall_clearance,
        passage_safety_margin=passage_safety_margin,
    )


def _curriculum_stage_max_active_static(stage_list, map_type: str) -> int:
    return max(
        int(stage.get("active_static", 0))
        for stage in stage_list
        if map_type in (stage.get("allowed_map_types") or [])
    )


def test_corridor_has_enough_fittable_static_types_for_its_hardest_stage():
    cfg = _load_yaml(ENV_CFG)
    env = cfg["environment"]
    stages = cfg["curriculum"]["stages"]
    catalog = _load_yaml(ASSET_CFG)["obstacles"]
    radii = {entry["key"]: float(entry["radius"]) for entry in catalog}

    fit = [
        key for key in mc.MAP_TYPE_ALLOWED_STATIC_KEYS["corridor"]
        if key in radii and _lane_fit(
            radii[key],
            float(env["map_corridor_width"]),
            float(env["map_corridor_passage_width"]),
            float(env["map_wall_clearance"]),
            float(env["map_passage_safety_margin"]),
        )
    ]
    need = _curriculum_stage_max_active_static(stages, "corridor")
    assert len(fit) >= need


def test_intersection_has_enough_fittable_static_types_for_its_hardest_stage():
    cfg = _load_yaml(ENV_CFG)
    env = cfg["environment"]
    stages = cfg["curriculum"]["stages"]
    catalog = _load_yaml(ASSET_CFG)["obstacles"]
    radii = {entry["key"]: float(entry["radius"]) for entry in catalog}

    fit = [
        key for key in mc.MAP_TYPE_ALLOWED_STATIC_KEYS["intersection"]
        if key in radii and _lane_fit(
            radii[key],
            float(env["map_intersection_width"]),
            float(env["map_intersection_passage_width"]),
            float(env["map_wall_clearance"]),
            float(env["map_passage_safety_margin"]),
        )
    ]
    need = _curriculum_stage_max_active_static(stages, "intersection")
    assert len(fit) >= need


def test_corridor_human_radius_fits_off_lane_spawn_band():
    env = _load_yaml(ENV_CFG)["environment"]
    human = _load_yaml(ASSET_CFG)["humans"][0]
    # Human spawn uses safety_margin=0.0 for the reserved passage check, but
    # the body itself must still fit outside the passage and off the walls.
    assert _lane_fit(
        float(human["radius"]),
        float(env["map_corridor_width"]),
        float(env["map_corridor_passage_width"]),
        float(env["map_wall_clearance"]),
        0.0,
    )


def test_known_corridor_allowed_keys_split_into_fit_and_no_fit_groups():
    env = _load_yaml(ENV_CFG)["environment"]
    radii = {
        entry["key"]: float(entry["radius"])
        for entry in _load_yaml(ASSET_CFG)["obstacles"]
    }
    fits = {
        key for key in mc.MAP_TYPE_ALLOWED_STATIC_KEYS["corridor"]
        if key in radii and _lane_fit(
            radii[key],
            float(env["map_corridor_width"]),
            float(env["map_corridor_passage_width"]),
            float(env["map_wall_clearance"]),
            float(env["map_passage_safety_margin"]),
        )
    }
    assert "warehouse_trash_can" in fits
    assert "hospital_mop_cart" in fits
    assert "hospital_drawer" not in fits
    assert "house_refrigerator" not in fits
