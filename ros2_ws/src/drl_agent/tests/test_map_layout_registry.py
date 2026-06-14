"""ROS-free unit tests for map_layout_registry.build_map_layouts (pure data)."""

import pytest

import map_layout_registry as reg


def _layouts():
    return reg.build_map_layouts(
        map_inner_lower=-5.0,
        map_inner_upper=5.0,
        map_wall_thickness=0.2,
        map_corridor_width=2.0,
        map_corridor_passage_width=1.0,
        map_intersection_width=2.0,
        map_intersection_passage_width=1.0,
        map_lobby_open_half_extent=2.5,
        map_start_band_depth=1.5,
    )


def test_all_map_types_present():
    layouts = _layouts()
    assert set(layouts) == {"lobby", "corridor", "intersection", "clutter"}


def test_wall_counts():
    layouts = _layouts()
    assert len(layouts["lobby"]["walls"]) == 0
    assert len(layouts["corridor"]["walls"]) == 2
    assert len(layouts["intersection"]["walls"]) == 4
    assert len(layouts["clutter"]["walls"]) == 4


def test_wall_names_match_wall_count():
    for mt, spec in _layouts().items():
        assert len(spec["wall_names"]) == len(spec["walls"])
        for i, name in enumerate(spec["wall_names"]):
            assert name == f"rl_wall_{mt}_{i:02d}"


def test_lobby_open_area_set():
    lobby = _layouts()["lobby"]
    assert lobby["open_area"] == (-2.5, 2.5, -2.5, 2.5)


def test_corridor_reserved_passage_and_regions():
    corridor = _layouts()["corridor"]
    assert corridor["open_area"] is None
    assert len(corridor["reserved_passages"]) == 1
    assert corridor["reserved_passages"][0]["axis"] == "x"
    # start/goal metadata exists for the structured curriculum
    assert {r["name"] for r in corridor["start_regions"]} == {"left", "right"}
    assert set(corridor["goal_regions"]) == {"left", "right"}


def test_intersection_has_four_arms_and_cross_passages():
    inter = _layouts()["intersection"]
    assert {r["name"] for r in inter["start_regions"]} == {"left", "right", "bottom", "top"}
    axes = {p["axis"] for p in inter["reserved_passages"]}
    assert axes == {"x", "y"}


def test_geometry_scales_with_params():
    # Larger corridor width → walls pushed further out (|cy| grows).
    narrow = reg.build_map_layouts(
        map_inner_lower=-5.0, map_inner_upper=5.0, map_wall_thickness=0.2,
        map_corridor_width=2.0, map_corridor_passage_width=1.0,
        map_intersection_width=2.0, map_intersection_passage_width=1.0,
        map_lobby_open_half_extent=2.5, map_start_band_depth=1.5,
    )["corridor"]["walls"][0]["cy"]
    wide = reg.build_map_layouts(
        map_inner_lower=-5.0, map_inner_upper=5.0, map_wall_thickness=0.2,
        map_corridor_width=4.0, map_corridor_passage_width=1.0,
        map_intersection_width=2.0, map_intersection_passage_width=1.0,
        map_lobby_open_half_extent=2.5, map_start_band_depth=1.5,
    )["corridor"]["walls"][0]["cy"]
    assert abs(wide) > abs(narrow)
