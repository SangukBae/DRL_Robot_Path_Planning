"""Tests for kinematic-pedestrian internal-wall collision handling.

Humans are teleported via set_pose (no physics), so without an explicit check
they walk straight through corridor / intersection / clutter internal walls.
``HumanMotionMixin._resolve_human_wall_collision`` is the per-step guard: it
rejects a step whose body would overlap an internal wall, sliding along the wall
axis-aligned when one component is free and holding (blocked=True) only when
fully boxed in.

The environment modules import ROS message packages unavailable in this ROS-free
suite, so the ROS deps are stubbed in sys.modules before import (the resolver
touches none of them — only ``_point_in_walls`` / ``current_layout_spec``).
"""

import sys
import types

import pytest


def _install_ros_stubs():
    class _Meta(type):
        _n = [0]

        def __getattr__(cls, name):
            _Meta._n[0] += 1
            return _Meta._n[0]

    def _dummy(name):
        return _Meta(name, (), {})

    for name in (
        "squaternion", "rclpy", "rclpy.parameter",
        "geometry_msgs", "geometry_msgs.msg",
        "nav_msgs", "nav_msgs.msg",
        "sensor_msgs", "sensor_msgs.msg",
        "visualization_msgs", "visualization_msgs.msg",
        "drl_agent_interfaces", "drl_agent_interfaces.msg",
        "ros_gz_interfaces", "ros_gz_interfaces.msg", "ros_gz_interfaces.srv",
    ):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__getattr__ = _dummy
            sys.modules[name] = mod


_install_ros_stubs()

import drl_agent.env.humans.human_motion_manager as hmm  # noqa: E402  (ROS deps now stubbed)
import drl_agent.env.humans.human_spawn_sampler as hss  # noqa: E402  (_point_in_human_regions source)
import drl_agent.env.simulation.map_layout_runtime as mlr  # noqa: E402  (_point_in_walls source)
import drl_agent.env.simulation.map_layout_registry as reg  # noqa: E402  (build real layouts)


class _Node(hmm.HumanMotionMixin, mlr.MapLayoutMixin):
    def __init__(self, walls):
        self.current_layout_spec = {"walls": walls} if walls is not None else None
        self.human_states = {}
        self.spawned_obstacle_records = {}
        self.human_static_obstacle_clearance = 0.2


class _RegionNode(hss.HumanSpawnMixin):
    def __init__(self, spec):
        self.current_layout_spec = spec


# A horizontal wall along x at y=0 (thickness 0.4 -> half 0.2).
_H_WALL = [{"cx": 0.0, "cy": 0.0, "sx": 10.0, "sy": 0.4}]
# An L-shaped concave corner (two perpendicular boxes).
_L_WALLS = [
    {"cx": 1.0, "cy": 0.0, "sx": 2.0, "sy": 0.4},   # x∈[0,2], y∈[-0.2,0.2]
    {"cx": 0.0, "cy": 1.0, "sx": 0.4, "sy": 2.0},   # x∈[-0.2,0.2], y∈[0,2]
]
R = 0.3


def test_lobby_no_walls_is_passthrough():
    node = _Node([])            # lobby: no internal walls
    out = node._resolve_human_wall_collision(0.0, -1.0, 0.0, 5.0, R)
    assert out == (0.0, 5.0, False)
    node2 = _Node(None)        # non-structured map
    assert node2._resolve_human_wall_collision(0.0, -1.0, 0.0, 5.0, R) == (0.0, 5.0, False)


def test_perpendicular_step_into_wall_is_not_penetrated():
    node = _Node(_H_WALL)
    # Approaching the wall from below; target would cross into the y-band.
    rx, ry, blocked = node._resolve_human_wall_collision(-1.0, -0.8, -1.0, -0.3, R)
    # Must NOT end up inside the wall (|y| < 0.2+0.3 = 0.5).
    assert not node._point_in_walls(rx, ry, R)
    assert abs(ry) >= 0.5 - 1e-9          # stayed on the near side


def test_step_along_wall_slides():
    node = _Node(_H_WALL)
    # Moving mostly along +x but dipping into the wall band: should keep x progress.
    rx, ry, blocked = node._resolve_human_wall_collision(-1.0, -0.8, -0.5, -0.3, R)
    assert not node._point_in_walls(rx, ry, R)
    assert rx == pytest.approx(-0.5)      # x advanced (slid along the wall)
    assert ry == pytest.approx(-0.8)      # y held off the wall
    assert blocked is False


def test_fully_boxed_corner_holds_and_flags_blocked():
    node = _Node(_L_WALLS)
    # (0.6,0.6) is free; stepping toward the concave corner (0.45,0.45) is blocked
    # on BOTH axis slides → hold at origin, blocked=True.
    rx, ry, blocked = node._resolve_human_wall_collision(0.6, 0.6, 0.45, 0.45, R)
    assert (rx, ry) == (0.6, 0.6)
    assert blocked is True
    assert not node._point_in_walls(rx, ry, R)


def test_free_step_unchanged():
    node = _Node(_H_WALL)
    out = node._resolve_human_wall_collision(-3.0, -2.0, -3.0, -1.5, R)
    assert out == (-3.0, -1.5, False)     # nowhere near the wall → passthrough


def test_step_into_static_obstacle_is_not_penetrated():
    node = _Node([])
    node.spawned_obstacle_records = {"rl_sta_001": (0.0, 0.0, 0.5)}
    rx, ry, blocked = node._resolve_human_static_obstacle_collision(
        -1.2, 0.0, -0.6, 0.0, R)
    assert not node._point_in_static_obstacle_for_human(rx, ry, R)
    assert rx == pytest.approx(-1.2)
    assert ry == pytest.approx(0.0)
    assert blocked is True


def test_static_obstacle_guard_ignores_other_human_records():
    node = _Node([])
    node.human_states = {"human_a": {}, "human_b": {}}
    node.spawned_obstacle_records = {
        "human_a": (0.0, 0.0, 0.3),
        "human_b": (1.0, 0.0, 0.3),
    }
    assert node._point_in_static_obstacle_for_human(0.0, 0.0, R) is False


def test_segment_hits_static_obstacle_for_human():
    node = _Node([])
    node.spawned_obstacle_records = {"rl_sta_001": (0.0, 0.0, 0.5)}
    assert node._segment_hits_static_obstacle_for_human(-2.0, 0.0, 2.0, 0.0, R)
    assert not node._segment_hits_static_obstacle_for_human(-2.0, 2.0, 2.0, 2.0, R)


def test_already_inside_wall_can_escape():
    node = _Node(_H_WALL)
    # Pathological: origin already in the wall band → allow the move out freely.
    assert node._point_in_walls(0.0, 0.0, R)
    rx, ry, blocked = node._resolve_human_wall_collision(0.0, 0.0, 0.0, 1.0, R)
    assert (rx, ry, blocked) == (0.0, 1.0, False)


# ── Goal reachability: _point_in_human_regions keeps goals on the navigable area
_LAYOUTS = reg.build_map_layouts(
    map_inner_lower=-11.8, map_inner_upper=11.8, map_wall_thickness=0.30,
    map_corridor_width=5.2, map_corridor_passage_width=1.6,
    map_intersection_width=5.2, map_intersection_passage_width=1.6,
    map_lobby_open_half_extent=6.0, map_start_band_depth=2.5,
)


def test_corridor_goal_must_stay_in_lane():
    node = _RegionNode(_LAYOUTS["corridor"])
    assert node._point_in_human_regions(0.0, 0.0) is True      # on the lane
    assert node._point_in_human_regions(0.0, 2.0) is True      # still in lane (half=2.6)
    assert node._point_in_human_regions(0.0, 8.0) is False     # across the wall → rejected


def test_intersection_goal_must_be_in_an_arm():
    node = _RegionNode(_LAYOUTS["intersection"])
    assert node._point_in_human_regions(8.0, 0.0) is True      # right arm
    assert node._point_in_human_regions(0.0, 8.0) is True      # top arm
    assert node._point_in_human_regions(8.0, 8.0) is False     # corner block → unreachable


def test_lobby_and_nonstructured_are_unconstrained():
    assert _RegionNode(_LAYOUTS["lobby"])._point_in_human_regions(5.0, 5.0) is True
    assert _RegionNode(None)._point_in_human_regions(99.0, 99.0) is True
