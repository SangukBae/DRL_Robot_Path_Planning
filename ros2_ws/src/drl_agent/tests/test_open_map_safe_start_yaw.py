"""Tests for the open-map (lobby/clutter) inward-safe start-yaw sampler.

The sampler (`Environment._sample_open_map_safe_yaw`) and its dispatch
(`Environment._choose_start_yaw`) are pure Python on the node, but live in
modules that import ROS at module load. We follow the project's established
pattern (see test_episode_active_counts_wiring.py): import `environment` with
ROS deps stubbed, then drive the REAL unbound methods on a minimal fake node.

Contracts pinned:
  * near a single wall  → yaw stays in the inward half-plane (never faces it),
    yet keeps diversity (a wide, non-degenerate arc);
  * near a corner       → yaw stays in the inward quadrant;
  * far from all walls  → unrestricted random yaw (legacy diversity);
  * flag OFF / structured region / non-open map → legacy paths unchanged;
  * a produced yaw always passes the existing _is_heading_toward_near_wall
    rejection (so it adds no extra rejection tries).
"""

import math
import sys
import types

import numpy as np
import pytest


_STUB_NAMES = (
    "rclpy", "rclpy.node", "rclpy.parameter", "rclpy.executors",
    "rclpy.callback_groups", "rclpy.qos",
    "rcl_interfaces", "rcl_interfaces.msg", "squaternion",
    "geometry_msgs", "geometry_msgs.msg", "nav_msgs", "nav_msgs.msg",
    "sensor_msgs", "sensor_msgs.msg", "visualization_msgs", "visualization_msgs.msg",
    "std_msgs", "std_msgs.msg",
    "drl_agent_interfaces", "drl_agent_interfaces.msg", "drl_agent_interfaces.srv",
    "ros_gz_interfaces", "ros_gz_interfaces.msg", "ros_gz_interfaces.srv",
)


def _import_environment_with_temp_stubs():
    class _Meta(type):
        _n = [0]

        def __getattr__(cls, name):
            _Meta._n[0] += 1
            return _Meta._n[0]

    saved = {n: sys.modules.get(n) for n in _STUB_NAMES}
    try:
        for n in _STUB_NAMES:
            if n not in sys.modules:
                mod = types.ModuleType(n)
                mod.__getattr__ = lambda name: _Meta(name, (), {})  # noqa: B023
                sys.modules[n] = mod
        import environment as env_mod
        return env_mod
    finally:
        for n, orig in saved.items():
            if orig is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = orig


env_mod = _import_environment_with_temp_stubs()
Environment = env_mod.Environment

WALL = 12.5
MARGIN = 1.0


def _node(**overrides):
    node = types.SimpleNamespace(
        _arena_wall_lower=-WALL,
        _arena_wall_upper=WALL,
        open_map_safe_yaw_wall_margin=MARGIN,
        open_map_safe_yaw_edge_margin=math.radians(10.0),
        open_map_safe_yaw_fallback_halfwidth=math.radians(25.0),
        open_map_safe_start_yaw_enabled=True,
        current_map_type="lobby",
    )
    for k, v in overrides.items():
        setattr(node, k, v)
    return node


def _yaw(node, x, y):
    return Environment._sample_open_map_safe_yaw(node, x, y)


def _samples(node, x, y, n=400):
    np.random.seed(123)
    return [_yaw(node, x, y) for _ in range(n)]


# ── near a single wall: inward half-plane, diversity preserved ──────────────

def test_near_right_wall_faces_inward():
    ys = _samples(_node(), x=WALL - 0.4, y=0.0)
    assert all(math.cos(a) < 0.0 for a in ys)          # never faces +x wall
    assert np.std(ys) > 0.3                              # not collapsed to one yaw
    assert max(ys) - min(ys) > 1.5                       # wide inward arc


def test_near_left_wall_faces_inward():
    ys = _samples(_node(), x=-WALL + 0.4, y=0.0)
    assert all(math.cos(a) > 0.0 for a in ys)


def test_near_top_wall_faces_inward():
    ys = _samples(_node(), x=0.0, y=WALL - 0.4)
    assert all(math.sin(a) < 0.0 for a in ys)


def test_near_bottom_wall_faces_inward():
    ys = _samples(_node(), x=0.0, y=-WALL + 0.4)
    assert all(math.sin(a) > 0.0 for a in ys)


# ── corner: inward quadrant ─────────────────────────────────────────────────

def test_corner_samples_inward_quadrant():
    ys = _samples(_node(), x=WALL - 0.3, y=WALL - 0.3)   # near right + top
    assert all(math.cos(a) < 0.0 and math.sin(a) < 0.0 for a in ys)
    assert np.std(ys) > 0.05                              # still has diversity


def test_other_corner_inward_quadrant():
    ys = _samples(_node(), x=-WALL + 0.3, y=-WALL + 0.3)  # near left + bottom
    assert all(math.cos(a) > 0.0 and math.sin(a) > 0.0 for a in ys)


# ── far from walls: unrestricted ────────────────────────────────────────────

def test_center_is_unrestricted():
    ys = _samples(_node(), x=0.0, y=0.0)
    assert any(math.cos(a) > 0.0 for a in ys)
    assert any(math.cos(a) < 0.0 for a in ys)
    assert any(math.sin(a) > 0.0 for a in ys)
    assert any(math.sin(a) < 0.0 for a in ys)
    assert max(ys) > 2.5 and min(ys) < -2.5              # spans ~[-pi, pi]


def test_far_from_wall_matches_legacy_random():
    # n_active == 0 path is exactly float(np.random.uniform(-pi, pi)).
    node = _node()
    np.random.seed(7)
    a = _yaw(node, 0.0, 0.0)
    np.random.seed(7)
    b = float(np.random.uniform(-np.pi, np.pi))
    assert a == pytest.approx(b)


# ── degenerate fallback keeps a (small) inward sector, never deterministic ───

def test_degenerate_edge_margin_falls_back_to_inward_sector():
    # Edge margin (50°) exceeds the corner half-width (45°) → fallback sector.
    node = _node(open_map_safe_yaw_edge_margin=math.radians(50.0))
    ys = _samples(node, x=WALL - 0.3, y=WALL - 0.3)
    assert all(math.cos(a) < 0.0 and math.sin(a) < 0.0 for a in ys)  # still inward
    assert np.std(ys) > 0.0                                          # not a fixed yaw


# ── consistency with the existing post-hoc rejection (no extra rejections) ──

@pytest.mark.parametrize("x,y", [
    (WALL - 0.4, 0.0), (-WALL + 0.4, 0.0),
    (0.0, WALL - 0.4), (0.0, -WALL + 0.4),
    (WALL - 0.3, WALL - 0.3), (-WALL + 0.3, -WALL + 0.3),
])
def test_sampled_yaw_passes_heading_toward_wall_check(x, y):
    node = _node()
    for a in _samples(node, x, y, n=200):
        assert not Environment._is_heading_toward_near_wall(node, x, y, a, MARGIN)


# ── _choose_start_yaw dispatch ──────────────────────────────────────────────

def test_choose_yaw_structured_region_uses_lane_aligned():
    node = _node()
    node._sample_lane_aligned_yaw = lambda region, x, y: 1.234
    # Even with the open-map flag on, a non-None region → lane-aligned path.
    assert Environment._choose_start_yaw(node, {"axis": "x"}, 10.0, 0.0) == 1.234


def test_choose_yaw_open_map_uses_safe_sampler():
    node = _node(current_map_type="clutter")
    node._sample_open_map_safe_yaw = types.MethodType(
        Environment._sample_open_map_safe_yaw, node)
    a = Environment._choose_start_yaw(node, None, WALL - 0.4, 0.0)
    assert math.cos(a) < 0.0          # inward-safe, not random into the wall


def test_choose_yaw_flag_off_is_legacy_random():
    node = _node(open_map_safe_start_yaw_enabled=False)
    np.random.seed(11)
    a = Environment._choose_start_yaw(node, None, WALL - 0.4, 0.0)
    np.random.seed(11)
    b = float(np.random.uniform(-np.pi, np.pi))
    assert a == pytest.approx(b)


def test_choose_yaw_non_open_map_is_legacy_random():
    # Open-map sampler must NOT fire on structured map types even via this path.
    node = _node(current_map_type="corridor")
    node._sample_open_map_safe_yaw = lambda x, y: pytest.fail("should not be called")
    np.random.seed(5)
    a = Environment._choose_start_yaw(node, None, WALL - 0.4, 0.0)
    np.random.seed(5)
    b = float(np.random.uniform(-np.pi, np.pi))
    assert a == pytest.approx(b)
