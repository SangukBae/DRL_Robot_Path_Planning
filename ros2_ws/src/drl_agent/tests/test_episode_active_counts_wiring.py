"""Wiring test for Environment._apply_episode_active_counts (the real method).

The base env resolves THIS episode's num_of_static_obstacles / num_of_humans from
the stage's per-map dicts (set by the curriculum subclass) AFTER the map_type is
chosen. The pure resolution math is covered in test_active_by_map; here we drive
the REAL method bound to a minimal fake node to confirm the wiring: it reads the
right attributes, prefers the by-map value, falls back to the single value, and
clamps to the pool sizes. environment.py imports ROS packages, so they are stubbed.
"""

import os
import sys
import types


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
    """Import the heavy `environment` module with ROS deps stubbed, then RESTORE
    sys.modules so the stubs do NOT leak into other test files (e.g.
    test_gym_parameter_client needs the REAL rcl_interfaces / rclpy). The imported
    module object keeps working — the method under test is pure Python and never
    touches the (now-removed) stub modules at call time."""
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
        import drl_agent.env.simulation.environment as env_mod
        return env_mod
    finally:
        for n, orig in saved.items():
            if orig is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = orig


env_mod = _import_environment_with_temp_stubs()


class _Logger:
    def info(self, *a, **k):
        pass

    def warn(self, *a, **k):
        pass


def _fake_node(**overrides):
    node = types.SimpleNamespace(
        current_map_type="corridor",
        _stage_active_static_by_map={"corridor": 5, "intersection": 7},
        _stage_active_static=6,
        _stage_active_humans_by_map={"corridor": 1, "intersection": 2},
        _stage_active_humans=3,
        obstacle_pool_static_size=24,
        obstacle_pool_human_size=8,
        _last_active_counts_logged=None,
        num_of_static_obstacles=0,
        num_of_humans=0,
    )
    node.get_logger = lambda: _Logger()
    for k, v in overrides.items():
        setattr(node, k, v)
    return node


def _apply(node):
    # call the REAL unbound method
    env_mod.Environment._apply_episode_active_counts(node)
    return node.num_of_static_obstacles, node.num_of_humans


def test_by_map_value_used_for_current_map():
    assert _apply(_fake_node(current_map_type="corridor")) == (5, 1)
    assert _apply(_fake_node(current_map_type="intersection")) == (7, 2)


def test_falls_back_to_single_when_map_absent():
    # lobby not in either by-map dict → stage single values
    assert _apply(_fake_node(current_map_type="lobby")) == (6, 3)


def test_empty_by_map_is_backward_compatible_single_value():
    node = _fake_node(_stage_active_static_by_map={}, _stage_active_humans_by_map={},
                      current_map_type="corridor")
    assert _apply(node) == (6, 3)


def test_clamped_to_pool_sizes():
    node = _fake_node(current_map_type="intersection",
                      obstacle_pool_static_size=4, obstacle_pool_human_size=1)
    # by-map asks 7 static / 2 humans → clamped to pool (4 / 1)
    assert _apply(node) == (4, 1)


def test_non_structured_empty_map_type_uses_single():
    node = _fake_node(current_map_type="", _stage_active_static_by_map={},
                      _stage_active_humans_by_map={})
    assert _apply(node) == (6, 3)
