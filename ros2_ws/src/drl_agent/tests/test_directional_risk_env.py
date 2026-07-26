"""Tests for the env-side PHASE2 plumbing on Environment
(`_compute_directional_risk`, `_append_aux_labels`'s action-risk prepend).

These are pure-Python methods on the ROS Node class but live in a module that
imports ROS at load time. Follows the project's established stub-import
pattern (see test_open_map_safe_start_yaw.py / test_episode_active_counts_
wiring.py): import `environment` with ROS deps stubbed, then drive the REAL
unbound methods on a minimal fake node (types.SimpleNamespace).

Covers:
  * _compute_directional_risk reads the pre-step human_states/GT pose and
    returns (risk_dir, min_dist_dir) sliced at the sector matching theta.
  * _append_aux_labels prepends the action-risk sentinel block ONLY when
    action_risk_head_env_enabled AND a target is supplied; absent (byte-
    identical to before PHASE2) when either is false/None.
  * reset() never gets this block (the trainer always calls _append_aux_labels
    with action_risk_target=None there — verified via the same default).
"""

import math
import sys
import threading
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
aux_labels = env_mod.aux_labels


def _directional_risk_cfg(**over):
    base = dict(enabled=True, num_sectors=16, horizons_sec=[1.0],
                risk_distance_scale=3.0, min_speed_for_motion=0.05)
    base.update(over)
    return aux_labels.AuxLabelConfig(base)


def _node(**overrides):
    node = types.SimpleNamespace(
        human_states={},
        _human_lock=threading.Lock(),
        gt_x=0.0, gt_y=0.0, gt_yaw=0.0,
        _directional_risk_cfg=_directional_risk_cfg(),
        action_risk_head_env_enabled=False,
        _aux_pred_enabled=False,
    )
    for k, v in overrides.items():
        setattr(node, k, v)
    return node


# --------------------------------------------------------------------------- #
#  _compute_directional_risk
# --------------------------------------------------------------------------- #
def test_no_humans_zero_risk_far_min_dist():
    node = _node()
    risk_dir, min_dist_dir = Environment._compute_directional_risk(node, 0.0)
    assert risk_dir == 0.0
    assert min_dist_dir == 1.0


def test_human_directly_ahead_high_risk_in_forward_sector():
    # Human 1m straight ahead (theta=0, robot frame), well inside D_c=3.0.
    node = _node(human_states={"h0": {"x": 1.0, "y": 0.0, "yaw": math.pi, "v": 0.0}})
    risk_dir, min_dist_dir = Environment._compute_directional_risk(node, 0.0)
    assert risk_dir > 0.5
    assert min_dist_dir < 1.0


def test_human_behind_low_risk_for_forward_theta():
    # Human directly BEHIND the robot must not raise the FORWARD sector's risk.
    node = _node(human_states={"h0": {"x": -1.0, "y": 0.0, "yaw": 0.0, "v": 0.0}})
    risk_dir, _ = Environment._compute_directional_risk(node, 0.0)
    assert risk_dir == 0.0


def test_min_dist_dir_is_sector_specific_not_global():
    # Regression for the reviewed bug: a human very close but BEHIND (a
    # different sector) must NOT pull the FORWARD sector's min_dist_dir down.
    # min_dist_dir must be the nearest human WITHIN theta's sector, not a
    # horizon-global nearest-human-anywhere distance.
    node = _node(human_states={"h0": {"x": -0.2, "y": 0.0, "yaw": 0.0, "v": 0.0}})
    risk_dir, min_dist_dir = Environment._compute_directional_risk(node, 0.0)  # forward
    assert risk_dir == 0.0
    # The old (buggy) global-min behaviour would have given ~0.2/3.0 = 0.067 here.
    assert min_dist_dir == 1.0


def test_far_human_no_risk():
    node = _node(human_states={"h0": {"x": 20.0, "y": 0.0, "yaw": 0.0, "v": 0.0}})
    risk_dir, min_dist_dir = Environment._compute_directional_risk(node, 0.0)
    assert risk_dir == 0.0
    assert min_dist_dir == 1.0


# --------------------------------------------------------------------------- #
#  _append_aux_labels: action-risk sentinel prepend
# --------------------------------------------------------------------------- #
def test_append_aux_labels_no_prepend_when_disabled():
    node = _node(action_risk_head_env_enabled=False)
    state = [1.0, 2.0, 3.0]
    out = Environment._append_aux_labels(node, state, action_risk_target=(0.5, 0.5))
    assert out == state  # unchanged: byte-identical when the env flag is off


def test_append_aux_labels_no_prepend_when_target_none():
    node = _node(action_risk_head_env_enabled=True)
    state = [1.0, 2.0, 3.0]
    out = Environment._append_aux_labels(node, state, action_risk_target=None)
    assert out == state  # unchanged: reset() calls with target=None


def test_append_aux_labels_prepends_sentinel_block_when_enabled():
    node = _node(action_risk_head_env_enabled=True)
    state = [1.0, 2.0, 3.0]
    out = Environment._append_aux_labels(node, state, action_risk_target=(0.42, 0.87))
    assert out[:3] == state
    assert out[3] == aux_labels.ACTION_RISK_WIRE_SENTINEL
    assert abs(out[4] - 0.42) < 1e-6
    assert abs(out[5] - 0.87) < 1e-6
    assert len(out) == 6  # no aux tail (aux disabled in this fixture)


def test_action_risk_block_composes_with_aux_tail():
    """Both PHASE2's action-risk block AND the aux label tail can coexist: the
    action-risk block comes first, the aux tail (self-describing) last, and
    EnvInterface's strip order (action-risk THEN aux) recovers both cleanly."""
    aux_cfg = _directional_risk_cfg(horizons_sec=[0.5, 1.0, 1.5])
    node = _node(
        action_risk_head_env_enabled=True,
        _aux_pred_enabled=True,
        _aux_label_cfg=aux_cfg,
        latest_actual_signed_speed=0.0,
        get_logger=lambda: types.SimpleNamespace(warn=lambda *a, **k: None),
    )
    state = [1.0, 2.0, 3.0]
    out = Environment._append_aux_labels(node, state, action_risk_target=(0.1, 0.9))
    assert out[:3] == state
    assert out[3] == aux_labels.ACTION_RISK_WIRE_SENTINEL

    remainder = out[3:]
    target, after_action_risk = aux_labels.strip_action_risk_wire(remainder)
    assert target is not None
    assert abs(target[0] - 0.1) < 1e-6
    assert abs(target[1] - 0.9) < 1e-6
    meta, label = aux_labels.parse_aux_wire(after_action_risk)
    assert meta["num_sectors"] == aux_cfg.num_sectors
    assert len(label) == aux_cfg.label_dim
