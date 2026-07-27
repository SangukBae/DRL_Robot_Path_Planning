#!/usr/bin/env python3
"""Compatibility shim — canonical module moved to drl_agent/env/simulation/gazebo_entity_manager.py.

Legacy flat bare-name imports (``import gazebo_entity_manager``) keep working: this file
aliases itself to the package module. New code should import
``drl_agent.env.simulation.gazebo_entity_manager`` directly.
"""
import os
import sys

try:
    import drl_agent.env.simulation.gazebo_entity_manager as _impl
except ModuleNotFoundError as _e:
    # Retry ONLY when the drl_agent package itself is unresolvable
    # (source-tree / flat-install execution without the built package on
    # sys.path: the ROS package root is two levels up from this file).
    # A missing third-party dep (e.g. torch) must propagate untouched —
    # purging drl_agent.* for it would break the identity of modules other
    # shims already aliased.
    if not (_e.name or "").startswith("drl_agent"):
        raise
    # Purge any partially-resolved namespace package before retrying.
    for _m in [m for m in list(sys.modules)
               if m == "drl_agent" or m.startswith("drl_agent.")]:
        del sys.modules[_m]
    _pkg_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    if os.path.isfile(os.path.join(_pkg_root, "drl_agent", "__init__.py")):
        if _pkg_root not in sys.path:
            sys.path.insert(0, _pkg_root)
    import drl_agent.env.simulation.gazebo_entity_manager as _impl

sys.modules[__name__] = _impl
