#!/usr/bin/env python3
"""Compatibility shim — canonical module moved to drl_agent/training/run_layout.py.

Legacy flat bare-name imports (``import run_layout``) keep working: this file
aliases itself to the package module. New code should import
``drl_agent.training.run_layout`` directly.
"""
import os
import sys

try:
    import drl_agent.training.run_layout as _impl
except ImportError:
    # Source-tree / flat-install execution without the built package on
    # sys.path: the ROS package root is two levels up from scripts/utils/.
    # Purge any partially-resolved namespace package before retrying.
    for _m in [m for m in list(sys.modules)
               if m == "drl_agent" or m.startswith("drl_agent.")]:
        del sys.modules[_m]
    _pkg_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    if os.path.isfile(os.path.join(_pkg_root, "drl_agent", "__init__.py")):
        if _pkg_root not in sys.path:
            sys.path.insert(0, _pkg_root)
    import drl_agent.training.run_layout as _impl

sys.modules[__name__] = _impl
