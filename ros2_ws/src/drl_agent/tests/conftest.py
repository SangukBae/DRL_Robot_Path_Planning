"""Pytest bootstrap for the ROS-free unit tests.

These tests import the ROS-free modules directly from the canonical
``drl_agent`` Python package (``drl_agent/drl_agent/...``) and run WITHOUT
ROS2 / Gazebo / a built workspace — the package root is put on ``sys.path``
here so ``import drl_agent.<...>`` resolves from the source tree without
requiring a prior ``colcon build``.

Modules that pull in ROS (e.g. gym_parameter_client → rcl_interfaces) are NOT
imported by these tests; they are exercised only on a built ROS2 workspace.
"""

import os
import sys

_PKG_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
