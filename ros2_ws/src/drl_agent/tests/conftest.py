"""Pytest bootstrap for the ROS-free unit tests.

These tests import the extracted pure / stateful-but-ROS-free modules directly
from ``scripts/`` and run WITHOUT ROS2 / Gazebo / a built workspace, so the
relevant script directories are put on ``sys.path`` here (the modules are
otherwise imported bare at runtime from a flat install dir).

Covered directories:
  * ``scripts/utils``       — pure helpers (geometry, seeding, config paths …)
  * ``scripts/environment`` — reward_calculator, collision_checker,
                              localization_noise (ROS-free numeric modules)
  * ``scripts/policy``      — curriculum_stage_logic, tqc_networks (torch-gated)

Modules that pull in ROS (e.g. gym_parameter_client → rcl_interfaces) are NOT
imported by these tests; they are exercised only on a built ROS2 workspace.
"""

import os
import sys

_PKG_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS = os.path.join(_PKG_ROOT, "scripts")
for _sub in ("utils", "environment", "policy"):
    _d = os.path.join(_SCRIPTS, _sub)
    if _d not in sys.path:
        sys.path.insert(0, _d)

# The importable ``drl_agent`` Python package (drl_agent/drl_agent/) — hosts
# the migrated pure modules (run_layout, config_paths→config.paths, …) plus the
# profile loader/validator. The bare-name shims in scripts/ resolve through it.
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
