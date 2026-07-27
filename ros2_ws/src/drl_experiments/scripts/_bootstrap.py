"""sys.path bootstrap shared by the drl_experiments launcher scripts.

Makes the ``drl_agent`` Python package importable when running straight from
the source tree (no colcon build needed): the sibling ROS package
``ros2_ws/src/drl_agent`` hosts it. After a build, site-packages provides it
and this is a no-op. Also exposes the legacy flat-script dirs for wrappers
that exec legacy analysis scripts.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# .../ros2_ws/src/drl_experiments/scripts -> .../ros2_ws/src
_SRC = os.path.normpath(os.path.join(_HERE, "..", ".."))
_DRL_AGENT_PKG = os.path.join(_SRC, "drl_agent")


def ensure_drl_agent_importable() -> str:
    """Return the drl_agent ROS-package root ("" if not found in source)."""
    try:
        import drl_agent  # noqa: F401
        return _DRL_AGENT_PKG if os.path.isdir(_DRL_AGENT_PKG) else ""
    except ImportError:
        pass
    if os.path.isfile(os.path.join(_DRL_AGENT_PKG, "drl_agent", "__init__.py")):
        if _DRL_AGENT_PKG not in sys.path:
            sys.path.insert(0, _DRL_AGENT_PKG)
        return _DRL_AGENT_PKG
    return ""


def legacy_utils_script(name: str) -> str:
    """Absolute path of a legacy ``drl_agent/scripts/utils/<name>`` script
    (source tree first, then the flat install dir)."""
    cand = os.path.join(_DRL_AGENT_PKG, "scripts", "utils", name)
    if os.path.isfile(cand):
        return cand
    try:
        from ament_index_python.packages import get_package_prefix
        cand = os.path.join(get_package_prefix("drl_agent"), "lib", "drl_agent", name)
        if os.path.isfile(cand):
            return cand
    except Exception:
        pass
    return ""
