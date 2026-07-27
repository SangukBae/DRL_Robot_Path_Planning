"""sys.path bootstrap shared by the drl_experiments launcher scripts.

Makes the ``drl_agent`` Python package importable when running straight from
the source tree (no colcon build needed): the sibling ROS package
``ros2_ws/src/drl_agent`` hosts it. After a build, site-packages provides it
and this is a no-op.
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


def canonical_module_file(module_name: str) -> str:
    """Absolute path of a canonical ``drl_agent.*`` module's own ``.py`` file
    (works for both source-tree and installed execution — see
    ``ensure_drl_agent_importable``, which must be called first)."""
    import importlib.util
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    if spec is not None and spec.origin and os.path.isfile(spec.origin):
        return spec.origin
    return ""
