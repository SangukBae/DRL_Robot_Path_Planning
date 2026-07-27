"""Installed-environment canonical-import contract.

The rest of the suite (via conftest.py) puts the SOURCE package root on
sys.path, so it can never detect a broken *installed* import path. This test
closes that gap by running imports in a SUBPROCESS that sources the ROS
underlay and THIS workspace's ``install/setup.bash`` ITSELF — it does not
rely on (or even inherit) whatever PYTHONPATH the parent pytest process
happens to already have (see the sourcing rationale in the module-level
comment inside ``_sourced_run`` below).

Locks two contracts on the INSTALLED package:
  1. ``import drl_agent.<canonical path>`` works from a neutral cwd with only
     the workspace sourced (no source tree needed).
  2. the retired bare names (``import tqc_agent``, ``import buffer``, …) do
     NOT resolve — bare-name import compatibility was intentionally dropped,
     so a regression that accidentally restores it (e.g. a stray script
     reappearing on PYTHONPATH) fails this test instead of going unnoticed.

Skips (rather than fails) when no built-and-sourceable install is found under
this workspace — e.g. host runs straight from the source tree without a
colcon build — and runs for real inside the docker workspace / under
``colcon test``.
"""

import glob
import os
import subprocess
import sys

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
# tests/ -> drl_agent (pkg) -> src -> <ws_root>
_WS_ROOT = os.path.normpath(os.path.join(_TESTS_DIR, "..", "..", ".."))
_INSTALL_SETUP = os.path.join(_WS_ROOT, "install", "setup.bash")


def _find_ros_underlay():
    """First existing ``/opt/ros/<distro>/setup.bash`` (env-hinted or globbed)."""
    candidates = []
    distro = os.environ.get("ROS_DISTRO", "").strip()
    if distro:
        candidates.append(f"/opt/ros/{distro}/setup.bash")
    candidates += sorted(glob.glob("/opt/ros/*/setup.bash"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return ""


_ROS_UNDERLAY = _find_ros_underlay()
_CAN_RUN = bool(_ROS_UNDERLAY) and os.path.isfile(_INSTALL_SETUP)

pytestmark = pytest.mark.skipif(
    not _CAN_RUN,
    reason=(
        "no built+sourceable drl_agent install found "
        f"(ROS underlay={'found' if _ROS_UNDERLAY else 'MISSING'}, "
        f"{_INSTALL_SETUP}={'exists' if os.path.isfile(_INSTALL_SETUP) else 'MISSING'}); "
        "colcon build this workspace to run this contract"
    ),
)

CANONICAL_PURE = [
    "drl_agent.training.run_layout", "drl_agent.config.paths",
    "drl_agent.common.geometry_utils", "drl_agent.common.seed_utils",
    "drl_agent.common.file_manager", "drl_agent.common.pure_pursuit",
]
CANONICAL_TORCH = [
    "drl_agent.rl.networks.tqc", "drl_agent.rl.replay.buffer",
    "drl_agent.rl.checkpointing.tqc_io", "drl_agent.rl.algorithms.tqc.agent",
]
CANONICAL_ROS_TORCH = [
    "drl_agent.env.simulation.environment", "drl_agent.training.train_rl",
]

RETIRED_BARE_NAMES = ["tqc_networks", "buffer", "tqc_io", "tqc_agent",
                      "train_rl", "environment", "config_paths"]


def _sourced_run(code, tmp_path):
    """Run ``code`` in a subprocess that sources the ROS underlay + this
    workspace's install/setup.bash ITSELF, with PYTHONPATH/AMENT_PREFIX_PATH/
    COLCON_PREFIX_PATH stripped from the inherited environment first — so the
    sourcing inside this subprocess is the only possible source of the import
    path (a test that merely inherits the parent's already-sourced env could
    stay green even if the install were broken)."""
    script = os.path.join(str(tmp_path), "check.py")
    with open(script, "w") as f:
        f.write(code)

    env = dict(os.environ)
    for k in ("PYTHONPATH", "AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH",
              "CMAKE_PREFIX_PATH"):
        env.pop(k, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    bash_cmd = (
        f'source "{_ROS_UNDERLAY}" && '
        f'source "{_INSTALL_SETUP}" && '
        f'exec "{sys.executable}" "{script}"'
    )
    return subprocess.run(
        ["bash", "-lc", bash_cmd], cwd=str(tmp_path), env=env,
        capture_output=True, text=True, timeout=120,
    )


def test_installed_pure_canonical_imports(tmp_path):
    code = "\n".join(f"import {m}" for m in CANONICAL_PURE) + "\nprint('OK')\n"
    proc = _sourced_run(code, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_installed_torch_canonical_imports(tmp_path):
    pytest.importorskip("torch")
    code = "\n".join(f"import {m}" for m in CANONICAL_TORCH) + "\nprint('OK')\n"
    proc = _sourced_run(code, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_installed_ros_torch_canonical_imports(tmp_path):
    pytest.importorskip("torch")
    if not os.path.isdir(os.path.join(_WS_ROOT, "install", "drl_agent_interfaces")):
        pytest.skip("drl_agent_interfaces not built in this workspace")
    code = "\n".join(f"import {m}" for m in CANONICAL_ROS_TORCH) + "\nprint('OK')\n"
    proc = _sourced_run(code, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_installed_bare_names_do_not_resolve(tmp_path):
    code = (
        "import importlib\n"
        f"names = {RETIRED_BARE_NAMES!r}\n"
        "still_working = []\n"
        "for n in names:\n"
        "    try:\n"
        "        importlib.import_module(n)\n"
        "        still_working.append(n)\n"
        "    except ModuleNotFoundError:\n"
        "        pass\n"
        "assert not still_working, still_working\n"
        "print('OK')\n"
    )
    proc = _sourced_run(code, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout
