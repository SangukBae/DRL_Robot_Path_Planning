"""Installed-environment bare-name import contract.

The rest of the suite (via conftest.py) puts the SOURCE scripts/ dirs on
sys.path, so it can never detect a broken *installed* import path. This test
closes that gap by running ``import <bare-name>`` in a SUBPROCESS that sources
the ROS underlay and THIS workspace's ``install/setup.bash`` ITSELF — it does
not rely on (or even inherit) whatever PYTHONPATH the parent pytest process
happens to already have.

Why sourcing inside the subprocess matters: under ``colcon test`` the pytest
runner process is typically launched with the workspace ALREADY sourced, so a
test that merely inspects ``os.environ["PYTHONPATH"]`` can pass for the wrong
reason — it would stay green even if the package's
``env-hooks/flat_legacy_scripts.dsv.in`` hook were broken or removed, because
some *other* mechanism put the flat dir on the parent's PYTHONPATH. Explicitly
stripping PYTHONPATH/AMENT_PREFIX_PATH/COLCON_PREFIX_PATH before the
subprocess sources the setup scripts makes the hook itself the only possible
source of the import path, so a regression there actually fails this test
instead of being silently masked by CI's ambient environment.

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
_INTERFACES_INSTALLED = os.path.isdir(
    os.path.join(_WS_ROOT, "install", "drl_agent_interfaces"))


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

# Torch-free shims — must import in ANY sourced environment.
PURE_BARE = ["run_layout", "config_paths", "geometry_utils", "seed_utils",
             "file_manager", "pure_pursuit", "episode_metrics",
             "curriculum_stage_logic", "map_catalog", "reward_calculator"]

# Heavier shims — required by the feedback contract; need torch (+ROS for the
# env/trainer stack), so they are asserted only where those deps exist.
TORCH_BARE = ["tqc_networks", "buffer", "tqc_io", "tqc_agent"]
ROS_TORCH_BARE = ["environment", "train_rl"]


def _sourced_import(names, tmp_path):
    """Import ``names`` in a subprocess that sources the ROS underlay + this
    workspace's install/setup.bash ITSELF, with PYTHONPATH/AMENT_PREFIX_PATH/
    COLCON_PREFIX_PATH stripped from the inherited environment first — so the
    sourcing inside this subprocess is the only possible source of the import
    path (see module docstring)."""
    script = os.path.join(str(tmp_path), "check_imports.py")
    code = (
        "import importlib\n"
        + "\n".join(f"m_{i} = importlib.import_module({n!r})"
                    for i, n in enumerate(names))
        + "\nimport drl_agent\n"
        + "\n".join(
            f"assert m_{i}.__name__.startswith('drl_agent.'), "
            f"({n!r}, m_{i}.__name__)" for i, n in enumerate(names))
        + "\nprint('OK')\n"
    )
    with open(script, "w") as f:
        f.write(code)

    env = dict(os.environ)
    for k in ("PYTHONPATH", "AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH",
              "CMAKE_PREFIX_PATH"):
        env.pop(k, None)
    # Never leave root-owned __pycache__ behind in a bind-mounted source tree
    # (this subprocess sources the install tree, not the source tree, but
    # keep the contract symmetric with the rest of the suite regardless).
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


def test_installed_pure_bare_imports(tmp_path):
    proc = _sourced_import(PURE_BARE, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_installed_torch_bare_imports(tmp_path):
    pytest.importorskip("torch")
    proc = _sourced_import(TORCH_BARE, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


@pytest.mark.skipif(not _INTERFACES_INSTALLED,
                    reason="drl_agent_interfaces not built in this workspace")
def test_installed_ros_torch_bare_imports(tmp_path):
    pytest.importorskip("torch")
    proc = _sourced_import(ROS_TORCH_BARE, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout
