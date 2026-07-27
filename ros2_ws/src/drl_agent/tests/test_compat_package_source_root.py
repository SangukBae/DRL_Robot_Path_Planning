"""Unit tests for drl_agent.common.compat.package_source_root().

The function must resolve the drl_agent SOURCE package root under every
execution mode the wrappers actually run in — source-tree execution AND
installed (site-packages) execution, where a naive "two dirs up from this
file" guess lands inside site-packages instead of the workspace's
``src/drl_agent``. This matters because CheckpointManager / ConfigValidator
use this value to locate ``runtime/experiments/`` for resume validation, and
must agree with what the legacy trainer's own
``_resolve_drl_agent_source_root`` would pick.
"""

import os
import sys
import types

import pytest

from drl_agent.common import compat


def _make_pkg_dir(base) -> str:
    """<base>/src/drl_agent/package.xml — a valid package dir for resolution."""
    d = os.path.join(str(base), "src", "drl_agent")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "package.xml"), "w").close()
    return d


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Every test controls its own env var / cwd / ament_index visibility."""
    monkeypatch.delenv("DRL_AGENT_SRC_PATH", raising=False)


def test_source_tree_execution_resolves_via_own_file_location():
    # Real repo: compat.py lives at .../src/drl_agent/drl_agent/common/compat.py
    root = compat.package_source_root()
    assert root, "expected the real source tree to resolve"
    assert os.path.basename(root) == "drl_agent"
    assert os.path.isfile(os.path.join(root, "package.xml"))


def test_installed_execution_traces_install_segment_to_source_tree(
        tmp_path, monkeypatch):
    pkg_dir = _make_pkg_dir(tmp_path / "ws")
    fake_here = os.path.join(
        str(tmp_path), "ws", "install", "drl_agent", "lib", "python3.10",
        "site-packages", "drl_agent", "common", "compat.py")
    monkeypatch.setattr(compat, "__file__", fake_here)
    monkeypatch.chdir(tmp_path)  # cwd/src/drl_agent must NOT also resolve here

    assert compat.package_source_root() == pkg_dir


def test_env_var_override_wins_over_everything(tmp_path, monkeypatch):
    pkg_dir = _make_pkg_dir(tmp_path / "override")
    # Point __file__ somewhere that would resolve to a DIFFERENT (nonexistent)
    # tree, to prove the env var short-circuits before that candidate is used.
    monkeypatch.setattr(compat, "__file__",
                        os.path.join(str(tmp_path), "elsewhere", "compat.py"))
    monkeypatch.setenv("DRL_AGENT_SRC_PATH", str(tmp_path / "override"))

    assert compat.package_source_root() == pkg_dir


def test_ament_index_prefix_traced_when_own_path_has_no_install_segment(
        tmp_path, monkeypatch):
    pkg_dir = _make_pkg_dir(tmp_path / "ws2")
    # __file__ has no "/install/" segment (e.g. a symlink-install layout) --
    # only the ament_index package-prefix lookup can find it.
    monkeypatch.setattr(compat, "__file__",
                        os.path.join(str(tmp_path), "symlinked", "compat.py"))
    monkeypatch.chdir(tmp_path)

    fake_prefix = os.path.join(str(tmp_path), "ws2", "install", "drl_agent")
    fake_pkgs_mod = types.ModuleType("ament_index_python.packages")
    fake_pkgs_mod.get_package_prefix = lambda name: fake_prefix
    fake_top_mod = types.ModuleType("ament_index_python")
    fake_top_mod.packages = fake_pkgs_mod
    monkeypatch.setitem(sys.modules, "ament_index_python", fake_top_mod)
    monkeypatch.setitem(sys.modules, "ament_index_python.packages", fake_pkgs_mod)

    assert compat.package_source_root() == pkg_dir


def test_cwd_fallback_when_file_path_gives_no_signal(tmp_path, monkeypatch):
    pkg_dir = _make_pkg_dir(tmp_path)
    monkeypatch.setattr(compat, "__file__",
                        os.path.join(str(tmp_path), "nowhere", "compat.py"))
    monkeypatch.setitem(sys.modules, "ament_index_python.packages", None)
    monkeypatch.chdir(tmp_path)

    assert compat.package_source_root() == pkg_dir


def test_returns_empty_when_nothing_resolves(tmp_path, monkeypatch):
    monkeypatch.setattr(compat, "__file__",
                        os.path.join(str(tmp_path), "nowhere", "compat.py"))
    monkeypatch.setitem(sys.modules, "ament_index_python.packages", None)
    empty_cwd = tmp_path / "empty"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)

    assert compat.package_source_root() == ""
