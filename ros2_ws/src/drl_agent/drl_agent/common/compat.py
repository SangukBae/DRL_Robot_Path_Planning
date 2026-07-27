"""Source-root resolution for the ``drl_agent`` package.

Pure stdlib — safe to import without ROS.
"""

import os


def _is_drl_agent_pkg_dir(path: str) -> bool:
    p = os.path.normpath(path) if path else ""
    return (bool(p)
            and os.path.basename(p) == "drl_agent"
            and os.path.isfile(os.path.join(p, "package.xml")))


def package_source_root() -> str:
    """Best-effort path of the ``drl_agent`` ROS package SOURCE root.

    Mirrors ``train_tqc_base.TrainTQCBase._resolve_drl_agent_source_root``'s
    precedence exactly, so a resume/run-dir decision made here (e.g.
    CheckpointManager, ConfigValidator) and the one the trainer itself makes
    always agree — including in the INSTALLED case (``ros2 run drl_agent
    train_node.py ... -p resume:=true``), where this file executes from
    site-packages and a naive "two dirs up" guess would land in site-packages,
    not the workspace's ``src/drl_agent``:

      1. ``DRL_AGENT_SRC_PATH`` env var (explicit override);
      2. this file's own path, when it already lives under
         ``.../src/drl_agent/drl_agent/...`` (source-tree execution, no
         colcon build yet — the common case during development);
      3. an ``.../install/...`` path traced back to the sibling
         ``.../src/drl_agent`` (installed execution — primary case this fixes);
      4. the same trace via an ament_index package-prefix lookup (covers
         install layouts where this file's own path lacks an "/install/"
         segment, e.g. some symlink-install trees);
      5. ``<cwd>/src/drl_agent`` (matches the legacy cwd-based fallback).

    Returns "" only when NONE of the above resolve to an existing package dir
    (``package.xml`` present) — callers must handle that explicitly rather
    than silently resolving paths against an unrelated cwd.
    """
    here = os.path.abspath(__file__)
    candidates = []

    src_env = os.environ.get("DRL_AGENT_SRC_PATH", "").strip()
    if src_env:
        src_env = os.path.expanduser(src_env)
        candidates += [
            src_env,
            os.path.join(src_env, "drl_agent"),
            os.path.join(src_env, "src", "drl_agent"),
        ]

    # Source-tree execution: .../src/drl_agent/drl_agent/common/compat.py
    candidates.append(os.path.normpath(os.path.join(os.path.dirname(here), "..", "..")))

    # Installed execution: trace an "/install/" segment back to the sibling
    # workspace source tree.
    token = f"{os.sep}install{os.sep}"
    if token in here:
        ws_root = here.split(token, 1)[0]
        candidates.append(os.path.join(ws_root, "src", "drl_agent"))

    try:
        from ament_index_python.packages import get_package_prefix
        prefix = os.path.abspath(get_package_prefix("drl_agent"))
        if token in prefix + os.sep:
            ws_root = (prefix + os.sep).split(token, 1)[0]
            candidates.append(os.path.join(ws_root, "src", "drl_agent"))
    except Exception:
        pass

    candidates.append(os.path.join(os.path.abspath(os.getcwd()), "src", "drl_agent"))

    for cand in candidates:
        if _is_drl_agent_pkg_dir(cand):
            return os.path.normpath(cand)
    return ""
