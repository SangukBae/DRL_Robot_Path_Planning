#!/usr/bin/env python3
"""Pure per-run directory-layout helpers (no ROS) for the curriculum trainer.

Replaces the old flat ``runtime/tqc/seed_<seed>/{pytorch_models,logs,...}``
layout (shared by every run of a given seed) with one self-contained folder
PER TRAINING RUN:

    runtime/experiments/<run_id>/
        configs/    -- copies of the 4 config files actually used this run
        logs/       -- all CSV/JSON/TensorBoard logs (plain filenames -- the
                       run folder itself carries the timestamp, so per-file
                       "_<tag>" suffixes are no longer needed)
        models/     -- checkpoints (*_actor.pth, *_critic.pth, ...)
        analysis/   -- scratch space for post-hoc analysis script output

    run_id = "<YYYYMMDD_HHMMSS>_<base_file_name>_seed<seed>"

Existing ``runtime/tqc/seed_<seed>/`` (legacy) runs are NEVER migrated: a
resume that finds ONLY a legacy checkpoint keeps using the legacy layout
(pytorch_models/final_models/results/logs, 4 separate dirs) for the rest of
that run -- see ``resolve_run_layout``. Only a genuinely fresh run (nothing to
resume, or ``load_model=false``) gets the new layout.

Unit-tested in ``tests/test_run_layout.py``.
"""

import csv
import glob
import os
import shutil
from datetime import datetime

# Keys expected in every layout dict this module returns.
LAYOUT_KEYS = ("run_dir", "is_legacy", "is_fresh", "run_id",
               "configs_dir", "logs_dir", "models_dir", "analysis_dir")


def make_run_id(base_file_name: str, seed: int, ts: str = None) -> str:
    """Build a run_id: ``<YYYYMMDD_HHMMSS>_<base_file_name>_seed<seed>``."""
    tag = ts if ts else datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{tag}_{base_file_name}_seed{int(seed)}"


def tagged_filename(base: str, ext: str, run_tag: str = "") -> str:
    """``<base><ext>`` when run_tag is empty (new per-run folder -- the folder
    name already carries the timestamp, so no suffix is needed), else the
    legacy ``<base>_<run_tag><ext>`` form. ``ext`` includes the leading dot
    (e.g. ``".csv"``)."""
    return f"{base}_{run_tag}{ext}" if run_tag else f"{base}{ext}"


def write_csv_header_if_new(path: str, header: list) -> bool:
    """Write ``header`` as the first row of a NEW csv at ``path`` -- but ONLY
    if the file does not already exist.

    A no-op when it does. This is what makes a RESUME into an existing
    new-structure run dir safe: because that run's CSVs use PLAIN filenames
    (``episode_rewards.csv``, no per-process ``_<timestamp>`` suffix -- see
    ``tagged_filename``), re-opening them in ``"w"`` mode unconditionally on
    every process start would silently truncate the prior rows. Every writer
    in this codebase must call this (or check the same way) before opening
    its CSV for append, so "resume" never means "lose the history so far".

    Returns True when the header was written (fresh file -- caller may now
    treat the file as freshly created), False when the file already existed
    (caller must open it in append ("a") mode, never "w").
    """
    if os.path.isfile(path):
        return False
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)
    return True


def experiments_root(package_root: str) -> str:
    return os.path.join(package_root, "runtime", "experiments")


def legacy_run_dir(package_root: str, seed: int) -> str:
    """The OLD per-seed run dir: ``runtime/tqc/seed_<seed>``."""
    return os.path.join(package_root, "runtime", "tqc", f"seed_{int(seed)}")


def _has_checkpoint(models_dir: str, base_file_name: str, seed: int) -> bool:
    """True when ``models_dir`` has at least one matching, non-mid-episode
    checkpoint (mirrors train_tqc_base.py's ``_find_latest_prefix`` glob,
    duplicated here so this module stays ROS-free and independently
    testable). Shared by both the legacy and the new-structure checks below."""
    pattern = os.path.join(models_dir, f"{base_file_name}_seed_{int(seed)}_*_actor.pth")
    return any(
        not os.path.basename(p).endswith("_checkpoint_actor.pth")
        for p in glob.glob(pattern)
    )


def find_latest_new_run_dir(package_root: str, base_file_name: str, seed: int):
    """Most-recent EXISTING new-structure run dir matching ``(base_file_name,
    seed)`` that actually has a checkpoint to resume from, or None.

    Two things this deliberately gets right (see the 2026-07 review that
    caught both):
      * Candidates WITHOUT a checkpoint in ``models/`` are excluded -- an
        empty/failed/smoke-test run dir must never be picked as a resume
        target and thereby block the legacy-checkpoint fallback below it.
      * "Most recent" is decided by the run_id's OWN
        ``<YYYYMMDD_HHMMSS>_...`` prefix (lexicographic == chronological for
        that format), NOT directory mtime -- a directory's mtime changes
        whenever ANY file inside is touched (a log append, a config copy),
        which does not necessarily track which run is actually newest.

    Lets a resume reuse the SAME run_id (and hence the same logs/models)
    across process restarts instead of minting a fresh timestamp -- exactly
    like the legacy layout's stable ``seed_<seed>`` dir let resumes find
    their checkpoint before this change.
    """
    pattern = os.path.join(
        experiments_root(package_root), f"*_{base_file_name}_seed{int(seed)}"
    )
    cands = [
        p for p in glob.glob(pattern)
        if os.path.isdir(p) and _has_checkpoint(os.path.join(p, "models"), base_file_name, seed)
    ]
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.basename(os.path.normpath(p)), reverse=True)
    return cands[0]


def legacy_has_checkpoint(package_root: str, base_file_name: str, seed: int) -> bool:
    """True when the OLD-structure ``pytorch_models`` dir has at least one
    matching, non-mid-episode checkpoint."""
    models_dir = os.path.join(legacy_run_dir(package_root, seed), "pytorch_models")
    return _has_checkpoint(models_dir, base_file_name, seed)


def new_layout(run_dir: str, *, is_fresh: bool, run_id: str = None) -> dict:
    """Layout dict for a new-structure run dir (existing or about to be created)."""
    return {
        "run_dir": run_dir,
        "is_legacy": False,
        "is_fresh": is_fresh,
        "run_id": run_id or os.path.basename(os.path.normpath(run_dir)),
        "configs_dir": os.path.join(run_dir, "configs"),
        "logs_dir": os.path.join(run_dir, "logs"),
        "models_dir": os.path.join(run_dir, "models"),
        "analysis_dir": os.path.join(run_dir, "analysis"),
    }


def legacy_layout(run_dir: str) -> dict:
    """Layout dict replicating the OLD 4-separate-dir structure exactly
    (pytorch_models / final_models / results / logs). ``configs_dir`` /
    ``analysis_dir`` are None -- callers must skip anything new-structure-only
    (config copying, analysis scratch dir) for a legacy run, so an existing
    seed folder never grows files it didn't have before this change."""
    return {
        "run_dir": run_dir,
        "is_legacy": True,
        "is_fresh": False,
        "run_id": None,
        "configs_dir": None,
        "logs_dir": os.path.join(run_dir, "logs"),
        "models_dir": os.path.join(run_dir, "pytorch_models"),
        "final_models_dir": os.path.join(run_dir, "final_models"),
        "results_dir": os.path.join(run_dir, "results"),
        "analysis_dir": None,
    }


def resolve_run_layout(package_root: str, base_file_name: str, seed: int,
                        *, load_model: bool, ts: str = None) -> dict:
    """Decide which run folder THIS process should use, and return its layout.

    Priority (only consulted when ``load_model`` is True -- resuming):
      1. An EXISTING new-structure run dir for (base_file_name, seed) --
         reuse it (same run_id, so logs/checkpoints keep accumulating there).
      2. An EXISTING legacy-structure checkpoint for (base_file_name, seed) --
         keep using the legacy layout for this run (never silently migrated).
      3. Neither found (fresh start despite load_model=True) -> new run dir.

    ``load_model=False`` always mints a fresh new-structure run dir.
    """
    if load_model:
        existing_new = find_latest_new_run_dir(package_root, base_file_name, seed)
        if existing_new:
            return new_layout(existing_new, is_fresh=False)
        if legacy_has_checkpoint(package_root, base_file_name, seed):
            return legacy_layout(legacy_run_dir(package_root, seed))

    run_id = make_run_id(base_file_name, seed, ts=ts)
    run_dir = os.path.join(experiments_root(package_root), run_id)
    return new_layout(run_dir, is_fresh=True, run_id=run_id)


def create_run_dirs(layout: dict) -> None:
    """mkdir -p every directory a ``resolve_run_layout`` result names.

    Legacy layouts get exactly their 4 original dirs (no configs/analysis);
    new layouts get the 4-subdir structure documented at module level.
    """
    if layout["is_legacy"]:
        dirs = [layout["run_dir"], layout["logs_dir"], layout["models_dir"],
                layout["final_models_dir"], layout["results_dir"]]
    else:
        dirs = [layout["run_dir"], layout["configs_dir"], layout["logs_dir"],
                layout["models_dir"], layout["analysis_dir"]]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def copy_run_configs(configs_dir: str, config_paths: dict) -> dict:
    """Copy each existing config file into ``configs_dir``.

    ``config_paths`` maps an arbitrary key (e.g. ``"train"``) to a source file
    path. Returns ``{key: copied_path}`` -- ``copied_path`` is ``""`` for a
    key whose source is empty/missing/uncopyable, so a run still starts even
    if one optional config could not be resolved (mirrors the rest of this
    codebase's "never let logging/bookkeeping abort a run" convention)."""
    out = {}
    for key, src in (config_paths or {}).items():
        if not src or not os.path.isfile(src):
            out[key] = ""
            continue
        dest = os.path.join(configs_dir, os.path.basename(src))
        try:
            shutil.copy2(src, dest)
            out[key] = dest
        except Exception:
            out[key] = ""
    return out
