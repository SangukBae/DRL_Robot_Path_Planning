#!/usr/bin/env python3
"""Pure path / config-discovery helpers (no ROS, no ament).

The trainers and test scripts repeat the same "given a location HINT (which may
be a file OR a directory), find <that-location>/<filename>" pattern when locating
config files. The semantics — matching the existing ``_find_config_file``
implementations — are:

  * hint is a FILE      -> look for ``filename`` in the file's DIRECTORY
                           (the hint names a sibling config, not the target);
  * hint is a DIRECTORY -> look for ``filename`` inside it;
  * hint is empty/other -> no candidate.

Only the pure, side-effect-light parts live here; the ament-share / ROS-parameter
resolution stays in the nodes. Unit-tested in ``tests/test_config_paths.py``.
"""

import os


def expand_user_path(path: str) -> str:
    """``os.path.expanduser`` that tolerates None/empty (returns "")."""
    if not path:
        return ""
    return os.path.expanduser(str(path))


def location_candidate(location: str, filename: str) -> str:
    """Resolve a file-or-directory location HINT to its ``<dir>/<filename>``
    candidate path. Returns "" when the hint is empty or is neither an existing
    file nor an existing directory.

    NOTE: a FILE hint resolves against its *directory* (so passing
    ``.../config/train_tqc_config.yaml`` as the hint still finds
    ``hyperparameters_tqc.yaml`` next to it) — this matches the existing
    ``_find_config_file`` behaviour exactly.
    """
    loc = expand_user_path(location)
    if not loc:
        return ""
    if os.path.isfile(loc):
        return os.path.join(os.path.dirname(loc), filename)
    if os.path.isdir(loc):
        return os.path.join(loc, filename)
    return ""


def first_existing_file(paths) -> str:
    """Return the first path in ``paths`` that is an existing file (after ~
    expansion), or "" if none exist. Empty/None entries are skipped."""
    for p in paths:
        ep = expand_user_path(p)
        if ep and os.path.isfile(ep):
            return ep
    return ""


def candidate_config_paths(filename: str, search_dirs) -> list:
    """Build an ordered candidate list of ``<dir>/<filename>`` for each search
    dir (skipping empty dirs). Pure string join — does not touch the filesystem,
    so it works for not-yet-existing directories too."""
    out = []
    for d in search_dirs:
        if not d:
            continue
        out.append(os.path.join(expand_user_path(d), filename))
    return out


def find_config_file(filename: str, locations) -> str:
    """Return the first existing ``<resolved-location>/<filename>``, or "".

    Each entry in ``locations`` is a file-or-directory HINT resolved via
    ``location_candidate`` (file -> its dir; dir -> itself). Returns "" when none
    resolve to an existing file, so callers can emit their own (ROS-aware) error.
    """
    for loc in locations:
        cand = location_candidate(loc, filename)
        if cand and os.path.isfile(cand):
            return cand
    return ""
