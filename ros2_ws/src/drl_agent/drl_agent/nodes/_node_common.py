"""Shared plumbing for the profile-based ROS entrypoint wrappers.

The wrappers (train_node.py, environment_curriculum_node.py, eval_node.py,
real_policy_node.py) do three things:

  1. parse the wrapper-only parameters (``profile`` / ``resume`` /
     ``validate_only`` — accepted BOTH as ``-p key:=value`` ROS params and as
     plain ``--key value`` CLI flags) and STRIP them from argv;
  2. resolve + validate the profile (ConfigValidator, hard fail before launch);
  3. ``os.execv`` the UNCHANGED legacy executable with translated legacy
     parameters (``train_config_file`` / ``config_file`` / ``load_model`` …)
     appended as an extra ``--ros-args`` section.

So a wrapper never re-implements training/env logic — compatibility with the
legacy entrypoints is structural, not duplicated.
"""

import json
import os
import sys

from ..common import compat
from ..config import ConfigValidator, ProfileError, ProfileLoader
from ..training import registry, run_manager

#: env var read by train_tqc_base.py to persist the profile manifest into the
#: run dir (additive hook; absent -> byte-identical legacy behaviour).
PROFILE_MANIFEST_ENV = "DRL_AGENT_PROFILE_MANIFEST"

_WRAPPER_KEYS = ("profile", "resume", "validate_only")


def _parse_bool(text: str) -> bool:
    return str(text).strip().lower() in ("true", "1", "yes", "on")


def _strip_empty_ros_args_sections(tokens):
    """Drop any ``--ros-args`` marker that introduces an EMPTY section (no
    tokens before the next ``--ros-args`` marker or the end of the list).

    Stripping the wrapper-only params (``profile`` / ``resume`` /
    ``validate_only``) out of a ``--ros-args -p profile:=... `` invocation can
    leave a dangling, now-empty ``--ros-args`` in the passthrough — ROS 2
    likely tolerates it, but ``exec_legacy`` then appends its OWN
    ``--ros-args`` section on top, so a plain ``-p profile:=x`` invocation
    would otherwise exec the legacy node with two ``--ros-args`` tokens back
    to back. Cheaper to just never emit the empty one.
    """
    out = []
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        if tok == "--ros-args":
            j = i + 1
            while j < n and tokens[j] != "--ros-args":
                j += 1
            if j == i + 1:
                i = j  # empty section -- drop the marker, resume scanning at j
                continue
        out.append(tok)
        i += 1
    return out


def parse_wrapper_args(argv):
    """Extract wrapper-only options from ``argv``.

    Returns ``(opts, passthrough)`` where ``opts`` has keys ``profile`` (str),
    ``resume`` (bool), ``validate_only`` (bool), ``seed`` (int or None; NOT
    stripped — the legacy trainers understand ``-p seed:=N`` natively), and
    ``passthrough`` is ``argv`` minus the wrapper-only tokens (and minus any
    ``--ros-args`` section left empty by that removal — see
    ``_strip_empty_ros_args_sections``).
    """
    opts = {"profile": "", "resume": False, "validate_only": False, "seed": None}
    out = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        nxt = argv[i + 1] if i + 1 < len(argv) else None

        # ROS-style: -p key:=value
        if tok == "-p" and nxt is not None and ":=" in nxt:
            key, _, val = nxt.partition(":=")
            if key in _WRAPPER_KEYS:
                if key == "profile":
                    opts["profile"] = val.strip()
                else:
                    opts[key] = _parse_bool(val)
                i += 2
                continue
            if key == "seed":
                try:
                    opts["seed"] = int(val)
                except ValueError:
                    pass
                # fall through: seed stays in passthrough for the legacy node
        # Plain CLI: --profile X / --resume / --validate-only / --seed N
        elif tok == "--profile" and nxt is not None:
            opts["profile"] = nxt.strip()
            i += 2
            continue
        elif tok == "--resume":
            opts["resume"] = True
            i += 1
            continue
        elif tok in ("--validate-only", "--validate"):
            opts["validate_only"] = True
            i += 1
            continue
        elif tok == "--seed" and nxt is not None:
            # Plain-CLI seed is wrapper-only (stripped): the wrapper re-emits
            # it as a proper ``-p seed:=N`` for the legacy node. A ROS-style
            # ``-p seed:=N`` above stays in passthrough untouched instead.
            try:
                opts["seed"] = int(nxt)
            except ValueError:
                pass
            i += 2
            continue

        out.append(tok)
        i += 1
    return opts, _strip_empty_ros_args_sections(out)


def load_and_validate(profile_name: str, *, resume: bool, seed):
    """Resolve the profile and run the strong pre-flight validation.

    Prints the validation report; exits(1) on errors. Returns
    ``(spec, entry)``.
    """
    loader = ProfileLoader()
    try:
        spec = loader.load(profile_name)
    except ProfileError as e:
        print(f"[profile] ERROR: {e}", file=sys.stderr)
        avail = loader.available_profiles()
        if avail:
            print("[profile] available profiles:", ", ".join(avail), file=sys.stderr)
        sys.exit(1)

    try:
        entry = registry.lookup(spec.algorithm, spec.trainer)
    except KeyError as e:
        print(f"[profile] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    rep = ConfigValidator(spec).validate(
        resume=resume, seed=seed, package_root=compat.package_source_root())
    print(f"[profile] {spec.name} ({spec.algorithm}/{spec.trainer}) "
          f"from {spec.profile_dir}")
    print(rep.summary())
    if not rep.ok:
        print("[profile] validation FAILED — refusing to launch.", file=sys.stderr)
        sys.exit(1)
    return spec, entry


def find_legacy_executable(name: str) -> str:
    """Locate a legacy entrypoint script by filename.

    Install case: the wrappers are installed FLAT next to the legacy scripts in
    ``lib/drl_agent`` → same directory. Source case: look under
    ``scripts/{policy,environment}`` of the source package.
    """
    here = os.path.dirname(os.path.abspath(sys.argv[0] or __file__))
    cands = [os.path.join(here, name)]
    root = compat.package_source_root()
    if root:
        cands += [os.path.join(root, "scripts", "policy", name),
                  os.path.join(root, "scripts", "environment", name)]
    for c in cands:
        if os.path.isfile(c):
            return c
    print(f"[profile] ERROR: legacy executable '{name}' not found "
          f"(tried {cands})", file=sys.stderr)
    sys.exit(1)


def export_profile_manifest_env(spec, *, resume: bool, seed) -> None:
    """Serialize the profile identity into PROFILE_MANIFEST_ENV so the trainer
    can persist it into the run dir it actually resolves."""
    payload = {
        "profile": spec.name,
        "profile_dir": spec.profile_dir,
        "output_prefix": spec.output_prefix,
        "resume": bool(resume),
        "seed_override": seed,
        "config_paths": spec.config_paths,
        "config_hashes": run_manager.config_hashes(spec.config_paths),
        "overrides": spec.overrides,
    }
    os.environ[PROFILE_MANIFEST_ENV] = json.dumps(payload)


def exec_legacy(exec_name: str, extra_ros_params: dict, passthrough) -> None:
    """Replace this process with the legacy executable.

    ``extra_ros_params`` are appended as one additional ``--ros-args`` section
    (ROS 2 accepts multiple), AFTER the user's passthrough args.
    """
    path = find_legacy_executable(exec_name)
    argv = [sys.executable, path] + list(passthrough)
    if extra_ros_params:
        argv.append("--ros-args")
        for k, v in extra_ros_params.items():
            argv += ["-p", f"{k}:={v}"]
    print(f"[profile] exec: {' '.join(argv[1:])}")
    sys.stdout.flush()
    os.execv(sys.executable, argv)
