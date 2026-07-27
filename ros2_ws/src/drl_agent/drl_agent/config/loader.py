"""ProfileLoader — resolve a profile NAME to a fully-resolved :class:`ProfileSpec`.

Profile roots are searched in order:

  1. ``DRL_EXPERIMENTS_DIR`` env var (either the ``profiles`` dir itself, or a
     dir containing ``profiles/``);
  2. the ``drl_experiments`` SOURCE package next to this package's source tree
     (``ros2_ws/src/drl_experiments/profiles``);
  3. the installed ``share/drl_experiments/profiles`` (via ament_index, when
     available).

A profile may also be given as a direct path (to the profile dir or to a
``profile.yaml`` file) — that always wins over name lookup.

Pure Python + PyYAML; no ROS required (ament_index is optional).
"""

import os

import yaml

from .schema import CONFIG_FILE_KEYS, KNOWN_OVERRIDE_KEYS, ProfileSpec
from ..common import compat


class ProfileError(RuntimeError):
    """Raised when a profile cannot be found or its manifest is malformed."""


def _candidate_profile_roots():
    roots = []

    env = os.environ.get("DRL_EXPERIMENTS_DIR", "").strip()
    if env:
        env = os.path.expanduser(env)
        roots.append(env if os.path.basename(os.path.normpath(env)) == "profiles"
                     else os.path.join(env, "profiles"))

    # Source tree: .../src/drl_agent (this package) -> sibling drl_experiments.
    pkg_root = compat.package_source_root()
    if pkg_root:
        roots.append(os.path.join(os.path.dirname(pkg_root),
                                  "drl_experiments", "profiles"))

    try:
        from ament_index_python.packages import get_package_share_directory
        roots.append(os.path.join(
            get_package_share_directory("drl_experiments"), "profiles"))
    except Exception:
        pass

    return [r for r in roots if os.path.isdir(r)]


class ProfileLoader:
    """Load ``profile.yaml`` manifests by name (``phase2/both``) or path."""

    def __init__(self, profiles_root: str = ""):
        self._explicit_root = os.path.expanduser(profiles_root) if profiles_root else ""

    def profiles_root(self) -> str:
        """The first existing profiles root, or "" when none is found."""
        if self._explicit_root:
            return self._explicit_root
        roots = _candidate_profile_roots()
        return roots[0] if roots else ""

    def available_profiles(self) -> list:
        """Names (``<group>/<variant>``) of every profile under the root."""
        root = self.profiles_root()
        if not root:
            return []
        out = []
        for dirpath, _dirnames, filenames in os.walk(root):
            if "profile.yaml" in filenames:
                out.append(os.path.relpath(dirpath, root).replace(os.sep, "/"))
        return sorted(out)

    # ------------------------------------------------------------------ #

    def _resolve_profile_dir(self, name_or_path: str) -> str:
        p = os.path.expanduser(str(name_or_path or "").strip())
        if not p:
            raise ProfileError("empty profile name")

        # Direct path (dir or profile.yaml file).
        if os.path.isfile(p) and os.path.basename(p) == "profile.yaml":
            return os.path.dirname(os.path.abspath(p))
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "profile.yaml")):
            return os.path.abspath(p)

        # Name lookup under the profile roots.
        tried = []
        roots = ([self._explicit_root] if self._explicit_root
                 else _candidate_profile_roots())
        for root in roots:
            cand = os.path.join(root, p)
            if os.path.isfile(os.path.join(cand, "profile.yaml")):
                return cand
            tried.append(cand)
        raise ProfileError(
            f"profile '{name_or_path}' not found. Tried: {tried or '(no profiles root found)'}"
        )

    def load(self, name_or_path: str) -> ProfileSpec:
        profile_dir = self._resolve_profile_dir(name_or_path)
        manifest_path = os.path.join(profile_dir, "profile.yaml")
        try:
            with open(manifest_path, "r") as f:
                raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ProfileError(f"malformed YAML in {manifest_path}: {e}")

        prof = raw.get("profile")
        if not isinstance(prof, dict):
            raise ProfileError(f"{manifest_path}: missing top-level 'profile:' mapping")

        for req in ("name", "algorithm", "trainer"):
            if not str(prof.get(req, "")).strip():
                raise ProfileError(f"{manifest_path}: missing required key 'profile.{req}'")

        config_paths = {}
        for key in CONFIG_FILE_KEYS:
            rel = str(prof.get(key, "")).strip()
            if not rel:
                continue  # optional (e.g. non-curriculum profile has no 'curriculum')
            config_paths[key] = os.path.normpath(
                rel if os.path.isabs(rel) else os.path.join(profile_dir, rel))

        overrides = {}
        raw_overrides = prof.get("overrides") or {}
        if not isinstance(raw_overrides, dict):
            raise ProfileError(f"{manifest_path}: 'profile.overrides' must be a mapping")
        for k, v in raw_overrides.items():
            if k not in KNOWN_OVERRIDE_KEYS:
                raise ProfileError(
                    f"{manifest_path}: unknown override '{k}' "
                    f"(known: {', '.join(KNOWN_OVERRIDE_KEYS)})")
            overrides[k] = bool(v)

        return ProfileSpec(
            name=str(prof["name"]).strip(),
            algorithm=str(prof["algorithm"]).strip().lower(),
            trainer=str(prof["trainer"]).strip().lower(),
            profile_dir=profile_dir,
            config_paths=config_paths,
            output_prefix=str(prof.get("output_prefix", "")).strip(),
            overrides=overrides,
        )
