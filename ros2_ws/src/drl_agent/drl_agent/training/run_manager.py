"""RunManager — one place for run identity + output-dir bookkeeping.

Wraps the battle-tested pure helpers in :mod:`drl_agent.training.run_layout`
(the module the curriculum trainer already uses, so behaviour is IDENTICAL to
a hand-launched run: same run_id format, same resume precedence, same "never
migrate a legacy dir" guarantee) and adds:

  * config sha256 hashing;
  * a ``profile_manifest.json`` written into the run's ``configs/`` dir
    recording profile name, config paths + hashes, overrides, seed.

The legacy trainers keep writing their own ``run_manifest.json`` via
aux_ablation_logging — this file is ADDITIVE (different filename), so nothing
existing changes.
"""

import hashlib
import json
import os
from datetime import datetime

from . import run_layout


def config_hashes(config_paths: dict) -> dict:
    """``{key: "sha256:<hex>"}`` for each existing config file ("" if missing)."""
    out = {}
    for key, path in (config_paths or {}).items():
        try:
            with open(path, "rb") as f:
                out[key] = "sha256:" + hashlib.sha256(f.read()).hexdigest()
        except OSError:
            out[key] = ""
    return out


class RunManager:
    def __init__(self, package_root: str, base_file_name: str, seed: int,
                 *, load_model: bool):
        self.package_root = package_root
        self.base_file_name = base_file_name
        self.seed = int(seed)
        self.load_model = bool(load_model)
        self.layout = None

    def resolve(self) -> dict:
        """Decide (and remember) this run's directory layout — see
        ``run_layout.resolve_run_layout`` for the exact precedence."""
        self.layout = run_layout.resolve_run_layout(
            self.package_root, self.base_file_name, self.seed,
            load_model=self.load_model)
        return self.layout

    def create_dirs(self) -> None:
        if self.layout is None:
            self.resolve()
        run_layout.create_run_dirs(self.layout)

    def write_profile_manifest(self, *, profile_name: str = "",
                               config_paths: dict = None,
                               overrides: dict = None,
                               extra: dict = None) -> str:
        """Write ``configs/profile_manifest.json`` (new-layout runs only —
        legacy dirs never grow new files). Returns the path, or ""."""
        if self.layout is None:
            self.resolve()
        configs_dir = self.layout.get("configs_dir")
        if not configs_dir:
            return ""  # legacy layout — leave untouched
        os.makedirs(configs_dir, exist_ok=True)
        manifest = {
            "written_at": datetime.now().isoformat(timespec="seconds"),
            "profile": profile_name,
            "run_id": self.layout.get("run_id"),
            "base_file_name": self.base_file_name,
            "seed": self.seed,
            "load_model": self.load_model,
            "config_paths": dict(config_paths or {}),
            "config_hashes": config_hashes(config_paths or {}),
            "overrides": dict(overrides or {}),
        }
        manifest.update(extra or {})
        path = os.path.join(configs_dir, "profile_manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        return path
