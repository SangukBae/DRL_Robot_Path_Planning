"""Typed schema for experiment profiles and their validation results.

A *profile* is a self-contained experiment definition living in
``drl_experiments/profiles/<group>/<variant>/profile.yaml`` next to the config
files it references (legacy filenames, so the profile dir can be handed to the
legacy trainers verbatim as their ``train_config_file``/``config_file`` dir
hint).
"""

from dataclasses import dataclass, field


#: profile.yaml keys that name a config file (relative to the profile dir).
CONFIG_FILE_KEYS = ("environment", "train", "hparams", "curriculum")

#: Override flags a profile may declare. Maps 1:1 onto the existing PHASE2
#: ``-p <flag>:=true|false`` CLI overrides (see config_paths.parse_bool_override).
KNOWN_OVERRIDE_KEYS = ("risk_map_reward_enabled", "action_risk_head_enabled")


@dataclass
class ProfileSpec:
    """A loaded profile manifest with every config path resolved to absolute."""

    name: str                      # e.g. "phase2/both"
    algorithm: str                 # e.g. "tqc"
    trainer: str                   # "curriculum" | "base"
    profile_dir: str               # absolute dir containing profile.yaml
    config_paths: dict             # key in CONFIG_FILE_KEYS -> absolute path
    output_prefix: str = ""        # must match train yaml's base_file_name
    overrides: dict = field(default_factory=dict)  # KNOWN_OVERRIDE_KEYS -> bool

    @property
    def environment_yaml(self) -> str:
        return self.config_paths.get("environment", "")

    @property
    def train_yaml(self) -> str:
        return self.config_paths.get("train", "")

    @property
    def hparams_yaml(self) -> str:
        return self.config_paths.get("hparams", "")

    @property
    def curriculum_yaml(self) -> str:
        return self.config_paths.get("curriculum", "")


@dataclass
class ValidationReport:
    """Outcome of ConfigValidator.validate(). ``ok`` == no errors (warnings
    never block a run; ``info`` records resolved facts for logging/manifests)."""

    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = []
        for e in self.errors:
            lines.append(f"[ERROR] {e}")
        for w in self.warnings:
            lines.append(f"[WARN ] {w}")
        for k in sorted(self.info):
            lines.append(f"[info ] {k} = {self.info[k]}")
        return "\n".join(lines) if lines else "[info ] validation: nothing to report"
