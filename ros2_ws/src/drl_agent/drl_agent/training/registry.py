"""Trainer registry — maps a profile's ``(algorithm, trainer)`` pair to the
canonical trainer/env MODULES that actually run it, and is also the single
source of truth for ``train_rl.py``'s flat ``MODEL_REGISTRY`` (its
``--rl_model``/``-p rl_model:=`` dispatch), which is derived from the
entries below at import time rather than re-listing each trainer_module path
a second time.

The profile system selects and validates configs, then the node wrapper
(``drl_agent/nodes/_node_common.py``) resolves each module's installed file
via ``importlib.util.find_spec`` and ``os.execv``'s a fresh ``python3
<resolved path>`` process for it — the SAME separate-process launch semantics
``ros2 run drl_agent <exec>`` always had, just targeting the canonical
package module instead of a flat legacy script name.

``class_name``/``description`` are only set on entries also reachable via
``--rl_model``/``-p rl_model:=<name>`` (every ``("<algo>", "curriculum")``
entry — the flat name is derived as ``f"{algorithm}_{trainer}"``, which
already matches every existing MODEL_REGISTRY key exactly). Left empty for
profile-only entries such as ``("tqc", "base")`` (the non-curriculum
ablation path, not selectable via ``--rl_model``) and ``derive_model_registry``
below skips any entry with an empty ``class_name``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerEntry:
    trainer_module: str    # dotted module, e.g. drl_agent.training.train_tqc_curriculum
    env_module: str         # dotted module, e.g. drl_agent.env.curriculum.environment_curriculum
    train_config_filename: str   # filename the trainer looks up via its dir hint
    env_config_filename: str     # filename the env node loads via config_file
    # Only set for entries also reachable via train_rl.py's flat
    # --rl_model dispatch — see module docstring.
    class_name: str = ""
    description: str = ""


TRAINERS = {
    ("tqc", "curriculum"): TrainerEntry(
        trainer_module="drl_agent.training.train_tqc_curriculum",
        env_module="drl_agent.env.curriculum.environment_curriculum",
        train_config_filename="train_tqc_config.yaml",
        env_config_filename="environment_curriculum.yaml",
        class_name="TrainTQCCurriculum",
        description="TQC + 10-stage curriculum (primary training path).",
    ),
    ("tqc", "base"): TrainerEntry(
        trainer_module="drl_agent.training.train_tqc_base",
        env_module="drl_agent.env.simulation.environment",
        train_config_filename="train_tqc_config.yaml",
        env_config_filename="environment.yaml",
    ),
    ("sac", "curriculum"): TrainerEntry(
        trainer_module="drl_agent.training.baselines.sac_curriculum",
        env_module="drl_agent.env.curriculum.environment_curriculum",
        train_config_filename="train_sac_config.yaml",
        env_config_filename="environment_curriculum.yaml",
        class_name="TrainSACCurriculum",
        description="SAC curriculum baseline.",
    ),
    ("td7", "curriculum"): TrainerEntry(
        trainer_module="drl_agent.training.baselines.td7_curriculum",
        env_module="drl_agent.env.curriculum.environment_curriculum",
        train_config_filename="train_td7_config.yaml",
        env_config_filename="environment_curriculum.yaml",
        class_name="TrainTD7Curriculum",
        description="TD7 curriculum baseline.",
    ),
    ("tqc_ieqn", "curriculum"): TrainerEntry(
        trainer_module="drl_agent.training.baselines.tqc_ieqn_curriculum",
        env_module="drl_agent.env.curriculum.environment_curriculum",
        train_config_filename="train_tqc_ieqn_config.yaml",
        env_config_filename="environment_curriculum.yaml",
        class_name="TrainTQCIEQNCurriculum",
        description="TQC + IEQn (inequality constraint) curriculum variant.",
    ),
    ("a3c", "curriculum"): TrainerEntry(
        trainer_module="drl_agent.training.baselines.a3c_curriculum",
        env_module="drl_agent.env.curriculum.environment_curriculum",
        train_config_filename="train_a3c_config.yaml",
        env_config_filename="environment_curriculum.yaml",
        class_name="TrainA3CCurriculum",
        description="A3C curriculum baseline.",
    ),
    ("sb3_sac", "curriculum"): TrainerEntry(
        trainer_module="drl_agent.training.baselines.sb3_sac_curriculum",
        env_module="drl_agent.env.curriculum.environment_curriculum",
        train_config_filename="train_sb3_sac_config.yaml",
        env_config_filename="environment_curriculum.yaml",
        class_name="TrainSB3SACCurriculum",
        description="Stable-Baselines3 SAC curriculum baseline.",
    ),
    ("sb3_td3", "curriculum"): TrainerEntry(
        trainer_module="drl_agent.training.baselines.sb3_td3_curriculum",
        env_module="drl_agent.env.curriculum.environment_curriculum",
        train_config_filename="train_sb3_td3_config.yaml",
        env_config_filename="environment_curriculum.yaml",
        class_name="TrainSB3TD3Curriculum",
        description="Stable-Baselines3 TD3 curriculum baseline.",
    ),
}


def derive_model_registry() -> dict:
    """Build train_rl.py's flat ``{model_name: {module, class_name,
    description}}`` MODEL_REGISTRY from TRAINERS above, so the trainer
    module path is listed exactly once (here) instead of twice. Only
    includes entries with a non-empty class_name (i.e. entries reachable
    via --rl_model — see module docstring); the flat name is
    ``f"{algorithm}_{trainer}"``, which already matches every existing
    MODEL_REGISTRY key exactly (verified: "tqc_curriculum", "sac_curriculum",
    "td7_curriculum", "tqc_ieqn_curriculum", "a3c_curriculum",
    "sb3_sac_curriculum", "sb3_td3_curriculum")."""
    return {
        f"{algorithm}_{trainer}": {
            "module": entry.trainer_module,
            "class_name": entry.class_name,
            "description": entry.description,
        }
        for (algorithm, trainer), entry in TRAINERS.items()
        if entry.class_name
    }


def lookup(algorithm: str, trainer: str) -> TrainerEntry:
    key = (str(algorithm).lower(), str(trainer).lower())
    if key not in TRAINERS:
        known = ", ".join(f"{a}/{t}" for a, t in sorted(TRAINERS))
        raise KeyError(f"no trainer registered for algorithm/trainer '{key[0]}/{key[1]}' "
                       f"(known: {known})")
    return TRAINERS[key]
