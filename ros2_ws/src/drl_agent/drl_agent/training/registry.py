"""Trainer registry — maps a profile's ``(algorithm, trainer)`` pair to the
canonical trainer/env MODULES that actually run it.

The profile system selects and validates configs, then the node wrapper
(``drl_agent/nodes/_node_common.py``) resolves each module's installed file
via ``importlib.util.find_spec`` and ``os.execv``'s a fresh ``python3
<resolved path>`` process for it — the SAME separate-process launch semantics
``ros2 run drl_agent <exec>`` always had, just targeting the canonical
package module instead of a flat legacy script name. ``train_rl.py``'s
MODEL_REGISTRY remains the lower-level per-algorithm registry (reachable
directly via ``-p rl_model:=<name>``); this table only adds the profile-facing
env/trainer pairing.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerEntry:
    trainer_module: str    # dotted module, e.g. drl_agent.training.train_tqc_curriculum
    env_module: str         # dotted module, e.g. drl_agent.env.curriculum.environment_curriculum
    train_config_filename: str   # filename the trainer looks up via its dir hint
    env_config_filename: str     # filename the env node loads via config_file


TRAINERS = {
    ("tqc", "curriculum"): TrainerEntry(
        trainer_module="drl_agent.training.train_tqc_curriculum",
        env_module="drl_agent.env.curriculum.environment_curriculum",
        train_config_filename="train_tqc_config.yaml",
        env_config_filename="environment_curriculum.yaml",
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
    ),
    ("td7", "curriculum"): TrainerEntry(
        trainer_module="drl_agent.training.baselines.td7_curriculum",
        env_module="drl_agent.env.curriculum.environment_curriculum",
        train_config_filename="train_td7_config.yaml",
        env_config_filename="environment_curriculum.yaml",
    ),
    ("tqc_ieqn", "curriculum"): TrainerEntry(
        trainer_module="drl_agent.training.baselines.tqc_ieqn_curriculum",
        env_module="drl_agent.env.curriculum.environment_curriculum",
        train_config_filename="train_tqc_ieqn_config.yaml",
        env_config_filename="environment_curriculum.yaml",
    ),
    ("a3c", "curriculum"): TrainerEntry(
        trainer_module="drl_agent.training.baselines.a3c_curriculum",
        env_module="drl_agent.env.curriculum.environment_curriculum",
        train_config_filename="train_a3c_config.yaml",
        env_config_filename="environment_curriculum.yaml",
    ),
    ("sb3_sac", "curriculum"): TrainerEntry(
        trainer_module="drl_agent.training.baselines.sb3_sac_curriculum",
        env_module="drl_agent.env.curriculum.environment_curriculum",
        train_config_filename="train_sb3_sac_config.yaml",
        env_config_filename="environment_curriculum.yaml",
    ),
    ("sb3_td3", "curriculum"): TrainerEntry(
        trainer_module="drl_agent.training.baselines.sb3_td3_curriculum",
        env_module="drl_agent.env.curriculum.environment_curriculum",
        train_config_filename="train_sb3_td3_config.yaml",
        env_config_filename="environment_curriculum.yaml",
    ),
}


def lookup(algorithm: str, trainer: str) -> TrainerEntry:
    key = (str(algorithm).lower(), str(trainer).lower())
    if key not in TRAINERS:
        known = ", ".join(f"{a}/{t}" for a, t in sorted(TRAINERS))
        raise KeyError(f"no trainer registered for algorithm/trainer '{key[0]}/{key[1]}' "
                       f"(known: {known})")
    return TRAINERS[key]
