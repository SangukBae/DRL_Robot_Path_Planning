"""Trainer registry — maps a profile's ``(algorithm, trainer)`` pair to the
legacy ROS entrypoints that actually run it.

This intentionally reuses the UNMODIFIED legacy executables (the same ones
``ros2 run drl_agent <exec>`` has always launched): the profile system selects
and validates configs, then delegates. ``train_rl.py``'s MODEL_REGISTRY remains
the lower-level per-algorithm registry; this table only adds the profile-facing
env/trainer pairing.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerEntry:
    trainer_exec: str      # ros2 run drl_agent <trainer_exec>
    env_exec: str          # ros2 run drl_agent <env_exec>
    train_config_filename: str   # filename the trainer looks up via its dir hint
    env_config_filename: str     # filename the env node loads via config_file


TRAINERS = {
    ("tqc", "curriculum"): TrainerEntry(
        trainer_exec="train_tqc_curriculum_agent.py",
        env_exec="environment_curriculum.py",
        train_config_filename="train_tqc_config.yaml",
        env_config_filename="environment_curriculum.yaml",
    ),
    ("tqc", "base"): TrainerEntry(
        trainer_exec="train_tqc_base.py",
        env_exec="environment.py",
        train_config_filename="train_tqc_config.yaml",
        env_config_filename="environment.yaml",
    ),
    ("sac", "curriculum"): TrainerEntry(
        trainer_exec="train_sac_curriculum_agent.py",
        env_exec="environment_curriculum.py",
        train_config_filename="train_sac_config.yaml",
        env_config_filename="environment_curriculum.yaml",
    ),
    ("td7", "curriculum"): TrainerEntry(
        trainer_exec="train_td7_curriculum_agent.py",
        env_exec="environment_curriculum.py",
        train_config_filename="train_td7_config.yaml",
        env_config_filename="environment_curriculum.yaml",
    ),
    ("tqc_ieqn", "curriculum"): TrainerEntry(
        trainer_exec="train_tqc_ieqn_curriculum_agent.py",
        env_exec="environment_curriculum.py",
        train_config_filename="train_tqc_ieqn_config.yaml",
        env_config_filename="environment_curriculum.yaml",
    ),
    ("a3c", "curriculum"): TrainerEntry(
        trainer_exec="train_a3c_curriculum_agent.py",
        env_exec="environment_curriculum.py",
        train_config_filename="train_a3c_config.yaml",
        env_config_filename="environment_curriculum.yaml",
    ),
    ("sb3_sac", "curriculum"): TrainerEntry(
        trainer_exec="train_sb3_sac_curriculum_agent.py",
        env_exec="environment_curriculum.py",
        train_config_filename="train_sb3_sac_config.yaml",
        env_config_filename="environment_curriculum.yaml",
    ),
    ("sb3_td3", "curriculum"): TrainerEntry(
        trainer_exec="train_sb3_td3_curriculum_agent.py",
        env_exec="environment_curriculum.py",
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
