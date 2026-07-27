"""Non-TQC baseline trainer implementations (canonical home of the flat legacy
``train_<algo>_agent.py`` / ``train_<algo>_curriculum_agent.py`` modules).

Each baseline is independent of TrainTQCBase (its own EnvInterface subclass,
mirroring the original scripts/ structure exactly) — this is deliberate: these
files back the paper's comparison/ablation baselines (SAC, TD7, A3C, TQC+IEQn,
SB3-SAC/TD3/PPO) and must stay reproducible byte-for-byte with how they always
trained, not be forced through TQC's aux/temporal/action-risk machinery.

  <algo>.py             single-phase trainer (base class)
  <algo>_curriculum.py  curriculum subclass (primary paper training path)
"""
