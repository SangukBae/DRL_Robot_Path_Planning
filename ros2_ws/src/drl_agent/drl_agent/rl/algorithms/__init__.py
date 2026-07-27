"""RL algorithm agents — every model used as a paper comparison/ablation
baseline lives here, alongside TQC (primary).

  tqc/       Truncated Quantile Critics (primary)
  tqc_ieqn/  TQC + IEQn variant
  sac/       Soft Actor-Critic
  td7/       TD7
  a3c/       A3C
  sb3/       Stable-Baselines3 baselines (sac, td3, ppo)

Each subpackage exports ``Agent`` lazily (torch required); nothing here imports
torch at package-import time.
"""

__all__ = ["tqc", "tqc_ieqn", "sac", "td7", "a3c", "sb3"]
