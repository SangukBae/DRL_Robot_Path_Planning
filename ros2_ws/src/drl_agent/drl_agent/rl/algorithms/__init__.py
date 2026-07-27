"""RL algorithm agents (canonical home of the flat legacy ``*_agent`` modules).

  tqc/       Truncated Quantile Critics (primary)      <- tqc_agent
  tqc_ieqn/  TQC + IEQn variant                        <- tqc_ieqn_agent
  sac/       Soft Actor-Critic                         <- sac_agent
  td7/       TD7                                       <- td7_agent
  a3c/       A3C                                       <- a3c_agent
  sb3/       Stable-Baselines3 baselines (sac, td3)    <- sb3_{sac,td3}_agent

Each subpackage exports ``Agent`` lazily (torch required); nothing here imports
torch at package-import time.
"""

__all__ = ["tqc", "tqc_ieqn", "sac", "td7", "a3c", "sb3"]
