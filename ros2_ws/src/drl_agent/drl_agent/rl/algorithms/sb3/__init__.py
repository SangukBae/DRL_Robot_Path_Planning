"""Stable-Baselines3 baseline agents (canonical home of the flat legacy
``sb3_sac_agent`` / ``sb3_td3_agent`` modules).

Submodules require torch (and, for actual training, stable_baselines3) — import
``drl_agent.rl.algorithms.sb3.sac`` / ``.td3`` directly; nothing is imported
eagerly here so environments without the optional SB3 dependency can still
import the package tree. ``sb3_ppo_agent`` remains a flat legacy module.
"""

__all__ = ["sac", "td3"]
