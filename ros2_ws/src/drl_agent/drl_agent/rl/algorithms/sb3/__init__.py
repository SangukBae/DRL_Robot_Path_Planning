"""Stable-Baselines3 baseline agents.

Submodules require torch (and, for actual training, stable_baselines3) — import
``drl_agent.rl.algorithms.sb3.sac`` / ``.td3`` / ``.ppo`` directly; nothing is
imported eagerly here so environments without the optional SB3 dependency can
still import the package tree.
"""

__all__ = ["sac", "td3", "ppo"]
