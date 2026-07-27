"""drl_agent — importable Python package for the DRL path-planning stack.

Structure (incremental migration target — see docs/overview/package_structure.md):

  drl_agent.config      profile/config loading + validation (new code)
  drl_agent.training    run layout / run manager / trainer registry (new code)
  drl_agent.rl          algorithm-side building blocks (checkpointing, …)
  drl_agent.env         environment-side building blocks (facade, migrating)
  drl_agent.common      pure helpers + flat-scripts compatibility bridge
  drl_agent.nodes       ROS entrypoint wrappers (profile-based train/env nodes)

Legacy note: the historical implementation lives in ``scripts/{environment,
policy,utils}`` as FLAT bare-name modules (installed into ``lib/drl_agent``).
Modules migrate here one at a time, each leaving a bare-name shim behind so
``import <name>`` keeps working for the legacy entrypoints.
"""

__all__ = ["common", "config", "training", "rl", "env", "nodes"]
