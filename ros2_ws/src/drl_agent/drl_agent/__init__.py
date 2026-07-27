"""drl_agent — importable Python package for the DRL path-planning stack.

Structure (incremental migration target — see docs/overview/package_structure.md):

  drl_agent.config      profile/config loading + validation, path discovery
  drl_agent.training    trainer bases (TQC base/curriculum, train_rl registry),
                        run layout/manager, curriculum stage logic + eval mixins
  drl_agent.rl          networks / replay buffer(+schema) / checkpointing /
                        algorithm agents (tqc, sac, td7, a3c, tqc_ieqn, sb3)
  drl_agent.evaluation  generalization / risk-map / sim-validation evals,
                        real-robot policy runner
  drl_agent.env         environment nodes + mixins (simulation, curriculum,
                        observation, rewards, spawning, humans)
  drl_agent.common      pure helpers + flat-scripts compatibility bridge
  drl_agent.nodes       ROS entrypoint wrappers (profile-based train/env nodes)

Legacy note: the historical FLAT bare-name modules in ``scripts/{environment,
policy,utils}`` (installed into ``lib/drl_agent``) are now thin shims aliasing
the canonical modules here (``sys.modules[__name__] = _impl``; entry points
dispatch to ``_impl.main()``), so ``import <name>`` and every legacy
``ros2 run`` path keep working. See tests/test_package_migration.py.
"""

__all__ = ["common", "config", "training", "rl", "evaluation", "env", "nodes"]
