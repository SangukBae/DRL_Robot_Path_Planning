"""drl_agent — importable Python package for the DRL path-planning stack.

This is the ONLY home of the implementation — see
docs/overview/package_structure.md for the full layout:

  drl_agent.config      profile/config loading + validation, path discovery
  drl_agent.training    trainer bases (TQC base/curriculum, train_rl registry),
                        run layout/manager, curriculum stage logic + eval
                        mixins, and every baseline trainer (drl_agent.training.
                        baselines — paper comparison/ablation models)
  drl_agent.rl          networks / replay buffer(+schema) / checkpointing /
                        algorithm agents (tqc, sac, td7, a3c, tqc_ieqn, sb3)
  drl_agent.evaluation  generalization / risk-map / sim-validation evals,
                        real-robot policy runner, live-sim run scripts
                        (drl_agent.evaluation.live), post-hoc analysis tools
                        (drl_agent.evaluation.analysis)
  drl_agent.env         environment nodes + mixins (simulation, curriculum,
                        observation, rewards, spawning, humans)
  drl_agent.common      pure helpers (source-root resolution, geometry/seed
                        utils, checkpointing/config I/O, pure pursuit)
  drl_agent.nodes       ROS entrypoint wrappers (profile-based train/env nodes)

There is no flat ``scripts/`` layer and no bare-name import compatibility
layer (the 2026-07 ``scripts/{environment,policy,utils}`` restructure is
complete and that directory has been removed) — ``import drl_agent.<...>`` is
the sole import path for every module in this package, whether or not it is
also installed as a ``ros2 run`` executable. Structural invariants (canonical
import success, retired-bare-name failure, no circular imports) are locked by
tests/test_package_migration.py; the installed-environment contract by
tests/test_installed_canonical_imports.py.
"""

__all__ = ["common", "config", "training", "rl", "evaluation", "env", "nodes"]
