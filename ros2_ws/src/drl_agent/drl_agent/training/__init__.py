"""Training-side logic: profile/run infrastructure, the TQC trainer (primary),
and every other algorithm's trainer (paper comparison/ablation baselines).

  registry.py / run_manager.py / run_layout.py   profile & run-dir plumbing
  train_tqc_base.py / train_tqc_curriculum.py    TQC trainer (primary path)
  train_rl.py                                    unified launcher (MODEL_REGISTRY)
  curriculum/                                    stage logic, metrics, eval mixins
  baselines/                                     SAC / TD7 / A3C / TQC+IEQn / SB3-*
  gym_parameter_client.py, episode_metrics.py, aux_ablation_logging.py,
  aux_eval_metrics.py, dynamic_avoidance_log.py  trainer-support utilities
"""
