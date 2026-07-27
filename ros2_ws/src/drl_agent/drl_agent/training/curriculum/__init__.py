"""Curriculum training support (canonical home of the flat ``curriculum_*`` modules).

  stage_logic.py  stage-advance promotion gate     <- curriculum_stage_logic
  metrics.py      eval metric accumulators (pure)  <- curriculum_metrics
  state_io.py     resume-state persistence         <- curriculum_state_io
  eval_runner.py  CurriculumEvalMixin (eval loop)  <- curriculum_eval_runner
  aux_eval.py     CurriculumAuxEvalMixin           <- curriculum_aux_eval

stage_logic / metrics / state_io are ROS-free; eval_runner / aux_eval need
torch + the env service interface — import those submodules directly.
"""
