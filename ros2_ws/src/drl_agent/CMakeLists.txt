cmake_minimum_required(VERSION 3.8)
project(drl_agent)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
# Gazebo Classic 전용 패키지였던 drl_agent_gazebo는 Fortress 환경에서는 사용하지 않으므로 제거
# find_package(drl_agent_gazebo REQUIRED)
find_package(drl_agent_interfaces REQUIRED)
# ros_gz_interfaces: SpawnEntity / DeleteEntity / SetEntityPose / ControlWorld
find_package(ros_gz_interfaces REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY
    config
    launch
    scripts
  DESTINATION
    share/${PROJECT_NAME}/
)

# Importable Python package (drl_agent/drl_agent/): profile loader/validator,
# trainer registry, run/checkpoint managers, and the migrated pure modules
# (run_layout, config_paths -> drl_agent.config.paths, geometry/seed utils).
# Installed into site-packages so both the flat legacy scripts (via their
# bare-name shims) and new code can `import drl_agent.*`.
ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    scripts/policy/td7_agent.py
    scripts/policy/a3c_agent.py
    scripts/policy/sac_agent.py
    scripts/policy/tqc_agent.py
    scripts/policy/tqc_ieqn_agent.py
    scripts/policy/train_sac_agent.py
    scripts/policy/train_td7_agent.py
    scripts/policy/test_td7_agent.py
    scripts/policy/test_tqc_agent.py
    scripts/policy/train_a3c_agent.py
    scripts/policy/train_tqc_base.py
    scripts/policy/train_tqc_ieqn_agent.py
    scripts/policy/sb3_sac_agent.py
    scripts/policy/sb3_td3_agent.py
    scripts/policy/sb3_ppo_agent.py
    scripts/policy/train_sb3_sac_agent.py
    scripts/policy/train_sb3_td3_agent.py
    scripts/policy/train_sb3_ppo_agent.py
    scripts/utils/buffer.py
    scripts/utils/point_cloud2.py
    scripts/utils/file_manager.py
    scripts/utils/episode_metrics.py
    scripts/utils/dynamic_avoidance_log.py     # DYN_AVOID consolidated per-episode CSV
    scripts/utils/pure_pursuit.py
    scripts/utils/geometry_utils.py    # pure helpers (extracted from environment.py)
    scripts/utils/seed_utils.py        # centralised RNG seeding
    scripts/utils/config_paths.py      # pure path/config-discovery helpers
    scripts/utils/run_layout.py        # RUN_LAYOUT: runtime/experiments/<run_id>/ per-run dirs
    scripts/utils/sim_validation.py            # SIM_VALIDATION
    scripts/utils/sim_validation_summary.py    # SIM_VALIDATION
    scripts/environment/environment_interface.py
    scripts/environment/environment.py
    scripts/environment/environment_360.py
    scripts/environment/environment_curriculum.py
    scripts/environment/reward_calculator.py     # extracted from environment.py
    scripts/environment/collision_checker.py     # RectSafetyChecker (from environment.py)
    scripts/environment/dynamic_avoidance_telemetry.py # DYN_AVOID per-episode pedestrian diagnostics
    scripts/environment/localization_noise.py    # loc/proprio noise models (from environment.py)
    scripts/environment/map_catalog.py           # static-obstacle catalog policy (pure)
    scripts/environment/map_layout_registry.py   # pure map-layout geometry builder
    scripts/environment/zone_tracker.py          # ZoneMixin (legacy zone tracking)
    scripts/environment/observation_builder.py   # ObservationMixin (state assembly)
    scripts/environment/gazebo_entity_manager.py # GazeboEntityMixin (spawn/delete)
    scripts/environment/gazebo_service_wait.py   # Gazebo service wait helpers
    scripts/environment/map_layout_runtime.py    # MapLayoutMixin (layout selection/walls)
    scripts/environment/start_sampler.py         # StartSamplerMixin
    scripts/environment/goal_sampler.py          # GoalSamplerMixin
    scripts/environment/human_spawn_sampler.py   # HumanSpawnMixin
    scripts/environment/human_motion_manager.py  # HumanMotionMixin
    scripts/environment/obstacle_catalog_spawner.py # ObstacleMixin
    scripts/policy/tqc_networks.py               # Actor/Critic/loss (from tqc_agent.py)
    scripts/policy/tqc_io.py                      # checkpoint save/load (from tqc_agent.py)
    scripts/policy/gym_parameter_client.py       # /gym_node param client (from curriculum trainer)
    scripts/policy/curriculum_stage_logic.py     # stage-advance gate (from curriculum trainer)
    scripts/policy/curriculum_state_io.py        # resume state persistence (from curriculum trainer)
    scripts/policy/curriculum_metrics.py         # eval metric accumulators (pure)
    scripts/policy/curriculum_aux_eval.py        # CurriculumAuxEvalMixin
    scripts/policy/curriculum_eval_runner.py     # CurriculumEvalMixin (eval loop)
    scripts/policy/train_tqc_curriculum_agent.py
    scripts/policy/train_rl.py                   # unified trainer launcher (MODEL_REGISTRY, see file header)
    scripts/policy/train_td7_curriculum_agent.py
    scripts/policy/train_sac_curriculum_agent.py
    scripts/policy/train_tqc_ieqn_curriculum_agent.py
    scripts/policy/train_a3c_curriculum_agent.py
    scripts/policy/train_sb3_sac_curriculum_agent.py
    scripts/policy/train_sb3_td3_curriculum_agent.py
    scripts/policy/generalization_eval.py
    scripts/policy/risk_map_eval.py               # PHASE1B: per-step risk-map dump eval harness
    scripts/policy/real_policy_runner.py
    scripts/policy/sim_validation_runner.py    # SIM_VALIDATION
    scripts/utils/aggregate_results.py
    scripts/policy/aux_prediction.py            # AUX_PRED
    scripts/policy/aux_prediction_losses.py     # AUX_PRED
    scripts/policy/aux_prediction_temporal.py   # AUX_PRED
    scripts/environment/aux_prediction_labels.py # AUX_PRED
    scripts/environment/obs_time_context.py      # observation time-context (frame stacking)
    scripts/utils/aux_ablation_logging.py       # AUX_ABLATION
    scripts/utils/aux_ablation_summary.py       # AUX_ABLATION
    scripts/utils/analyze_yield_freezing.py     # PHASE1A: yield precision/recall/freezing post-processing
    scripts/utils/analyze_aux_correlation.py    # PHASE1A: aux-accuracy vs avoidance correlation post-processing
    scripts/utils/risk_map_dump.py               # PHASE1B: per-step risk-map dump writer (JSONL)
    scripts/utils/aux_eval_metrics.py           # AUX_PRED formal eval metrics
    scripts/policy/action_risk_head.py          # PHASE2: Critic-connected Action-Risk Head
    # PROFILE: new profile-based entrypoint wrappers (delegate to the legacy
    # executables above after ConfigValidator pre-flight — see drl_agent/nodes/).
    drl_agent/nodes/train_node.py
    drl_agent/nodes/environment_curriculum_node.py
    drl_agent/nodes/eval_node.py
    drl_agent/nodes/real_policy_node.py
  DESTINATION
    lib/${PROJECT_NAME}
)

ament_package()
