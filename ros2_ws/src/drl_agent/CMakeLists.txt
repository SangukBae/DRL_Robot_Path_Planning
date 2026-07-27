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
  # The legacy code base predates flake8/pep257 conventions (thousands of
  # accumulated style-only hits, no functional signal) — skip them like
  # copyright/cpplint above so `colcon test` reflects FUNCTIONAL health.
  # The functional coverage is the pytest suite registered right below.
  set(ament_cmake_flake8_FOUND TRUE)
  set(ament_cmake_pep257_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  # Run the full ROS-free unit suite (tests/, ~600 cases incl. the
  # migration-invariant and replay-schema tests) under `colcon test`.
  # pytest.ini at the package root scopes collection; torch/ROS-gated tests
  # self-skip where those deps are unavailable.
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(drl_agent_pytests tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 600
    # Collection imports every module under drl_agent/; without this, running
    # the suite against a bind-mounted / shared-ownership source tree (e.g.
    # the docker dev container) leaves root-owned __pycache__/*.pyc behind
    # that a later host-side `compileall` or `git clean` chokes on.
    ENV PYTHONDONTWRITEBYTECODE=1
  )
endif()

install(
  DIRECTORY
    config
    launch
  DESTINATION
    share/${PROJECT_NAME}/
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

# Importable Python package (drl_agent/drl_agent/) — the CANONICAL and ONLY
# implementation: config (profile loader/validator, paths), rl (networks,
# replay buffer+schema, checkpointing incl. tqc_io, algorithms tqc/sac/td7/
# a3c/tqc_ieqn/sb3), training (trainer bases + curriculum modules + train_rl +
# baselines), evaluation (generalization/risk-map/sim-validation, live-sim
# runners, post-hoc analysis), env (environment node bodies, mixins,
# samplers, rewards, observation), common (pure helpers). Installed into
# site-packages, importable as `import drl_agent.<...>` from any sourced
# environment — this is the only mechanism by which drl_agent code is
# importable (no flat/bare-name compatibility layer).
ament_python_install_package(${PROJECT_NAME})

# ROS executables (`ros2 run drl_agent <name>.py`). Each entry below is
# installed under its OWN basename into lib/${PROJECT_NAME}; the canonical
# implementation lives at the listed drl_agent/ path (ament_python_install_package
# above already makes it importable — this list only grants direct `ros2 run`
# access). Runtime dispatch between them (e.g. the profile wrappers below
# selecting a trainer/env module) resolves the canonical module via
# importlib.util.find_spec, not by relying on this flat layout — see
# drl_agent/nodes/_node_common.py.
install(
  PROGRAMS
    # Profile-based entrypoint wrappers (resolve + validate a profile, then
    # exec the canonical trainer/env/eval module for it).
    drl_agent/nodes/train_node.py
    drl_agent/nodes/environment_curriculum_node.py
    drl_agent/nodes/eval_node.py
    drl_agent/nodes/real_policy_node.py
    # Unified trainer launcher (MODEL_REGISTRY; --rl_model / -p rl_model:=...).
    drl_agent/training/train_rl.py
    # TQC (primary): curriculum path + non-curriculum/ablation path.
    drl_agent/training/train_tqc_curriculum.py
    drl_agent/training/train_tqc_base.py
    # Environment nodes.
    drl_agent/env/curriculum/environment_curriculum.py
    drl_agent/env/simulation/environment.py
    drl_agent/env/simulation/environment_360.py   # legacy Classic-Gazebo variant
    # Evaluation / deploy harnesses.
    drl_agent/evaluation/generalization_eval.py
    drl_agent/evaluation/risk_map_eval.py
    drl_agent/evaluation/real_policy_runner.py
    drl_agent/evaluation/sim_validation_runner.py
    drl_agent/evaluation/live/tqc_live_runner.py   # used by launch/test_tqc.launch.py
    drl_agent/evaluation/live/td7_live_runner.py   # used by launch/test_td7.launch.py
    # Post-hoc analysis / result aggregation.
    drl_agent/evaluation/analysis/aggregate_results.py
    drl_agent/evaluation/analysis/aux_ablation_summary.py
    drl_agent/evaluation/analysis/analyze_yield_freezing.py
    drl_agent/evaluation/analysis/analyze_aux_correlation.py
    drl_agent/evaluation/analysis/sim_validation_summary.py
    # Paper comparison/ablation baselines — SAC, TD7, A3C, TQC+IEQn, and the
    # Stable-Baselines3 SAC/TD3/PPO wrappers — each with a curriculum variant
    # where one exists (PPO has none). Reachable directly here AND via
    # train_rl.py's MODEL_REGISTRY.
    drl_agent/training/baselines/sac.py
    drl_agent/training/baselines/sac_curriculum.py
    drl_agent/training/baselines/td7.py
    drl_agent/training/baselines/td7_curriculum.py
    drl_agent/training/baselines/a3c.py
    drl_agent/training/baselines/a3c_curriculum.py
    drl_agent/training/baselines/tqc_ieqn.py
    drl_agent/training/baselines/tqc_ieqn_curriculum.py
    drl_agent/training/baselines/sb3_sac.py
    drl_agent/training/baselines/sb3_sac_curriculum.py
    drl_agent/training/baselines/sb3_td3.py
    drl_agent/training/baselines/sb3_td3_curriculum.py
    drl_agent/training/baselines/sb3_ppo.py
  DESTINATION
    lib/${PROJECT_NAME}
)

ament_package()
