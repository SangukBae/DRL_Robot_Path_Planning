cmake_minimum_required(VERSION 3.8)
project(hunter_se_gazebo)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY urdf launch config rviz worlds meshes
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/hunter_se_teleop_key.py
  scripts/hunter_se_cmd_prefilter.py
  scripts/hunter_se_steering_monitor.py
  scripts/hunter_se_dynamics_monitor.py
  scripts/hunter_se_mesh_extent.py
  scripts/hunav_pose_csv_logger.py
  scripts/hunter_se_validation_driver.py
  scripts/hunter_se_validation_logger.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
