cmake_minimum_required(VERSION 3.8)
project(aws_robomaker_small_house_world)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch models worlds maps photos param routes env-hooks
    DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(ament_cmake)

# Ignition Gazebo 환경 변수 설정을 위한 env-hooks
ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/aws_robomaker_small_house_world.dsv.in")

ament_package()
