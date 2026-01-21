cmake_minimum_required(VERSION 3.5)
project(ouster_description)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

# Install launch, urdf, meshes, and worlds directories (if exist)
# foreach(dir launch meshes urdf worlds rviz)
#   if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}")
#     install(DIRECTORY ${dir}
#       DESTINATION share/${PROJECT_NAME}/${dir})
#   endif()
# endforeach()

# Install test script
install(PROGRAMS
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY meshes urdf worlds rviz launch
  DESTINATION share/${PROJECT_NAME}
)

# Export package
ament_package()
