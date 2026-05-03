"""
Launch file for OctoMap Server with Scout robot in Gazebo Ignition.

Usage:
    1. First launch Gazebo simulation:
       ros2 launch agilex_scout simulate_control_gazebo_ignition.launch.py rviz:=false

    2. Then launch this file:
       ros2 launch octomap_server octomap_scout_ignition.launch.py

    3. (Optional) Launch with LIO-SAM for better odometry:
       ros2 launch lio_sam run_scout_ignition.launch.py
       # Then change frame_id to 'map' in params or via command line
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directory
    octomap_server_dir = get_package_share_directory('octomap_server')

    # Parameters file
    params_file = os.path.join(
        octomap_server_dir,
        'params',
        'scout_ignition.yaml'
    )

    # Launch arguments
    frame_id_arg = DeclareLaunchArgument(
        'frame_id',
        default_value='odom',
        description='Fixed frame for OctoMap (use "odom" for Gazebo, "map" for LIO-SAM)'
    )

    resolution_arg = DeclareLaunchArgument(
        'resolution',
        default_value='0.05',
        description='OctoMap resolution in meters'
    )

    max_range_arg = DeclareLaunchArgument(
        'max_range',
        default_value='20.0',
        description='Maximum sensor range for ray casting'
    )

    # OctoMap Server node
    octomap_server_node = Node(
        package='octomap_server',
        executable='octomap_server_node',
        name='octomap_server',
        output='screen',
        parameters=[
            params_file,
            {
                'frame_id': LaunchConfiguration('frame_id'),
                'resolution': LaunchConfiguration('resolution'),
                'sensor_model.max_range': LaunchConfiguration('max_range'),
                'use_sim_time': True,
            }
        ],
        remappings=[
            # Remap input topic to Gazebo Ignition pointcloud
            ('cloud_in', '/points'),
        ],
    )

    return LaunchDescription([
        frame_id_arg,
        resolution_arg,
        max_range_arg,
        octomap_server_node,
    ])
