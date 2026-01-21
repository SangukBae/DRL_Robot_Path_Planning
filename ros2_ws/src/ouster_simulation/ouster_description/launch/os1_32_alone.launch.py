import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, EmitEvent, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    this_directory = get_package_share_directory('ouster_description')
    xacro_path = os.path.join(this_directory, 'urdf', 'os1_32_example.urdf.xacro')
    world = os.path.join(this_directory, 'worlds', 'test.world')
    rviz_config_file = os.path.join(this_directory, 'rviz', 'test.rviz')

    # Robot description (no gpu parameter needed for Ignition)
    robot_description = Command(['xacro ', xacro_path])

    # Robot state publisher
    start_robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': robot_description
        }]
    )

    # Spawn robot in Gazebo Ignition
    spawn_example_cmd = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'example',
            '-topic', 'robot_description',
        ],
        output='screen',
    )

    # RViz
    start_rviz_cmd = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    exit_event_handler = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=start_rviz_cmd,
            on_exit=EmitEvent(event=Shutdown(reason='rviz exited'))
        )
    )

    # Gazebo Ignition (Gazebo Sim) launch
    declare_gui_cmd = DeclareLaunchArgument(
        'gui',
        default_value='True',
        description='Whether to launch the Gazebo GUI or not (headless)')
    gui = LaunchConfiguration('gui')

    # Use ros_gz_sim instead of gazebo_ros
    start_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': ['-r ', world],
        }.items()
    )

    # Bridge for PointCloud2 from Ignition to ROS2
    # Ignition topic: /os/points -> ROS2 topic: /ouster/points
    bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/os/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        output='screen'
    )

    ld = LaunchDescription()

    # Add the actions
    ld.add_action(declare_gui_cmd)
    ld.add_action(start_gazebo)
    ld.add_action(start_robot_state_publisher_cmd)
    ld.add_action(spawn_example_cmd)
    ld.add_action(bridge_cmd)
    ld.add_action(start_rviz_cmd)
    ld.add_action(exit_event_handler)

    return ld
