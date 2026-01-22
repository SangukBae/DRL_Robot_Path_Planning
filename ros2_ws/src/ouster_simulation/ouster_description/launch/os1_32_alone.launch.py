import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node
from launch.substitutions import Command, LaunchConfiguration


def generate_launch_description():
    pkg = get_package_share_directory('ouster_description')
    xacro_path = os.path.join(pkg, 'urdf', 'os1_32_example.urdf.xacro')
    world = os.path.join(pkg, 'worlds', 'test.world')
    rviz_config_file = os.path.join(pkg, 'rviz', 'test.rviz')

    use_sim_time = True

    # More standard xacro Command pattern
    robot_description = Command(['xacro', ' ', xacro_path])

    # Robot state publisher (also republishes /robot_description topic by default in many setups)
    start_robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_description,
        }]
    )

    # Spawn robot in Gazebo Sim
    spawn_example_cmd = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'example',
            '-topic', 'robot_description',
        ],
        output='screen',
    )

    # RViz (use_sim_time recommended when bridging /clock)
    start_rviz_cmd = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    exit_event_handler = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=start_rviz_cmd,
            on_exit=EmitEvent(event=Shutdown(reason='rviz exited'))
        )
    )

    # GUI option (actually used)
    declare_gui_cmd = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Launch Gazebo GUI'
    )
    gui = LaunchConfiguration('gui')

    ros_gz_sim_share = get_package_share_directory('ros_gz_sim')
    gz_launch = PythonLaunchDescriptionSource(
        os.path.join(ros_gz_sim_share, 'launch', 'gz_sim.launch.py')
    )

    # Gazebo (GUI)
    start_gazebo_gui = IncludeLaunchDescription(
        gz_launch,
        launch_arguments={
            'gz_args': ['-r ', world],
            'on_exit_shutdown': 'true',
        }.items(),
        condition=IfCondition(gui),
    )

    # Gazebo (headless/server only)
    start_gazebo_headless = IncludeLaunchDescription(
        gz_launch,
        launch_arguments={
            'gz_args': ['-r -s ', world],
            'on_exit_shutdown': 'true',
        }.items(),
        condition=UnlessCondition(gui),
    )

    # Bridge (Ignition topic: /ouster/pointcloud/points -> ROS2: /ouster/points)
    bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/ouster/pointcloud/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
        ],
        remappings=[
            ('/ouster/pointcloud/points', '/ouster/points'),
        ],
        output='screen'
    )

    ld = LaunchDescription()
    ld.add_action(declare_gui_cmd)
    ld.add_action(start_gazebo_gui)
    ld.add_action(start_gazebo_headless)
    ld.add_action(start_robot_state_publisher_cmd)
    ld.add_action(spawn_example_cmd)
    ld.add_action(bridge_cmd)
    ld.add_action(start_rviz_cmd)
    ld.add_action(exit_event_handler)
    return ld
