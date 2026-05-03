"""
Hunter SE – empty world keyboard teleop launch file.

Starts:
  1. Ignition Gazebo   – empty world (no external plugins required)
  2. robot_state_publisher
  3. hunter_se_cmd_prefilter – Unity-like throttle / steering shaping
  4. ros_gz_bridge     – filtered cmd_vel / odometry / tf / joint_states / clock
  5. spawn hunter_se   – at origin, delayed 3 s after Gazebo starts
  6. RViz2             – optional (rviz:=true|false, default true)
  7. hunter_se_teleop_key.py – focused teleop window (W/S=fwd/bck, A/D=steer)

Usage:
  ros2 launch hunter_se_gazebo keyboard_teleop_empty.launch.py
  ros2 launch hunter_se_gazebo keyboard_teleop_empty.launch.py rviz:=false
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    RegisterEventHandler,
    Shutdown,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
)


def generate_launch_description():

    # ------------------------------------------------------------------ #
    # Launch arguments
    # ------------------------------------------------------------------ #
    rviz_arg = DeclareLaunchArgument(
        name="rviz",
        default_value="true",
        description="Launch RViz2",
        choices=["true", "false"],
    )

    # ------------------------------------------------------------------ #
    # Package paths
    # ------------------------------------------------------------------ #
    pkg = get_package_share_directory("hunter_se_gazebo")
    resource_root = os.path.dirname(pkg)
    world_file = os.path.join(pkg, "worlds", "empty.world")

    # ------------------------------------------------------------------ #
    # Ignition Gazebo (empty world – no RGL plugin needed)
    # ------------------------------------------------------------------ #
    gazebo_process = ExecuteProcess(
        cmd=["ign", "gazebo", "-v", "1", "-r", world_file],
        output="screen",
        additional_env={
            "GZ_SIM_RESOURCE_PATH": ":".join(filter(None, [
                resource_root,
                pkg,
                os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
            ])),
            "IGN_GAZEBO_RESOURCE_PATH": ":".join(filter(None, [
                resource_root,
                pkg,
                os.environ.get("IGN_GAZEBO_RESOURCE_PATH", ""),
            ])),
        },
        shell=False,
        on_exit=Shutdown(),
    )

    # ------------------------------------------------------------------ #
    # Robot description (XACRO → URDF)
    # ------------------------------------------------------------------ #
    description_content = Command([
        FindExecutable(name="xacro"),
        " ", os.path.join(pkg, "urdf", "robot.urdf.xacro"),
        " load_gazebo:=true",
    ])
    description_param = {
        "robot_description": ParameterValue(description_content, value_type=str)
    }

    # ------------------------------------------------------------------ #
    # robot_state_publisher
    # ------------------------------------------------------------------ #
    rsp_node = Node(
        name="robot_state_publisher",
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True}, description_param],
        remappings=[
            ("/joint_states",      "/hunter_se/joint_states"),
            ("/robot_description", "/hunter_se/robot_description"),
        ],
    )

    # ------------------------------------------------------------------ #
    # ROS2 ↔ Gazebo bridge
    # ------------------------------------------------------------------ #
    bridge_node = Node(
        name="ros2_gz_bridge",
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{
            "config_file": os.path.join(pkg, "config", "ros2_gz_bridge_empty_config.yaml"),
            "qos_overrides./tf_static.publisher.durability": "transient_local",
        }],
        output="screen",
    )

    prefilter_node = Node(
        name="hunter_se_cmd_prefilter",
        package="hunter_se_gazebo",
        executable="hunter_se_cmd_prefilter.py",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            os.path.join(pkg, "config", "hunter_se_cmd_prefilter.yaml"),
        ],
    )

    # ------------------------------------------------------------------ #
    # Spawn hunter_se at origin (delayed 3 s – empty world loads fast)
    # z = 0.20 m keeps wheels clear of ground during spawn
    # ------------------------------------------------------------------ #
    spawn_node = Node(
        name="spawn_hunter_se",
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name",  "hunter_se",
            "-topic", "/hunter_se/robot_description",
            "-x", "0", "-y", "0", "-z", "0.20",
            "-R", "0", "-P", "0", "-Y", "0",
        ],
        output="screen",
    )
    spawn_after_gazebo = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=gazebo_process,
            on_start=[TimerAction(period=3.0, actions=[spawn_node])],
        )
    )

    # ------------------------------------------------------------------ #
    # RViz2
    # ------------------------------------------------------------------ #
    rviz2_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", os.path.join(pkg, "rviz", "model_display.rviz")],
        parameters=[{"use_sim_time": True}, description_param],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    # ------------------------------------------------------------------ #
    # Teleop keyboard (xterm window)
    # Custom Ackermann teleop: 'j'/'l' adjust steering while preserving speed,
    # and 'i'/',' set speed while auto-straightening the steering.
    # ------------------------------------------------------------------ #
    teleop_node = Node(
        name="teleop",
        package="hunter_se_gazebo",
        executable="hunter_se_teleop_key.py",
        output="screen",
    )

    return LaunchDescription([
        rviz_arg,
        rsp_node,
        gazebo_process,
        spawn_after_gazebo,
        prefilter_node,
        bridge_node,
        rviz2_node,
        teleop_node,
    ])
