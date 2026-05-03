"""
Launch file for AgileX Hunter SE (sensor-free) in Gazebo Ignition Fortress.

Starts:
  1. Ignition Gazebo with hospital world
  2. robot_state_publisher  – publishes URDF-based TF
  3. hunter_se_cmd_prefilter – Unity-like throttle / steering shaping
  4. ros_gz_bridge          – bridges filtered cmd_vel / odometry / tf / joint_states
  5. hunter_se_steering_monitor – prints front wheel steering angles
  6. spawn_entity           – spawns hunter_se into Gazebo (delayed 5 s)
  7. RViz2                  – optional (rviz:=true|false)
  8. hunter_se_teleop_key.py – focused teleop window (W/S=fwd/bck, A/D=steer)

Usage:
  ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py
  ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false

NOTE: RGL plugin must be installed at:
  ~/DRL_Robot_Path_Planning/third_party/rgl/RGLGazeboPlugin/install
"""

import os
from os import environ

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
        description="Launch RViz2 with model display configuration",
        choices=["true", "false"],
    )

    # ------------------------------------------------------------------ #
    # Package paths / world file
    # ------------------------------------------------------------------ #
    hunter_se_pkg = get_package_share_directory("hunter_se_gazebo")
    hunter_se_resource_root = os.path.dirname(hunter_se_pkg)

    hospital_world_file = os.path.join(
        get_package_share_directory("aws_robomaker_hospital_world"),
        "worlds",
        "hospital_ignition.world",
    )

    # ------------------------------------------------------------------ #
    # RGL plugin paths (same as Scout v2 setup)
    # ------------------------------------------------------------------ #
    rgl_install_dir = os.path.join(
        os.path.expanduser("~"),
        "DRL_Robot_Path_Planning",
        "third_party", "rgl", "RGLGazeboPlugin", "install",
    )
    rgl_server_plugin_path = os.path.join(rgl_install_dir, "RGLServerPlugin")
    rgl_gui_plugin_path    = os.path.join(rgl_install_dir, "RGLVisualize")

    gz_env = {
        # Fortress resolves package mesh URIs through Gazebo resource paths.
        # `package://hunter_se_gazebo/...` gets translated to
        # `model://hunter_se_gazebo/...`, so Gazebo must see the parent
        # directory that contains the `hunter_se_gazebo/` folder.
        "GZ_SIM_RESOURCE_PATH": ":".join(filter(None, [
            hunter_se_resource_root,
            hunter_se_pkg,
            environ.get("GZ_SIM_RESOURCE_PATH", ""),
        ])),
        "IGN_GAZEBO_RESOURCE_PATH": ":".join(filter(None, [
            hunter_se_resource_root,
            hunter_se_pkg,
            environ.get("IGN_GAZEBO_RESOURCE_PATH", ""),
        ])),
        "IGN_GAZEBO_SYSTEM_PLUGIN_PATH": ":".join(filter(None, [
            rgl_server_plugin_path,
            environ.get("IGN_GAZEBO_SYSTEM_PLUGIN_PATH", ""),
            environ.get("LD_LIBRARY_PATH", ""),
        ])),
        "IGN_GUI_PLUGIN_PATH": ":".join(filter(None, [
            rgl_gui_plugin_path,
            environ.get("IGN_GUI_PLUGIN_PATH", ""),
        ])),
        "LD_LIBRARY_PATH": ":".join(filter(None, [
            rgl_server_plugin_path,
            environ.get("LD_LIBRARY_PATH", ""),
        ])),
    }

    # ------------------------------------------------------------------ #
    # Ignition Gazebo
    # ------------------------------------------------------------------ #
    gazebo_process = ExecuteProcess(
        cmd=["ign", "gazebo", "-v", "1", "-r", hospital_world_file],
        output="screen",
        additional_env=gz_env,
        shell=False,
        on_exit=Shutdown(),
    )

    # ------------------------------------------------------------------ #
    # Robot description
    # ------------------------------------------------------------------ #
    description_file = os.path.join(hunter_se_pkg, "urdf", "robot.urdf.xacro")

    description_content = Command([
        FindExecutable(name="xacro"),
        " ", description_file,
        " load_gazebo:=true",
    ])
    description_param = {
        "robot_description": ParameterValue(description_content, value_type=str)
    }

    # ------------------------------------------------------------------ #
    # robot_state_publisher
    # ------------------------------------------------------------------ #
    robot_state_publisher_node = Node(
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
            "config_file": os.path.join(hunter_se_pkg, "config", "ros2_gz_bridge_config.yaml"),
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
            os.path.join(hunter_se_pkg, "config", "hunter_se_cmd_prefilter.yaml"),
        ],
    )

    steering_monitor_node = Node(
        name="hunter_se_steering_monitor",
        package="hunter_se_gazebo",
        executable="hunter_se_steering_monitor.py",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # ------------------------------------------------------------------ #
    # Spawn hunter_se (delayed until Gazebo world is ready)
    # spawn z = wheel_radius - wheel_vertical_offset = 0.136 + 0.060 = 0.196 m
    # ------------------------------------------------------------------ #
    spawn_node = Node(
        name="spawn_hunter_se",
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name",  "hunter_se",
            "-topic", "/hunter_se/robot_description",
            "-x", "4.5", "-y", "0", "-z", "0.20",
            "-R", "0",   "-P", "0", "-Y", "0",
        ],
        output="screen",
    )
    spawn_after_gazebo = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=gazebo_process,
            on_start=[TimerAction(period=5.0, actions=[spawn_node])],
        )
    )

    # ------------------------------------------------------------------ #
    # RViz2
    # ------------------------------------------------------------------ #
    rviz2_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", os.path.join(hunter_se_pkg, "rviz", "model_display.rviz")],
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
        robot_state_publisher_node,
        gazebo_process,
        spawn_after_gazebo,
        prefilter_node,
        bridge_node,
        steering_monitor_node,
        rviz2_node,
        teleop_node,
    ])
