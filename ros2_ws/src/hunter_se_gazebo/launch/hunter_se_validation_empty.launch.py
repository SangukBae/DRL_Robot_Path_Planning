"""
Hunter SE validation launch for an empty world.

Purpose:
  Spawn the robot in a simple map and record the key drive diagnostics to CSV
  so steering, speed, and wheel slip can be checked after the run.

Starts:
  1. Ignition Gazebo empty world
  2. robot_state_publisher
  3. hunter_se_cmd_prefilter
  4. ros_gz_bridge
  5. spawn hunter_se
  6. hunter_se_teleop_key.py
  7. hunter_se_validation_logger.py
  8. hunter_se_steering_monitor.py   (optional terminal monitor)
  9. hunter_se_dynamics_monitor.py   (optional terminal monitor)
  10. RViz2 (optional)

Usage:
  ros2 launch hunter_se_gazebo hunter_se_validation_empty.launch.py
  ros2 launch hunter_se_gazebo hunter_se_validation_empty.launch.py rviz:=false
  ros2 launch hunter_se_gazebo hunter_se_validation_empty.launch.py drive_mode:=manual
  ros2 launch hunter_se_gazebo hunter_se_validation_empty.launch.py print_monitors:=true
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
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PythonExpression,
)


def generate_launch_description():
    rviz_arg = DeclareLaunchArgument(
        name="rviz",
        default_value="false",
        description="Launch RViz2",
        choices=["true", "false"],
    )
    drive_mode_arg = DeclareLaunchArgument(
        name="drive_mode",
        default_value="auto",
        description="auto = scripted validation drive, manual = keyboard teleop",
        choices=["auto", "manual"],
    )
    print_monitors_arg = DeclareLaunchArgument(
        name="print_monitors",
        default_value="false",
        description="Print steering and dynamics monitors to the terminal",
        choices=["true", "false"],
    )

    pkg = get_package_share_directory("hunter_se_gazebo")
    resource_root = os.path.dirname(pkg)
    world_file = os.path.join(pkg, "worlds", "empty.world")
    repo_root = os.path.abspath(os.path.join(pkg, "..", "..", "..", ".."))
    validation_output_dir = os.path.join(repo_root, "hunter_se_validation_logs")

    resource_env = {
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
    }

    gazebo_process = ExecuteProcess(
        cmd=["ign", "gazebo", "-v", "1", "-r", world_file],
        output="screen",
        additional_env=resource_env,
        shell=False,
        on_exit=Shutdown(),
    )

    description_content = Command([
        FindExecutable(name="xacro"),
        " ", os.path.join(pkg, "urdf", "robot.urdf.xacro"),
        " load_gazebo:=true",
    ])
    description_param = {
        "robot_description": ParameterValue(description_content, value_type=str)
    }

    rsp_node = Node(
        name="robot_state_publisher",
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True}, description_param],
        remappings=[
            ("/joint_states", "/hunter_se/joint_states"),
            ("/robot_description", "/hunter_se/robot_description"),
        ],
    )

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

    validation_logger_node = Node(
        name="hunter_se_validation_logger",
        package="hunter_se_gazebo",
        executable="hunter_se_validation_logger.py",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "output_dir": validation_output_dir,
        }],
    )

    steering_monitor_node = Node(
        name="hunter_se_steering_monitor",
        package="hunter_se_gazebo",
        executable="hunter_se_steering_monitor.py",
        output="screen",
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("print_monitors")),
    )

    dynamics_monitor_node = Node(
        name="hunter_se_dynamics_monitor",
        package="hunter_se_gazebo",
        executable="hunter_se_dynamics_monitor.py",
        output="screen",
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("print_monitors")),
    )

    spawn_node = Node(
        name="spawn_hunter_se",
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "hunter_se",
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

    rviz2_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", os.path.join(pkg, "rviz", "model_display.rviz")],
        parameters=[{"use_sim_time": True}, description_param],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    teleop_node = Node(
        name="teleop",
        package="hunter_se_gazebo",
        executable="hunter_se_teleop_key.py",
        output="screen",
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration("drive_mode"), "' == 'manual'"])
        ),
    )

    validation_driver_node = Node(
        name="hunter_se_validation_driver",
        package="hunter_se_gazebo",
        executable="hunter_se_validation_driver.py",
        output="screen",
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration("drive_mode"), "' == 'auto'"])
        ),
    )

    shutdown_after_validation = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=validation_driver_node,
            on_exit=[Shutdown(reason="Hunter SE validation sequence completed")],
        )
    )

    return LaunchDescription([
        rviz_arg,
        drive_mode_arg,
        print_monitors_arg,
        rsp_node,
        gazebo_process,
        spawn_after_gazebo,
        prefilter_node,
        bridge_node,
        validation_logger_node,
        steering_monitor_node,
        dynamics_monitor_node,
        rviz2_node,
        teleop_node,
        validation_driver_node,
        shutdown_after_validation,
    ])
