"""
Launch file for AgileX Hunter SE (sensor-free) in Gazebo Ignition Fortress.

Starts:
  1. Gazebo Sim via ros_gz_sim with the selected world (default: drl_arena)
  2. robot_state_publisher  – publishes URDF-based TF
  3. hunter_se_cmd_prefilter – Unity-like throttle / steering shaping
  4. ros_gz_bridge          – bridges filtered cmd_vel / odometry / tf / joint_states
  5. spawn_entity           – spawns hunter_se into Gazebo (5 s after Gazebo starts)
  6. RViz2                  – optional (rviz:=true|false)
  7. hunter_se_teleop_key.py – optional focused teleop window

Supported worlds
─────────────────
  drl_arena   (default) – 15×15m enclosed arena, world name="default". Compatible
                          with ros2_gz_bridge_config.yaml and environment.py.
  hospital              – AWS hospital world, also world name="default".
  <full path>           – Any .world file whose SDF world name is "default".

NOTE: ros2_gz_bridge_config.yaml hardcodes /world/default/..., so only worlds
      whose SDF <world name="default"> are compatible with this launch file.
      In particular, worlds/empty.world (name="empty") is NOT supported here;
      use hunter_se_validation_empty.launch.py for keyboard-drive testing.

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
    OpaqueFunction,
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

    teleop_arg = DeclareLaunchArgument(
        name="teleop",
        default_value="false",
        description="Launch keyboard teleop window. Keep false during DRL training.",
        choices=["true", "false"],
    )

    world_arg = DeclareLaunchArgument(
        name="world",
        default_value="drl_arena",
        description=(
            "World to simulate. Supported names: 'drl_arena' (default), 'hospital'. "
            "Or pass a full path to any .world file with <world name=\"default\">. "
            "NOTE: worlds/empty.world is NOT compatible (uses name=\"empty\"); "
            "use hunter_se_validation_empty.launch.py instead."
        ),
    )

    # ------------------------------------------------------------------ #
    # Package paths
    # ------------------------------------------------------------------ #
    hunter_se_pkg = get_package_share_directory("hunter_se_gazebo")
    hunter_se_resource_root = os.path.dirname(hunter_se_pkg)

    hospital_world_file = os.path.join(
        get_package_share_directory("aws_robomaker_hospital_world"),
        "worlds",
        "hospital_ignition.world",
    )

    # AWS world package share dirs for model:// URI resolution
    try:
        warehouse_share = get_package_share_directory("aws_robomaker_small_warehouse_world")
    except Exception:
        warehouse_share = ""
    try:
        hospital_share = get_package_share_directory("aws_robomaker_hospital_world")
    except Exception:
        hospital_share = ""
    try:
        bookstore_share = get_package_share_directory("aws_robomaker_bookstore_world")
    except Exception:
        bookstore_share = ""
    try:
        house_share = get_package_share_directory("aws_robomaker_small_house_world")
    except Exception:
        house_share = ""

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

    # DrlModelPosePlugin — built into drl_obstacle_assets lib dir
    try:
        drl_obstacle_assets_lib = os.path.join(
            os.path.dirname(get_package_share_directory("drl_obstacle_assets")),
            "..", "lib",
        )
        drl_obstacle_assets_lib = os.path.realpath(drl_obstacle_assets_lib)
    except Exception:
        drl_obstacle_assets_lib = ""

    # Include both package-share root AND models/ sub-directory for each AWS
    # package so that model:// URIs resolve correctly via GZ_SIM_RESOURCE_PATH.
    aws_resource_paths = [p for p in [
        warehouse_share,
        os.path.join(warehouse_share, "models") if warehouse_share else "",
        hospital_share,
        os.path.join(hospital_share, "models") if hospital_share else "",
        os.path.join(hospital_share, "fuel_models") if hospital_share else "",
        bookstore_share,
        os.path.join(bookstore_share, "models") if bookstore_share else "",
        house_share,
        os.path.join(house_share, "models") if house_share else "",
    ] if p]

    gz_env = {
        # Fortress resolves package mesh URIs through Gazebo resource paths.
        # `package://hunter_se_gazebo/...` gets translated to
        # `model://hunter_se_gazebo/...`, so Gazebo must see the parent
        # directory that contains the `hunter_se_gazebo/` folder.
        "GZ_SIM_RESOURCE_PATH": ":".join(filter(None, [
            hunter_se_resource_root,
            hunter_se_pkg,
            *aws_resource_paths,
            environ.get("GZ_SIM_RESOURCE_PATH", ""),
        ])),
        "IGN_GAZEBO_RESOURCE_PATH": ":".join(filter(None, [
            hunter_se_resource_root,
            hunter_se_pkg,
            *aws_resource_paths,
            environ.get("IGN_GAZEBO_RESOURCE_PATH", ""),
        ])),
        "IGN_GAZEBO_SYSTEM_PLUGIN_PATH": ":".join(filter(None, [
            rgl_server_plugin_path,
            drl_obstacle_assets_lib,
            environ.get("IGN_GAZEBO_SYSTEM_PLUGIN_PATH", ""),
            environ.get("LD_LIBRARY_PATH", ""),
        ])),
        "IGN_GUI_PLUGIN_PATH": ":".join(filter(None, [
            rgl_gui_plugin_path,
            environ.get("IGN_GUI_PLUGIN_PATH", ""),
        ])),
        "LD_LIBRARY_PATH": ":".join(filter(None, [
            rgl_server_plugin_path,
            drl_obstacle_assets_lib,
            environ.get("LD_LIBRARY_PATH", ""),
        ])),
    }

    # ------------------------------------------------------------------ #
    # Known world name → file path mapping (only worlds with name="default")
    # ------------------------------------------------------------------ #
    _known_worlds = {
        "drl_arena": os.path.join(hunter_se_pkg, "worlds", "drl_arena.world"),
        "hospital":  hospital_world_file,
    }

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
    # ROS2 ↔ Gazebo bridge (topics)
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

    # ------------------------------------------------------------------ #
    # ROS2 ↔ Gazebo bridge (world services)
    # Bridges Ignition world services as ROS2 services so environment.py
    # can call /world/default/control, /set_pose, /create, /remove.
    # Must use CLI argument syntax; YAML service bridge is not supported
    # in this version of ros_gz_bridge.
    # ------------------------------------------------------------------ #
    service_bridge_node = Node(
        name="ros2_gz_service_bridge",
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/world/default/control@ros_gz_interfaces/srv/ControlWorld",
            "/world/default/set_pose@ros_gz_interfaces/srv/SetEntityPose",
            "/world/default/create@ros_gz_interfaces/srv/SpawnEntity",
            "/world/default/remove@ros_gz_interfaces/srv/DeleteEntity",
        ],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # pointcloud_to_laserscan
    # Converts Ouster OS1-128 PointCloud2 (/ouster/points) to LaserScan
    # (/scan) so the DRL environment node can use obs_source:=scan.
    # Height filter mirrors environment.py cloud callback (z > -0.2 m in
    # sensor frame). Full 360° at ~0.176° resolution matches Ouster 2048
    # samples/rotation.
    # ------------------------------------------------------------------ #
    pointcloud_to_laserscan_node = Node(
        name="pointcloud_to_laserscan",
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            # Scan is referenced to the Ouster frame itself so collision checks
            # are centered at the LiDAR origin, not at the robot body centre.
            "target_frame": "ouster_lidar",
            "transform_tolerance": 0.01,
            # Height filter in the Ouster frame.
            # Sensor origin is about 0.50 m above ground, so this keeps points
            # roughly 0.045–0.750 m above ground:
            #   0.045 - 0.500 = -0.455
            #   0.750 - 0.500 =  0.250
            "min_height": -0.455,
            "max_height": 0.25,
            "angle_min": -3.14159265,  # full 360°
            "angle_max":  3.14159265,
            "angle_increment": 0.00306796,  # ≈ 2π/2048 ≈ 0.176°
            "scan_time": 0.1,          # 10 Hz
            # Match OS1-128 minimum range; farther ranges are still capped to the
            # RL observation horizon below.
            "range_min": 0.3,
            "range_max": 50.0,         # RL scan horizon; sensor itself is configured to 120 m
            "use_inf": True,
        }],
        remappings=[
            ("cloud_in", "/ouster/points"),
            ("scan",     "/scan"),
        ],
    )

    prefilter_node = Node(
        name="hunter_se_cmd_prefilter",
        package="hunter_se_gazebo",
        executable="hunter_se_cmd_prefilter.py",
        output="screen",
        parameters=[
            # Wall-clock time: timer fires at real 50 Hz regardless of sim RTF.
            # If use_sim_time=True and RTF≈0.001, the 50 Hz timer fires only
            # once per ~100 wall-clock steps (≈20 s), so /cmd_vel_filtered is
            # never published and the robot never moves.
            {"use_sim_time": False},
            os.path.join(hunter_se_pkg, "config", "hunter_se_cmd_prefilter.yaml"),
        ],
    )

    # ------------------------------------------------------------------ #
    # Spawn hunter_se
    # spawn z = 0.02 m clearance (base_footprint is model root, wheel
    # bottoms land at z=0 when base_footprint is at z=0)
    # ------------------------------------------------------------------ #
    spawn_node = Node(
        name="spawn_hunter_se",
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name",  "hunter_se",
            "-topic", "/hunter_se/robot_description",
            "-x", "4.5", "-y", "0", "-z", "0.02",
            "-R", "0",   "-P", "0", "-Y", "0",
        ],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # Gazebo Sim via ros_gz_sim (world resolved at launch time; spawn attached
    # to the resulting launch process via OnProcessStart so timing stays robust
    # even for slow-loading worlds or heavy plugin initialisation)
    # ------------------------------------------------------------------ #
    def start_gazebo(context, *args, **kwargs):
        world = LaunchConfiguration("world").perform(context)
        world_file = _known_worlds.get(world, world)  # fallback: treat as file path
        gz_process = ExecuteProcess(
            cmd=[
                "ros2", "launch", "ros_gz_sim", "gz_sim.launch.py",
                f"gz_args:=-r -v 1 {world_file}",
                "on_exit_shutdown:=true",
            ],
            output="screen",
            additional_env=gz_env,
            shell=False,
            on_exit=Shutdown(),
        )
        # Wait for Gazebo to start, then delay 5 s before spawning the robot.
        # This is robust to variable world / plugin load times.
        spawn_after_gazebo = RegisterEventHandler(
            event_handler=OnProcessStart(
                target_action=gz_process,
                on_start=[TimerAction(period=5.0, actions=[spawn_node])],
            )
        )
        return [gz_process, spawn_after_gazebo]

    gazebo_opaque = OpaqueFunction(function=start_gazebo)

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
        condition=IfCondition(LaunchConfiguration("teleop")),
    )

    return LaunchDescription([
        rviz_arg,
        teleop_arg,
        world_arg,
        robot_state_publisher_node,
        gazebo_opaque,
        prefilter_node,
        bridge_node,
        service_bridge_node,
        pointcloud_to_laserscan_node,
        rviz2_node,
        teleop_node,
    ])
