"""
Hunter SE + HuNav unified launch file for Gazebo Ignition Fortress.

Startup sequence
────────────────
  1. hunav_loader         — reads <agent_config>.yaml, declares agent params
  2. [+2 s] hunav_gazebo_world_generator
                          — injects HuNav actors+plugin into drl_arena.world,
                            writes generatedWorld.sdf alongside the base world
  3. Gazebo               — started independently via ros_gz_sim after a short
                            fixed delay, matching the basic Hunter launch style
  4. [+2 s] spawn_hunter_se — spawns robot after Gazebo is up
  5. Immediately (parallel):
       • robot_state_publisher
       • hunter_se_cmd_prefilter
       • ros_gz_bridge (topics)
       • ros_gz_bridge (world services)
       • pointcloud_to_laserscan
       • hunav_agent_manager  (BehaviorTree-driven pedestrian nav)

HuNav parameters wired for Hunter SE
──────────────────────────────────────
  robot_name              hunter_se
  global_frame_to_publish map
  use_navgoal_to_start    false
  ignore_models           ground_plane sun

Note: a static map→odom TF is published here so HuNav/global-frame consumers
and CSV logging can resolve the LiDAR tree into the map frame.
"""

import os
import tempfile
from os import environ

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
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
    launch_dir = os.path.dirname(os.path.abspath(__file__))

    # ------------------------------------------------------------------ #
    # Launch arguments
    # ------------------------------------------------------------------ #
    rviz_arg = DeclareLaunchArgument(
        name="rviz",
        default_value="true",
        description="Launch RViz2",
        choices=["true", "false"],
    )
    agent_conf_arg = DeclareLaunchArgument(
        name="agent_config",
        default_value="drl_arena_agents.yaml",
        description="Agent scenario YAML in hunav_gazebo_fortress_wrapper/scenarios/",
    )
    csv_log_arg = DeclareLaunchArgument(
        name="log_hunav_pose_csv",
        default_value="false",
        description="Log HuNav actor vs point cloud positions to CSV",
        choices=["true", "false"],
    )
    csv_path_arg = DeclareLaunchArgument(
        name="hunav_pose_csv_path",
        default_value=os.path.join(launch_dir, "hunter_se_hunav_pose_log.csv"),
        description="CSV output path for HuNav actor vs point cloud pose logging",
    )

    # ------------------------------------------------------------------ #
    # Package paths (resolved at description-generation time)
    # ------------------------------------------------------------------ #
    hunter_se_pkg   = get_package_share_directory("hunter_se_gazebo")
    hunav_wrap_pkg  = get_package_share_directory("hunav_gazebo_fortress_wrapper")
    hunter_se_resource_root = os.path.dirname(hunter_se_pkg)

    # Base world — drl_arena.world lives in hunter_se_gazebo
    drl_arena_world = os.path.join(hunter_se_pkg, "worlds", "drl_arena.world")
    # Write the generated world into a writable temp directory instead of
    # install/share, which is often read-only in sourced workspaces.
    generated_world_dir = os.path.join(tempfile.gettempdir(), "hunter_se_hunav")
    os.makedirs(generated_world_dir, exist_ok=True)
    generated_world = os.path.join(generated_world_dir, "generatedWorld.sdf")

    # AWS world resources (for GZ_SIM_RESOURCE_PATH, unchanged from original launch)
    try:
        hospital_share  = get_package_share_directory("aws_robomaker_hospital_world")
    except Exception:
        hospital_share  = ""

    # ------------------------------------------------------------------ #
    # RGL plugin paths
    # ------------------------------------------------------------------ #
    rgl_install_dir = os.path.join(
        os.path.expanduser("~"),
        "DRL_Robot_Path_Planning",
        "third_party", "rgl", "RGLGazeboPlugin", "install",
    )
    rgl_server_plugin_path = os.path.join(rgl_install_dir, "RGLServerPlugin")
    rgl_gui_plugin_path    = os.path.join(rgl_install_dir, "RGLVisualize")

    # Hunav worlds directory (provides actor skin meshes)
    hunav_worlds_dir = os.path.join(hunav_wrap_pkg, "worlds")

    # HuNavSystemPluginIGN .so is installed to <prefix>/lib/ by colcon.
    # Derive the lib path from the share path (share/../lib).
    hunav_lib_dir = os.path.join(os.path.dirname(hunav_wrap_pkg), "..", "lib")

    gz_env = {
        "GZ_SIM_RESOURCE_PATH": ":".join(filter(None, [
            hunter_se_resource_root,
            hunter_se_pkg,
            hunav_worlds_dir,
            hospital_share,
            environ.get("GZ_SIM_RESOURCE_PATH", ""),
        ])),
        "IGN_GAZEBO_RESOURCE_PATH": ":".join(filter(None, [
            hunter_se_resource_root,
            hunter_se_pkg,
            hunav_worlds_dir,
            hospital_share,
            environ.get("IGN_GAZEBO_RESOURCE_PATH", ""),
        ])),
        "IGN_GAZEBO_SYSTEM_PLUGIN_PATH": ":".join(filter(None, [
            rgl_server_plugin_path,
            hunav_lib_dir,
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
        "GAZEBO_RESOURCE_PATH": ":".join(filter(None, [
            hunav_worlds_dir,
            environ.get("GAZEBO_RESOURCE_PATH", ""),
        ])),
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
    # Spawn node (created early; referenced by Gazebo OnProcessStart handler)
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
    # Gazebo launcher (same style as the basic Hunter launch)
    # ------------------------------------------------------------------ #
    def start_gazebo_with_generated_world(context, *args, **kwargs):
        gz_process = ExecuteProcess(
            cmd=[
                "ros2", "launch", "ros_gz_sim", "gz_sim.launch.py",
                f"gz_args:=-r -v 1 {generated_world}",
                "on_exit_shutdown:=true",
            ],
            output="screen",
            additional_env=gz_env,
            shell=False,
            on_exit=Shutdown(),
        )
        # Spawn robot shortly after Gazebo starts. HuNav's world plugin waits
        # for the robot entity, so keeping this delay small avoids noisy retry
        # logs while still giving Gazebo time to advertise services.
        spawn_after_gz = RegisterEventHandler(
            event_handler=OnProcessStart(
                target_action=gz_process,
                on_start=[TimerAction(period=2.0, actions=[spawn_node])],
            )
        )
        return [gz_process, spawn_after_gz]

    gazebo_opaque = OpaqueFunction(function=start_gazebo_with_generated_world)

    # ------------------------------------------------------------------ #
    # HuNav pipeline — resolved inside OpaqueFunction so agent_config arg
    # can be evaluated at launch time (not description-generation time).
    # ------------------------------------------------------------------ #
    def hunav_pipeline(context, *args, **kwargs):
        agent_config = LaunchConfiguration("agent_config").perform(context)
        agent_conf_file = os.path.join(hunav_wrap_pkg, "scenarios", agent_config)

        # 1. HuNav loader — reads <agent_config>.yaml
        hunav_loader_node = Node(
            package="hunav_agent_manager",
            executable="hunav_loader",
            output="screen",
            parameters=[agent_conf_file],
        )

        # 2. HuNav world generator — injects actors + plugin into drl_arena.world
        hunav_worldgen_node = Node(
            package="hunav_gazebo_fortress_wrapper",
            executable="hunav_gazebo_world_generator",
            output="screen",
            parameters=[{
                "base_world":              drl_arena_world,
                "output_world":            generated_world,
                "use_gazebo_obs":          True,
                "update_rate":             100.0,
                "robot_name":              "hunter_se",
                "global_frame_to_publish": "map",
                "use_navgoal_to_start":    False,
                "ignore_models":           "ground_plane sun",
                "plugin_position":         0,
            }],
        )

        # 3. Start worldgen 2 s after loader (loader must declare params first)
        worldgen_after_loader = RegisterEventHandler(
            OnProcessStart(
                target_action=hunav_loader_node,
                on_start=[
                    LogInfo(msg="HuNavLoader ready — starting world generator in 2 s ..."),
                    TimerAction(period=2.0, actions=[hunav_worldgen_node]),
                ],
            )
        )

        # 4. Start Gazebo independently a short time after loader start.
        #    World generation is quick (<1 s in practice) and the generator
        #    stays alive to serve /get_agents, so we avoid relying on process
        #    exit / stdout event semantics here.
        gazebo_after_loader = RegisterEventHandler(
            OnProcessStart(
                target_action=hunav_loader_node,
                on_start=[
                    TimerAction(
                        period=4.0,
                        actions=[
                            LogInfo(msg="Starting Gazebo with generated HuNav world ..."),
                            gazebo_opaque,
                        ],
                    ),
                ],
            )
        )

        return [
            hunav_loader_node,
            worldgen_after_loader,
            gazebo_after_loader,
        ]

    # ------------------------------------------------------------------ #
    # Fixed nodes (independent of agent_config)
    # ------------------------------------------------------------------ #

    # robot_state_publisher
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

    # Static map -> odom TF so map-frame HuNav states and odom-frame robot
    # sensors live in the same TF tree.
    map_to_odom_tf_node = Node(
        name="map_to_odom_static_tf",
        package="tf2_ros",
        executable="static_transform_publisher",
        output="screen",
        arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
        parameters=[{"use_sim_time": True}],
    )

    # cmd_vel prefilter
    prefilter_node = Node(
        name="hunter_se_cmd_prefilter",
        package="hunter_se_gazebo",
        executable="hunter_se_cmd_prefilter.py",
        output="screen",
        parameters=[
            {"use_sim_time": False},
            os.path.join(hunter_se_pkg, "config", "hunter_se_cmd_prefilter.yaml"),
        ],
    )

    # ROS2 ↔ Gazebo bridge (topics)
    bridge_node = Node(
        name="ros2_gz_bridge",
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{
            "config_file": os.path.join(
                hunter_se_pkg, "config", "ros2_gz_bridge_config.yaml"
            ),
            "qos_overrides./tf_static.publisher.durability": "transient_local",
        }],
        output="screen",
    )

    # ROS2 ↔ Gazebo bridge (world services)
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

    # pointcloud_to_laserscan
    pointcloud_to_laserscan_node = Node(
        name="pointcloud_to_laserscan",
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        output="screen",
        parameters=[{
            "use_sim_time":      True,
            "target_frame":      "ouster_lidar",
            "transform_tolerance": 0.01,
            "min_height":        -0.455,
            "max_height":         0.25,
            "angle_min":         -3.14159265,
            "angle_max":          3.14159265,
            "angle_increment":    0.00306796,
            "scan_time":          0.1,
            "range_min":          0.3,
            "range_max":         50.0,
            "use_inf":            True,
        }],
        remappings=[
            ("cloud_in", "/ouster/points"),
            ("scan",     "/scan"),
        ],
    )

    # HuNav agent manager (BehaviorTree pedestrian controller)
    hunav_manager_node = Node(
        package="hunav_agent_manager",
        executable="hunav_agent_manager",
        name="hunav_agent_manager",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # RViz2 (optional)
    rviz2_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", os.path.join(hunter_se_pkg, "rviz", "model_display.rviz")],
        parameters=[{"use_sim_time": True}, description_param],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    # CSV logger for comparing HuNav actor positions vs point cloud centroids
    hunav_pose_logger_node = Node(
        package="hunter_se_gazebo",
        executable="hunav_pose_csv_logger.py",
        output="screen",
        condition=IfCondition(LaunchConfiguration("log_hunav_pose_csv")),
        parameters=[{
            "use_sim_time": True,
            "csv_path": LaunchConfiguration("hunav_pose_csv_path"),
            "human_states_topic": "/human_states",
            "pointcloud_topic": "/ouster/points",
            "target_frame": "map",
            "agent_names": ["agent1"],
            "cluster_xy_radius": 0.55,
            "min_height_above_agent_base": 0.05,
            "max_height_above_agent_base": 1.60,
            "torso_min_height_above_agent_base": 0.35,
            "max_points_per_agent": 192,
            "min_cluster_points": 5,
            "xy_center_correction_radius": 0.30,
        }],
    )

    return LaunchDescription([
        rviz_arg,
        agent_conf_arg,
        csv_log_arg,
        csv_path_arg,
        # HuNav world generation pipeline (agent_config resolved at launch time)
        OpaqueFunction(function=hunav_pipeline),
        # Hunter SE stack (start immediately; bridge/sensor wait for Gazebo topics)
        robot_state_publisher_node,
        map_to_odom_tf_node,
        prefilter_node,
        bridge_node,
        service_bridge_node,
        pointcloud_to_laserscan_node,
        # HuNav manager
        hunav_manager_node,
        hunav_pose_logger_node,
        # Visualisation
        rviz2_node,
    ])
