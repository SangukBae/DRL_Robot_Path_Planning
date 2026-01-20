# python imports
import os
from ament_index_python.packages import get_package_share_directory
from math import pi

# ros2 imports
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch.substitutions import (
    LaunchConfiguration,
    PythonExpression,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # ============================================================
    # Gazebo resource paths 설정 (scout_description meshes 찾기)
    # ============================================================
    scout_description_prefix = get_package_share_directory("scout_description")

    # scout_description_prefix: .../install/scout_description/share/scout_description
    # Gazebo는 상위 share 디렉토리까지 필요
    scout_resource_path = os.path.join(scout_description_prefix, "..")
    scout_resource_path = os.path.abspath(scout_resource_path)

    # GZ_SIM_RESOURCE_PATH 설정 (Gazebo Garden/Harmonic)
    if "GZ_SIM_RESOURCE_PATH" in os.environ:
        gz_paths = os.environ["GZ_SIM_RESOURCE_PATH"].split(":")
        if scout_resource_path not in gz_paths:
            os.environ["GZ_SIM_RESOURCE_PATH"] = (
                scout_resource_path + ":" + os.environ["GZ_SIM_RESOURCE_PATH"]
            )
    else:
        os.environ["GZ_SIM_RESOURCE_PATH"] = scout_resource_path

    # IGN_GAZEBO_RESOURCE_PATH 설정 (구버전 Ignition 호환용)
    if "IGN_GAZEBO_RESOURCE_PATH" in os.environ:
        ign_paths = os.environ["IGN_GAZEBO_RESOURCE_PATH"].split(":")
        if scout_resource_path not in ign_paths:
            os.environ["IGN_GAZEBO_RESOURCE_PATH"] = (
                scout_resource_path + ":" + os.environ["IGN_GAZEBO_RESOURCE_PATH"]
            )
    else:
        os.environ["IGN_GAZEBO_RESOURCE_PATH"] = scout_resource_path

    # GZ_SIM_SYSTEM_PLUGIN_PATH 설정
    os.environ["GZ_SIM_SYSTEM_PLUGIN_PATH"] = ":".join(
        [
            os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH", default=""),
            os.environ.get("LD_LIBRARY_PATH", default=""),
        ]
    )

    # ============================================================
    # Launch configuration variables
    # ============================================================
    # odometry source (지금은 실제로 사용되진 않지만 원본 구조 유지)
    odometry_source_arg = DeclareLaunchArgument(
        name="odometry_source",
        default_value="ground_truth",
        description="Odometry source (ground_truth or wheel encoders)",
        choices=["encoders", "ground_truth"],
    )

    # RViz 실행 여부
    rviz_arg = DeclareLaunchArgument(
        name="rviz",
        default_value="true",
        description="Open RViz with model display configuration",
        choices=["true", "false"],
    )

    # LiDAR 타입 선택 (3D pointcloud → laserscan, 또는 2D lidar)
    lidar_type_arg = DeclareLaunchArgument(
        name="lidar_type",
        default_value="3d",
        description="choose lidar type: pointcloud with 3d lidar or laserscan with 2d lidar",
        choices=["3d", "2d"],
    )

    # ============================================================
    # Gazebo world: AWS small warehouse
    # ============================================================
    aws_small_warehouse_dir = get_package_share_directory(
        "aws_robomaker_small_warehouse_world"
    )
    warehouse_world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                aws_small_warehouse_dir,
                "/launch/no_roof_small_warehouse.launch.py",
            ]
        )
    )

    # ============================================================
    # ROS <-> Gazebo bridge 설정 (Scout Mini용)
    # ============================================================
    ros2_gz_bridge_file = os.path.join(
        get_package_share_directory("agilex_scout"), # agilex_scout , scout_gazebo_sim
        "config",
        "scout_mini_bridge_config.yaml", # scout_mini_bridge_config , scout_mini_bridge_ros_gz
    )

    bridge = Node(
        name="ros2_gz_bridge",
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[
            {
                "config_file": ros2_gz_bridge_file,
                "qos_overrides./tf_static.publisher.durability": "transient_local",
            }
        ],
        output="screen",
    )

    # ============================================================
    # Scout Mini robot_state_publisher & spawn (scout_gazebo_sim 방식)
    # ============================================================
    pkg_name = "scout_gazebo_sim"
    namespace = "scout_mini"
    launch_file_dir = os.path.join(get_package_share_directory(pkg_name), "launch")

    # scout_mini_robot_state_publisher.launch.py include
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, "scout_mini_robot_state_publisher.launch.py")
        ),
        launch_arguments={
            "use_sim_time": "true",
            "namespace": namespace,
        }.items(),
    )

    # spawn_scout_mini.launch.py include
    robot_spawn_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, "spawn_scout_mini.launch.py")
        ),
        launch_arguments={
            "namespace": namespace,
            "x_pose": "0.0",
            "y_pose": "0.0",
            "yaw_pose": "0.0",  # 필요하면 3.14 등으로 변경 가능
        }.items(),
    )

    # ============================================================
    # RViz2
    # ============================================================
    rviz2_file = os.path.join(
        get_package_share_directory("scout_description"),
        "rviz",
        "scout_mini.rviz",
    )

    rviz2_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz2_file],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    # ============================================================
    # static transform: world -> map
    # ============================================================
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x",
            "0.0",
            "--y",
            "0.0",
            "--z",
            "0.0",
            "--yaw",
            "0.0",
            "--pitch",
            "0.0",
            "--roll",
            "0.0",
            "--frame-id",
            "world",
            "--child-frame-id",
            "map",
        ],
        parameters=[{"use_sim_time": True}],
    )

    # ============================================================
    # static transform: map -> odom
    # ============================================================
    # static transform from map to odom (identity)
    static_tf_map_odom = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", "0.0",
            "--y", "0.0",
            "--z", "0.0",
            "--yaw", "0.0",
            "--pitch", "0.0",
            "--roll", "0.0",
            "--frame-id", "map",
            "--child-frame-id", "odom",
        ],
        parameters=[{"use_sim_time": True}],
    )

    # ============================================================
    # Teleop (키보드 제어)
    # ============================================================
    teleop_keyboard_node = Node(
        name="teleop",
        package="teleop_twist_keyboard",
        executable="teleop_twist_keyboard",
        output="screen",
        prefix="xterm -e",
    )

    # ============================================================
    # pointcloud_to_laserscan (3D LiDAR → 2D LaserScan)
    # ============================================================
    pointcloud_to_laserscan_node = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan_node",
        remappings=[
            ("cloud_in", "/points"),
            ("scan", "/laser_scan"),
        ],
        parameters=[
            {
                "transform_tolerance": 0.05,
                "min_height": 0.0,
                "max_height": 1.0,
                "angle_min": -pi,
                "angle_max": pi,
                "angle_increment": pi / 180.0 / 2.0,  # 0.5 deg
                "scan_time": 1 / 10,  # 10Hz
                "range_min": 0.1,
                "range_max": 100.0,
                "use_inf": True,
            }
        ],
        condition=IfCondition(
            PythonExpression(
                ["'", LaunchConfiguration("lidar_type"), "'", " == '3d'"]
            )
        ),
    )

    # ============================================================
    # LaunchDescription 반환
    # ============================================================
    return LaunchDescription(
        [
            odometry_source_arg,
            rviz_arg,
            lidar_type_arg,
            static_tf,
            static_tf_map_odom,
            warehouse_world_launch,
            robot_state_publisher_cmd,
            robot_spawn_cmd,
            bridge,
            rviz2_node,
            teleop_keyboard_node,
            pointcloud_to_laserscan_node,
        ]
    )
