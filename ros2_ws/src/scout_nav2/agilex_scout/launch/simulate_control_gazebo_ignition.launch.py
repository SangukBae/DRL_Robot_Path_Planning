# python imports
import os
from os import environ
from ament_index_python.packages import get_package_share_directory
from math import pi

# ros2 imports
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, Shutdown, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import (
	Command,
	FindExecutable,
	LaunchConfiguration,
	PythonExpression,
)


def generate_launch_description():
	# Launch configuration variables specific to simulation

	# where to get odometry information from
	# NOTE: odometry source wheel encoders doesn't work for a skid steering kinematics robot yet
	odometry_source_arg = DeclareLaunchArgument(
		name="odometry_source",
		default_value="ground_truth",
		description="Odometry source (ground_truth or wheel encoders)",
		choices=["encoders", "ground_truth"],
	)

	# whether to launch rviz with this launch file or not
	rviz_arg = DeclareLaunchArgument(
		name="rviz",
		default_value="true",
		description="Open RViz with model display configuration",
		choices=["true", "false"],
	)

	lidar_type_arg = DeclareLaunchArgument(
		name="lidar_type",
		default_value="3d",
		description="choose lidar type: pointcloud with 3d lidar or laserscan with 2d lidar",
		choices=["3d", "2d"]
	)

	# Launch Ignition Gazebo with hospital world directly (avoids IncludeLaunchDescription
	# which prevents RegisterEventHandler from referencing the Gazebo process)
	hospital_world_dir = get_package_share_directory(
		"aws_robomaker_hospital_world"
	)
	hospital_world_file = os.path.join(
		hospital_world_dir, "worlds", "hospital_ignition.world"
	)

	# RGL (Robotec GPU Lidar) plugin paths
	rgl_install_dir = os.path.join(
		os.path.expanduser("~"), "DRL_Robot_Path_Planning",
		"third_party", "rgl", "RGLGazeboPlugin", "install"
	)
	rgl_server_plugin_path = os.path.join(rgl_install_dir, "RGLServerPlugin")
	rgl_gui_plugin_path = os.path.join(rgl_install_dir, "RGLVisualize")

	gz_env = {
		# Server plugins: RGLServerPluginManager, RGLServerPluginInstance
		"IGN_GAZEBO_SYSTEM_PLUGIN_PATH": ":".join(filter(None, [
			rgl_server_plugin_path,
			environ.get("IGN_GAZEBO_SYSTEM_PLUGIN_PATH", ""),
			environ.get("LD_LIBRARY_PATH", ""),
		])),
		# GUI plugin: RGLVisualize (point cloud visualization in Gazebo GUI)
		"IGN_GUI_PLUGIN_PATH": ":".join(filter(None, [
			rgl_gui_plugin_path,
			environ.get("IGN_GUI_PLUGIN_PATH", ""),
		])),
		# libRobotecGPULidar.so dependency resolution
		"LD_LIBRARY_PATH": ":".join(filter(None, [
			rgl_server_plugin_path,
			environ.get("LD_LIBRARY_PATH", ""),
		])),
	}
	hospital_world_launch = ExecuteProcess(
		cmd=["ign", "gazebo", "-v", "1", "-r", hospital_world_file],
		output="screen",
		additional_env=gz_env,
		shell=False,
		on_exit=Shutdown(),
	)

	# bridge configuration file
	ros2_gz_bridge_file = os.path.join(
		get_package_share_directory("agilex_scout"),
		"config",
		"ros2_gz_bridge_config.yaml",
	)

	# bridge between ROS2 and Gazebo topics (utility service)
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

	# Scout robot description XACRO + gazebo definitions
	scout_description_file = os.path.join(
		get_package_share_directory("agilex_scout"),
		"urdf",
		"robot.urdf.xacro"
	)
	scout_description_content = Command(
		[
			FindExecutable(name="xacro"),
			" ",
			scout_description_file,
			" odometry_source:=", LaunchConfiguration("odometry_source"),
			" load_gazebo:=true",
			" simulation:=true",
			" lidar_type:=", LaunchConfiguration("lidar_type")
		]
	)
	scout_description = {
		"robot_description": ParameterValue(scout_description_content, value_type=str)
	}

	# robot state publisher node
	robot_state_publisher_node = Node(
		name="robot_state_publisher",
		package="robot_state_publisher",
		executable="robot_state_publisher",
		output="screen",
		parameters=[{"use_sim_time": True}, scout_description],
		# arguments=[scout_description_file],
		remappings=[
			("/joint_states", "/scout/joint_states"),
			("/robot_description", "/scout/robot_description"),
		],
	)

	# spawn Scout robot after Gazebo world is ready
	# TimerAction delays spawn until Gazebo has finished loading the world
	spawn_robot_urdf_node = Node(
		name="spawn_robot_urdf",
		package="ros_gz_sim",
		executable="create",
		arguments=[
			"-name",
			"scout_v2",
			"-topic",
			"/scout/robot_description",
			"-x",
			"4.5",
			"-y",
			"0",
			"-z",
			"0.2346",
			"-R",
			"0",
			"-P",
			"0",
			"-Y",
			"0",
		],
		output="screen",
	)
	spawn_robot_after_gazebo = RegisterEventHandler(
		event_handler=OnProcessStart(
			target_action=hospital_world_launch,
			on_start=[
				TimerAction(
					period=5.0,
					actions=[spawn_robot_urdf_node],
				)
			],
		)
	)

	rviz2_file = os.path.join(
		get_package_share_directory("agilex_scout"),
		"rviz",
		"model_display.rviz",
	)

	rviz2_node = Node(
		package="rviz2",
		executable="rviz2",
		arguments=["-d", rviz2_file],
		parameters=[{"use_sim_time": True}, scout_description],
		condition=IfCondition(LaunchConfiguration("rviz")),
	)

	# static transform from world to map
	# NOTE: Disabled for LIO-SAM - creates disconnected TF tree
	# LIO-SAM's odom frame serves as the world reference
	# Uncomment if you need world->map for other purposes
	# static_tf = Node(
	# 	package="tf2_ros",
	# 	executable="static_transform_publisher",
	# 	arguments=[
	# 		"--x",
	# 		"0.0",
	# 		"--y",
	# 		"0.0",
	# 		"--z",
	# 		"0.0",
	# 		"--yaw",
	# 		"0.0",
	# 		"--pitch",
	# 		"0.0",
	# 		"--roll",
	# 		"0.0",
	# 		"--frame-id",
	# 		"world",
	# 		"--child-frame-id",
	# 		"map",
	# 	],
	# 	parameters=[{"use_sim_time": True}]
	# )

	# simulate robot remote control
	teleop_keyboard_node = Node(
		name="teleop",
		package="teleop_twist_keyboard",
		executable="teleop_twist_keyboard",
		output="screen",
		prefix="xterm -e",
	)

	pointcloud_to_laserscan_node = Node(
		package='pointcloud_to_laserscan',
		executable='pointcloud_to_laserscan_node',
		name='pointcloud_to_laserscan_node',
		remappings=[('cloud_in', "/points"),
					('scan', "/laser_scan")],
		parameters=[{
			'transform_tolerance': 0.05,
			'min_height': 0.0,
			'max_height': 1.0,
			'angle_min': -pi,
			'angle_max': pi,
			'angle_increment': pi / 180.0 / 2.0,
			'scan_time': 1/10, # 10Hz
			'range_min': 0.1,
			'range_max': 100.0,
			'use_inf': True,
		}],
		condition=IfCondition(PythonExpression(["'", LaunchConfiguration("lidar_type"), "'", " == '3d'"]))
	)

	return LaunchDescription(
		[
			odometry_source_arg,
			rviz_arg,
			lidar_type_arg,
			# static_tf,  # Disabled for LIO-SAM
			robot_state_publisher_node,
			hospital_world_launch,
			spawn_robot_after_gazebo,
			bridge,
			rviz2_node,
			teleop_keyboard_node,
			pointcloud_to_laserscan_node
		]
	)
