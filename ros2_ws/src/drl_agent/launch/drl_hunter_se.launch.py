"""
DRL environment node launch file for AgileX Hunter SE.

Run Gazebo separately first:
  ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py

Then launch the environment node:
  ros2 launch drl_agent drl_hunter_se.launch.py                  # train
  ros2 launch drl_agent drl_hunter_se.launch.py mode:=test       # test

Then start the agent in another terminal:
  ros2 run drl_agent train_tqc_agent.py

This launch file does not start RViz. If RViz is enabled in the Hunter SE
simulation launch, the DRL marker topics will appear in that existing window.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    mode_arg = DeclareLaunchArgument(
        name="mode",
        default_value="train",
        choices=["train", "test", "random_test"],
        description="DRL run mode",
    )

    environment_node = Node(
        package="drl_agent",
        executable="environment.py",
        name="environment_node",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "use_sim_time": True,
            "environment_mode": LaunchConfiguration("mode"),
            # Hunter SE topics (cmd_vel handled by prefilter, odometry bridged to /odometry)
            "cmd_vel_topic":   "/cmd_vel",
            "odom_topic":      "/odometry",
            # 2-D LiDAR bridged to /scan
            "obs_source":      "scan",
            "scan_topic":      "/scan",
            # Must match the SDF <world name="..."> in the running world file.
            # drl_arena.world and hospital_ignition.world both use name="default".
            "world_name":      "default",
        }],
    )

    return LaunchDescription([
        mode_arg,
        environment_node,
    ])
