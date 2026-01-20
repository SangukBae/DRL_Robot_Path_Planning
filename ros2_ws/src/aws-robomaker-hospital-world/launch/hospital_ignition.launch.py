import os
from os import environ
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, Shutdown
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory("aws_robomaker_hospital_world")

    # Launch configuration variables
    world = LaunchConfiguration("world")

    # World file path
    world_file = os.path.join(
        pkg_dir, "worlds", "hospital_ignition.world"
    )

    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        name="use_sim_time",
        default_value="True",
        description="Use simulation (Gazebo) clock if true",
    )

    world_arg = DeclareLaunchArgument(
        name="world",
        default_value=world_file,
        description="Full path to world file",
    )

    # Ignition Gazebo environment variables
    env = {
        "IGN_GAZEBO_SYSTEM_PLUGIN_PATH": ":".join([
            environ.get("IGN_GAZEBO_SYSTEM_PLUGIN_PATH", default=""),
            environ.get("LD_LIBRARY_PATH", default=""),
        ]),
    }

    # Launch Ignition Gazebo
    gazebo_sim = ExecuteProcess(
        cmd=["ign", "gazebo", "-v", "1", "-r", world],
        output="screen",
        additional_env=env,
        shell=False,
        on_exit=Shutdown(),
    )

    return LaunchDescription([
        use_sim_time_arg,
        world_arg,
        gazebo_sim,
    ])
