import os
from os import environ
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, Shutdown
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch.substitutions import PythonExpression


def generate_launch_description():
    pkg_dir = get_package_share_directory('aws_robomaker_small_warehouse_world')

    world_arg = DeclareLaunchArgument(
        'world',
        default_value='small_warehouse',
        description='World to load: small_warehouse or no_roof_small_warehouse',
        choices=['small_warehouse', 'no_roof_small_warehouse'],
    )

    world_name = LaunchConfiguration('world')

    rgl_install_dir = os.path.join(
        os.path.expanduser('~'), 'DRL_Robot_Path_Planning',
        'third_party', 'rgl', 'RGLGazeboPlugin', 'install'
    )
    rgl_server_plugin_path = os.path.join(rgl_install_dir, 'RGLServerPlugin')
    rgl_gui_plugin_path = os.path.join(rgl_install_dir, 'RGLVisualize')

    gz_env = {
        'IGN_GAZEBO_SYSTEM_PLUGIN_PATH': ':'.join(filter(None, [
            rgl_server_plugin_path,
            environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', ''),
            environ.get('LD_LIBRARY_PATH', ''),
        ])),
        'IGN_GUI_PLUGIN_PATH': ':'.join(filter(None, [
            rgl_gui_plugin_path,
            environ.get('IGN_GUI_PLUGIN_PATH', ''),
        ])),
        'LD_LIBRARY_PATH': ':'.join(filter(None, [
            rgl_server_plugin_path,
            environ.get('LD_LIBRARY_PATH', ''),
        ])),
        'IGN_GAZEBO_RESOURCE_PATH': ':'.join(filter(None, [
            os.path.join(pkg_dir, 'models'),
            os.path.join(pkg_dir, 'worlds'),
            environ.get('IGN_GAZEBO_RESOURCE_PATH', ''),
        ])),
    }

    # small_warehouse
    gazebo_sim_small = ExecuteProcess(
        cmd=[
            'ign', 'gazebo', '-v', '1', '-r',
            os.path.join(pkg_dir, 'worlds', 'small_warehouse', 'small_warehouse_ignition.world'),
        ],
        output='screen',
        additional_env=gz_env,
        shell=False,
        on_exit=Shutdown(),
        condition=IfCondition(PythonExpression(["'", world_name, "' == 'small_warehouse'"])),
    )

    # no_roof_small_warehouse
    gazebo_sim_no_roof = ExecuteProcess(
        cmd=[
            'ign', 'gazebo', '-v', '1', '-r',
            os.path.join(pkg_dir, 'worlds', 'no_roof_small_warehouse', 'no_roof_small_warehouse_ignition.world'),
        ],
        output='screen',
        additional_env=gz_env,
        shell=False,
        on_exit=Shutdown(),
        condition=IfCondition(PythonExpression(["'", world_name, "' == 'no_roof_small_warehouse'"])),
    )

    return LaunchDescription([
        world_arg,
        gazebo_sim_small,
        gazebo_sim_no_roof,
    ])
