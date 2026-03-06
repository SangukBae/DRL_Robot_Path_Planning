import os
from os import environ
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, Shutdown


def generate_launch_description():
    pkg_dir = get_package_share_directory('aws_robomaker_bookstore_world')

    world_file = os.path.join(pkg_dir, 'worlds', 'bookstore_ignition.world')

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

    gazebo_sim = ExecuteProcess(
        cmd=['ign', 'gazebo', '-v', '1', '-r', world_file],
        output='screen',
        additional_env=gz_env,
        shell=False,
        on_exit=Shutdown(),
    )

    return LaunchDescription([
        gazebo_sim,
    ])
