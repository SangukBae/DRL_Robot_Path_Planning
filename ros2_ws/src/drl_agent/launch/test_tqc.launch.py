from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    gym_env_node = Node(
        package="drl_agent",
        executable="environment.py",
        name="environment_node",
        output="screen",
        emulate_tty=True,
        parameters=[{"environment_mode": "test"}],
    )

    test_tqc_node = Node(
        package="drl_agent",
        executable="tqc_live_runner.py",
        name="test_tqc_node",
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            gym_env_node,
            test_tqc_node,
        ]
    )
