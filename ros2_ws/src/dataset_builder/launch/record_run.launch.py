#!/usr/bin/env python3
"""Launch file for dataset recording run."""
import os
from datetime import datetime

import pytz
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_run_id():
    """Generate run_id with Asia/Seoul timezone."""
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    return f"run_{now.strftime('%Y-%m-%d_%H-%M-%S')}"


def _load_yaml_file(path: str) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}
    except Exception as e:
        print(f"[dataset_builder] Failed to load YAML: {path} ({e})")
        return {}


def load_run_defaults() -> dict:
    """Load default values from run_defaults.yaml in package share."""
    try:
        pkg_share = get_package_share_directory('dataset_builder')
        defaults_yaml = os.path.join(pkg_share, 'configs', 'run_defaults.yaml')
        return _load_yaml_file(defaults_yaml)
    except Exception as e:
        print(f"[dataset_builder] Failed to locate run_defaults.yaml: {e}")
        return {}


def load_topics_list() -> list:
    """Load topics list from topics.yaml in package share."""
    try:
        pkg_share = get_package_share_directory('dataset_builder')
        topics_yaml = os.path.join(pkg_share, 'configs', 'topics.yaml')
        data = _load_yaml_file(topics_yaml)
        return data.get('topics', [])
    except Exception as e:
        print(f"[dataset_builder] Failed to locate topics.yaml: {e}")
        return []


def launch_setup(context, *args, **kwargs):
    """Setup launch with evaluated configurations."""
    dataset_root = LaunchConfiguration('dataset_root').perform(context)
    run_id_arg = LaunchConfiguration('run_id').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    segment_duration_sec = LaunchConfiguration('segment_duration_sec').perform(context)
    world_name = LaunchConfiguration('world_name').perform(context)
    notes = LaunchConfiguration('notes').perform(context)
    pose_source = LaunchConfiguration('pose_source').perform(context)
    base_frame = LaunchConfiguration('base_frame').perform(context)

    # Generate run_id if not provided
    if run_id_arg == '' or run_id_arg == 'auto':
        run_id = generate_run_id()
    else:
        run_id = run_id_arg

    print(f"[dataset_builder] Run ID: {run_id}")
    print(f"[dataset_builder] Dataset root: {dataset_root}")

    # Convert use_sim_time to boolean
    use_sim_time_bool = use_sim_time.lower() in ['true', '1', 'yes']

    # Run manager node
    run_manager_node = Node(
        package='dataset_builder',
        executable='run_manager',
        name='run_manager',
        output='screen',
        parameters=[{
            'dataset_root': dataset_root,
            'use_sim_time': use_sim_time_bool,
            'run_id': run_id,
            'world_name': world_name,
            'notes': notes,
            # keep run_meta split_policy consistent with logger interval
            'segment_duration_sec': int(segment_duration_sec),
        }]
    )

    # Metadata logger node
    metadata_logger_node = Node(
        package='dataset_builder',
        executable='metadata_logger',
        name='metadata_logger',
        output='screen',
        parameters=[{
            'dataset_root': dataset_root,
            'run_id': run_id,
            'segment_duration_sec': int(segment_duration_sec),
            'pose_source': pose_source,
            'base_frame': base_frame,
            'fixed_frame': 'odom',   # fixed per requirement
            'use_sim_time': use_sim_time_bool,
        }]
    )

    topics = load_topics_list()

    # Rosbag output path
    rosbag_output = os.path.join(dataset_root, 'runs', run_id, 'segments', 'rosbag')
    segments_dir = os.path.dirname(rosbag_output)

    # Ensure segments directory exists before rosbag starts
    mkdir_segments = ExecuteProcess(
        cmd=['mkdir', '-p', segments_dir],
        output='screen',
        shell=False
    )

    rosbag_record = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            *topics,
            '--output', rosbag_output,
        ],
        output='screen',
        shell=False
    )

    return [
        run_manager_node,
        metadata_logger_node,
        mkdir_segments,
        rosbag_record,
    ]


def generate_launch_description():
    """Generate launch description."""
    defaults = load_run_defaults()

    dataset_root_default = str(defaults.get('dataset_root', '/root/DRL_Robot_Path_Planning/ros2_ws/data'))
    use_sim_time_default = 'true' if bool(defaults.get('use_sim_time', True)) else 'false'
    segment_duration_default = str(int(defaults.get('segment_duration_sec', 600)))
    base_frame_default = str(defaults.get('base_frame', 'base_link'))
    pose_source_default = str(defaults.get('pose_source', 'odom'))

    # Declare launch arguments
    dataset_root_arg = DeclareLaunchArgument(
        'dataset_root',
        default_value=dataset_root_default,
        description='Root directory for dataset storage'
    )

    run_id_arg = DeclareLaunchArgument(
        'run_id',
        default_value='auto',
        description='Run identifier (auto = generate timestamp-based ID)'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value=use_sim_time_default,
        description='Use simulation time'
    )

    segment_duration_sec_arg = DeclareLaunchArgument(
        'segment_duration_sec',
        default_value=segment_duration_default,
        description='Segment duration in seconds'
    )

    world_name_arg = DeclareLaunchArgument(
        'world_name',
        default_value='unknown_world',
        description='Name of the Ignition Gazebo world'
    )

    notes_arg = DeclareLaunchArgument(
        'notes',
        default_value='',
        description='Additional notes for this run'
    )

    pose_source_arg = DeclareLaunchArgument(
        'pose_source',
        default_value=pose_source_default,
        description='Pose source: odom or tf'
    )

    base_frame_arg = DeclareLaunchArgument(
        'base_frame',
        default_value=base_frame_default,
        description='Robot base frame'
    )

    return LaunchDescription([
        dataset_root_arg,
        run_id_arg,
        use_sim_time_arg,
        segment_duration_sec_arg,
        world_name_arg,
        notes_arg,
        pose_source_arg,
        base_frame_arg,
        OpaqueFunction(function=launch_setup)
    ])
