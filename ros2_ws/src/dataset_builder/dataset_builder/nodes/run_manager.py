#!/usr/bin/env python3
"""Run manager node for dataset collection."""
import os
import shutil
from datetime import datetime

import pytz
import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterAlreadyDeclaredException
from ament_index_python.packages import get_package_share_directory

from dataset_builder.utils.paths import generate_run_id, get_run_dir, ensure_dir
from dataset_builder.utils.yaml_io import save_yaml, load_yaml


class RunManager(Node):
    """Manages run creation and metadata for dataset collection."""

    def __init__(self):
        super().__init__('run_manager')

        # Load run defaults from installed share/configs
        defaults = self._load_run_defaults()

        # Declare parameters (do NOT redeclare use_sim_time if already declared by rclpy)
        self.declare_parameter('dataset_root', str(defaults.get('dataset_root', '/root/drl_path_final/ros2_ws/data')))

        try:
            self.declare_parameter('use_sim_time', bool(defaults.get('use_sim_time', True)))
        except ParameterAlreadyDeclaredException:
            # rclpy may have already declared it
            pass

        self.declare_parameter('run_id', '')
        self.declare_parameter('world_name', 'unknown_world')
        self.declare_parameter('notes', '')
        # Used to keep split_policy consistent with actual segment interval
        self.declare_parameter('segment_duration_sec', int(defaults.get('segment_duration_sec', 600)))

        # Get parameters
        self.dataset_root = str(self.get_parameter('dataset_root').value)
        self.use_sim_time = bool(self.get_parameter('use_sim_time').value)
        run_id_param = str(self.get_parameter('run_id').value)
        self.world_name = str(self.get_parameter('world_name').value)
        self.notes = str(self.get_parameter('notes').value)
        self.segment_duration_sec = int(self.get_parameter('segment_duration_sec').value)

        # Keep split_policy base in defaults (expanded at write time)
        self.split_policy_base = str(defaults.get('split_policy', 'time_based'))

        # Generate or use provided run_id
        if (not run_id_param) or (run_id_param == 'auto'):
            self.run_id = generate_run_id()
        else:
            self.run_id = run_id_param

        # Initialize run
        self.initialize_run()

        self.get_logger().info(f'Run manager initialized for run_id: {self.run_id}')

    def _load_run_defaults(self) -> dict:
        try:
            pkg_share = get_package_share_directory('dataset_builder')
            defaults_path = os.path.join(pkg_share, 'configs', 'run_defaults.yaml')
            data = load_yaml(defaults_path)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            self.get_logger().warn(f'Failed to load run_defaults.yaml: {e}')
            return {}

    def _compute_split_policy(self) -> str:
        base = (self.split_policy_base or 'time_based').strip()
        if base == 'time_based':
            return f"time_based_{self.segment_duration_sec}s"
        return base

    def initialize_run(self):
        """Initialize run directory and metadata."""
        run_dir = get_run_dir(self.dataset_root, self.run_id)
        ensure_dir(run_dir)

        # Load topics configuration
        try:
            pkg_share = get_package_share_directory('dataset_builder')
            topics_yaml_path = os.path.join(pkg_share, 'configs', 'topics.yaml')
            topics_data = load_yaml(topics_yaml_path)
            record_topics = topics_data.get('topics', []) if isinstance(topics_data, dict) else []

            # Copy topics.yaml to run directory
            topics_dest = os.path.join(run_dir, 'topics.yaml')
            shutil.copy(topics_yaml_path, topics_dest)
            self.get_logger().info(f'Copied topics.yaml to {topics_dest}')
        except Exception as e:
            self.get_logger().error(f'Failed to load topics.yaml: {e}')
            record_topics = []

        # Create run_meta.yaml
        kst = pytz.timezone('Asia/Seoul')
        created_at = datetime.now(kst).isoformat()

        run_meta = {
            'run_id': self.run_id,
            'created_at_iso': created_at,
            'dataset_root': self.dataset_root,
            'ros_distro': os.environ.get('ROS_DISTRO', 'unknown'),
            'world_name': self.world_name,
            'notes': self.notes,
            'record_topics': record_topics,
            'split_policy': self._compute_split_policy(),
            'use_sim_time': self.use_sim_time,
            'git': {
                'commit': None,
                'branch': None,
                'dirty': None
            }
        }

        run_meta_path = os.path.join(run_dir, 'run_meta.yaml')
        save_yaml(run_meta, run_meta_path)
        self.get_logger().info(f'Created run_meta.yaml at {run_meta_path}')

        self.get_logger().info(f'Run directory: {run_dir}')
        self.get_logger().info(f'World: {self.world_name}')
        self.get_logger().info(f'Topics to record: {len(record_topics)}')


def main(args=None):
    rclpy.init(args=args)
    node = RunManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
