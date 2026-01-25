#!/usr/bin/env python3
"""Metadata logger node for segment-level sidecar files.

Creates management-only segments (seg_XXXX) on a fixed wall-clock interval
(ROS clock; sim time supported) and writes sidecar.yaml per segment.
"""
import os

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterAlreadyDeclaredException
from nav_msgs.msg import Odometry
import tf2_ros
from tf2_ros import TransformException

from dataset_builder.utils.paths import (
    ensure_dir,
    format_segment_id,
    get_segment_dir,
)
from dataset_builder.utils.yaml_io import save_yaml, load_yaml


class MetadataLogger(Node):
    """Logs segment-level metadata (sidecar.yaml) for dataset collection."""

    def __init__(self):
        super().__init__('metadata_logger')

        # Declare parameters (guard use_sim_time because rclpy may have already declared it)
        self.declare_parameter('dataset_root', '/root/drl_path_final/ros2_ws/data')
        self.declare_parameter('run_id', '')
        self.declare_parameter('segment_duration_sec', 600)
        self.declare_parameter('pose_source', 'odom')   # 'odom' or 'tf'
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('fixed_frame', 'odom')   # requirement: fixed to 'odom'
        try:
            self.declare_parameter('use_sim_time', True)
        except ParameterAlreadyDeclaredException:
            pass

        # Get parameters
        self.dataset_root = str(self.get_parameter('dataset_root').value)
        self.run_id = str(self.get_parameter('run_id').value)
        self.segment_duration_sec = int(self.get_parameter('segment_duration_sec').value)
        self.pose_source = str(self.get_parameter('pose_source').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        fixed_frame_param = str(self.get_parameter('fixed_frame').value)
        self.use_sim_time = bool(self.get_parameter('use_sim_time').value)

        if not self.run_id:
            self.get_logger().error('run_id parameter is required')
            raise ValueError('run_id parameter must be provided')

        if self.segment_duration_sec <= 0:
            self.get_logger().error('segment_duration_sec must be > 0')
            raise ValueError('segment_duration_sec must be > 0')

        if self.pose_source not in ('odom', 'tf'):
            self.get_logger().error("pose_source must be either 'odom' or 'tf'")
            raise ValueError("pose_source must be either 'odom' or 'tf'")

        # Enforce requirement: fixed_frame is always 'odom'
        if fixed_frame_param != 'odom':
            self.get_logger().warning(
                f"fixed_frame must be 'odom' per requirement. Overriding '{fixed_frame_param}' -> 'odom'."
            )
        self.fixed_frame = 'odom'

        # Initialize state
        self.segment_counter = 0
        self.current_segment_start_time = None
        self.current_segment_start_sim_time = None  # Track sim time separately
        self.latest_odom = None
        self.clock_initialized = False  # Wait for stable clock

        # Setup TF buffer if using TF
        self.tf_buffer = None
        self.tf_listener = None
        if self.pose_source == 'tf':
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Setup odom subscriber if using odom
        # Note: Ignition Gazebo bridges odometry to /odometry (not /odom)
        self.odom_sub = None
        if self.pose_source == 'odom':
            self.odom_sub = self.create_subscription(
                Odometry,
                '/odometry',
                self.odom_callback,
                10
            )

        # Timer for segment management - use short interval and check sim time manually
        # This avoids issues with sim time jumps causing premature segment transitions
        self.check_interval_sec = 1.0  # Check every 1 second (wall clock)
        self.segment_timer = self.create_timer(
            self.check_interval_sec,
            self.segment_timer_callback,
            clock=self.get_clock()  # Use node's clock (respects use_sim_time)
        )

        self.get_logger().info(f'Metadata logger initialized for run_id: {self.run_id}')
        self.get_logger().info(f'Dataset root: {self.dataset_root}')
        self.get_logger().info(f'Use sim time: {self.use_sim_time}')
        self.get_logger().info(f'Segment duration: {self.segment_duration_sec}s')
        self.get_logger().info(f'Pose source: {self.pose_source}')
        self.get_logger().info(f'Fixed frame: {self.fixed_frame}, Base frame: {self.base_frame}')

        # Don't create first segment here - wait for clock to stabilize
        self.get_logger().info('Waiting for clock to stabilize before creating first segment...')

    def odom_callback(self, msg: Odometry) -> None:
        """Store latest odometry message."""
        self.latest_odom = msg

    def get_current_pose(self):
        """Get current robot pose based on pose_source.

        Returns:
            dict or None: Pose data or None if unavailable
        """
        if self.pose_source == 'odom':
            if self.latest_odom is None:
                return None

            pose = self.latest_odom.pose.pose
            return {
                'position': {
                    'x': float(pose.position.x),
                    'y': float(pose.position.y),
                    'z': float(pose.position.z),
                },
                'orientation': {
                    'x': float(pose.orientation.x),
                    'y': float(pose.orientation.y),
                    'z': float(pose.orientation.z),
                    'w': float(pose.orientation.w),
                },
                'frame_id': str(self.latest_odom.header.frame_id),
                'child_frame_id': str(self.latest_odom.child_frame_id),
            }

        if self.pose_source == 'tf':
            try:
                # time=0 => latest available transform
                transform = self.tf_buffer.lookup_transform(
                    self.fixed_frame,
                    self.base_frame,
                    rclpy.time.Time()
                )
                return {
                    'position': {
                        'x': float(transform.transform.translation.x),
                        'y': float(transform.transform.translation.y),
                        'z': float(transform.transform.translation.z),
                    },
                    'orientation': {
                        'x': float(transform.transform.rotation.x),
                        'y': float(transform.transform.rotation.y),
                        'z': float(transform.transform.rotation.z),
                        'w': float(transform.transform.rotation.w),
                    },
                    'frame_id': str(transform.header.frame_id),
                    'child_frame_id': str(transform.child_frame_id),
                }
            except TransformException as e:
                self.get_logger().warning(
                    f'Failed to get transform {self.fixed_frame}->{self.base_frame}: {e}'
                )
                return None

        return None

    def _segment_id(self, index: int) -> str:
        return format_segment_id(index=index, width=4, prefix='seg')

    def _sidecar_path(self, segment_dir: str) -> str:
        return os.path.join(segment_dir, 'sidecar.yaml')

    def create_new_segment(self) -> None:
        """Create a new segment and save sidecar.yaml."""
        # Close previous segment if exists
        if self.current_segment_start_time is not None:
            self.close_current_segment()

        # Start new segment
        self.segment_counter += 1
        segment_id = self._segment_id(self.segment_counter)
        now = self.get_clock().now()
        self.current_segment_start_time = now
        self.current_segment_start_sim_time = now.nanoseconds  # Track for duration check

        # Get pose at segment start
        pose_at_start = self.get_current_pose()

        # Create segment directory
        segment_dir = get_segment_dir(self.dataset_root, self.run_id, segment_id)
        ensure_dir(segment_dir)

        sidecar_data = {
            'run_id': self.run_id,
            'segment_id': segment_id,
            'start_time_ros': int(self.current_segment_start_time.nanoseconds),
            'end_time_ros': None,
            'pose_at_start': pose_at_start,
            'notes': '',
        }

        sidecar_path = self._sidecar_path(segment_dir)
        save_yaml(sidecar_data, sidecar_path)

        self.get_logger().info(f'Created segment: {segment_id}')
        self.get_logger().info(f'Sidecar saved: {sidecar_path}')

    def close_current_segment(self) -> None:
        """Update sidecar.yaml with end time."""
        if self.current_segment_start_time is None:
            return

        segment_id = self._segment_id(self.segment_counter)
        end_time = self.get_clock().now()

        segment_dir = get_segment_dir(self.dataset_root, self.run_id, segment_id)
        sidecar_path = self._sidecar_path(segment_dir)

        try:
            sidecar_data = load_yaml(sidecar_path)
            if not isinstance(sidecar_data, dict):
                sidecar_data = {}

            sidecar_data['run_id'] = self.run_id
            sidecar_data['segment_id'] = segment_id
            sidecar_data['end_time_ros'] = int(end_time.nanoseconds)

            save_yaml(sidecar_data, sidecar_path)
            self.get_logger().info(f'Closed segment: {segment_id}')
        except Exception as e:
            self.get_logger().error(f'Failed to close segment {segment_id}: {e}')

    def segment_timer_callback(self) -> None:
        """Timer callback to check if segment duration has elapsed.

        Uses simulation time elapsed since segment start to determine
        when to create a new segment. This approach handles:
        - Sim time jumps at Gazebo startup
        - Variable RTF (Real-Time Factor)
        - Clock initialization delays
        """
        current_time = self.get_clock().now()

        # Wait for clock to be valid (non-zero for sim time)
        if self.use_sim_time:
            current_time_sec = current_time.nanoseconds / 1e9
            if current_time_sec < 1.0:
                # Clock not yet initialized or just started
                return

        # Initialize clock and create first segment
        if not self.clock_initialized:
            self.clock_initialized = True
            self.get_logger().info(
                f'Clock stabilized at {current_time.nanoseconds / 1e9:.2f}s, creating first segment'
            )
            self.create_new_segment()
            return

        # Check if segment duration has elapsed
        if self.current_segment_start_sim_time is not None:
            elapsed_ns = current_time.nanoseconds - self.current_segment_start_sim_time
            elapsed_sec = elapsed_ns / 1e9

            if elapsed_sec >= self.segment_duration_sec:
                self.get_logger().info(
                    f'Segment duration reached ({elapsed_sec:.1f}s >= {self.segment_duration_sec}s)'
                )
                self.create_new_segment()


def main(args=None):
    rclpy.init(args=args)
    node = MetadataLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Close last segment on shutdown
        try:
            if node.current_segment_start_time is not None:
                node.close_current_segment()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
