#!/usr/bin/env python3

import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from drl_agent_interfaces.srv import Step, Reset, Seed, GetDimensions, SampleActionSpace


class EnvInterface(Node):
    def __init__(self, node_name):
        super().__init__(node_name)

        # Create service clients
        self.clients_callback_group = MutuallyExclusiveCallbackGroup()
        self.reset_client = self.create_client(Reset, "reset")
        self.step_client = self.create_client(Step, "step")
        self.seed_client = self.create_client(Seed, "seed")
        self.actio_space_sample_client = self.create_client(
            SampleActionSpace, "action_space_sample"
        )
        self.dimensions_client = self.create_client(GetDimensions, "get_dimensions")

    def reset(self):
        """Resets the environment to its initial state using /reset service"""
        request = Reset.Request()
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /reset not available, waiting again...")
        try:
            future = self.reset_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
        except Exception as e:
            self.get_logger().error(f"Service call /reset failed: {e}")
        return future.result().state

    def step(self, action):
        """Takes a step in the environment with the given action and the observed state"""
        request = Step.Request()
        # Pass actions directly in [-1, 1] normalized space.
        # environment.py._map_action_to_twist() maps [-1,1] → [actions_low, actions_high].
        # Hunter SE supports bidirectional motion (actions_low[0] = -1.333).
        request.action = np.array(
            [action[0], action[1]], dtype=np.float32
        ).tolist()
        while not self.step_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /step not available, waiting again...")
        try:
            future = self.step_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
        except Exception as e:
            self.get_logger().error(f"Service call /step failed: {e}")
        response = future.result()
        return response.state, response.reward, response.done, response.target

    def get_dimensions(self):
        """Get the dimensions of the environment"""
        request = GetDimensions.Request()
        while not self.dimensions_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Service /get_dimensions not available, waiting again..."
            )
        try:
            future = self.dimensions_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
        except Exception as e:
            self.get_logger().error(f"Service call /get_dimensions failed: {e}")
        response = future.result()
        return response.state_dim, response.action_dim, response.max_action, response.environment_dim, response.agent_dim

    def sample_action_space(self):
        """Sample an action from the action space"""
        request = SampleActionSpace.Request()
        while not self.actio_space_sample_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Service /action_space_sample not available, waiting again..."
            )
        try:
            future = self.actio_space_sample_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
        except Exception as e:
            self.get_logger().error(f"Service call /action_space_sample failed: {e}")
        return np.array(future.result().action)

    def _resolve_seed_override(self, default_seed: int) -> int:
        """Return the seed to use, allowing a per-run override for multi-seed sweeps.

        Priority: ROS2 parameter ``seed`` (>=0) > env var ``DRL_AGENT_SEED`` >
        ``default_seed`` (the value read from the YAML config). This lets the same
        config file be swept across seeds without editing it — required for the
        paper protocol (>=3 seeds, mean ± std):

            ros2 run drl_agent <trainer>.py --ros-args -p seed:=1
            DRL_AGENT_SEED=1 ros2 run drl_agent <trainer>.py
        """
        seed = int(default_seed)
        if not self.has_parameter("seed"):
            self.declare_parameter("seed", -1)
        ovr = self.get_parameter("seed").get_parameter_value().integer_value
        if ovr is None or ovr < 0:
            env_seed = os.environ.get("DRL_AGENT_SEED", "").strip()
            ovr = int(env_seed) if env_seed.lstrip("-").isdigit() else -1
        if ovr is not None and ovr >= 0 and int(ovr) != seed:
            seed = int(ovr)
            self.get_logger().info(f"[Seed] Config seed overridden → {seed}")
        return seed

    def set_env_seed(self, seed):
        """Set the seed of the environment"""
        request = Seed.Request()
        request.seed = seed
        while not self.seed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /seed not available, waiting again...")
        try:
            future = self.seed_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
        except Exception as e:
            self.get_logger().error(f"Service call /seed failed: {e}")
        self.get_logger().info(
            f"Environment seed set to: {seed}, Success: {future.result().success}"
        )
