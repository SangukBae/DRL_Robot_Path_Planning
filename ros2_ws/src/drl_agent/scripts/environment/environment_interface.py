#!/usr/bin/env python3

import os
import numpy as np
import rclpy
from rclpy.node import Node
from drl_agent_interfaces.srv import Step, Reset, Seed, GetDimensions, SampleActionSpace
# AUX_PRED: shared wire-format parser (geometry header + label).  Pure module,
# no ROS/torch deps; installed alongside this file.
import aux_prediction_labels as _aux_labels


class EnvInterface(Node):
    def __init__(self, node_name):
        super().__init__(node_name)

        # Create service clients
        self.reset_client = self.create_client(Reset, "reset")
        self.step_client = self.create_client(Step, "step")
        self.seed_client = self.create_client(Seed, "seed")
        self.actio_space_sample_client = self.create_client(
            SampleActionSpace, "action_space_sample"
        )
        self.dimensions_client = self.create_client(GetDimensions, "get_dimensions")

        # AUX_PRED: when the environment runs with auxiliary prediction enabled,
        # /reset and /step return a state of length (rl_state_dim + aux_label_dim)
        # -- the privileged future-risk label is appended after the RL state.
        # get_dimensions() still reports the true rl_state_dim (87), so this
        # common layer slices the tail off for EVERY client (trainers, test,
        # generalization_eval, ...) and stashes it in self.last_aux_label.  When
        # aux is disabled the returned length equals rl_state_dim and nothing is
        # stripped, so the baseline behaviour is byte-for-byte identical.
        self._rl_state_dim = None      # cached by get_dimensions()
        self.last_aux_label = None     # set on every reset()/step() (label only)
        self.last_aux_meta = None      # parsed geometry header (or None)

    def _strip_aux_label(self, state):
        """AUX_PRED: split any appended auxiliary wire-tail off the env state.

        The tail is ``[geometry header][label]`` (see aux_prediction_labels).
        Returns the RL state (length rl_state_dim); sets self.last_aux_label to
        the label (header removed) and self.last_aux_meta to the parsed geometry
        (num_sectors / num_horizons / horizons_sec) so the trainer can verify
        the STRUCTURE, not just the total length.  Passes the original object
        through untouched when there is nothing to strip.
        """
        if self._rl_state_dim is None:
            self.last_aux_label = None
            self.last_aux_meta = None
            return state
        arr = np.asarray(state, dtype=np.float32).ravel()
        if arr.shape[0] > self._rl_state_dim:
            tail = arr[self._rl_state_dim:]
            meta, label = _aux_labels.parse_aux_wire(tail)
            self.last_aux_meta = meta
            self.last_aux_label = label
            return arr[: self._rl_state_dim]
        self.last_aux_label = None
        self.last_aux_meta = None
        return state

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
        # AUX_PRED: strip the appended label (no-op when aux disabled).
        return self._strip_aux_label(future.result().state)

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
        # AUX_PRED: strip the appended label (no-op when aux disabled).
        state = self._strip_aux_label(response.state)
        return state, response.reward, response.done, response.target

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
        # AUX_PRED: cache the true RL state dim so reset()/step() can slice off
        # any appended auxiliary label.
        self._rl_state_dim = int(response.state_dim)
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
