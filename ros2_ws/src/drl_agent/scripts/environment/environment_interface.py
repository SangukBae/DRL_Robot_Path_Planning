#!/usr/bin/env python3

import os
import numpy as np
import rclpy
from rclpy.node import Node
from drl_agent_interfaces.srv import Step, Reset, Seed, GetDimensions, SampleActionSpace
# AUX_PRED: shared wire-format parser (geometry header + label).  Pure module,
# no ROS/torch deps; installed alongside this file.
import aux_prediction_labels as _aux_labels


class EnvServiceError(RuntimeError):
    """Raised when an environment service call fails definitively (unavailable,
    timed out, or errored beyond the retry budget). Trainers catch this to
    fail-fast (optionally after saving a checkpoint) instead of hanging."""


class EnvInterface(Node):
    def __init__(self, node_name):
        super().__init__(node_name)

        # Service-call robustness knobs (overridable as ROS params so a slow
        # Gazebo can be given more headroom without code edits). Bounded waits +
        # a finite retry budget replace the old unbounded `while not available`
        # loops, so a dead env/Gazebo surfaces a clear error instead of hanging.
        def _p(name, default):
            if not self.has_parameter(name):
                self.declare_parameter(name, default)
            return self.get_parameter(name).value
        self._svc_wait_timeout = float(_p("service_wait_timeout_sec", 5.0))
        self._svc_call_timeout = float(_p("service_call_timeout_sec", 30.0))
        self._svc_max_retries  = int(_p("service_max_retries", 3))

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

    def _call_service(
        self,
        client,
        request,
        name,
        *,
        wait_timeout=None,
        call_timeout=None,
        max_retries=None,
    ):
        """Robustly call a service: bounded availability wait, bounded completion
        wait, finite retries, explicit logging. Returns the response or raises
        EnvServiceError after the retry budget is exhausted (never hangs).

        name: leading-slash-free service name, used only for log/error messages."""
        wait_timeout = (
            self._svc_wait_timeout if wait_timeout is None else float(wait_timeout)
        )
        call_timeout = (
            self._svc_call_timeout if call_timeout is None else float(call_timeout)
        )
        retries = self._svc_max_retries if max_retries is None else int(max_retries)
        total = retries + 1
        last_err = "unknown error"
        for attempt in range(1, total + 1):
            # The whole wait/call/spin block is guarded so that REAL ROS-level
            # failures (executor or context shutdown, rmw/rcl errors raised by
            # wait_for_service / call_async / spin_until_future_complete) are
            # normalized into EnvServiceError too — not just the soft
            # timeout/None/exception-on-future cases. Otherwise such a raw
            # exception would escape this layer and bypass the trainer's
            # checkpoint-on-failure path, breaking the fail-fast/resume contract.
            try:
                if not client.wait_for_service(timeout_sec=wait_timeout):
                    last_err = f"unavailable after {wait_timeout:.1f}s wait"
                else:
                    future = client.call_async(request)
                    rclpy.spin_until_future_complete(
                        self, future, timeout_sec=call_timeout)
                    if not future.done():
                        try:
                            future.cancel()
                        except Exception:
                            pass
                        last_err = f"call timed out after {call_timeout:.1f}s"
                    elif future.exception() is not None:
                        last_err = f"call raised: {future.exception()}"
                    elif future.result() is None:
                        last_err = "call returned None result"
                    else:
                        if attempt > 1:
                            self.get_logger().info(
                                f"Service /{name} succeeded on attempt {attempt}/{total}")
                        return future.result()
            except Exception as e:
                last_err = f"ROS-level error {type(e).__name__}: {e}"
            self.get_logger().warn(
                f"Service /{name} {last_err} (attempt {attempt}/{total})")
        self.get_logger().error(
            f"Service /{name} FAILED after {total} attempts: {last_err}")
        raise EnvServiceError(f"/{name}: {last_err} (after {total} attempts)")

    def reset(self):
        """Resets the environment to its initial state using /reset service"""
        result = self._call_service(self.reset_client, Reset.Request(), "reset")
        # AUX_PRED: strip the appended label (no-op when aux disabled).
        return self._strip_aux_label(result.state)

    def step(self, action):
        """Takes a step in the environment with the given action and the observed state"""
        request = Step.Request()
        # Pass actions directly in [-1, 1] normalized space.
        # environment.py._map_action_to_twist() maps [-1,1] → [actions_low, actions_high].
        # Hunter SE supports bidirectional motion (actions_low[0] = -1.333).
        request.action = np.array(
            [action[0], action[1]], dtype=np.float32
        ).tolist()
        response = self._call_service(self.step_client, request, "step")
        # AUX_PRED: strip the appended label (no-op when aux disabled).
        state = self._strip_aux_label(response.state)
        return state, response.reward, response.done, response.target

    def get_dimensions(self):
        """Get the dimensions of the environment"""
        response = self._call_service(
            self.dimensions_client, GetDimensions.Request(), "get_dimensions")
        # AUX_PRED: cache the true RL state dim so reset()/step() can slice off
        # any appended auxiliary label.
        self._rl_state_dim = int(response.state_dim)
        return response.state_dim, response.action_dim, response.max_action, response.environment_dim, response.agent_dim

    def sample_action_space(self):
        """Sample an action from the action space"""
        response = self._call_service(
            self.actio_space_sample_client, SampleActionSpace.Request(),
            "action_space_sample")
        return np.array(response.action)

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
        result = self._call_service(self.seed_client, request, "seed")
        self.get_logger().info(
            f"Environment seed set to: {seed}, Success: {result.success}"
        )
