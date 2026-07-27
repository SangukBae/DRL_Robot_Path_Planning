#!/usr/bin/env python3
# SIM_VALIDATION: simulation verification runner (localization-aware RL).
# VALIDATION_ONLY — drives a few short episodes (NO training, NO model) against
# the running environment so it emits loc_validation_*.csv. Remove with this file.
"""Short validation-mode driver.

Runs N episodes of M steps with simple actions (random / zero / forward) — no
policy, no learning — purely to exercise the environment so the SIM_VALIDATION
logging in environment.py records obs-vs-gt / reset-jump / stale / curriculum
data. Optionally sweeps curriculum stages via /gym_node/set_parameters.

Prereq: launch the environment with `-p enable_sim_validation_logging:=true`
(``ros2 run drl_agent environment.py`` or ``environment_curriculum.py``, or
their profile-based ``environment_curriculum_node.py`` wrapper).

Usage:
  ros2 run drl_agent sim_validation_runner.py --ros-args \
    -p episodes:=5 -p max_steps:=80 -p action_mode:=random
  # sweep curriculum stages 0, 2, 4 (requires the curriculum environment node):
  ros2 run drl_agent sim_validation_runner.py --ros-args \
    -p episodes:=3 -p max_steps:=80 -p stages:="[0,2,4]"
"""

import os
import sys
import time

import numpy as np
import rclpy
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType


from drl_agent.env.environment_interface import EnvInterface


class SimValidationRunner(EnvInterface):
    def __init__(self):
        super().__init__("sim_validation_runner")
        self.declare_parameter("episodes", 5)
        self.declare_parameter("max_steps", 80)
        self.declare_parameter("action_mode", "random")   # random | zero | forward
        self.declare_parameter("stages", [-1])             # [-1] = current stage only
        self.declare_parameter("seed", 0)
        gp = lambda n: self.get_parameter(n).get_parameter_value()
        self.episodes = int(gp("episodes").integer_value) or 5
        self.max_steps = int(gp("max_steps").integer_value) or 80
        self.action_mode = gp("action_mode").string_value.strip().lower() or "random"
        self.stages = [int(s) for s in gp("stages").integer_array_value] or [-1]
        self.seed = int(gp("seed").integer_value)

        np.random.seed(self.seed)
        self.set_env_seed(self.seed)
        self.state_dim, self.action_dim, *_ = self.get_dimensions()
        self._param_set = self.create_client(SetParameters, "/gym_node/set_parameters")

    def _set_stage(self, stage):
        if not self._param_set.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("[SIM_VALIDATION] /gym_node/set_parameters unavailable; "
                                   "ignoring stage (is the curriculum environment node running?)")
            return False
        req = SetParameters.Request()
        req.parameters = [Parameter(
            name="curriculum_stage",
            value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=int(stage)))]
        fut = self._param_set.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        ok = fut.result() is not None and all(r.successful for r in fut.result().results)
        self.get_logger().info(f"[SIM_VALIDATION] set curriculum_stage={stage} ok={ok}")
        time.sleep(1.0)
        return ok

    def _action(self):
        if self.action_mode == "zero":
            return np.zeros(self.action_dim, dtype=np.float32)
        if self.action_mode == "forward":
            a = np.zeros(self.action_dim, dtype=np.float32); a[0] = 1.0  # max waypoint dist, straight
            return a
        return self.sample_action_space()

    def run(self):
        for stage in self.stages:
            if stage >= 0:
                self._set_stage(stage)
            for ep in range(self.episodes):
                self.reset()
                for _ in range(self.max_steps):
                    _, _, done, _ = self.step(self._action())
                    if done:
                        break
                self.get_logger().info(
                    f"[SIM_VALIDATION] stage={stage} episode {ep + 1}/{self.episodes} done"
                )
        self.get_logger().info("[SIM_VALIDATION] validation run complete — see loc_validation_*.csv")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SimValidationRunner()
        node.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[SIM_VALIDATION] error: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
