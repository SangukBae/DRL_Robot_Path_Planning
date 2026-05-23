#!/usr/bin/env python3
"""Curriculum-learning subclass of Environment.

No existing file is modified. This file only adds:
  - curriculum_stage  ROS parameter (integer, writable at runtime by the trainer)
  - _load_curriculum_config()        parses the curriculum: block from self.config
  - _apply_curriculum_stage(idx)     overrides active counts + motion params
  - reset_callback override          reads the parameter and applies the stage
                                     before every episode's obstacle placement

Launch with:
  ros2 run drl_agent environment_curriculum.py \\
    --ros-args -p config_file:=<path>/environment_curriculum.yaml

The trainer (train_tqc_curriculum_agent.py) advances the stage by writing to
the /gym_node/set_parameters service after each successful evaluation.
The trainer also reads /gym_node/get_parameters::curriculum_num_stages to stay
in sync with the actual number of stages in the config that was passed here.
"""

import os
import sys

# Make sure the environment directory is on the path so we can import Environment.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
import rclpy.executors
from rcl_interfaces.msg import SetParametersResult

from environment import Environment   # base class — not modified


def _with_default_curriculum_config(args=None):
    """Inject `config_file:=environment_curriculum.yaml` when the caller did
    not explicitly provide one.

    This must happen before rclpy.init(), otherwise the base Environment class
    will fall back to the installed default `environment.yaml`.
    """
    arg_list = list(sys.argv[1:] if args is None else args)
    joined = " ".join(arg_list)
    if "config_file:=" in joined:
        return arg_list

    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = get_package_share_directory("drl_agent")
        default_cfg = os.path.join(share_dir, "config", "environment_curriculum.yaml")
        if os.path.isfile(default_cfg):
            arg_list.extend(["--ros-args", "-p", f"config_file:={default_cfg}"])
    except Exception:
        pass
    return arg_list


class EnvironmentCurriculum(Environment):
    """Environment with runtime curriculum stage control.

    The pool is always built to the maximum size (from yaml).
    _apply_curriculum_stage() just changes how many pool slots are activated
    and what motion parameters are used — no pool rebuild ever happens.
    """

    def __init__(self):
        # Prefer the curriculum config by default. The base Environment loader
        # checks the explicit ROS parameter first, so a user-provided
        # `-p config_file:=...` still overrides this fallback cleanly.
        if "DRL_AGENT_CONFIG" not in os.environ:
            try:
                from ament_index_python.packages import get_package_share_directory
                share_dir = get_package_share_directory("drl_agent")
                default_cfg = os.path.join(
                    share_dir, "config", "environment_curriculum.yaml"
                )
                if os.path.isfile(default_cfg):
                    os.environ["DRL_AGENT_CONFIG"] = default_cfg
            except Exception:
                pass

        super().__init__()                      # full base init (config, pool, timers …)
        self._load_curriculum_config()
        # Declare AFTER super() so we don't conflict with base parameter declarations.
        self.declare_parameter("curriculum_stage", self.curriculum_initial_stage)
        # curriculum_num_stages is read-only: lets the trainer query the actual
        # stage count from the config file the environment was launched with,
        # avoiding any mismatch between the two processes.
        self.declare_parameter("curriculum_num_stages", len(self.curriculum_stages))
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._current_stage = self.curriculum_initial_stage
        self._last_stage_apply_logged = None
        self._apply_curriculum_stage(self._current_stage)
        self.get_logger().info(
            f"[Curriculum] Ready — stage {self._current_stage} "
            f"('{self._stage_name(self._current_stage)}') | "
            f"total stages: {len(self.curriculum_stages)}"
        )

    # ------------------------------------------------------------------ #
    #  Config loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_curriculum_config(self):
        """Parse the curriculum: block from self.config (loaded by super().__init__)."""
        cfg = self.config.get("curriculum", {})
        self.curriculum_enabled       = bool(cfg.get("enabled", False))
        self.curriculum_initial_stage = int(cfg.get("initial_stage", 0))
        self.curriculum_stages        = cfg.get("stages", [])
        if not self.curriculum_stages:
            self.curriculum_enabled = False
            self.get_logger().warn(
                "[Curriculum] No stages found in config — curriculum disabled. "
                "Make sure you are using environment_curriculum.yaml."
            )

    def _on_set_parameters(self, params):
        """Reject writes to curriculum_num_stages — it is read-only."""
        for p in params:
            if p.name == "curriculum_num_stages":
                return SetParametersResult(
                    successful=False,
                    reason="curriculum_num_stages is read-only",
                )
        return SetParametersResult(successful=True)

    def _stage_name(self, idx: int) -> str:
        if 0 <= idx < len(self.curriculum_stages):
            return self.curriculum_stages[idx].get("name", f"stage_{idx}")
        return f"stage_{idx}"

    # ------------------------------------------------------------------ #
    #  Stage application                                                    #
    # ------------------------------------------------------------------ #

    def _apply_curriculum_stage(self, idx: int):
        """Override active obstacle counts and motion parameters for stage idx.

        Pool sizes are fixed at startup (from yaml obstacle_pool_*_size).
        This method only changes the *active* subset; inactive pool slots
        are returned to their parking positions by _activate_random_obstacles.
        """
        if not self.curriculum_enabled:
            return
        idx = max(0, min(idx, len(self.curriculum_stages) - 1))
        stage = self.curriculum_stages[idx]

        # Active counts — never exceed pre-allocated pool sizes
        self.num_of_static_obstacles = min(
            int(stage.get("active_static",  self.num_of_static_obstacles)),
            self.obstacle_pool_static_size,
        )
        self.num_of_dynamic_obstacles = min(
            int(stage.get("active_dynamic", self.num_of_dynamic_obstacles)),
            self.obstacle_pool_dynamic_size,
        )
        self.num_of_humans = min(
            int(stage.get("active_humans",  self.num_of_humans)),
            self.obstacle_pool_human_size,
        )

        # Dynamic obstacle speed range
        if "dynamic_speed_min" in stage:
            self.dynamic_speed_min = float(stage["dynamic_speed_min"])
        if "dynamic_speed_max" in stage:
            self.dynamic_speed_max = float(stage["dynamic_speed_max"])

        # Human sensor noise / dropout (domain randomisation intensity)
        if "human_scan_noise_std" in stage:
            self.human_scan_noise_std = float(stage["human_scan_noise_std"])
        if "human_scan_dropout_prob" in stage:
            self.human_scan_dropout_prob = float(stage["human_scan_dropout_prob"])
        self.human_placement_mode = str(stage.get("human_placement_mode", "quadrants"))

        if self._last_stage_apply_logged != idx:
            self.get_logger().info(
                f"[Curriculum] Stage {idx} '{self._stage_name(idx)}' applied — "
                f"static={self.num_of_static_obstacles} "
                f"dynamic={self.num_of_dynamic_obstacles} "
                f"humans={self.num_of_humans} "
                f"human_place={self.human_placement_mode} "
                f"dyn_spd=[{self.dynamic_speed_min:.2f}, {self.dynamic_speed_max:.2f}] "
                f"noise={self.human_scan_noise_std:.3f} "
                f"dropout={self.human_scan_dropout_prob:.3f}"
            )
            self._last_stage_apply_logged = idx

    # ------------------------------------------------------------------ #
    #  reset_callback override                                              #
    # ------------------------------------------------------------------ #

    def reset_callback(self, request, response):
        """Read curriculum_stage parameter and apply it before each episode.

        The trainer writes /gym_node/set_parameters between episodes.
        This override detects the change and applies the new stage config
        before _activate_random_obstacles() is called in the base class.
        """
        stage = int(
            self.get_parameter("curriculum_stage")
            .get_parameter_value()
            .integer_value
        )
        if stage != self._current_stage:
            self.get_logger().info(
                f"[Curriculum] Stage transition: "
                f"{self._current_stage} ('{self._stage_name(self._current_stage)}') → "
                f"{stage} ('{self._stage_name(stage)}')"
            )
            self._current_stage = stage
        # Always re-apply so dynamic_speed_* and noise settings take effect
        # even on the very first call or after a parameter write.
        self._apply_curriculum_stage(self._current_stage)

        return super().reset_callback(request, response)


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main(args=None):
    rclpy.init(args=_with_default_curriculum_config(args))
    node = None
    try:
        node = EnvironmentCurriculum()
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
