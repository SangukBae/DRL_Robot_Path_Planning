#!/usr/bin/env python3
"""Curriculum-learning subclass of TrainTD7.

Same curriculum mechanism as drl_agent.training.train_tqc_curriculum:
  - Loads curriculum_settings from train_td7_curriculum_config.yaml
  - evaluate_and_print() returns success/collision/timeout rates (dict)
  - Automatic stage advancement via /gym_node/set_parameters
  - curriculum_stage column in per-episode CSV log
  - curriculum_state.json checkpoint for resume/inspection

The actual stage-advancement / eval / resume / CSV-logging machinery is
shared with SAC/SB3-SAC/SB3-TD3 via BaselineCurriculumTrainerMixin (see
drl_agent/training/curriculum/trainer_base.py) — this file only supplies the
TD7-specific config filename and its select_action call convention: TD7's
select_action has no use_exploration kwarg (unlike SAC's), for both eval and
the online training loop.

Usage:
  ros2 run drl_agent train_rl.py --ros-args -p rl_model:=td7_curriculum
  # or, once a matching drl_experiments profile exists:
  ros2 run drl_agent train_node.py --ros-args -p profile:=<group/variant>

The curriculum environment node must be running first:
  ros2 run drl_agent environment_curriculum_node.py
"""

import numpy as np
import rclpy

from drl_agent.training.baselines.td7 import TrainTD7
from drl_agent.training.curriculum.trainer_base import BaselineCurriculumTrainerMixin


class TrainTD7Curriculum(BaselineCurriculumTrainerMixin, TrainTD7):
    """TD7 trainer with automatic curriculum stage advancement.

    Inherits setup, training loop, and model I/O from TrainTD7; stage
    advancement / eval / resume / CSV logging come from
    BaselineCurriculumTrainerMixin (shared with SAC/SB3-SAC/SB3-TD3).
    """

    CURRICULUM_CONFIG_FILENAME = "train_td7_curriculum_config.yaml"
    # DEFAULT_MIN_STAGE_STEPS and _maybe_train_and_checkpoint use the mixin's
    # native-torch-agent defaults unchanged — TD7 has the same
    # train_and_checkpoint ensembling hook as SAC.

    def __init__(self):
        super().__init__()   # TrainTD7.__init__ (calls _init_csv_loggers, from the mixin)
        self._init_curriculum_state()

    def _select_action_training(self, state):
        """TD7's select_action has no use_exploration kwarg (unlike SAC's)."""
        return self.rl_agent.select_action(np.array(state))


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TrainTD7Curriculum()
        node.train_online()
        while rclpy.ok() and not node.done_training:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("\n[Curriculum] Training interrupted by user.")
        if node is not None:
            try:
                node._save_curriculum_state(getattr(node, "_last_global_t", 0))
            except Exception:
                pass
    except Exception as e:
        print(f"[Curriculum] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
