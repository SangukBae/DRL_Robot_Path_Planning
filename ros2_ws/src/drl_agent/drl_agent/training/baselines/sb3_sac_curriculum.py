#!/usr/bin/env python3
"""Curriculum-learning subclass of TrainSB3SAC.

Same curriculum mechanism as drl_agent.training.baselines.sac_curriculum:
  - Loads curriculum_settings from train_sb3_sac_curriculum_config.yaml
  - evaluate_and_print() returns success/collision/timeout rates (dict)
  - Automatic stage advancement via /gym_node/set_parameters
  - curriculum_stage column in per-episode CSV log
  - curriculum_state.json checkpoint for resume/inspection

The actual stage-advancement / eval / resume / CSV-logging machinery is
shared with SAC/TD7/SB3-TD3 via BaselineCurriculumTrainerMixin (see
drl_agent/training/curriculum/trainer_base.py) — this file only supplies the
SB3-SAC-specific config filename/default and its select_action call
convention: the SB3 wrapper's select_action takes the raw state (no
np.array() wrap) and no use_checkpoint kwarg, and has no
train_and_checkpoint ensembling hook.

Usage:
  ros2 run drl_agent train_rl.py --ros-args -p rl_model:=sb3_sac_curriculum
  # or, once a matching drl_experiments profile exists:
  ros2 run drl_agent train_node.py --ros-args -p profile:=<group/variant>

The curriculum environment node must be running first:
  ros2 run drl_agent environment_curriculum_node.py
"""

import rclpy

from drl_agent.training.baselines.sb3_sac import TrainSB3SAC
from drl_agent.training.curriculum.trainer_base import BaselineCurriculumTrainerMixin


class TrainSB3SACCurriculum(BaselineCurriculumTrainerMixin, TrainSB3SAC):
    """SB3 SAC trainer with automatic curriculum stage advancement.

    Inherits setup, training loop, and model I/O from TrainSB3SAC; stage
    advancement / eval / resume / CSV logging come from
    BaselineCurriculumTrainerMixin (shared with SAC/TD7/SB3-TD3).
    """

    CURRICULUM_CONFIG_FILENAME = "train_sb3_sac_curriculum_config.yaml"
    DEFAULT_MIN_STAGE_STEPS = 30000

    def __init__(self):
        super().__init__()   # TrainSB3SAC.__init__ (calls _init_csv_loggers, from the mixin)
        self._init_curriculum_state()

    def _select_action_eval(self, state):
        return self.rl_agent.select_action(state, use_exploration=False)

    def _select_action_training(self, state):
        return self.rl_agent.select_action(state)

    def _maybe_train_and_checkpoint(self, ep_timesteps, ep_total_reward, train_ready):
        pass  # SB3 wrapper has no checkpoint-ensembling hook.


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TrainSB3SACCurriculum()
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
