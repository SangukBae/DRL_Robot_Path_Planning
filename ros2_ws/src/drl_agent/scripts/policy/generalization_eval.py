#!/usr/bin/env python3
"""Generalization evaluation harness.

Loads a trained TQC model and evaluates it (deterministic policy) under a set of
*conditions* without any further training, so that the paper can show the policy
is not over-fit to a single setting. Each condition is a curriculum stage, which
bundles static-obstacle density, human (pedestrian) count and perception noise —
i.e. the density-shift / noise-shift axes asked for by the paper guide. The world axis
(train world vs unseen test world) is selected at Gazebo launch time; pass a
``--world`` label so the output records which world produced the numbers.

Every condition reports the same metrics as training/eval (success, collision,
timeout, SPL, CTE, jerk, ...) via the shared EpisodeMetrics, so generalization
rows are directly comparable to the in-distribution eval rows.

Prerequisites
-------------
  * Gazebo running with the desired world.
  * environment_curriculum.py running (so curriculum_stage is settable).

Usage
-----
    ros2 run drl_agent generalization_eval.py --ros-args \
        -p weight_prefix:=tqc_agent_seed_0_20260601 \
        -p weights_dir:=<run_dir>/final_models \
        -p world:=aws_hospital \
        -p eval_eps_override:=20 \
        -p conditions:="[0,1,2,3,4]"

If ``conditions`` is empty the harness sweeps every stage the environment reports.
Results: <run_dir>/logs/generalization_<world>_<tag>.csv
"""

import os
import sys
import csv
import time
from datetime import datetime

import numpy as np
import rclpy
from rcl_interfaces.srv import GetParameters, SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "environment"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))

from train_tqc_base import TrainTQCBase
from episode_metrics import EpisodeMetrics, PaperMetricsCSV, METRIC_COLUMNS
# AUX_ABLATION: run-identity columns (seed / aux_enabled / aux_version).
import aux_ablation_logging as aux_log


class GeneralizationEval(TrainTQCBase):
    """Evaluate a fixed trained policy across curriculum conditions + worlds."""

    # AUX_PRED: evaluation of an aux-trained policy is allowed -- it restores the
    # encoder via Agent.load(...) and never computes the auxiliary loss, so it
    # only runs the encoder+actor inference path.
    AUX_SUPPORTED = True

    def __init__(self):
        super().__init__()   # builds agent, env interface, dirs, CSV loggers

        self.declare_parameter("weight_prefix", "")
        self.declare_parameter("weights_dir", "")
        self.declare_parameter("world", "unknown")
        self.declare_parameter("conditions", [-1])         # int list of stages; [-1]=all
        self.declare_parameter("eval_eps_override", -1)
        self.declare_parameter("near_collision_dist_m", 0.5)
        self.declare_parameter("metric_time_delta", 0.1)

        weight_prefix = self.get_parameter("weight_prefix").get_parameter_value().string_value.strip()
        weights_dir = self.get_parameter("weights_dir").get_parameter_value().string_value.strip()
        self.world = self.get_parameter("world").get_parameter_value().string_value.strip() or "unknown"
        self.conditions = list(self.get_parameter("conditions").get_parameter_value().integer_array_value)
        eval_override = self.get_parameter("eval_eps_override").get_parameter_value().integer_value
        if eval_override and eval_override > 0:
            self.eval_eps = eval_override
        ncd = self.get_parameter("near_collision_dist_m").get_parameter_value().double_value
        mdt = self.get_parameter("metric_time_delta").get_parameter_value().double_value

        if not weight_prefix:
            raise ValueError("generalization_eval requires -p weight_prefix:=<model prefix>")
        weights_dir = os.path.expanduser(weights_dir) if weights_dir else self.pytorch_models_dir
        self.rl_agent.load(weights_dir, weight_prefix,
                           load_optimizer_state=False, load_replay_buffer=False)
        self.get_logger().info(f"[Gen] Loaded model '{weight_prefix}' from {weights_dir}")

        self._em = EpisodeMetrics(
            self.environment_dim,
            time_delta=mdt if mdt > 0 else 0.1,
            near_collision_dist_m=ncd if ncd > 0 else 0.5,
        )

        self._param_set = self.create_client(SetParameters, "/gym_node/set_parameters")
        self._param_get = self.create_client(GetParameters, "/gym_node/get_parameters")

        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._out_csv = os.path.join(self.log_dir, f"generalization_{self.world}_{tag}.csv")
        header = [
            "world", "condition_stage", "eval_eps",
            "mean_reward", "std_reward", "success_rate", "collision_rate",
            "timeout_rate", "mean_goal_dist",
        ] + METRIC_COLUMNS + aux_log.META_COLUMN_NAMES   # AUX_ABLATION
        with open(self._out_csv, "w", newline="") as f:
            csv.writer(f).writerow(header)
        self.get_logger().info(f"[Gen] Output: {self._out_csv}")
        self.done = False

    # ------------------------------------------------------------------ #
    def _fetch_num_stages(self) -> int:
        if not self._param_get.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("[Gen] /gym_node/get_parameters unavailable — defaulting 5 stages.")
            return 5
        req = GetParameters.Request(); req.names = ["curriculum_num_stages"]
        fut = self._param_get.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        if fut.result() is None or not fut.result().values:
            return 5
        return max(1, int(fut.result().values[0].integer_value))

    def _set_stage(self, stage: int) -> bool:
        if not self._param_set.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("[Gen] /gym_node/set_parameters unavailable — is environment_curriculum.py running?")
            return False
        req = SetParameters.Request()
        req.parameters = [Parameter(
            name="curriculum_stage",
            value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=int(stage)))]
        fut = self._param_set.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        return fut.result() is not None and all(r.successful for r in fut.result().results)

    def _eval_condition(self, stage: int) -> dict:
        ENV_DIM = self.environment_dim
        rewards, final_dists, per_ep = [], [], []
        sc = cc = tc = 0
        for _ in range(self.eval_eps):
            state = self.reset()
            done = False; steps = 0; ep_rew = 0.0
            self._em.reset(state)
            while not done and steps < self.max_episode_steps:
                action = self.rl_agent.select_action(state, use_checkpoint=False, use_exploration=False)
                state, reward, done, info = self.step(action)
                self._em.update(state, action)
                ep_rew += reward; steps += 1
            rewards.append(ep_rew)
            final_dists.append(float(np.asarray(state, dtype=np.float32).ravel()[ENV_DIM]))
            success = bool(done and info)
            per_ep.append(self._em.compute(success))
            if success:   sc += 1
            elif done:    cc += 1
            else:         tc += 1
        n = self.eval_eps
        base = {
            "mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
            "success_rate": sc / n, "collision_rate": cc / n, "timeout_rate": tc / n,
            "mean_goal_dist": float(np.mean(final_dists)),
        }
        return base, PaperMetricsCSV.aggregate(per_ep)

    def run(self):
        num_stages = self._fetch_num_stages()
        conditions = [c for c in self.conditions if 0 <= c < num_stages]
        if not conditions:   # [-1] / empty / out-of-range → sweep every stage
            conditions = list(range(num_stages))
        self.get_logger().info(f"[Gen] World='{self.world}' | conditions={conditions} | eval_eps={self.eval_eps}")

        for stage in conditions:
            if not self._set_stage(stage):
                raise RuntimeError(f"[Gen] Could not set curriculum_stage={stage}. Is environment_curriculum.py running?")
            time.sleep(1.0)   # let the env apply the stage before the first reset
            t0 = time.time()
            base, agg = self._eval_condition(stage)
            self.get_logger().info(
                f"[Gen] stage {stage}: Success {base['success_rate']*100:.1f}% "
                f"Collision {base['collision_rate']*100:.1f}% Timeout {base['timeout_rate']*100:.1f}% "
                f"SPL {agg['spl']:.3f} CTE {agg['mean_cross_track_error_m']:.3f}m "
                f"({time.time()-t0:.0f}s)"
            )
            row = [
                self.world, stage, self.eval_eps,
                round(base["mean_reward"], 4), round(base["std_reward"], 4),
                round(base["success_rate"], 4), round(base["collision_rate"], 4),
                round(base["timeout_rate"], 4), round(base["mean_goal_dist"], 4),
            ] + [agg[c] for c in METRIC_COLUMNS] + self._aux_log_meta_cols()  # AUX_ABLATION
            with open(self._out_csv, "a", newline="") as f:
                csv.writer(f).writerow(row)

        self.get_logger().info(f"[Gen] Done. Results: {self._out_csv}")
        self.done = True


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GeneralizationEval()
        node.run()
    except KeyboardInterrupt:
        print("\n[Gen] Interrupted.")
    except Exception as e:
        print(f"[Gen] Error: {e}")
        import traceback; traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
