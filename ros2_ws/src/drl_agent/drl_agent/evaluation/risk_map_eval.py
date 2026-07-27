#!/usr/bin/env python3
"""Phase 1b: risk-map dump evaluation harness.

Loads a trained TQC checkpoint (deterministic policy, no further training) and
rolls out N episodes under the curriculum env, dumping one JSONL record per
step: robot GT pose, the selected waypoint (r, theta) + its risk-map sector
index, the GT (label-side, privileged) risk_map/min_dist, the aux HEAD's
PREDICTED risk_map/min_dist (kept separate from the GT fields), and a
lightweight human-state summary. See drl_agent.evaluation.risk_map_dump for the
record schema and drl_agent.training.aux_eval_metrics for split_label().

This is the "risk map figure" data source for Slide 9's item (6) and the
qualitative figure planned in Slide 15/16 (Directional Risk-map Reward
Shaping candidate): LiDAR + real human positions + predicted risk sectors +
selected action, all in one dumped record per step.

Works gracefully with aux fully disabled (every aux-related field in the dump
is simply ``null``/None -- see risk_map_dump.py) and with an action-conditioned
aux head (whose live per-step prediction is architecturally undefined without
already-known future actions, so aux_predict_eval() correctly returns None
there too -- the GT fields are still dumped either way).

IMPORTANT -- ``pred_risk_map`` is null by default: every shipped TQC config
(the main config AND experiments/A1-A4) has ``action_conditioned_aux: true``
(hyperparameters_tqc.yaml). A live per-step call has no future actions to
condition on, so ``aux_predict_eval()`` returns None for every step with any
of those checkpoints -- this is NOT a bug, it is an architectural property of
action-conditioned aux heads. With the current default configs this script
produces a GT-risk-map dump (gt_risk_map/gt_min_dist), NOT a predicted-
risk-map visualization; ``pred_risk_map``/``pred_min_dist`` only populate for
a checkpoint trained with ``action_conditioned_aux: false``. The script logs
a one-time warning at startup when it detects this so it is not a silent gap.

ALIGNMENT -- every record pairs the action/GT-risk/pred-risk/pose/humans with
the SAME state s_t the policy actually conditioned on when it chose that
action (decision-time / pre-step), not the state s_{t+1} the action produced.
This matters for "selected waypoint direction's risk at decision time"
figures and for the Phase-2 risk-map-shaping design (Slide 15's pre-step
risk_dir) -- pairing the action with s_{t+1} instead would misattribute which
risk value the policy could actually have reacted to. The step's reward/done
outcome (necessarily only known AFTER executing the action) is logged
separately under ``extra`` as ``reward``/``done``/``post_robot_pose`` for
reference, clearly distinguished from the decision-time fields above.

Prerequisites
-------------
  * Gazebo running with the desired world.
  * environment_curriculum.py running (so curriculum_stage / eval params are
    settable via /gym_node/set_parameters).

Usage
-----
    ros2 run drl_agent risk_map_eval.py --ros-args \\
        -p weight_prefix:=tqc_agent_seed_0_20260722 \\
        -p weights_dir:=<run_dir>/final_models \\
        -p episodes:=5 \\
        -p stage:=5

    # with the Phase 1b eval-only presets:
    ros2 run drl_agent risk_map_eval.py --ros-args \\
        -p weight_prefix:=tqc_agent_seed_0_20260722 \\
        -p weights_dir:=<run_dir>/final_models \\
        -p episodes:=10 -p stage:=8 \\
        -p use_fixed_eval_suite:=true -p fixed_eval_suite_base_seed:=20260722 \\
        -p use_hard_pedestrian_eval:=true \\
        -p use_robot_reactive_pedestrian:=false

Output: <run_dir>/logs/risk_map_dump_<tag>.jsonl (or -p dump_path:=<path>)
"""

import os
import sys
import time
import json
from datetime import datetime

import numpy as np
import rclpy
from rcl_interfaces.srv import GetParameters, SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType


from drl_agent.training.train_tqc_base import TrainTQCBase
import drl_agent.training.aux_eval_metrics as aux_metrics
import drl_agent.evaluation.risk_map_dump as rmdump


def _to_param_value(value):
    if isinstance(value, bool):
        return ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=value)
    if isinstance(value, int):
        return ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=value)
    if isinstance(value, float):
        return ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=value)
    if isinstance(value, str):
        return ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=value)
    raise TypeError(f"unsupported parameter value type: {type(value)!r}")


class RiskMapEval(TrainTQCBase):
    """Roll out a fixed trained policy and dump per-step risk-map records."""

    # AUX_PRED: evaluation of an aux-trained policy is allowed -- it restores
    # the encoder via Agent.load(...) and never computes the auxiliary loss.
    AUX_SUPPORTED = True

    def __init__(self):
        super().__init__()   # builds agent, env interface, dirs, CSV loggers

        self.declare_parameter("weight_prefix", "")
        self.declare_parameter("weights_dir", "")
        self.declare_parameter("episodes", 5)
        self.declare_parameter("stage", -1)          # -1 = leave curriculum_stage unchanged
        self.declare_parameter("dump_path", "")
        # PHASE1B eval-only presets, forwarded to /gym_node as ROS params.
        self.declare_parameter("use_fixed_eval_suite", False)
        self.declare_parameter("fixed_eval_suite_base_seed", 0)
        self.declare_parameter("use_hard_pedestrian_eval", False)
        self.declare_parameter("use_robot_reactive_pedestrian", False)

        weight_prefix = self.get_parameter("weight_prefix").get_parameter_value().string_value.strip()
        weights_dir = self.get_parameter("weights_dir").get_parameter_value().string_value.strip()
        if not weight_prefix:
            raise ValueError("risk_map_eval requires -p weight_prefix:=<model prefix>")
        weights_dir = os.path.expanduser(weights_dir) if weights_dir else self.pytorch_models_dir
        self.rl_agent.load(weights_dir, weight_prefix,
                           load_optimizer_state=False, load_replay_buffer=False)
        self.get_logger().info(f"[RiskMapEval] Loaded model '{weight_prefix}' from {weights_dir}")

        self.episodes = int(self.get_parameter("episodes").value)
        self.stage = int(self.get_parameter("stage").value)
        self._warned_action_conditioned = False

        self._param_set = self.create_client(SetParameters, "/gym_node/set_parameters")
        self._param_get = self.create_client(GetParameters, "/gym_node/get_parameters")

        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_path = self.get_parameter("dump_path").get_parameter_value().string_value.strip()
        self.dump_path = dump_path or os.path.join(self.log_dir, f"risk_map_dump_{tag}.jsonl")
        self.writer = rmdump.RiskMapDumpWriter(self.dump_path)
        self.get_logger().info(f"[RiskMapEval] Output: {self.dump_path}")

    # ------------------------------------------------------------------ #
    def _set_env_params(self, params: dict) -> bool:
        if not self._param_set.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                "[RiskMapEval] /gym_node/set_parameters unavailable — "
                "is environment_curriculum.py running?"
            )
            return False
        req = SetParameters.Request()
        req.parameters = [
            Parameter(name=name, value=_to_param_value(value))
            for name, value in params.items()
        ]
        fut = self._param_set.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        return fut.result() is not None and all(r.successful for r in fut.result().results)

    def _get_env_param_value(self, name):
        """Best-effort read of a single scalar /gym_node parameter (used to
        snapshot curriculum_stage before overwriting it, so run() can restore
        it in `finally`). Returns None on any failure."""
        if not self._param_get.wait_for_service(timeout_sec=3.0):
            return None
        req = GetParameters.Request()
        req.names = [name]
        fut = self._param_get.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        if fut.result() is None or not fut.result().values:
            return None
        pv = fut.result().values[0]
        getters = {
            ParameterType.PARAMETER_BOOL: "bool_value",
            ParameterType.PARAMETER_INTEGER: "integer_value",
            ParameterType.PARAMETER_DOUBLE: "double_value",
            ParameterType.PARAMETER_STRING: "string_value",
        }
        attr = getters.get(pv.type)
        return getattr(pv, attr) if attr else None

    def _get_step_debug_state(self) -> dict:
        """Best-effort read of the env's step_debug_state parameter (see
        environment.py::_publish_step_debug_state). Returns {} on any failure
        so a transient service hiccup degrades to a robot_pose=None record
        instead of crashing the eval run."""
        if not self._param_get.wait_for_service(timeout_sec=3.0):
            return {}
        req = GetParameters.Request()
        req.names = ["step_debug_state"]
        fut = self._param_get.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        if fut.result() is None or not fut.result().values:
            return {}
        try:
            return json.loads(fut.result().values[0].string_value or "{}")
        except (json.JSONDecodeError, IndexError, AttributeError):
            return {}

    def _gt_risk_and_min_dist(self):
        """Slice the GT label off self.last_aux_label/last_aux_meta (already
        parsed by EnvInterface._strip_aux_label -- no-op / None when aux is
        disabled at the env). Returns (risk_list, min_dist_list, num_sectors)
        or (None, None, None)."""
        label, meta = self.last_aux_label, self.last_aux_meta
        if label is None or meta is None:
            return None, None, None
        num_sectors = int(meta.get("num_sectors", 0))
        num_horizons = int(meta.get("num_horizons", 0))
        if num_sectors <= 0 or num_horizons <= 0:
            return None, None, None
        risk, md = aux_metrics.split_label(label, num_horizons, num_sectors)
        return risk[0].reshape(-1).tolist(), md[0].tolist(), num_sectors

    def _pred_risk_and_min_dist(self, state):
        """Best-effort aux-head prediction for the state the policy is ABOUT
        TO ACT ON (pre-step / decision-time -- see the module docstring's
        ALIGNMENT note). Returns (None, None) when aux is disabled, or when
        the head is action-conditioned (a live per-step call has no future
        actions yet -- an architectural limitation, not a bug;
        aux_predict_eval already returns None in that case; see the
        module docstring's IMPORTANT note, and the one-time warning below)."""
        if not getattr(self.rl_agent, "aux_eval_enabled", False):
            return None, None
        preds = self.rl_agent.aux_predict_eval(np.asarray(state, dtype=np.float32)[None, :])
        if preds is None:
            if getattr(self.rl_agent, "aux_action_conditioned", False) and not self._warned_action_conditioned:
                self._warned_action_conditioned = True
                self.get_logger().warn(
                    "[RiskMapEval] aux head is action-conditioned (action_conditioned_aux: "
                    "true) -- pred_risk_map/pred_min_dist will be null for EVERY step of "
                    "this run (no future actions are known yet at decision time). "
                    "gt_risk_map/gt_min_dist are still populated normally. This is expected "
                    "with the shipped TQC configs, not a bug -- see the module docstring."
                )
            return None, None
        risk = preds["risk_map"][0].reshape(-1).tolist()
        min_dist = preds["min_dist"][0].tolist() if "min_dist" in preds else None
        return risk, min_dist

    @staticmethod
    def _pose_from_debug(dbg: dict):
        if dbg.get("robot_x") is None:
            return None
        return (dbg["robot_x"], dbg["robot_y"], dbg["robot_yaw"])

    def run(self):
        # Snapshot the exact env params BEFORE touching anything. The `finally`
        # below restores these values verbatim (where readable), not merely a
        # safe-off approximation, so this helper can be run against an env that was
        # intentionally left in a non-default eval/debug mode.
        restore_names = [
            "curriculum_eval_mode",
            "fixed_eval_suite_enabled",
            "fixed_eval_suite_base_seed",
            "fixed_eval_suite_reset_token",
            "risk_map_dump_enabled",
            "hard_pedestrian_eval_enabled",
            "human_robot_avoid_enabled",
            "curriculum_stage",
        ]
        restore_params = {
            name: value
            for name in restore_names
            for value in [self._get_env_param_value(name)]
            if value is not None
        }

        try:
            # Force a clean OFF baseline first, THEN apply the real config. This
            # guarantees environment.py observes a False->True transition on
            # curriculum_eval_mode regardless of whatever state a prior run left
            # it in. The explicit reset token below is the stronger guarantee, but
            # the toggle keeps the existing map-cursor reset path working too.
            baseline_off = {
                "curriculum_eval_mode": False,
                "fixed_eval_suite_enabled": False,
                "risk_map_dump_enabled": False,
                "hard_pedestrian_eval_enabled": False,
                "human_robot_avoid_enabled": False,
            }
            if not self._set_env_params(baseline_off):
                raise RuntimeError(
                    "[RiskMapEval] could not configure /gym_node parameters. "
                    "Is environment_curriculum.py running?"
                )

            env_params = {
                "curriculum_eval_mode": True,
                "risk_map_dump_enabled": True,
                "fixed_eval_suite_enabled": bool(self.get_parameter("use_fixed_eval_suite").value),
                "fixed_eval_suite_base_seed": int(self.get_parameter("fixed_eval_suite_base_seed").value),
                # Force environment.py to restart the suite's episode-index counter
                # at 0 for THIS run, independent of curriculum_eval_mode's toggle
                # history (which a crashed/uncleaned prior run could have left
                # stuck at True -- see environment.py's declare_parameter comment).
                "fixed_eval_suite_reset_token": int(time.time() * 1000) % (2**31 - 1),
                "hard_pedestrian_eval_enabled": bool(self.get_parameter("use_hard_pedestrian_eval").value),
                "human_robot_avoid_enabled": bool(self.get_parameter("use_robot_reactive_pedestrian").value),
            }
            if self.stage >= 0:
                env_params["curriculum_stage"] = self.stage
            if not self._set_env_params(env_params):
                raise RuntimeError(
                    "[RiskMapEval] could not configure /gym_node parameters. "
                    "Is environment_curriculum.py running?"
                )
            time.sleep(1.0)   # let the env apply the params before the first reset
            self.get_logger().info(
                f"[RiskMapEval] episodes={self.episodes} stage={self.stage} params={env_params}"
            )

            for ep in range(self.episodes):
                # `state`/`dbg` here are the DECISION-TIME (pre-step) values --
                # what the policy is about to condition its action on. For
                # step 0 these come from reset(); for every later step they
                # are carried over from the previous iteration's post-step
                # read (see the loop's tail below) rather than re-read after
                # step() executes the action (see the ALIGNMENT note in the
                # module docstring for why that pairing would be wrong).
                state = self.reset()
                dbg = self._get_step_debug_state()
                done = False
                step = 0
                while not done and step < self.max_episode_steps:
                    gt_risk, gt_min_dist, num_sectors = self._gt_risk_and_min_dist()
                    pred_risk, pred_min_dist = self._pred_risk_and_min_dist(state)
                    robot_pose = self._pose_from_debug(dbg)
                    humans = rmdump.human_state_summary(dbg.get("humans"))

                    action = self.rl_agent.select_action(
                        state, use_checkpoint=False, use_exploration=False)
                    next_state, reward, done, info = self.step(action)
                    next_dbg = self._get_step_debug_state()
                    # action_r/action_theta are only known AFTER step() maps
                    # the normalized action through Pure Pursuit, but they
                    # describe the action just chosen FROM `state` above, so
                    # they belong with the pre-step fields, not next_dbg's.
                    action_r = next_dbg.get("action_r")
                    action_theta = next_dbg.get("action_theta")
                    post_pose = self._pose_from_debug(next_dbg)

                    self.writer.write_step(
                        episode=ep, step=step,
                        robot_pose=robot_pose,                      # pre-step (decision-time)
                        action_r=action_r, action_theta=action_theta,
                        num_sectors=num_sectors,
                        gt_risk_map=gt_risk, gt_min_dist=gt_min_dist,        # pre-step
                        pred_risk_map=pred_risk, pred_min_dist=pred_min_dist,  # pre-step
                        humans=humans,                              # pre-step
                        extra={
                            "reward": float(reward),
                            "done": bool(done),
                            # Post-step outcome, kept separate from the
                            # decision-time fields above (see ALIGNMENT note).
                            "post_robot_x": None if post_pose is None else post_pose[0],
                            "post_robot_y": None if post_pose is None else post_pose[1],
                            "post_robot_yaw": None if post_pose is None else post_pose[2],
                        },
                    )
                    state, dbg = next_state, next_dbg
                    step += 1
                self.get_logger().info(
                    f"[RiskMapEval] episode {ep + 1}/{self.episodes}: {step} steps, "
                    f"success={bool(done and info)} collision={bool(done and not info)}"
                )
            self.get_logger().info(f"[RiskMapEval] Done. Dump: {self.dump_path}")
        finally:
            self.writer.close()
            # Always restore the exact readable pre-run parameter snapshot, even
            # if setup failed halfway through. If the snapshot could not be read
            # at all, fall back to a conservative safe-off cleanup for the flags
            # this script owns.
            restore = restore_params or {
                "curriculum_eval_mode": False,
                "fixed_eval_suite_enabled": False,
                "risk_map_dump_enabled": False,
                "hard_pedestrian_eval_enabled": False,
                "human_robot_avoid_enabled": False,
            }
            if not self._set_env_params(restore):
                self.get_logger().warn(
                    "[RiskMapEval] failed to restore /gym_node parameters after the "
                    "run -- the env may be left in eval/hard-pedestrian/"
                    "robot-reactive/fixed-suite mode. Restart environment_curriculum.py "
                    "before the next training/eval run to be safe."
                )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RiskMapEval()
        node.run()
    except KeyboardInterrupt:
        print("\n[RiskMapEval] Interrupted.")
    except Exception as e:
        print(f"[RiskMapEval] Error: {e}")
        import traceback; traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
