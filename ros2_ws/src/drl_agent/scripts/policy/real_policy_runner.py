#!/usr/bin/env python3
"""Real-robot inference node for a trained TQC (or compatible) actor.

Drives the robot toward a goal using the SAME observation/action convention as
training, but from real sensor topics — no /step or /reset services. It:

  • loads a saved actor (.pth),
  • subscribes to /scan, a localization odom, a proprioception odom,
    /joint_states and /goal,
  • computes goal_distance / heading_error from TF (goal frame → base frame,
    works for goal frames `map` OR `odom`),
  • assembles the state at a fixed rate following the SAME contract as training,
  • runs the actor (deterministic) and converts the action with the shared
    Pure-Pursuit helper into /cmd_vel (then the existing prefilter shapes it).

Safety: every input is freshness-guarded; if any required input is stale (or a
TF lookup fails) the node either holds the last command or publishes zero
(configurable), and a watchdog publishes zero when no fresh goal exists.

State layout (must match training) — base 87-D frame:
  state[0:80]  LiDAR obs bins (front 180°, nearest range per angular bin)
  state[80]    goal distance [m]            (TF, base frame)
  state[81]    heading error [rad]          (TF, base frame)
  state[82,83] previous normalized action (r, theta)
  state[84]    actual speed  [m/s]          (proprio odom)
  state[85]    actual yaw rate [rad/s]      (proprio odom)
  state[86]    center steering [rad]        (joint_states)

Observation time-context (frame stacking): when the env config's
`observation_time_context.enabled` is set, the state is the SAME stacked vector
used in training — the current 87-D frame above, then the last (N-1) obs frames
appended ([obs_t, agent_t, obs_{t-1}, ...]). The runner reuses the env's
ObsTimeContext, re-seeding the history (first-frame repeat) on each new goal, so
a stacked-policy checkpoint deploys with NO contract mismatch. Disabled -> the
exact 87-D state above. Point env_config at the file the policy was TRAINED with;
a mismatch is caught when the shared encoder fails to load.
"""

import os
import sys
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import tf2_ros
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, JointState

# Allow direct execution + ros2 run (utils/policy/environment on path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "environment"))

import torch
import pure_pursuit
from file_manager import load_yaml
from tqc_agent import Agent
# Observation time-context (frame stacking): the SAME ROS-free helper the env
# uses, so training and deployment build the identical stacked-state contract.
from obs_time_context import ObsTimeContext


def _yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


class RealPolicyRunner(Node):
    def __init__(self):
        super().__init__("real_policy_runner")

        # ---- parameters ----
        self.declare_parameter("actor_path", "")
        self.declare_parameter("env_config", "")            # environment.yaml path (or dir)
        self.declare_parameter("hparams_config", "")        # hyperparameters_tqc.yaml path (or dir)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("proprio_odom_topic", "/odometry")
        self.declare_parameter("joint_states_topic", "/hunter_se/joint_states")
        self.declare_parameter("goal_topic", "/goal")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("base_frame", "base_footprint")   # or base_link
        self.declare_parameter("control_rate_hz", 10.0)
        self.declare_parameter("steering_joint_left", "front_left_steering")
        self.declare_parameter("steering_joint_right", "front_right_steering")
        self.declare_parameter("stale_timeout_sec", 0.5)         # input freshness window
        self.declare_parameter("stale_action", "hold")           # "hold" | "zero"
        self.declare_parameter("goal_timeout_sec", 5.0)          # watchdog: no goal → zero
        self.declare_parameter("stop_at_goal", True)

        gp = lambda n: self.get_parameter(n).get_parameter_value()
        self.actor_path = gp("actor_path").string_value.strip()
        self.base_frame = gp("base_frame").string_value.strip() or "base_footprint"
        self.cmd_vel_topic = gp("cmd_vel_topic").string_value
        self.rate_hz = float(gp("control_rate_hz").double_value) or 10.0
        self.j_left = gp("steering_joint_left").string_value
        self.j_right = gp("steering_joint_right").string_value
        self.stale_timeout = float(gp("stale_timeout_sec").double_value)
        self.stale_action = gp("stale_action").string_value.strip().lower() or "hold"
        self.goal_timeout = float(gp("goal_timeout_sec").double_value)
        self.stop_at_goal = bool(gp("stop_at_goal").bool_value)

        if not self.actor_path or not os.path.isfile(os.path.expanduser(self.actor_path)):
            raise FileNotFoundError(f"actor_path not found: {self.actor_path!r}")

        # ---- config (environment.yaml) ----
        env_cfg = self._find_config("environment.yaml", gp("env_config").string_value.strip())
        cfg = load_yaml(env_cfg)
        e = cfg["environment"]
        self.env_dim   = int(e.get("environment_state_dim", 80))
        self.agent_dim = int(e.get("agent_state_dim", 7))
        self.action_dim = int(e.get("action_dim", 2))
        self.max_action = float(e.get("max_action", 1.0))
        self.actions_low  = list(e.get("actions_low",  [0.8, -0.524]))
        self.actions_high = list(e.get("actions_high", [2.0,  0.524]))
        self.wheelbase = float(e.get("vehicle_wheelbase_m", 0.547696))
        self.steer_limit = math.radians(float(e.get("vehicle_steering_limit_deg", 21.58)))
        self.cruise = float(e.get("controller_cruise_speed_mps", 2.0))
        self.min_speed = float(e.get("controller_min_speed_mps", 0.3))
        self.speed_steer_factor = float(e.get("controller_speed_steer_factor", 0.6))
        # Stop/yield capability (must match training config for contract parity).
        self.low_speed_distance = float(e.get("controller_low_speed_distance_m", 0.0))
        self.lidar_max_range = float(cfg.get("threshold_parameters", {}).get("lidar_max_range", 50))
        self.goal_threshold = float(cfg.get("threshold_parameters", {}).get("goal_threshold", 0.42))

        # Front-180° observation bins (must match environment.py exactly)
        obs_eps = 0.03
        obs_width = math.pi / self.env_dim
        obs_start = -math.pi / 2 - obs_eps
        self.obs_bins = [(obs_start + i * obs_width, obs_start + (i + 1) * obs_width)
                         for i in range(self.env_dim)]
        self.obs_bins[-1] = (self.obs_bins[-1][0], self.obs_bins[-1][1] + obs_eps)

        # ---- observation time-context (frame stacking) ----
        # Deployment MUST follow the same stacked-state contract as training, so
        # the actor/encoder receive a state of the exact width they were trained
        # with. Read the SAME `observation_time_context` block from the env config
        # the policy was trained with (point env_config at that file). Absent /
        # disabled -> 87-D state, byte-for-byte the legacy runner behaviour. A
        # config that disagrees with the checkpoint is caught loudly when the
        # shared encoder (input width == state_dim) fails to load below.
        _otc = dict(e.get("observation_time_context", {}) or {})
        self._otc = ObsTimeContext(
            self.env_dim, self.agent_dim,
            enabled=bool(_otc.get("enabled", False)),
            obs_frame_stack=int(_otc.get("obs_frame_stack", 1)),
            stack_agent_state=bool(_otc.get("stack_agent_state", False)),
        )
        self._state_dim = self._otc.stacked_state_dim()
        # Re-seed the obs history (first-frame repeat) on each new goal so a goal
        # = one "episode", exactly as the env reset does.
        self._needs_history_reset = True

        # ---- actor ----
        hp_cfg = self._find_config("hyperparameters_tqc.yaml", gp("hparams_config").string_value.strip())
        hp = load_yaml(hp_cfg)["hyperparameters"]
        # TEMPORAL_ACTOR: pass per-frame obs/agent dims so the agent can split the
        # stacked deploy state into current(87) + scan history. The runner does not
        # set a curriculum stage, so the temporal gain stays 1.0 (full temporal),
        # which is the intended deployment behaviour.
        self.agent = Agent(self._state_dim, self.action_dim, self.max_action, hp,
                           env_obs_dim=self.env_dim, env_agent_dim=self.agent_dim)
        self._load_actor(os.path.expanduser(self.actor_path))

        # ---- runtime state / caches ----
        now = self._now()
        self._obs = np.ones(self.env_dim, dtype=np.float32) * self.lidar_max_range
        self._speed = 0.0
        self._yaw_rate = 0.0
        self._steer = 0.0
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self._goal = None                 # PoseStamped (in its own frame)
        self._t_scan = self._t_prop = self._t_joint = self._t_goal = -1e9

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- pub / sub ----
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        qos_be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, qos)
        self.create_subscription(LaserScan, gp("scan_topic").string_value, self._on_scan, qos_be)
        self.create_subscription(Odometry, gp("proprio_odom_topic").string_value, self._on_proprio, qos)
        self.create_subscription(JointState, gp("joint_states_topic").string_value, self._on_joint, qos)
        self.create_subscription(PoseStamped, gp("goal_topic").string_value, self._on_goal, qos)

        self.create_timer(1.0 / self.rate_hz, self._control_step)
        self.get_logger().info(
            f"[RealRunner] ready — actor={os.path.basename(self.actor_path)} "
            f"base_frame={self.base_frame} rate={self.rate_hz}Hz stale={self.stale_timeout}s "
            f"({self.stale_action}) state_dim={self._state_dim}"
            f"{' (obs_frame_stack=%d)' % self._otc.obs_frame_stack if self._otc.enabled else ''}"
        )

    # ------------------------------------------------------------------ #
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _find_config(self, name, user):
        if user:
            p = os.path.expanduser(user)
            if os.path.isfile(p):
                return p
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, name)):
                return os.path.join(p, name)
        try:
            from ament_index_python.packages import get_package_share_directory
            cand = os.path.join(get_package_share_directory("drl_agent"), "config", name)
            if os.path.isfile(cand):
                return cand
        except Exception:
            pass
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.normpath(os.path.join(here, "..", "..", "config", name))
        if os.path.isfile(cand):
            return cand
        raise FileNotFoundError(f"Could not find config '{name}'")

    def _load_actor(self, path):
        try:
            sd = torch.load(path, map_location=self.agent.device, weights_only=True)
        except Exception:
            obj = torch.load(path, map_location=self.agent.device)
            sd = obj.get("actor", obj) if isinstance(obj, dict) else obj
        self.agent.actor.load_state_dict(sd, strict=False)
        if getattr(self.agent, "checkpoint_actor", None) is not None:
            self.agent.checkpoint_actor.load_state_dict(self.agent.actor.state_dict(), strict=False)
        self.agent.actor.eval()
        self.get_logger().info(f"[RealRunner] loaded actor weights from {path}")

        # AUX_PRED: real-robot inference runs state -> encoder -> actor, so an
        # aux-enabled policy must restore its shared encoder (derived from the
        # actor path).  No-op for baseline (aux-disabled) checkpoints.
        if not self.agent.load_encoder_for_inference(path):
            raise FileNotFoundError(
                "Aux-enabled agent but matching '_encoder.pth' not found next to "
                f"'{path}'. The deployed policy would be broken."
            )
        if self.agent.encoder.has_params():
            self.get_logger().info("[RealRunner] loaded encoder weights (aux-enabled policy).")

    # ---- callbacks (cache latest + timestamp) ----
    def _on_scan(self, msg: LaserScan):
        obs = np.ones(self.env_dim, dtype=np.float32) * self.lidar_max_range
        angle = float(msg.angle_min)
        inc = float(msg.angle_increment)
        rmin, rmax = float(msg.range_min), float(msg.range_max)
        for r in msg.ranges:
            if math.isfinite(r) and rmin <= r <= rmax:
                dist = min(float(r), self.lidar_max_range)
                for j, (lo, hi) in enumerate(self.obs_bins):
                    if lo <= angle < hi:
                        if dist < obs[j]:
                            obs[j] = dist
                        break
            angle += inc
        self._obs = obs
        self._t_scan = self._now()

    def _on_proprio(self, msg: Odometry):
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self._speed = math.hypot(vx, vy)
        self._yaw_rate = float(msg.twist.twist.angular.z)
        self._t_prop = self._now()

    def _on_joint(self, msg: JointState):
        try:
            li = msg.name.index(self.j_left)
            ri = msg.name.index(self.j_right)
            self._steer = 0.5 * (float(msg.position[li]) + float(msg.position[ri]))
            self._t_joint = self._now()
        except (ValueError, IndexError):
            pass

    def _on_goal(self, msg: PoseStamped):
        self._goal = msg
        self._t_goal = self._now()
        # New goal = new "episode": re-seed the obs history (first-frame repeat)
        # on the next control step so the stacked state matches the env's reset.
        self._needs_history_reset = True
        self.get_logger().info(
            f"[RealRunner] new goal in '{msg.header.frame_id}': "
            f"({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
        )

    # ---- goal-relative via TF ----
    def _goal_in_base(self):
        """Return (distance, heading_error) of the goal in the base frame.

        Returns None if the lookup fails OR the transform is STALE — so a frozen
        localization (TF stops advancing while still serving the last transform)
        is caught by the freshness guard rather than driving on old pose data.
        """
        if self._goal is None:
            return None
        goal_frame = self._goal.header.frame_id or self.base_frame
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, goal_frame, rclpy.time.Time())
        except Exception:
            return None
        # Freshness: reject a transform older than stale_timeout (frozen loc).
        stamp = tf.header.stamp
        tf_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if tf_time > 0.0 and (self._now() - tf_time) > self.stale_timeout:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = _yaw_from_quat(q.x, q.y, q.z, q.w)
        gx, gy = self._goal.pose.position.x, self._goal.pose.position.y
        # point in goal_frame → base_frame
        bx = math.cos(yaw) * gx - math.sin(yaw) * gy + t.x
        by = math.sin(yaw) * gx + math.cos(yaw) * gy + t.y
        return math.hypot(bx, by), math.atan2(by, bx)

    # ---- control loop ----
    def _publish(self, v, steering):
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(steering)
        self.cmd_pub.publish(cmd)

    def _safe_stop_or_hold(self, reason):
        if self.stale_action == "zero":
            self._publish(0.0, 0.0)
        # else "hold": publish nothing (downstream prefilter has its own timeout)
        self.get_logger().warn(f"[RealRunner] {reason} → {self.stale_action}", throttle_duration_sec=1.0)

    def _control_step(self):
        now = self._now()
        # Watchdog: no (fresh) goal → stop.
        if self._goal is None or (now - self._t_goal) > self.goal_timeout:
            self._publish(0.0, 0.0)
            return
        # Freshness guards for required inputs.
        stale = [name for name, t in (("scan", self._t_scan),
                                      ("proprio", self._t_prop),
                                      ("joint", self._t_joint))
                 if (now - t) > self.stale_timeout]
        if stale:
            self._safe_stop_or_hold(f"stale inputs: {stale}")
            return
        gm = self._goal_in_base()
        if gm is None:
            self._safe_stop_or_hold("TF goal transform unavailable or stale (frozen localization?)")
            return
        goal_dist, heading_err = gm
        if self.stop_at_goal and goal_dist < self.goal_threshold:
            self._publish(0.0, 0.0)
            self.get_logger().info("[RealRunner] goal reached → stop", throttle_duration_sec=2.0)
            return

        # New goal = new "episode": match the env reset semantics EXACTLY. The env
        # starts an episode with the previous-action slots at 0 (_rebuild_agent_state
        # + per-episode memory clear), so zero _prev_action here BEFORE building the
        # agent vector — otherwise the first inference state would still carry the
        # last goal's final action.
        if self._needs_history_reset:
            self._prev_action = np.zeros(self.action_dim, dtype=np.float32)

        # Assemble the (optionally stacked) state — SAME contract as training:
        # current [obs, agent] frame first, older obs frames appended. Disabled
        # -> exactly the 87-D vector. Mirrors env reset/step: on a new goal seed
        # the history (first-frame repeat) and DON'T advance; otherwise advance.
        agent = np.array([
            goal_dist, heading_err,
            float(self._prev_action[0]), float(self._prev_action[1]),
            self._speed, self._yaw_rate, self._steer,
        ], dtype=np.float32)
        obs = self._obs.astype(np.float32)
        if self._needs_history_reset:
            self._otc.reset(obs, agent)
            state = self._otc.assemble(obs, agent, advance=False)
            self._needs_history_reset = False
        else:
            state = self._otc.assemble(obs, agent, advance=True)

        # Deterministic actor inference.
        action = self.agent.select_action(state, use_checkpoint=False, use_exploration=False)
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        self._prev_action = action

        # Action → waypoint → Pure-Pursuit command (shared with environment.py).
        _, _, x_wp, y_wp = pure_pursuit.action_to_waypoint(action, self.actions_low, self.actions_high)
        v, steering = pure_pursuit.waypoint_to_command(
            x_wp, y_wp, self.wheelbase, self.steer_limit,
            self.cruise, self.min_speed, self.speed_steer_factor,
            low_speed_distance_m=self.low_speed_distance)
        self._publish(v, steering)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RealPolicyRunner()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[RealRunner] fatal: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            try:
                node.cmd_pub.publish(Twist())   # stop on exit
            except Exception:
                pass
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
