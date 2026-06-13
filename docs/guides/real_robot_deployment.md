# Real-Robot Deployment & Localization-Error Robustness

How to (a) train a policy that tolerates real localization error and (b) run a
trained actor on the real Hunter SE with `real_policy_runner.py`.

The trained policy is **mapless**: it avoids obstacles from LiDAR and only needs
to know *where the goal is relative to itself*. On hardware that relative pose
comes from **localization (odometry / SLAM)**, which has error — so we both
emulate that error in sim and provide a real-robot inference node.

---

## 1. Localization-error emulation (training)

`environment.py` now separates three pose sources and corrupts only the policy
observation:

| source | param | feeds |
|---|---|---|
| ground truth | `gt_odom_topic` | reward + goal-reached/done geometry |
| localization estimate | `loc_odom_topic` | policy `goal_distance` / `heading_error` (noisy) |
| proprioception | `proprio_odom_topic` | `actual_speed`, `actual_yaw_rate` |

All three default to the single `odom_topic` (`/odometry`) → with no extra config
behaviour is identical to before.

The localization noise is configured in the `localization:` block of
`environment.yaml` / `environment_curriculum.yaml`:

```yaml
localization:
  enabled: false            # master switch (OFF = ground truth)
  sigma_xy_m / sigma_yaw_rad        # per-step Gaussian noise
  bias_xy_m / bias_yaw_rad          # per-EPISODE constant bias (std)
  random_walk_xy_mps / _yaw_rps     # accumulating drift
  delay_steps                       # localization latency (RL steps)
  jump_prob / jump_xy_m / jump_yaw_rad   # rare pose jumps
  use_gt_for_reward: true           # keep reward/done on ground truth (recommended)
  use_gt_for_done:   true
```

**Defaults are conservative (off).** Reward/done stay on ground truth for stable
early training; flip `use_gt_for_*` to study estimated-pose reward/done.

**Curriculum ramp** (`environment_curriculum.yaml`, per-stage `localization:`):
Stage 0–1 off → Stage 2 gaussian + 1-step delay → Stage 3 drift + delay →
Stage 4 drift + delay + rare jump. Each stage is applied as *base + override*;
omitting the block inherits the base (no cross-stage leakage). All per-episode
noise state (bias, random walk, latency buffer, jump) is reset every episode and
the latency buffer is padded with the initial pose.

---

## 2. Real-robot inference node — `real_policy_runner.py`

Drives the robot toward a goal from real topics (no `/step` `/reset` services).

### Required topics
| topic (param) | type | purpose |
|---|---|---|
| `scan_topic` (`/scan`) | `LaserScan` | obstacle observation (front-180° → 80 bins) |
| `proprio_odom_topic` (`/odometry`) | `Odometry` | actual speed, yaw rate |
| `joint_states_topic` (`/hunter_se/joint_states`) | `JointState` | center steering angle |
| `goal_topic` (`/goal`) | `PoseStamped` | target (in `map` **or** `odom` frame) |
| `cmd_vel_topic` (`/cmd_vel`) | `Twist` (pub) | → prefilter → base |

### Required frames / TF
`goal_distance` / `heading_error` are computed via **TF**: the goal is
transformed from its header frame (`map` or `odom`) into `base_frame`
(`base_footprint` by default; set to `base_link` if that is your training base).
TF lookup failure → safe hold/zero (configurable). Provide a TF tree:
`map → odom → base_footprint → sensors`.

### Safety params
`stale_timeout_sec` (0.5): if `/scan`, proprio odom, or `/joint_states` is older
than this → `stale_action` = `hold` or `zero`. `goal_timeout_sec` (5.0): no fresh
goal → publish zero (watchdog). `stop_at_goal` (true): stop within
`goal_threshold`. Uses **freshness guards** (latest-value cache), not strict sync.

### Run
```bash
# 1) Base driver + odom (wheel+IMU)      2) LiDAR → /scan (same params as training!)
# 3) Localization                        4) goal publisher (RViz "2D Goal Pose" → /goal)
ros2 run drl_agent real_policy_runner.py --ros-args \
  -p actor_path:=<run_dir>/final_models/tqc_agent_seed_0_<date>_actor.pth \
  -p base_frame:=base_footprint \
  -p proprio_odom_topic:=/odometry \
  -p scan_topic:=/scan \
  -p goal_topic:=/goal \
  -p control_rate_hz:=10.0 \
  -p stale_action:=hold
```

> The LiDAR → `/scan` pipeline (height filter, 360°, resolution) MUST match
> training (`environment.yaml`), or the 80-bin observation will not match the
> policy. The node reuses the exact front-180° binning and the shared
> `utils/pure_pursuit.py` action→cmd convention.

---

## 3. Localization options (indoor)

| option | notes |
|---|---|
| **wheel + IMU EKF** (`robot_localization`) | simplest, no infrastructure; drift over distance. Goal in `odom`. Good first choice when an IMU is present. |
| **LIO-SAM** (in repo) | LiDAR+IMU SLAM → drift-corrected pose. Recommended with Ouster+IMU; needs IMU↔LiDAR extrinsic calibration. Goal in `map`. |
| **KISS-ICP / slam_toolbox / AMCL** | alternative LiDAR localization (AMCL needs a prebuilt map). |
| external (UWB/mocap/RTK) | most accurate; needs infrastructure. |

Whatever the source, publish it as the node's `proprio_odom_topic` (speed/yaw)
and keep a TF `goal_frame → base_frame`. Match units/sign/frame to training
(`heading` = robot-front-relative, CCW +, `[-π, π]`).

---

## 4. Recommended bring-up

1. **Low speed first**: lower `controller_cruise_speed_mps` (env config) and test
   in open space; verify `/cmd_vel` sane and steering not saturating.
2. **E-stop** wired to the base; keep `stale_action: zero` for the first runs so a
   sensor dropout halts the robot.
3. Confirm `/scan` bin count/orientation, then `goal` TF, then closed-loop driving.
4. Only then enable harder localization (longer SLAM runs, faster speed).
