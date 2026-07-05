# 실로봇 배포 & 위치추정 오차 강건성 (Real-Robot Deployment)

(a) 실제 localization 오차를 견디는 정책을 학습하는 방법과, (b) 학습된 actor를 실제 Hunter SE에서
`real_policy_runner.py`로 구동하는 방법을 설명한다.

학습된 정책은 **mapless**다 — LiDAR로 장애물을 피하며, *목표가 자신에 대해 어디에 있는지*만 알면 된다.
하드웨어에서 이 상대 pose는 **localization(odometry / SLAM)**에서 오는데 오차가 있다 — 그래서 시뮬에서
그 오차를 emulate하고, 실로봇용 추론 노드를 함께 제공한다.

---

## 1. Localization 오차 emulation (학습)

`environment.py`는 세 pose 소스를 분리하고, 정책의 관측만 오염시킨다:

| 소스 | param | 쓰임 |
|---|---|---|
| ground truth | `gt_odom_topic` | reward + goal-reached/done 판정 기하 |
| localization estimate | `loc_odom_topic` | 정책의 `goal_distance` / `heading_error`(noisy) |
| proprioception | `proprio_odom_topic` | `actual_speed`, `actual_yaw_rate` |

셋 다 기본값이 단일 `odom_topic`(`/odometry`)이라 → 추가 설정 없이는 이전과 동작이 동일하다.

localization noise는 `environment.yaml` / `environment_curriculum.yaml`의 `localization:` 블록에서 설정한다:

```yaml
localization:
  enabled: false            # master switch (OFF = ground truth)
  sigma_xy_m / sigma_yaw_rad        # 스텝별 Gaussian noise
  bias_xy_m / bias_yaw_rad          # episode별 상수 bias (std)
  random_walk_xy_mps / _yaw_rps     # 누적 drift
  delay_steps                       # localization latency (RL 스텝)
  jump_prob / jump_xy_m / jump_yaw_rad   # 드문 pose 점프
  use_gt_for_reward: true           # reward/done을 ground truth로 유지 (권장)
  use_gt_for_done:   true
```

**기본값은 보수적(off)이다.** reward/done은 초반 학습 안정을 위해 ground truth를 유지한다.
추정 pose 기반 reward/done을 연구하려면 `use_gt_for_*`를 뒤집는다.

**Curriculum ramp**(`environment_curriculum.yaml`, stage별 `localization_profile:`):
Stage 0–3 `clean` → Stage 4–7 `weak_goal_noise` → Stage 8 `drift_goal_noise` →
Stage 9 `robustness_train`(full correlated noise + bias + drift + delay + 드문 relocalization 점프;
**yaw flip 없음** — flip은 eval 전용 `stress_eval` 프로파일에만 존재). 각 stage는 *base + override*로
적용되고, 블록을 생략하면 base를 상속한다(stage 간 누수 없음). 모든 episode별 noise 상태(bias,
random walk, latency buffer, jump)는 매 episode 리셋되고, latency buffer는 초기 pose로 패딩된다.

---

## 2. 실로봇 추론 노드 — `real_policy_runner.py`

실제 topic으로부터 목표를 향해 로봇을 구동한다(`/step` `/reset` 서비스 없음).

### 필요한 topic
| topic (param) | 타입 | 용도 |
|---|---|---|
| `scan_topic` (`/scan`) | `LaserScan` | 장애물 관측 (전방 180° → 80 bins) |
| `proprio_odom_topic` (`/odometry`) | `Odometry` | 실제 speed, yaw rate |
| `joint_states_topic` (`/hunter_se/joint_states`) | `JointState` | 중심 조향각 |
| `goal_topic` (`/goal`) | `PoseStamped` | 목표 (`map` **또는** `odom` frame) |
| `cmd_vel_topic` (`/cmd_vel`) | `Twist` (pub) | → prefilter → base |

### 필요한 frame / TF
`goal_distance` / `heading_error`는 **TF**로 계산한다: 목표를 header frame(`map` 또는 `odom`)에서
`base_frame`(기본 `base_footprint`; 학습 base가 `base_link`면 그렇게 설정)으로 변환한다. TF lookup
실패 → 안전한 hold/zero(설정 가능). TF tree 필요: `map → odom → base_footprint → sensors`.

### 안전 param
`stale_timeout_sec`(0.5): `/scan`, proprio odom, `/joint_states`가 이보다 오래되면 →
`stale_action` = `hold` 또는 `zero`. `goal_timeout_sec`(5.0): 새 목표가 없으면 → zero 발행(watchdog).
`stop_at_goal`(true): `goal_threshold` 안에서 정지. 엄격한 sync가 아니라 **freshness guard**
(최신값 캐시)를 사용한다.

### 실행
```bash
# 1) Base driver + odom (wheel+IMU)      2) LiDAR → /scan (학습과 동일 param!)
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

> LiDAR → `/scan` 파이프라인(height filter, 360°, resolution)은 학습(`environment.yaml`)과
> **반드시 일치**해야 한다. 아니면 80-bin 관측이 정책과 맞지 않는다. 이 노드는 학습과 동일한 전방
> 180° binning과 공유 `utils/pure_pursuit.py`의 action→cmd 규약을 그대로 재사용한다.

---

## 3. Localization 옵션 (실내)

| 옵션 | 비고 |
|---|---|
| **wheel + IMU EKF** (`robot_localization`) | 가장 단순, 인프라 불필요; 거리에 따라 drift. 목표는 `odom`. IMU가 있으면 첫 선택으로 좋음. |
| **LIO-SAM** (repo 내) | LiDAR+IMU SLAM → drift 보정 pose. Ouster+IMU와 함께 권장; IMU↔LiDAR extrinsic 캘리브레이션 필요. 목표는 `map`. |
| **KISS-ICP / slam_toolbox / AMCL** | 대안 LiDAR localization (AMCL은 사전 지도 필요). |
| external (UWB/mocap/RTK) | 가장 정확; 인프라 필요. |

어떤 소스든 노드의 `proprio_odom_topic`(speed/yaw)으로 발행하고 TF `goal_frame → base_frame`을
유지한다. 학습과 단위/부호/frame을 맞춘다(`heading` = 로봇 전방 기준, CCW +, `[-π, π]`).

---

## 4. 권장 bring-up 순서

1. **저속 먼저**: `controller_cruise_speed_mps`(env config)를 낮추고 열린 공간에서 테스트. `/cmd_vel`이
   정상이고 조향이 포화하지 않는지 확인.
2. base에 **E-stop** 연결; 첫 실행은 `stale_action: zero`로 두어 센서 dropout 시 로봇이 멈추게.
3. `/scan` bin 수/방향 → `goal` TF → closed-loop 주행 순으로 확인.
4. 그다음에야 더 어려운 localization(긴 SLAM run, 빠른 속도)을 활성화.
