# Architecture

## 서비스 기반 환경 인터페이스

에이전트와 환경 노드는 ROS2 서비스로 통신한다. 토픽 대신 서비스를 사용하므로 동기적 스텝 제어가 가능하다.

```
Environment Node (environment_curriculum.py)   Agent Node (train_tqc_curriculum_agent.py)
├── /reset               ←─────────────────── 에피소드 시작 시 초기화
├── /step                ←─────────────────── 액션(Waypoint) 전달 + 상태/보상 반환
├── /get_dimensions      ←─────────────────── state_dim, action_dim 조회
├── /seed                ←─────────────────── 랜덤 시드 설정
└── /action_space_sample ←─────────────────── 랜덤 액션 샘플 (워밍업)
```

서비스 정의: `drl_agent_interfaces/srv/`

`GetDimensions.srv`는 `state_dim`, `action_dim`, `max_action`, `environment_dim`, `agent_dim` 5개 필드를 반환한다.

---

## 상태/액션 공간

### State (87D)

| 인덱스 | 내용 | 단위 |
|--------|------|------|
| `[0:80]` | LiDAR 80 빈 (전방 180°, `obs_state`), 빈당 최근접 장애물 거리 | m |
| `[80]` | 목표까지 거리 | m |
| `[81]` | 목표 방향 오차 θ | rad |
| `[82]` | 이전 액션 r (웨이포인트 거리), 정규화 | — |
| `[83]` | 이전 액션 θ (웨이포인트 각도), 정규화 | — |
| `[84]` | 실제 선속도 (오도메트리) | m/s |
| `[85]` | 실제 요레이트 (오도메트리) | rad/s |
| `[86]` | 중심 조향각 (조인트 스테이트) | rad |

### Action (2D) — Waypoint 명령 (Pure Pursuit)

| 인덱스 | 내용 | 범위 |
|--------|------|------|
| `action[0]` | 웨이포인트 거리 r (전진, 로봇 프레임) | [0.8, 2.0] m |
| `action[1]` | 웨이포인트 각도 θ (로봇 프레임) | [-0.524, 0.524] rad (±30°) |

정책 출력은 `[-1, 1]`로 정규화되며 `environment.py`가 물리 단위로 스케일 변환 후 Pure Pursuit 컨트롤러를 구동해 `cmd_vel`을 생성한다.

---

## 커맨드 파이프라인

```
RL policy
  → /cmd_vel (Twist)
  → hunter_se_cmd_prefilter  (스로틀/조향 셰이핑, 50 Hz, use_sim_time=false)
  → /cmd_vel_filtered
  → ros_gz_bridge
  → Gazebo (/hunter_se/cmd_vel)
```

프리필터는 벽시계 50 Hz로 동작하므로 시뮬레이션 RTF가 낮아도 지속 실행된다.

---

## LiDAR 파이프라인

```
Gazebo Ouster RGL 플러그인
  → /hunter_se/pointcloud/points  (Ignition 내부 토픽)
  → ros_gz_bridge
  → /ouster/points                (ROS2 PointCloud2, ~10 Hz)
  → pointcloud_to_laserscan
      height filter: z ∈ [-0.455, 0.250] m (센서 프레임 기준)
      = 지상 약 0.045–0.850 m
  → /scan                         (ROS2 LaserScan, 360°, 0.176°/bin)
  → environment.py
      environment_state: 360° (충돌 판정용)
      obs_state: 전방 180° (RL 관측 입력, 80 빈)
```

Height filter 파라미터: `environment.yaml`의 `obs_z_min_sensor_m`, `obs_z_max_sensor_m`.

### 주요 토픽 Hz

| 토픽 | 목표 Hz |
|------|--------|
| `/ouster/points` | ~10 |
| `/scan` | ~10 |
| `/odometry` | ~50 |

---

## 환경 구현

| 파일 | 시뮬레이터 | 리셋 방식 |
|------|-----------|---------|
| `environment.py` | Ignition Fortress | `ros_gz_interfaces/SetEntityPose` |
| `environment_curriculum.py` | Ignition Fortress | 동일 + 커리큘럼 스테이지 관리 |
| `environment_360.py` | Classic Gazebo | `gazebo_msgs/SetEntityState` |

실행 중인 시뮬레이터에 맞는 환경 파일을 사용해야 한다. 세 파일 모두 동일한 ROS2 서비스를 노출한다.

---

## 패키지 구성

| 패키지 | 경로 | 설명 |
|--------|------|------|
| `drl_agent` | `src/drl_agent/` | DRL 환경, 정책, 학습/테스트 스크립트 |
| `drl_agent_interfaces` | `src/drl_agent_interfaces/` | ROS2 서비스/액션/메시지 정의 |
| `hunter_se_gazebo` | `src/hunter_se_gazebo/` | Hunter SE URDF, Gazebo launch, worlds |
| `drl_obstacle_assets` | `src/drl_obstacle_assets/` | Gazebo 장애물 모델 라이브러리 (38종) |
| `ouster_description` | `src/ouster_simulation/ouster_description/` | OS1-64 LiDAR (RGL 플러그인) |
| `aws-robomaker-hospital-world` | `src/aws-robomaker-hospital-world/` | Hospital 시뮬레이션 환경 |
| `aws-robomaker-bookstore-world` | `src/aws-robomaker-bookstore-world/` | Bookstore 시뮬레이션 환경 |
| `aws-robomaker-small-house-world` | `src/aws-robomaker-small-house-world/` | Small House 시뮬레이션 환경 |
| `aws-robomaker-small-warehouse-world` | `src/aws-robomaker-small-warehouse-world/` | Small Warehouse 시뮬레이션 환경 |

---

## Gazebo-ROS2 브릿지

브릿지 설정 파일: `hunter_se_gazebo/config/ros2_gz_bridge_config.yaml`

| ROS2 토픽 | 방향 | Gazebo 토픽 |
|-----------|------|-------------|
| `/cmd_vel_filtered` | → | `/hunter_se/cmd_vel` |
| `/odometry` | ← | `/hunter_se/odometry` |
| `/ouster/points` | ← | `/hunter_se/pointcloud/points` |
| (joint states) | ← | `/hunter_se/joint_states` |
