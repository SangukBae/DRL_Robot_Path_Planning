# ugv_gazebo_sim

## Overview

AgileX UGV(Scout Mini, Bunker) Gazebo Ignition 시뮬레이션 패키지.
Scout Mini (4륜 스키드 스티어링)와 Bunker (트랙형) 로봇 모델을 포함한다.

이 패키지는 기본 로봇 시뮬레이션용이며, 현재 프로젝트는 `agilex_scout` 패키지의 Scout V2를 주로 사용한다.

## Quick Start

```bash
# 1) Scout Mini 시뮬레이션
ros2 launch scout_gazebo_sim scout_mini_empty_world.launch.py use_rviz:=true

# 2) Bunker 시뮬레이션
ros2 launch bunker_gazebo_sim bunker_empty_world.launch.py use_rviz:=true

# 3) 초기 위치 지정
ros2 launch scout_gazebo_sim scout_mini_empty_world.launch.py x_pose:=1.0 y_pose:=2.0
```

## Interfaces

### Topics (Scout Mini)

| ROS 토픽 | 타입 | 설명 |
|---------|------|------|
| `/scout_mini/cmd_vel` | `Twist` | 속도 명령 |
| `/scout_mini/odom` | `Odometry` | 오도메트리 |
| `/scout_mini/scan` | `LaserScan` | LiDAR 스캔 |
| `/scout_mini/joint_states` | `JointState` | 휠 조인트 상태 |

### Topics (Bunker)

| ROS 토픽 | 타입 | 설명 |
|---------|------|------|
| `/bunker/cmd_vel` | `Twist` | 속도 명령 |
| `/bunker/odom` | `Odometry` | 오도메트리 |

### TF Frames

`odom` → `base_footprint` → `base_link` → `*_wheel_link`

## Configuration

### Scout Mini

| 파일 | 역할 |
|------|------|
| `scout/scout_description/urdf/scout_mini.xacro` | Scout Mini URDF |
| `scout/scout_gazebo_sim/config/scout_mini_bridge_ros_gz.yaml` | ROS-Gazebo 브릿지 |
| `scout/scout_gazebo_sim/worlds/empty.world` | 시뮬레이션 월드 |

### Bunker

| 파일 | 역할 |
|------|------|
| `bunker/bunker_description/urdf/bunker.xacro` | Bunker URDF |
| `bunker/bunker_gazebo_sim/config/bunker_bridge_ros_gz.yaml` | ROS-Gazebo 브릿지 |

**Scout Mini 사양:**
- 크기: 612×580×245mm
- 질량: 23kg
- 휠 반지름: 0.08m
- 트랙 폭: 0.49m
- 구동: 4륜 스키드 스티어링

**Bunker 사양:**
- 질량: 22kg (본체)
- 구동: 트랙형 (4×4 롤러 어셈블리)

## Dependencies / Assumptions

### 의존성

- `ros_gz_bridge`, `ros_gz_sim`
- `robot_state_publisher`, `xacro`
- `twist_mux`

### 전제조건

- Gazebo Ignition Fortress 설치
- 현재 프로젝트에서는 `agilex_scout` 패키지의 Scout V2를 주로 사용
- Scout Mini / Bunker는 별도 테스트용

## Troubleshooting

| 증상 | 조치 |
|------|------|
| 로봇 스폰 안됨 | `ign gazebo` 실행 확인, world 파일 경로 확인 |
| 브릿지 토픽 없음 | `ros_gz_bridge` 노드 실행 확인 |
| 휠 회전 안됨 | `/cmd_vel` 토픽에 메시지 발행 확인 |

## 이 README에서 다루지 않음

- Scout V2 시뮬레이션: `agilex_scout` 패키지 README 참고 (현재 프로젝트 주력)
- Nav2 네비게이션: `scout_nav2` 패키지 README 참고
- DRL 학습: `drl_agent` 패키지 README 참고
