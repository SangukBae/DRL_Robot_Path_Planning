# DRL Robot Path Planning

ROS2 Humble + Gazebo Ignition Fortress 기반 **AgileX Hunter SE** 자율주행 경로계획 시뮬레이션.

TQC 커리큘럼 강화학습으로 Ackermann 조향 로봇이 장애물/보행자 혼합 환경에서 목표 지점까지 자율 주행을 학습한다.

**Tech Stack**: ROS2 Humble · Gazebo Ignition Fortress · PyTorch 2.4.1 (CUDA 11.8) · Python 3.10

---

## Prerequisites

| 항목 | 버전 |
|------|------|
| Ubuntu | 22.04 |
| ROS2 | Humble |
| Gazebo | Ignition Fortress |
| CUDA | 11.8 |
| Python | 3.10 |

> RGL LiDAR 플러그인 설치 필요 → [Installation](docs/installation.md#rgl-lidar-plugin)

---

## Quick Install

```bash
# 1. 의존성 설치
cd ros2_ws
rosdep install --from-paths src -yi --rosdistro humble \
  --skip-keys='libgraphicsmagick++1-dev graphicsmagick-libmagick-dev-compat'

# 2. 빌드
source /opt/ros/humble/setup.bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# 3. 소스
source install/setup.bash

# 4.
xhost +local:
```

---

## Quick Start

```bash
# [터미널 1] Gazebo 시뮬레이션
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false

# [터미널 2] 커리큘럼 환경 노드
ros2 run drl_agent environment_curriculum.py

# [터미널 3] TQC 커리큘럼 학습
ros2 run drl_agent train_tqc_curriculum_agent.py

# TensorBoard 모니터링
tensorboard --logdir <run_dir>/logs
```

---

## Documentation

| 문서 | 내용 |
|------|------|
| [Installation](docs/installation.md) | 상세 빌드, Docker, RGL LiDAR 설치 |
| [Training](docs/training.md) | 커리큘럼 학습, 재개 방법 |
| [Architecture](docs/architecture.md) | 서비스 인터페이스, 상태/액션 공간, 파이프라인 |
| [Algorithms](docs/algorithms.md) | TQC, TD7, SAC, A3C 알고리즘 설명 |
| [Configuration](docs/configuration.md) | config 파일별 파라미터 레퍼런스 |
| [Troubleshooting](docs/troubleshooting.md) | 환경 변수, 디버깅, 흔한 오류 |

---

## Repository Layout

```
/
├── ros2_ws/src/
│   ├── drl_agent/                          # DRL 환경/정책/학습 스크립트
│   ├── drl_agent_interfaces/               # ROS2 srv/action 정의
│   ├── hunter_se_gazebo/                   # Hunter SE URDF, Gazebo launch, worlds
│   ├── drl_obstacle_assets/                # Gazebo 장애물 모델 라이브러리 (38종)
│   ├── ouster_simulation/ouster_description/  # OS1-64 LiDAR (RGL)
│   ├── aws-robomaker-hospital-world/
│   ├── aws-robomaker-bookstore-world/
│   ├── aws-robomaker-small-house-world/
│   └── aws-robomaker-small-warehouse-world/
├── docs/                                   # 상세 문서
├── Dockerfile
└── cyclonedds_config.xml
```

---

## License

MIT
