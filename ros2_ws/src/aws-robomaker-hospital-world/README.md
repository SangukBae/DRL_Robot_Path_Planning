# aws-robomaker-hospital-world

## Overview

AWS RoboMaker Hospital 시뮬레이션 환경 패키지.
1~3층 병원 월드와 80개 모델(의료장비, 가구, 환자 등)을 포함한다.

Gazebo Classic과 Ignition Gazebo 두 버전을 지원한다.

## Quick Start

```bash
# 1) Hospital world (Gazebo Classic)
ros2 launch aws_robomaker_hospital_world hospital.launch.py gui:=true

# 2) Hospital world (Ignition Gazebo)
ros2 launch aws_robomaker_hospital_world hospital_ignition.launch.py

# 3) 시각적 확인용 뷰어
ros2 launch aws_robomaker_hospital_world view_hospital.launch.py

# 4) Scout 로봇과 함께 실행
ros2 launch agilex_scout simulate_control_gazebo_ignition.launch.py rviz:=true
```

## Interfaces

### World Files

| 파일 | 층수 | 용도 |
|------|------|------|
| `worlds/hospital.world` | 1층 | 기본 시뮬레이션 |
| `worlds/hospital_two_floors.world` | 2층 | 다층 네비게이션 |
| `worlds/hospital_three_floors.world` | 3층 | 복잡한 환경 |
| `worlds/hospital_ignition.world` | 1층 | Ignition Gazebo용 |

### Gazebo Plugins (hospital_ignition.world)

| 플러그인 | 역할 |
|---------|------|
| `RGLServerPluginManager` | RGL LiDAR 지원 |
| `gz::sim::systems::Physics` | 물리 시뮬레이션 |
| `gz::sim::systems::Sensors` | 센서 시뮬레이션 |

## Configuration

| 경로 | 역할 |
|------|------|
| `models/` | 커스텀 병원 모델 37개 (벽, 바닥, 커튼 등) |
| `fuel_models/` | Ignition Fuel 모델 43개 (의료장비, 환자 등) |
| `env-hooks/` | Gazebo 리소스 경로 설정 |

**주요 모델:**
- 병원 구조: `hospital_floor_01_*`, `hospital_elevator_*`, `hospital_curtain_*`
- 의료장비: `TrolleyBed`, `IVStand`, `XRayMachine`, `AnesthesiaMachine`
- 가구: `Chair`, `BedsideTable`, `MetalCabinet`

## Dependencies / Assumptions

### 의존성

- `gazebo_ros`, `gazebo`, `gazebo_plugins` (Classic)
- Ignition Gazebo Fortress (Ignition 버전)
- Python: `docopt`, `requests`, `lxml` (Fuel 모델 다운로드용)

### 전제조건

- RGL LiDAR 사용 시 `hospital_ignition.world`에 `RGLServerPluginManager` 플러그인 필요
- Fuel 모델은 `setup.sh` 또는 `fuel_utility.py`로 사전 다운로드 필요
- 환경변수 `IGN_GAZEBO_RESOURCE_PATH`에 모델 경로 포함 필요

## Troubleshooting

| 증상 | 조치 |
|------|------|
| 모델 로드 실패 | `IGN_GAZEBO_RESOURCE_PATH` 환경변수 확인, `source install/setup.bash` 실행 |
| Fuel 모델 없음 | `python3 fuel_utility.py download -m [MODEL] -d fuel_models` 실행 |
| Ignition 실행 안됨 | `ign gazebo --version` 확인, Fortress 버전 필요 |
| RGL LiDAR 동작 안함 | world 파일에 `RGLServerPluginManager` 플러그인 추가 여부 확인 |

## 이 README에서 다루지 않음

- Fuel 모델 상세 목록: `fuel_models/database.config` 참고
- 로봇 스폰 및 네비게이션: `agilex_scout`, `scout_nav2` 패키지 README 참고
- World 파일 커스터마이징: `worlds/*.world` 파일 직접 편집
