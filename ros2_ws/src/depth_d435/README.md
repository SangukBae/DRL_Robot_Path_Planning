# depth_d435

## Overview

Intel RealSense D435 RGBD 카메라 시뮬레이션 패키지.
Differential drive 로봇에 D435 카메라, LiDAR, IMU를 장착한 테스트용 로봇 모델을 포함한다.

Gazebo Ignition 환경에서 동작한다.

## Quick Start

```bash
# 1) 기본 시뮬레이션 실행
ros2 launch depth_d435 one_robot_ign_launch.py

# 2) 텔레옵 키보드 제어
ros2 run depth_d435 omni_teleop_keyboard.py

# 3) 특정 world 지정
ros2 launch depth_d435 one_robot_ign_launch.py world:=warehouse.sdf
```

## Interfaces

### Topics (발행)

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/camera/image_raw` | `Image` | RGB 이미지 (1920×1080) |
| `/depth_camera/depth_image` | `Image` | Depth 이미지 (1280×720) |
| `/depth_camera/points` | `PointCloud2` | Depth 포인트클라우드 |
| `/scan` | `LaserScan` | 2D LiDAR 스캔 |
| `/odom` | `Odometry` | 오도메트리 |

### Topics (구독)

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/cmd_vel` | `Twist` | 속도 명령 |

### TF Frames

`base_footprint` → `base_link` → `camera_link` → `camera_depth_link`

## Configuration

| 파일 | 역할 |
|------|------|
| `urdf/sensors_diffbot.xacro` | 로봇 + 센서 URDF 정의 |
| `meshes/d435.dae` | D435 카메라 3D 메시 |
| `worlds/*.sdf` | 시뮬레이션 월드 (empty, warehouse) |
| `rviz/robot_display.rviz` | RViz 설정 |

**D435 센서 사양:**
- RGB: 1920×1080, 30Hz
- Depth: 1280×720, FOV 62°
- Range: 0.1~10m

**로봇 사양:**
- 휠 간격: 0.13m
- 휠 지름: 0.07m
- 최대 속도: 0.5 m/s (선속도), 1.0 rad/s (각속도)

## Dependencies / Assumptions

### 의존성

- `ros_gz_sim`, `ros_gz_bridge`
- `robot_state_publisher`, `xacro`
- `rviz2`

### 전제조건

- Gazebo Ignition Fortress 설치
- 이 패키지는 독립 테스트용으로, Scout 로봇과는 별개

## Troubleshooting

| 증상 | 조치 |
|------|------|
| 카메라 이미지 없음 | `ros2 topic list`에서 `/camera/image_raw` 확인 |
| depth 데이터 노이즈 | `<noise>` 파라미터 조정 (URDF) |
| 로봇 움직이지 않음 | `/cmd_vel` 토픽에 메시지 발행 확인 |

## 이 README에서 다루지 않음

- Scout 로봇에 D435 장착: `agilex_scout` 패키지에서 D455 사용 (D435 대신)
- 상세 URDF 파라미터: `urdf/sensors_diffbot.xacro` 직접 참고
