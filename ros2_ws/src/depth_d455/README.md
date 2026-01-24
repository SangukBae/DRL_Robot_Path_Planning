# depth_d455

## Overview

Intel RealSense D455 RGBD 카메라 시뮬레이션 패키지.
D435 대비 넓은 FOV(87°)와 개선된 depth 정확도를 제공한다.

Scout 로봇(`agilex_scout`)에서 기본 RGBD 센서로 사용된다.

## Quick Start

```bash
# 1) 기본 시뮬레이션 실행
ros2 launch depth_d455 one_robot_ign_launch.py

# 2) 텔레옵 키보드 제어
ros2 run depth_d455 omni_teleop_keyboard.py

# 3) Scout 로봇에서 D455 사용 (agilex_scout 패키지)
ros2 launch agilex_scout simulate_control_gazebo_ignition.launch.py rviz:=true
```

## Interfaces

### Topics (발행)

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/depth_camera/image` | `Image` | RGB 이미지 (1280×720) |
| `/depth_camera/depth_image` | `Image` | Depth 이미지 |
| `/depth_camera/points` | `PointCloud2` | Depth 포인트클라우드 |
| `/depth_camera/camera_info` | `CameraInfo` | 카메라 intrinsic |

### Topics (Scout 로봇 브릿지)

| ROS 토픽 | Gazebo 토픽 |
|---------|------------|
| `/depth_camera/image` | `/scout/depth_camera/image` |
| `/depth_camera/depth_image` | `/scout/depth_camera/depth_image` |
| `/depth_camera/points` | `/scout/depth_camera/points` |

### TF Frames

`front_mount` → `rgbd_camera_frame`

## Configuration

| 파일 | 역할 |
|------|------|
| `meshes/d455.stl` | D455 카메라 3D 메시 |
| `urdf/sensors_diffbot.xacro` | 테스트용 로봇 URDF |

**D455 센서 사양 (Scout 로봇 적용):**
- RGB/Depth: 1280×720, 30Hz
- Horizontal FOV: 87° (1.518 rad)
- Depth Range: 0.52~6.0m (ideal: 0.6~6m)
- Focal Length: fx=fy=674.29

**D435 대비 차이점:**
- FOV: 87° vs 62°
- Depth Min: 0.52m vs 0.1m
- 깊이 정확도 향상

## Dependencies / Assumptions

### 의존성

- `ros_gz_sim`, `ros_gz_bridge`
- `robot_state_publisher`, `xacro`

### 전제조건

- Gazebo Ignition Fortress 설치
- Scout 로봇 사용 시 `agilex_scout` 패키지에서 URDF 통합됨

## Troubleshooting

| 증상 | 조치 |
|------|------|
| depth 이미지 검정 | depth range 내 객체 없음 (0.52~6m 확인) |
| 토픽 안보임 | `ros2_gz_bridge_config.yaml`에 depth_camera 항목 확인 |
| 카메라 위치 이상 | `scout_v2.urdf.xacro`의 `rgbd_camera_joint` origin 확인 |

## 이 README에서 다루지 않음

- Scout 로봇 URDF 설정: `agilex_scout/urdf/mobile_robot/scout_v2.urdf.xacro` 참고
- Gazebo 센서 플러그인 설정: `agilex_scout/urdf/mobile_robot/scout_v2.gazebo` 참고
- 브릿지 설정: `agilex_scout/config/ros2_gz_bridge_config.yaml` 참고
