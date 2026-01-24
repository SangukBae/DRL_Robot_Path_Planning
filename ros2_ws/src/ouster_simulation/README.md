# ouster_simulation

## Overview

Ouster OS1-64 LiDAR 시뮬레이션 패키지.
RGL(Robotec GPU Lidar) 플러그인을 사용하여 GPU 가속 3D 포인트클라우드를 생성한다.

Scout 로봇의 주 LiDAR 센서로 사용된다.

## Quick Start

```bash
# 1) OS1-64 단독 테스트
ros2 launch ouster_description os1_64_alone.launch.py gui:=true

# 2) OS1-32 단독 테스트
ros2 launch ouster_description os1_32_alone.launch.py gui:=true

# 3) Scout 로봇에서 사용 (agilex_scout 패키지)
ros2 launch agilex_scout simulate_control_gazebo_ignition.launch.py rviz:=true

# 4) Headless 모드
ros2 launch ouster_description os1_64_alone.launch.py gui:=false
```

## Interfaces

### Topics (발행)

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/points` | `PointCloud2` | 3D 포인트클라우드 (브릿지 후) |
| `/scout/pointcloud/points` | `PointCloud2` | Gazebo 내부 토픽 |

### TF Frames

`mobile_robot_base_link` → `os_lidar`

## Configuration

| 파일 | 역할 |
|------|------|
| `ouster_description/urdf/os1_64.urdf.xacro` | OS1-64 URDF 매크로 |
| `ouster_description/urdf/os1_32.urdf.xacro` | OS1-32 URDF 매크로 |
| `ouster_description/meshes/os1_64.dae` | LiDAR 3D 메시 |
| `ouster_description/worlds/test.world` | 테스트 월드 |

**OS1-64 사양:**
- 채널: 64
- 수평 샘플: 1024 (설정 가능)
- 수직 FOV: -21.2° ~ +21.2°
- Range: 0.5~100m
- Update Rate: 10Hz

**URDF 매크로 파라미터:**

```xml
<xacro:os1_64 parent="mobile_robot_base"
              name="os"
              hz="10"
              samples="1024"
              min_range="0.5"
              max_range="100.0"
              min_angle_v="-21.2"
              max_angle_v="21.2"
              noise="0.01">
  <origin xyz="0 0 0.5" rpy="0 0 0"/>
</xacro:os1_64>
```

## Dependencies / Assumptions

### 의존성

- `ros_gz_sim`, `ros_gz_bridge`
- `robot_state_publisher`, `xacro`
- RGL Gazebo Plugin (`RGLServerPluginManager`)

### 전제조건

- Gazebo Ignition Fortress 설치
- RGL 플러그인 설치 및 `RGL_PATTERNS_DIR` 환경변수 설정
- World 파일에 `RGLServerPluginManager` 플러그인 포함 필요

**RGL 플러그인 설정 (world 파일):**

```xml
<plugin name='rgl::RGLServerPluginManager' filename='RGLServerPluginManager'>
  <do_ignore_entities_in_lidar_link>true</do_ignore_entities_in_lidar_link>
</plugin>
```

## Troubleshooting

| 증상 | 조치 |
|------|------|
| `/points` 토픽 없음 | World 파일에 `RGLServerPluginManager` 플러그인 확인 |
| 포인트클라우드 비어있음 | RGL 패턴 파일 경로 확인 (`RGL_PATTERNS_DIR`) |
| 브릿지 토픽 불일치 | `os1_64.urdf.xacro`의 topic 파라미터 확인 |
| LiDAR 위치 이상 | `scout_v2.urdf.xacro`의 `<origin>` 확인 |

## 이 README에서 다루지 않음

- RGL 플러그인 설치: [RGL GitHub](https://github.com/RobotecAI/RGLGazeboPlugin) 참고
- Scout 로봇 통합: `agilex_scout/urdf/mobile_robot/scout_v2.urdf.xacro` 참고
- Gazebo Classic 플러그인: `ouster_gazebo_plugins` (COLCON_IGNORE 상태)
