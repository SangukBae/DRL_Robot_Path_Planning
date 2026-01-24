# scout_nav2

## Overview

Scout 로봇 Nav2 네비게이션 스택 메타패키지.
4개 서브패키지로 구성: `agilex_scout`, `scout_nav2`, `nav2_bringup`, `aws-robomaker-small-warehouse-world`.

Gazebo Ignition 시뮬레이션과 Nav2 자율 주행을 지원한다.

## Quick Start

```bash
# 1) Gazebo 시뮬레이션 (Hospital world, RGL LiDAR)
ros2 launch agilex_scout simulate_control_gazebo_ignition.launch.py rviz:=true

# 2) Gazebo 시뮬레이션 (Warehouse world)
ros2 launch agilex_scout simulate_control_gazebo.launch.py lidar_type:=3d rviz:=true

# 3) Nav2 + AMCL 로컬라이제이션
ros2 launch scout_nav2 nav2.launch.py simulation:=true slam:=false localization:=amcl

# 4) Nav2 + SLAM 매핑
ros2 launch scout_nav2 nav2.launch.py simulation:=true slam:=true

# 5) Nav2 + SLAM Toolbox 로컬라이제이션
ros2 launch scout_nav2 nav2.launch.py simulation:=true slam:=false localization:=slam_toolbox

# 6) 실제 로봇 + Ouster LiDAR
ros2 launch agilex_scout scout_robot_lidar.launch.py
```

## Interfaces

### Topics (브릿지)

| ROS 토픽 | Gazebo 토픽 | 타입 | 설명 |
|---------|------------|------|------|
| `/cmd_vel` | `/scout/cmd_vel` | `Twist` | 속도 명령 |
| `/odometry` | `/scout/odometry` | `Odometry` | 오도메트리 |
| `/laser_scan` | `/scout/laser_scan` | `LaserScan` | 2D 스캔 |
| `/points` | `/scout/pointcloud/points` | `PointCloud2` | 3D 포인트클라우드 |
| `/depth_camera/*` | `/scout/depth_camera/*` | `Image, CameraInfo` | RGBD 카메라 |

### TF Frames

`world` → `map` → `odom` → `base_footprint` → `base_link` → `mobile_robot_base_link`

### Nav2 Services

- `/navigate_to_pose` - 단일 목표 네비게이션
- `/navigate_through_poses` - 다중 웨이포인트

## Configuration

### agilex_scout

| 파일 | 역할 |
|------|------|
| `urdf/mobile_robot/scout_v2.urdf.xacro` | Scout V2 로봇 URDF |
| `urdf/mobile_robot/scout_v2.gazebo` | Gazebo 플러그인 설정 |
| `config/ros2_gz_bridge_config.yaml` | ROS-Gazebo 토픽 브릿지 |

### scout_nav2

| 파일 | 역할 |
|------|------|
| `params/sim_lidar3d_amcl.yaml` | 3D LiDAR + AMCL 파라미터 |
| `params/sim_slam_localization.yaml` | SLAM Toolbox 파라미터 |
| `params/bt_nav2pose.xml` | 네비게이션 Behavior Tree |
| `maps/warehouse/` | Warehouse 맵 파일 |

**주요 파라미터:**
- 로봇 풋프린트: 0.45m × 0.35m
- MPPI 최대 속도: 0.5 m/s (선속도), 1.0 rad/s (각속도)
- 전역 플래너: Hybrid-A* (SmacPlannerHybrid)
- 로컬 코스트맵: 8×8m, 해상도 0.05m

## Dependencies / Assumptions

### 의존성

- `navigation2`, `nav2_bringup`, `slam_toolbox`
- `ros_gz_bridge`, `ros_gz_sim`
- `ouster_ros` (실제 로봇)
- `pointcloud_to_laserscan`

### 전제조건

- Gazebo Ignition Fortress 설치
- Hospital world 사용 시 RGL 플러그인 필요
- `ouster_description` 패키지 빌드 필요 (OS1-64 LiDAR)

## Troubleshooting

| 증상 | 조치 |
|------|------|
| `/points` 토픽 없음 | RGL 플러그인 설치 확인, world 파일에 `RGLServerPluginManager` 포함 여부 확인 |
| AMCL 초기화 실패 | `/map` 토픽 발행 확인, 초기 위치 파라미터 점검 |
| MPPI 컨트롤러 진동 | `vx_std`, `wz_std` 파라미터 조정 |
| 코스트맵 장애물 미반영 | observation_sources 토픽명 확인 |

## 이 README에서 다루지 않음

- Nav2 파라미터 상세 튜닝: `params/*.yaml` 파일 직접 참고
- URDF 센서 설정 상세: `ouster_description`, `depth_d455` 패키지 README 참고
- DRL 학습/테스트: `drl_agent` 패키지 README 참고
