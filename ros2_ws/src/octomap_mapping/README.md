octomap_mapping ![CI](https://github.com/OctoMap/octomap_mapping/workflows/CI/badge.svg)
===============

ROS stack for mapping with OctoMap, contains the `octomap_server` package.

The main branch for ROS1 Kinetic, Melodic, and Noetic is `kinetic-devel`.

The main branch for ROS2 Foxy and newer is `ros2`.

### Usage

#### Save octomap

```
ros2 run octomap_server octomap_saver_node --ros-args -p octomap_path:=(path for saving octomap)
```
Note: The extension of octomap path should be `.bt` or `.ot`

### Scout Ignition 파라미터 설명

아래 표는 `octomap_server/params/scout_ignition.yaml`의 각 파라미터가 의미하는 바를 정리한 것입니다.

| 파라미터 | 값 | 의미 |
| --- | --- | --- |
| frame_id | `odom` | OctoMap의 고정 프레임. Ignition에서는 `odom` 기준으로 TF가 제공됨 |
| base_frame_id | `base_footprint` | 센서/로봇 기준 프레임 |
| resolution | `0.05` | OctoMap 해상도(보xel 크기, m) |
| sensor_model.max_range | `20.0` | 센서 최대 측정 거리(m) |
| sensor_model.hit | `0.7` | 히트 시 점유 확률 업데이트 값 |
| sensor_model.miss | `0.4` | 미스 시 점유 확률 업데이트 값 |
| sensor_model.min | `0.12` | 점유 확률 최소 클램프 값 |
| sensor_model.max | `0.97` | 점유 확률 최대 클램프 값 |
| point_cloud_min_z | `-1.0` | 포인트클라우드 z 하한(이 값보다 낮은 점 제거) |
| point_cloud_max_z | `3.0` | 포인트클라우드 z 상한(이 값보다 높은 점 제거) |
| occupancy_min_z | `0.1` | 2D 점유 격자 투영용 z 하한 |
| occupancy_max_z | `2.0` | 2D 점유 격자 투영용 z 상한 |
| filter_ground_plane | `true` | 지면 평면 필터링 사용 여부 |
| ground_filter.distance | `0.1` | 지면 검출 거리 임계값(m) |
| ground_filter.angle | `0.15` | 수평면 대비 각도 임계값 |
| ground_filter.plane_distance | `0.1` | 지면 평면에서 허용되는 최대 거리(m) |
| use_height_map | `true` | 시각화에서 높이 기반 컬러맵 사용 |
| colored_map | `false` | 포인트클라우드 색상 사용 여부 |
| filter_speckles | `false` | 고립된 보xel 제거 여부 |
| compress_map | `true` | 퍼블리시 시 맵 압축 여부 |
| publish_free_space | `false` | 자유 공간 마커 퍼블리시 여부 |
| latch | `true` | 맵 토픽 래치 사용 여부 |
