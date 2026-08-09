# ROS Interface Reference

환경↔에이전트 **서비스**, **토픽**, **Gazebo 브릿지**, **파이프라인** 정확한 표.

## 서비스 (환경 노드가 노출)
| 서비스 | 용도 |
|--|--|
| `/reset` | 에피소드 초기화 + 초기 state 반환 |
| `/step` | action 전달 → (state, reward, done) 반환 |
| `/get_dimensions` | `state_dim, action_dim, max_action, environment_dim, agent_dim` |
| `/seed` | 랜덤 시드 설정 |
| `/action_space_sample` | 랜덤 action 샘플(워밍업) |

- 정의: `ros2_ws/src/drl_agent_interfaces/srv/`
- 토픽이 아닌 **서비스**라 한 스텝씩 동기 제어 가능.
- 커리큘럼/평가 제어용 ROS 파라미터: `curriculum_stage`(쓰기), `curriculum_num_stages`(읽기), `current_map_type`(읽기), `curriculum_eval_mode`(쓰기).

## 커맨드 파이프라인
```
policy → /cmd_vel(Twist) → hunter_se_cmd_prefilter (50Hz, use_sim_time=false)
       → /cmd_vel_filtered → ros_gz_bridge → Gazebo (/hunter_se/cmd_vel)
```

## LiDAR 파이프라인
```
Gazebo Ouster RGL → /hunter_se/pointcloud/points → ros_gz_bridge
  → /ouster/points (PointCloud2, ~10Hz)
  → pointcloud_to_laserscan (height filter z∈[-0.455,0.250] m, 센서 프레임)
  → /scan (LaserScan, 360°, 0.176°/bin)
  → environment.py (obs_state 전방 180° 80빈[정책 입력] + environment_state 360° 80빈[충돌 판정])
```

## Gazebo–ROS2 브릿지
설정: `hunter_se_gazebo/config/ros2_gz_bridge_config.yaml`
| ROS2 토픽 | 방향 | Gazebo 토픽 |
|--|--|--|
| `/cmd_vel_filtered` | → | `/hunter_se/cmd_vel` |
| `/odometry` | ← | `/hunter_se/odometry` |
| `/ouster/points` | ← | `/hunter_se/pointcloud/points` |
| joint states | ← | `/hunter_se/joint_states` |

서비스 브릿지가 `/world/default/...`를 하드코딩하므로, world의 SDF `<world name="default">` 여야 호환된다.

## 토픽 Hz (정상 동작 기준)
| 토픽 | 목표 Hz |
|--|--|
| `/ouster/points` | ~10 |
| `/scan` | ~10 |
| `/odometry` | ~50 |

## Where in code
- 환경 서비스: `env/simulation/environment.py`
- 서비스 클라이언트(에이전트): `env/environment_interface.py`
- 프리필터: `hunter_se_gazebo/scripts/hunter_se_cmd_prefilter.py`
- launch: `hunter_se_gazebo/launch/simulate_hunter_se_ignition.launch.py`
