# State / Action Reference

정책이 보는 **state(87차원)**와 내는 **action(2차원)**의 정확한 정의표. (개념 설명은 [../overview/training_pipeline.md](../overview/training_pipeline.md))

## State — 87D
`state = [ obs_state(80) | agent_state(7) ]`

| 인덱스 | 내용 | 단위 | 비고 |
|--|--|--|--|
| `[0:80]` | LiDAR 80빈, 빈당 최근접 장애물 거리 (전방 180°, `obs_state`) | m | 충돌 판정엔 별도 360° `environment_state` 사용 |
| `[80]` | 목표까지 거리 | m | localization noise 주입 대상 |
| `[81]` | 목표 방향 오차 θ | rad | localization noise 주입 대상 |
| `[82]` | 이전 action r (정규화) | — | |
| `[83]` | 이전 action θ (정규화) | — | |
| `[84]` | 실제 선속도 (odom) | m/s | proprio noise 대상(옵션) |
| `[85]` | 실제 요레이트 (odom) | rad/s | proprio noise 대상(옵션) |
| `[86]` | 중심 조향각 (joint state) | rad | proprio noise 대상(옵션) |

- 차원 분리: `environment_state_dim=80`, `agent_state_dim=7` (`config/environment*.yaml`).
- LiDAR 거리는 `lidar_max_range`(50 m)로 클램프.

## Action — 2D (Pure Pursuit waypoint)
| 인덱스 | 내용 | 물리 범위 |
|--|--|--|
| `action[0]` | waypoint 거리 r (전진, 로봇 프레임) | [0.8, 2.0] m |
| `action[1]` | waypoint 각도 θ (로봇 프레임) | [-0.524, 0.524] rad (±30°) |

- 정책은 `[-1, 1]` 정규화 action을 내고, `environment.py`가 `actions_low/high`(config)로 물리 단위 스케일 후 **Pure Pursuit**로 추종해 `cmd_vel` 생성.
- `GetDimensions.srv`는 `state_dim, action_dim, max_action, environment_dim, agent_dim` 반환.

## aux 라벨 (학습 전용, state에 미포함)
aux prediction이 켜지면 env가 state 뒤에 **미래 위험 라벨**을 임시로 덧붙여 보내고, trainer가 떼어내 보조 손실에 쓴다. **정책 입력 87D는 그대로**다. → [../design/aux_prediction_design.md](../design/aux_prediction_design.md)

## Where in code
- state 조립: `environment/environment.py` (`get_obs_state`, `_rebuild_agent_state`, `step_callback`)
- action→waypoint→cmd: `environment/environment.py::_map_action_to_waypoint` + `pure_pursuit.py`
- 차원/범위 설정: `config/environment.yaml`, `config/environment_curriculum.yaml`
