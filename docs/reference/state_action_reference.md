# State / Action Reference

정책이 보는 **state(기본 87차원)**와 profile별 **action 계약**의 정확한 정의표. 기본 curriculum은
3D waypoint_yield을 쓴다.
(개념 설명은 [../overview/training_pipeline.md](../overview/training_pipeline.md))

## State — 87D
`state = [ obs_state(80) | agent_state(7) ]`

| 인덱스 | 내용 | 단위 | 비고 |
|--|--|--|--|
| `[0:80]` | LiDAR 80빈, 빈당 최근접 장애물 거리 (전방 180°, `obs_state`) | m | 충돌 판정엔 별도 360° `environment_state` 사용 |
| `[80]` | 목표까지 거리 | m | localization noise 주입 대상 |
| `[81]` | 목표 방향 오차 θ | rad | localization noise 주입 대상 |
| `[82]` | 이전 action 0축 (정규화) | — | waypoint mode에서는 r, speed_steering에서는 speed |
| `[83]` | 이전 action 1축 (정규화) | — | waypoint mode에서는 θ, speed_steering에서는 steering |
| `[84]` | 실제 선속도 (odom) | m/s | proprio noise 대상(옵션) |
| `[85]` | 실제 요레이트 (odom) | rad/s | proprio noise 대상(옵션) |
| `[86]` | 중심 조향각 (joint state) | rad | proprio noise 대상(옵션) |

- 차원 분리: `environment_state_dim=80`, `agent_state_dim=7` (`config/environment*.yaml`).
- LiDAR 거리는 `lidar_max_range`(50 m)로 클램프.
- 이전 action 슬롯 `[82]`/`[83]`은 **주행 2축만** 담는다. `waypoint_yield`의 yield 축은 state로 되먹이지 않는다.

### 시간 맥락(temporal context) — 옵션, 커리큘럼 기본 ON
`observation_time_context`가 켜지면 최근 `obs_frame_stack`개 `obs_state`(전방 180° 80빈)를 쌓아
state가 `80×N + 7`이 된다(기본 N=4 → **327D**). 현재 87D 프레임이 맨 앞이라 위 표는 그대로 유효하고
뒤에 과거 스캔만 붙는다(에이전트가 압축해 사용). 꺼지면 87D로 baseline과 동일.

## Action modes

`environment.action_mode`가 있으면 그 값을 우선한다. 없으면 legacy 규칙으로 `action_dim>=3`은
`waypoint_yield`, 그 외는 `waypoint`로 해석한다.

### `waypoint_yield` — 3D (하이브리드 stop/yield, Pure Pursuit)
| 인덱스 | 내용 | 물리 범위 |
|--|--|--|
| `action[0]` | 전진 waypoint 거리 r (로봇 프레임) | [0.0, 2.0] m |
| `action[1]` | waypoint 각도 θ (로봇 프레임) | [-0.524, 0.524] rad (±30°) |
| `action[2]` | yield(정지) 스칼라 | [-1, 1] (`≥ action_threshold`이면 YIELD) |

- **MOVE 모드**(yield 축 < 임계값): 회피/주행. r은 `controller_lookahead_min_m`(0.8 m), 전진 속도는
  `controller_v_move_min_mps`(0.35 m/s)로 바닥값 보정 → **MOVE 중 정지 불가**.
- **YIELD 모드**(`action[2] ≥ action_threshold`, 기본 0.3): 전진 속도를 `controller_yield_creep_mps`
  (기본 0.0 → **완전 정지**)로 제한(조향은 유지). → 회피와 정지를 분리한 "브레이크 달린 차".
- 정책이 `[-1,1]`을 내면 `pure_pursuit.action_to_waypoint`(물리 스케일) → `hybrid_action_to_command`(Pure Pursuit)로 `cmd_vel` 생성.
- yield 축은 **Stage 0–4 봉인**(`yield_reward.action_enabled=false`), **Stage 5 해제**. → [curriculum_design](../design/curriculum_design.md)
- 비커리큘럼 baseline(`environment.yaml`)은 **2D**(yield 없음, ablation용).

### `waypoint` — 2D (legacy ablation)
| 인덱스 | 내용 | 물리 범위 |
|--|--|--|
| `action[0]` | 전진 waypoint 거리 r (로봇 프레임) | [0.0, 2.0] m |
| `action[1]` | waypoint 각도 θ (로봇 프레임) | [-0.524, 0.524] rad (±30°) |

yield 축이 없으므로 항상 MOVE 계약이다. 전진 속도는 `controller_v_move_min_mps` 바닥값 보정을 받는다.

### `speed_steering` — 2D
| 인덱스 | 내용 | 물리 범위 |
|--|--|--|
| `action[0]` | 목표 speed | [0.0, `controller_cruise_speed_mps`] m/s |
| `action[1]` | 중심 steering | [-`vehicle_steering_limit_rad`, +`vehicle_steering_limit_rad`] rad |

정책 출력 `[-1,1]`은 `pure_pursuit.speed_steering_action_to_command`로 바로 `cmd_vel`에 매핑된다.
yield 축과 Stage 5 계약 변경이 없으므로 `reset_buffer_on_promote_to`도 필요하지 않다.
- `GetDimensions.srv` → `state_dim, action_dim, max_action, environment_dim, agent_dim`.

## aux 라벨 (학습 전용, state에 미포함)
aux prediction이 켜지면 env가 state 뒤에 **미래 위험 라벨**을 임시로 덧붙여 보내고, trainer가 떼어내 보조 손실에 쓴다. **정책 입력 87D는 그대로**다. → [../design/aux_prediction_design.md](../design/aux_prediction_design.md)

## Where in code
- state 조립: `env/observation/observation_builder.py`(obs_state 360°/180°), `env/simulation/environment.py`(`_rebuild_agent_state`), `env/simulation/step_pipeline.py`(`/step` 조립), `env/observation/obs_time_context.py`(프레임 스택)
- action→cmd: `env/simulation/step_pipeline.py::_step_callback_impl` + `common/pure_pursuit.py`
  (`action_to_waypoint`, `hybrid_action_to_command`, `speed_steering_action_to_command`)
- 차원/범위 설정: `config/environment.yaml`(2D waypoint baseline), `config/environment_curriculum.yaml`(3D 하이브리드)
