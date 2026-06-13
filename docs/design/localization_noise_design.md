# Localization Noise Design

이 문서는 **실로봇의 위치추정 오차를 시뮬레이션에서 흉내 내는** 방식을 설명한다.

## What
실로봇은 자기 위치를 정확히 모른다(AMCL/SLAM/odom 오차). 이 프로젝트는 그 오차를 **정책의 goal 관측**(목표까지 거리 state[80], 방향 state[81])에 주입한다. 그래야 정책이 "내 위치가 약간 틀려도" 견디도록 학습된다.

## Why — 정확히 무엇이 오염되나
- 이건 "전체 localization 스택"을 모델링하는 게 아니라, **localization 불확실성이 goal 관측에 반영되는 것**이다(goal-conditioned observation corruption).
- **LiDAR(state[0:80])는 깨끗하다** — 실로봇에서도 LiDAR는 ego-centric 실측이라 위치오차와 무관.
- **보상·종료·충돌은 항상 ground-truth 좌표** 사용 → 학습 신호는 깨끗, 관측만 노이즈.
- proprio 관측(속도/요레이트/조향, state[84:86])은 **별도 축**(`proprio_noise`)으로 따로 켠다.

## How — 오차 모델 (축별로, 0이면 자동 비활성)
| 성분 | 의미 |
|--|--|
| **bias** | 에피소드마다 고정된 정합 오프셋 |
| **OU(sigma, tau)** | 시간상관 측정오차. `sigma`=정상상태 std, `tau`=상관시간. **tau=0이면 기존 white noise와 동일**(하위호환) |
| **drift** | 느린 random walk. `drift_*_mps/radps`는 **초당 강도**(누적 std ≈ rate·√t) |
| **delay_steps** | 관측 지연(latency) — 독립 축 |
| **jump** | 드문 재측위 점프(작은 snap + 옵션 큰 실패) |
| **yaw flip** | 대칭 맵(corridor)에서 ±π 미러 재측위 — **stress-test 전용**, 평소 학습엔 OFF |
| **map_type_multipliers** | 맵 종류별 세기 배수. corridor는 길이축(x)이 더 부정확 → 이방성 증폭 |

- **ablation 플래그**(`noise_goal_enabled`/`noise_jump_enabled`/`noise_flip_enabled`/`noise_delay_enabled`)로 축을 독립적으로 끄고 켤 수 있다.
- reset 시 buffer를 초기 추정으로 **seed**해서 clean→noisy 점프가 없게 한다(goal·proprio 모두).

## Train vs Stress-test 분리
- **학습 프로파일**: clean → weak_goal_noise → drift_goal_noise → robustness_train(드문 jump만, **flip 없음**).
- **stress_eval 프로파일**(어떤 학습 stage도 미사용): 강한 jump/delay + corridor yaw flip. 평가에서만 선택하도록 구조만 열려 있다.

## 파라미터 이름 = 물리 단위
`sigma_xy_m`, `sigma_yaw_rad`, `drift_xy_mps`, `drift_yaw_radps`, `bias_xy_m`, `bias_yaw_rad`, `jump_prob`, `jump_xy_m`, `jump_yaw_rad`, `delay_steps` — 실로봇 로그(LIO-SAM/AMCL/odom)와 1:1 대응.

## Where in code
- 주입/모델: `environment/environment.py` (`_loc_emulator_step`, `_reset_localization`, `_loc_map_multiplier`, proprio: `_proprio_emulator_step`)
- stage별 프로파일 선택: `environment/environment_curriculum.py::_resolve_noise_override`
- 프로파일/파라미터: `config/environment_curriculum.yaml` (`localization`, `localization_profiles`, `proprio_noise*`)
- 실로봇 배포·검증: [../guides/real_robot_deployment.md](../guides/real_robot_deployment.md), [../experiments/simulation_validation.md](../experiments/simulation_validation.md)
