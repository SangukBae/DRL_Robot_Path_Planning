# Training Pipeline

이 문서는 **학습이 한 스텝 한 스텝 실제로 어떻게 도는지** 데이터 흐름 중심으로 설명한다. "파일이 뭐가 있나"가 아니라 "데이터가 어디로 흐르나"를 본다.

## What — 한 줄 요약
`reset → state → action → step → replay buffer → train → (주기적) eval` 루프를 수백만 스텝 반복한다.

## How — 한 에피소드의 흐름

```
[에피소드 시작]
  trainer ── /reset ──▶ env
                         ├─ 커리큘럼 stage 적용 (장애물/사람 수, 맵, noise 프로파일)
                         ├─ 맵 레이아웃 선택 + 로봇/목표/장애물 배치
                         └─ 초기 state(87D) 반환
  trainer ◀── state ──

[매 스텝 반복]
  trainer: action = policy(state)        # profile별 action, [-1,1] 정규화
  trainer ── /step(action) ──▶ env
                         ├─ action → waypoint_yield 또는 speed_steering decode → cmd_vel
                         ├─ Gazebo 0.1초 진행, 보행자 이동
                         ├─ LiDAR/odom → 다음 state(87D)
                         ├─ goal 관측에 localization noise 주입 (옵션)
                         └─ 보상 + 충돌/도달/타임아웃 판정
  trainer ◀── (next_state, reward, done) ──
  trainer: replay_buffer.add(state, action, next_state, reward, done [, aux_label])
  trainer: if 워밍업 끝났으면 → agent.train()   # 신경망 1회 업데이트
  state = next_state
  done 이면 → 에피소드 종료, /reset

[주기적 (eval_freq 스텝마다)]
  trainer: evaluate_and_print()          # 탐험 없이 N 에피소드 평가
                                         # 성공률/충돌률/SPL/aux 지표 집계
  통과하면 → 커리큘럼 다음 stage 로 진급
```

## 각 단계 설명 (짧게)
- **reset**: 에피소드 초기화. 환경이 stage에 맞춰 장애물·맵·noise를 세팅하고 초기 state를 준다.
- **state (기본 87D)**: 전방 180° LiDAR 80빈 + 목표거리/방향 + 이전 action 2축 + 실제 속도/요레이트/조향. → [state 구조](../reference/state_action_reference.md)
- **action (profile별)**: 기본 curriculum은 3D `waypoint_yield`(거리 r, 각도 θ + yield 축)를 쓴다. 정책은
  `[-1,1]`로 내고, 환경이 profile action mode에 맞춰 물리 명령으로 바꾼다.
- **step**: action 실행 → 0.1초 시뮬레이션 → 다음 state·보상·done.
- **replay buffer**: 경험 `(s,a,s',r,done)`을 저장. off-policy 학습이라 과거 경험을 재사용한다. (`rl/replay/buffer.py`, LAP 우선순위)
- **train**: 버퍼에서 미니배치를 뽑아 actor/critic(+aux head)을 1회 업데이트. (`rl/algorithms/tqc/agent.py::train`)
- **eval**: 학습을 멈추고 결정론적으로 평가. 지표가 기준을 넘으면 stage 진급.

## 어디에 무엇이 끼어드나
- **커리큘럼 stage**: `reset` 시점에 장애물 수·맵 종류·noise 세기를 바꾼다. → [curriculum_design](../design/curriculum_design.md)
- **localization noise**: `step`의 *goal 관측*(state[80],[81])에만 섞인다. 보상/종료는 정답 좌표 사용. → [localization_noise_design](../design/localization_noise_design.md)
- **aux prediction**: `train` 시점에 공유 인코더가 "미래 위험"을 같이 예측하도록 보조 손실을 추가한다. state 차원은 안 바뀐다. → [aux_prediction_design](../design/aux_prediction_design.md)

## Where in code
- 학습 루프: `training/train_tqc_curriculum.py::train_online`
- reset/step 서비스 클라이언트: `env/environment_interface.py`
- 신경망 업데이트: `rl/algorithms/tqc/agent.py::train`
- 환경 step/reset: `env/simulation/environment.py::step_callback`, `reset_callback`
