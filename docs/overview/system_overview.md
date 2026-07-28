# System Overview

이 문서는 **이 프로젝트가 전체적으로 무엇인지** 한 장으로 설명한다. 강화학습(RL)이나 ROS2를 처음 보는 사람을 위한 출발점이다.

## What — 무엇을 하는가
- 바퀴형 로봇(**AgileX Hunter SE**)이 **LiDAR만 보고** 장애물(벽·가구·사람)을 피해 목표 지점까지 가도록 **강화학습으로 주행 정책**을 학습한다.
- 지도를 미리 만들지 않는 **mapless** 방식이다. 로봇은 "주변 장애물 거리(LiDAR)"와 "내가 목표로부터 어디에 있는지"만 알면 된다.
- 시뮬레이터는 **Gazebo Ignition Fortress**, 미들웨어는 **ROS2 Humble**, 학습 프레임워크는 **PyTorch**다.

## Why — 왜 이렇게 구성했나
- **두 프로세스 분리**: 시뮬레이션을 돌리는 *환경 노드*와, 신경망을 학습하는 *에이전트 노드*를 분리한다. 그래야 학습 코드가 시뮬레이터와 독립적으로 관리된다.
- **서비스 기반 동기 제어**: 토픽(비동기) 대신 ROS2 *서비스*로 `reset`/`step`을 주고받아, 한 스텝씩 정확히 맞물려 학습한다.
- **커리큘럼**: 처음부터 어려운 환경을 주면 학습이 안 되므로, 빈 맵 → 장애물 → 사람 → 복잡한 맵 순으로 난이도를 자동으로 올린다.

## How — 큰 그림
```
┌─────────────────────────┐         ROS2 services          ┌──────────────────────────────┐
│  Environment node        │  ◀── /reset, /step, /seed ──   │  Agent (trainer) node          │
│  environment_curriculum  │                                │  train_tqc_curriculum.py       │
│   .py                    │  ── state, reward, done ──▶    │   ├─ TQC 신경망 (actor/critic) │
│   ├─ Gazebo 제어         │                                │   ├─ replay buffer             │
│   ├─ LiDAR → state       │                                │   └─ aux prediction head       │
│   └─ 보상/충돌 판정      │                                │                                │
└─────────────────────────┘                                └──────────────────────────────┘
            │                                                            
            ▼  cmd_vel → prefilter → Gazebo (로봇이 실제로 움직임)
```
- **환경 노드**가 Gazebo를 돌리고, 로봇 관측을 **state 벡터(87차원)**로 만들어 준다.
- **에이전트 노드**가 state를 받아 **action(3차원: 전진 waypoint 거리·각도 + yield/정지 축)**을 정하고, 환경에 `step`으로 보낸다.
- 환경은 action을 실행해 다음 state·보상·종료여부를 돌려준다. 이 과정을 수백만 번 반복하며 정책이 좋아진다.

## 핵심 개념 5가지 (어디서 더 읽나)
1. **학습 파이프라인** (reset→state→action→step→train→eval): [training_pipeline.md](training_pipeline.md)
2. **환경/커리큘럼 노드 역할 분리**: [../design/environment_design.md](../design/environment_design.md), [../design/curriculum_design.md](../design/curriculum_design.md)
3. **state/action 구조**: [../reference/state_action_reference.md](../reference/state_action_reference.md)
4. **aux network**(미래 위험 예측 보조 과제): [../design/aux_prediction_design.md](../design/aux_prediction_design.md)
5. **localization noise**(실로봇 위치오차 모사): [../design/localization_noise_design.md](../design/localization_noise_design.md)

## Where in code
- 환경: `ros2_ws/src/drl_agent/drl_agent/env/simulation/environment.py`, `env/curriculum/environment_curriculum.py`
- 학습: `ros2_ws/src/drl_agent/drl_agent/training/train_tqc_curriculum.py`, `rl/algorithms/tqc/agent.py`
- 설정: `ros2_ws/src/drl_agent/config/`
- 로봇/시뮬: `ros2_ws/src/hunter_se_gazebo/`
