# Paper Preparation Guide

현재 프레임워크로 TQC 계열 모델을 개선해 논문으로 연결하기 위한 작업 가이드. 기준: 단순 성능이 아니라
**논문 기여로 설명 가능한 수정**, 저장소 구조를 크게 깨지 않는 확장, `TQC / TQC+IEQN / 제안 방법` + 외부
baseline(`SAC / TD3`) 비교 가능.

이 문서의 범위는 **시스템 구축·로깅·평가 지표·baseline 비교·환경/보상 검증**까지다. `TQC 내부 알고리즘
수정`은 후속 단계(사용자 직접 진행)로 둔다.

---

## 1. 추천 논문 방향

| 방향 | 핵심 | 이유 |
|---|---|---|
| **A. Risk-aware TQC** | quantile 분포를 위험 측면으로 사용(tail quantile 강조), 동적 환경 collision rate 감소, quantile spread를 uncertainty로 curriculum에 활용 | 초반 동적-객체 stage에서 성공률은 오르나 충돌이 병목 → 분포 정보 활용 스토리 |
| **B. Curriculum-aware Safety TQC** | 단계 증가형 → 위험도 기반 curriculum(collision 유형·clearance·uncertainty로 stage 조절) | 현재 고정 임계값 승급은 기여로 약함 |
| **C. Distributional Replay** | priority를 mean TD가 아니라 tail risk / quantile spread 기반으로 | replay priority가 기본 off이고 정교하지 않음 |

**A + B**를 묶는 방향을 추천. 제목 예: *Risk-Aware Truncated Quantile Critics with Adaptive Curriculum
for Dynamic Robot Path Planning*.

---

## 2. 수정 대상 코드 영역

### 2.1 알고리즘 핵심 (`rl/algorithms/tqc/agent.py`, `rl/algorithms/tqc_ieqn/agent.py`)
거의 표준형 TQC(고정 개수 top-quantile drop, actor는 평균 Q 최대화). 논문용 포인트:
- **Adaptive truncation** — 위험도/proximity/quantile variance에 따라 drop 개수를 동적 조절.
- **Risk-sensitive actor** — 평균 Q 대신 CVaR / lower-tail mean / worst-k quantile mean.
- **Quantile spread regularization** — 분산이 큰 상태를 불확실로 보고 actor/aux penalty에 반영.
- 비교군: TQC baseline / TQC+IEQN / Proposed, 그리고 외부 SAC·TD3(1차 필수), TD7(선택).

메시지: "평균 Q 기반 RL보다 distributional RL이 동적 환경에서 안전하고, 제안 방법은 그 안에서도 tail
risk를 더 잘 제어한다."

### 2.2 보상 함수 · 안전 cost (`env/simulation/environment.py`, `env/rewards/reward_calculator.py`)
현재 reward = 진행 + heading + obstacle penalty + time penalty, terminal goal `+20`/collision `-30`.
논문용 포인트:
- near-collision penalty(clearance 기반 연속 벌점), time-to-collision/predicted safety margin(상대 속도
  반영), steering jerk/제어 smoothness cost, **reward와 안전 metric 분리 로깅**.
- **리밸런싱은 필요할 때만.** 현재 보상으로 학습이 안정적이면 유지하고, timeout/freeze/oscillation이
  반복될 때만 progress reward·step penalty·동적 proximity penalty를 최소 조정. 보상만 계속 만지면
  "reward engineering" 논문처럼 보이므로 알고리즘 수정의 보조 수단으로 둔다.

### 2.3 Curriculum (`env/curriculum/environment_curriculum.py`, `training/train_tqc_curriculum.py`, 관련 config)
현재 고정 순서 + success/collision 임계값 승급. 논문용 포인트:
- **Adaptive curriculum** — uncertainty를 TQC quantile로 명시 정의(예 `U_spread = Q_0.9 − Q_0.1`),
  승급/유지/강등 조건에 반영, EMA + 히스테리시스로 진동 방지.
- Failure-type aware(충돌↑ → density↓·속도 유지 / timeout↑ → goal 거리·clutter 조정),
  hard-case replay(실패한 start-goal 재노출), domain randomization 세분화.
- baseline = 고정 단계, 제안 = quantile-spread aware adaptive curriculum.

### 2.4 Replay buffer (`rl/replay/buffer.py`, `rl/algorithms/tqc/agent.py`)
현재 prioritized replay는 옵션, priority는 mean TD, IS weight 없음. 논문용: tail-TD priority,
uncertainty priority, importance sampling 보정, collision-aware stratified replay. 가장 현실적 구현은
**tail-TD priority + importance sampling**.

### 2.5 관측 상태 (`env/simulation/environment.py` + config)
현재 전방 180° LiDAR 요약 + goal/state scalar, 동적 장애물 속도는 직접 관측 안 됨. 논문용:
- **1차: frame stacking 또는 scan difference**(motion cue), safety feature(front clearance, 좌우
  비대칭, predicted margin). **2차: recurrent encoder.**
- 동적 환경 논문에서 상태 표현을 그대로 두고 알고리즘만 바꾸면 설득력이 약하므로 시간성 도입은 중요하다.
  (단, 1차에서 RNN까지 동시에 넣으면 기여가 섞이므로 frame stacking부터.)

### 2.6 로깅 인프라 (`training/train_tqc_base.py`, `training/train_tqc_curriculum.py`)
알고리즘을 바꾸기 전에 로깅부터 보강. 역할 분리:
- **TensorBoard**(고주파 내부 상태): Q mean/max/min, discounted return, `Q_est − G_true` bias, entropy,
  alpha, critic loss/TD error, quantile spread, action jerk.
- **episode-level CSV**(논문 표/후처리): success/collision/SPL/path length/q_bias/mean_alpha 등.
- 평가 루프에서 true discounted return을 계산해 추정 Q와 같은 시점 기준으로 비교 가능하게.

### 2.7 실험 스위치 (`train_tqc_config.yaml` + trainer)
재현성·ablation을 위해 하드코딩이 아니라 config 스위치 중심: `algo`(tqc/sac/td3/...),
`use_curriculum`, `fixed_stage`, 로깅 프로파일 플래그.

---

## 3. 실험 설계 필수 사항

- **외부 baseline**: SAC·TD3·TQC(필수) + TQC+IEQN·TD7(권장). 공정 비교 원칙 = 동일
  state/action·curriculum·episode 길이·eval protocol·seed. "왜 SAC/TD3보다 TQC인가, 왜 그 위에
  risk-aware가 필요한가"에 답해야 한다.
- **다중 seed**: 단일 seed 금지. 최소 3, 권장 5, mean±std 보고.
- **저장 원칙**: 논문 원본은 **episode-level CSV** 한 줄/episode를 기본으로 하고, success/collision rate,
  sample-efficiency curve 등 집계는 후처리로 만든다(step-level 전체 저장은 기본 정책이 아님).

### 평가 지표 — 5축 정리
안전성 논문에서 collision rate 하나로는 부족하다. 다음 축으로 정리한다(지표 정의·저장 위치는
[metrics_reference](../reference/metrics_reference.md) / [experiment_protocol](experiment_protocol.md)):

| 축 | 핵심 지표 | 메시지 |
|---|---|---|
| **A. 가치 추정 정확도** | `Q_est`, true discounted return, bias `Q_est − G_true`, overestimation ratio | 동적 환경 과대추정을 tail을 보수적으로 써 줄임 (collision/success episode 분리 분석) |
| **B. 샘플 효율성** | reward/success/collision vs timesteps, 목표 성능 도달 step 수 | 동일 샘플에서 더 높은 성공·낮은 충돌, 더 이른 수렴 |
| **C. Exploration** | policy entropy, alpha, stage 전환 전후 변화 | 난이도 증가 후에도 적절한 entropy 유지 |
| **D. Critic 안정성** | critic loss, TD error, loss/spread variance | 동적 stage에서 loss 진동폭 감소 |
| **E. 제어 품질** | action jerk, waypoint angle change, steering/speed smoothness | 불필요한 steering oscillation 감소(실로봇 부담↓) |
| **F. 네비게이션 표준** | SPL, heading error, CTE | 충돌 감소뿐 아니라 더 짧고 효율적인 경로 |

### Ablation & Generalization
- **Ablation**: SAC / TD3 / TQC / TQC+shaping / TQC+risk-actor / TQC+adaptive-curriculum / Full proposed
  (+ 가능하면 TQC+IEQN, Proposed+IEQN).
- **Generalization**: train world vs unseen world, obstacle density/human randomness/sensor noise shift를
  분리 → 같은 맵에서만 잘 되는 모델이 아님을 보인다.

---

## 4. 권장 작업 순서

1. **Baseline 고정** — TQC curriculum + SAC/TD3 동일 조건 프로토콜(seed/config/run_dir/eval) 고정, 표·곡선 확보.
2. **로깅 확장** — `train_tqc_base.py` TensorBoard(Q/bias/alpha/entropy/loss/jerk).
3. **episode-level 논문 지표 확장** — near-collision/clearance/steering/path efficiency/rollout return/
   entropy·alpha/critic loss 분산/SPL·heading·CTE.
4. **실험 스위치 구조화** — `algo` / `use_curriculum` / `fixed_stage`.
5. **보상 검증 + 필요 시 최소 리밸런싱**.
6. **Ablation + seed 반복** — 최소 3 seed, stage별 집계, reward보다 safety metric 중심.
7. **Generalization** — unseen map, 강한 noise, 빠른 동적 장애물.

`TQC 내부 알고리즘 수정`·`제안 방법 구현`은 위 시스템/baseline이 끝난 뒤의 후속 단계다.

---

## 5. 추천 논문 구조

- **Intro** — 동적 환경은 충돌 회피 + 목표 도달 동시 만족, 평균 Q RL은 위험 미반영, 고정 curriculum의 한계.
- **Related Work** — SAC/TD3/TQC navigation, safe RL, distributional RL, curriculum learning.
- **Method** — baseline TQC → risk-aware objective → quantile-spread uncertainty → adaptive curriculum → safety-aware replay/cost.
- **Setup** — ROS2 + Gazebo + Hunter SE, state/action, curriculum stage, train/test protocol.
- **Results** — success/collision/timeout, safety metric, ablation, generalization.
- **Discussion** — 어느 stage에서 효과가 컸는지, collision reduction의 의미, 실로봇 적용.

---

## 6. 1차 구현 목표 & 피해야 할 것

**1차(시스템 구축)**: `environment.py`(frame stacking/scan diff + clearance 로깅, 필요 시 near-collision
penalty), `train_tqc_base.py`(TensorBoard 확장), `train_tqc_curriculum.py`(episode-level 지표 +
SPL/heading/CTE + stage별 기록), `train_tqc_config.yaml`(스위치 정리), 실험군 SAC/TD3/TQC/TQC+IEQN/Proposed.
→ 목적: 동일 조건 baseline 비교, 안전·효율·가치추정 지표 저장, 병목 정량화, 후속 알고리즘 수정의 기준 확보.

**피해야 할 것**: reward weight만 반복 조정 / 단일 seed 결론 / success rate만 강조하고 collision severity
무시 / baseline 비교 없는 제안 / 같은 맵에서만 평가.

**바로 다음**: ① SAC·TD3·TQC 공통 프로토콜 + config 스위치 고정, ② TensorBoard + episode-level CSV 로깅
확장, ③ 보상 재현성 검증 후 필요 시에만 최소 rebalancing. 그다음 baseline 결과를 보고 curriculum/알고리즘
후속 연구로.
