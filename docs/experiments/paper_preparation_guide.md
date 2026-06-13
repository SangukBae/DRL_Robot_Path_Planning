# Paper Preparation Guide

이 문서는 현재 DRL 로봇 경로계획 프레임워크를 기반으로 TQC 계열 모델을 개선하고, 그 결과를 논문으로 연결하기 위해 무엇을 수정해야 하는지 정리한 작업 가이드다.

기준은 다음과 같다.

- 단순 성능 향상보다 논문 기여로 설명 가능한 수정일 것
- 현재 저장소 구조를 크게 깨지 않고 확장 가능할 것
- `TQC`, `TQC + IEQN`, `제안 방법`의 비교 실험이 가능할 것
- `SAC`, `TD3` 같은 외부 actor-critic baseline과의 비교가 가능할 것

현재 문서의 실행 범위는 우선 `시스템 구축`, `로깅`, `평가 지표`, `baseline 비교 구조`, `환경/보상 검증`까지로 둔다.
직접적인 `TQC 내부 알고리즘 수정`은 후속 단계로 미루며, 그 단계의 구현은 사용자가 별도로 진행한다.

---

## 1. 추천 논문 방향

현재 코드와 로그를 기준으로 가장 적합한 주제는 다음 셋 중 하나다.

### A. Risk-aware TQC for Dynamic Robot Path Planning

핵심 아이디어:

- TQC의 quantile 분포를 평균값이 아니라 위험 측면으로 사용
- 충돌 가능성이 높은 tail quantile을 더 강하게 반영
- 동적 장애물과 보행자 환경에서 collision rate를 줄이는 것이 목표
- quantile spread를 uncertainty로 해석해 curriculum 제어에도 사용

추천 이유:

- 현재 커리큘럼 학습이 `stage 2` 부근에서 성공률은 올라가지만 충돌률이 병목이 되는 구조다
- TQC의 분포 정보 자체를 활용하는 논문 스토리를 만들기 쉽다

### B. Curriculum-aware Safety TQC

핵심 아이디어:

- 단순 단계 증가형 curriculum을 위험도 기반 curriculum으로 변경
- 성공/실패만 보지 말고 collision 유형, minimum clearance, uncertainty를 이용해 stage를 조절

추천 이유:

- 현재 curriculum은 고정 임계값 승급 구조라서 논문 기여로는 다소 약하다
- 환경 난이도 제어와 학습 안정화의 관계를 강조할 수 있다

### C. Distributional Replay for Safe Navigation

핵심 아이디어:

- prioritized replay를 distributional RL에 맞게 재설계
- mean TD error가 아니라 tail risk 또는 quantile spread 기반 priority 사용

추천 이유:

- 현재 replay priority는 기본적으로 꺼져 있고, 켜도 논문 수준으로 정교하지 않다
- 알고리즘 블록만 수정해도 분명한 ablation 구성이 가능하다

이 셋 중에서는 `A + B`를 묶는 방향이 가장 좋다. 제목 예시는 다음과 같다.

- `Risk-Aware Truncated Quantile Critics with Adaptive Curriculum for Dynamic Robot Path Planning`
- `Safety-Aware Distributional Reinforcement Learning for Ackermann Robot Navigation in Dynamic Environments`

---

## 2. 우선 수정해야 할 코드 영역

### 2.1 알고리즘 핵심

파일:

- `ros2_ws/src/drl_agent/scripts/policy/tqc_agent.py`
- `ros2_ws/src/drl_agent/scripts/policy/tqc_ieqn_agent.py`

현재 상태:

- `tqc_agent.py`는 거의 표준형 TQC 구조다
- critic target에서 quantile을 정렬한 뒤 상위 quantile 일부를 고정 개수로 잘라낸다
- actor loss는 critic의 평균 Q를 사용한다

논문용 수정 포인트:

1. Adaptive truncation
- 현재 `top_quantiles_to_drop_per_net`은 고정값이다
- 상태 위험도, obstacle proximity, quantile variance에 따라 drop 개수를 동적으로 조절하도록 바꾼다

2. Risk-sensitive actor objective
- 현재 actor는 quantile 평균을 최대화한다
- 이를 CVaR, lower-tail mean, worst-k quantile mean으로 바꿔서 보수적인 정책을 학습시킨다

3. Quantile spread regularization
- critic quantile 분산이 큰 상태를 불확실 상태로 해석
- actor loss 또는 auxiliary penalty에 uncertainty 항을 추가한다

4. TQC와 IEQN의 통합 비교
- `tqc_ieqn_agent.py`는 이미 확장 축이 있다
- 논문에서는 최소한 다음 3개를 비교해야 한다
  - TQC baseline
  - TQC + IEQN
  - Proposed method

5. 외부 baseline과의 비교
- 논문에서는 "왜 distributional RL 기반인 TQC를 써야 하는가"를 보여줘야 한다
- 따라서 내부 ablation만으로는 부족하고 다음 외부 baseline이 필요하다
  - SAC
  - TD3
- 가능하면 TD7도 추가 비교군으로 둘 수 있지만, 1차 필수군은 `SAC`, `TD3`, `TQC`다

추천 구현 순서:

1. `tqc_agent.py`에 risk-aware actor loss 추가
2. critic quantile spread 로깅 추가
3. adaptive truncation 추가
4. 필요하면 IEQN 경로에도 동일 논리 적용
5. 동일 상태/액션/curriculum 조건에서 `SAC`, `TD3` baseline 재실험

논문에서 만들고 싶은 핵심 비교 구도는 다음과 같다.

- SAC: 평균 Q 기반 actor-critic baseline
- TD3: deterministic actor-critic baseline
- TQC: distributional RL baseline
- TQC + IEQN: 기존 확장 baseline
- Proposed Risk-aware TQC: 제안 방법

즉, 논문 메시지는 아래처럼 정리되어야 한다.

- 평균 Q 기반 RL보다 distributional RL이 동적 환경에서 더 안전하다
- 제안 방법은 distributional RL 내부에서도 tail risk를 더 잘 제어한다

---

### 2.2 보상 함수와 안전 cost

파일:

- `ros2_ws/src/drl_agent/scripts/environment/environment.py`

현재 상태:

- reward는 진행 보상, heading 보너스, obstacle penalty, time penalty 중심이다
- terminal reward는 goal `+20`, collision `-30`이다
- `curvature penalty`와 `smooth penalty`는 사실상 약하거나 꺼져 있다
- step별 reward term 로그는 이미 잘 남고 있다

논문용 수정 포인트:

1. Near-collision penalty 추가
- 실제 collision 전에 very close call을 별도 cost로 벌점화
- minimum clearance 기반 연속 penalty를 강화

2. Time-to-collision 또는 predicted safety margin 추가
- 현재는 순간 거리 중심이다
- 동적 장애물에 대해 상대 속도까지 반영한 위험 지표를 만들면 논문 기여가 커진다

3. Steering jerk / control smoothness 추가
- Ackermann 로봇 특성상 급격한 steering 변화는 실제 주행 품질을 떨어뜨린다
- waypoint angle 변화와 실제 steering rate를 cost에 포함한다

4. Reward와 safety metric 분리
- 논문에서는 reward 하나만으로 설명하면 약하다
- 학습 보상과 별도로 안전 metric을 로그로 남겨야 한다

5. Reward rebalancing은 필요할 때만 최소 수정
- 현재 보상으로 학습이 안정적으로 진행되고 성공률/충돌률/timeout이 충분히 관리된다면, 보상 함수는 우선 유지하는 편이 낫다
- timeout 비율이 높거나 로봇이 멈추거나 제자리에서 주저하는 현상이 반복될 때만 상대 가중치를 다시 맞춘다

권장 조정 방향:

- progress reward 강화
  - 목표와의 거리를 줄일 때 주는 양의 보상을 더 키워 정지 행동을 억제
- step penalty 소폭 강화
  - 오래 머무를수록 불리하게 만들어 빠른 목표 도달을 유도
- 동적 장애물 proximity penalty 완화-재설계
  - "멈춤"보다 "우회"를 선택하도록 penalty 곡선을 더 부드럽게 조정
- heading / steering 관련 shaping 보조
  - 목표 방향 정렬과 부드러운 회피를 유도하되 reward hacking은 피해야 함

권장 추가 지표:

- episode minimum clearance
- near-collision count
- mean steering change
- path length
- goal reaching time

주의:

- 보상만 많이 건드리면 "reward engineering" 논문처럼 보이기 쉽다
- 보상 수정은 알고리즘 수정의 보조 수단으로 두는 것이 좋다
- 현재 보상으로 이미 잘 학습된다면, 일단은 수정하지 말고 baseline 재현성과 안전성 지표를 먼저 검증하는 편이 낫다

---

### 2.3 Curriculum 설계

파일:

- `ros2_ws/src/drl_agent/scripts/environment/environment_curriculum.py`
- `ros2_ws/src/drl_agent/scripts/policy/train_tqc_curriculum_agent.py`
- `ros2_ws/src/drl_agent/config/environment_curriculum.yaml`
- `ros2_ws/src/drl_agent/config/train_tqc_curriculum_config.yaml`

현재 상태:

- stage는 고정된 순서로 증가한다
- 승급 조건은 success rate와 collision rate 임계값이다
- stage별 차이는 장애물 수, 사람 수, 속도, 센서 노이즈 정도다

논문용 수정 포인트:

1. Adaptive curriculum
- 성공률만 보지 말고 uncertainty, near-collision, timeout pattern까지 반영
- 성능이 불안정하면 stage 유지 또는 rollback

특히 여기서 `uncertainty`는 모호한 표현으로 두지 말고 TQC의 quantile 분포에서 직접 정의하는 것이 좋다.

예시 정의:

- 상태 `s`와 정책 행동 `a = pi(s)`에 대해 critic이 예측한 quantile 집합을 `Z(s,a)`라고 두자
- 불확실성 지표는 다음 중 하나로 정의할 수 있다
  - `U_spread(s) = Q_0.9(s,a) - Q_0.1(s,a)`
  - `U_iqr(s) = Q_0.75(s,a) - Q_0.25(s,a)`
  - critic 간 평균 분산

에피소드 단위 지표 예시:

- `\bar{U}_ep = (1/T) * sum_t U(s_t)`
- `U_max,ep = max_t U(s_t)`

stage 판정 예시:

- 승급 조건:
  - success rate >= `tau_s`
  - collision rate <= `tau_c`
  - mean uncertainty <= `tau_u`
- 유지 조건:
  - success rate는 만족하지만 uncertainty가 높음
- 강등 조건:
  - collision rate > `tau_c,down` 또는 uncertainty > `tau_u,down`

실제로는 이동평균과 히스테리시스를 넣는 것이 좋다.

- `EMA_U(k) = beta * EMA_U(k-1) + (1-beta) * \bar{U}_ep`
- 승급 임계값과 강등 임계값을 다르게 둬서 진동을 막는다

2. Failure-type aware curriculum
- collision이 많은 경우 obstacle density를 낮추는 대신 dynamic speed만 유지
- timeout이 많은 경우 goal distance 또는 clutter 구조를 조정

3. Hard-case replay curriculum
- 실패한 start-goal pair나 obstacle 배치를 별도로 저장
- 일정 비율로 다시 노출해 학습 난제를 반복 훈련

4. Domain randomization curriculum
- 후반 stage에서만 noise/dropout을 주는 현재 구조를 더 세분화
- perception noise, human motion randomness, obstacle speed variance를 단계적으로 증가

추천 방향:

- `고정 단계 curriculum`을 baseline으로 두고
- `quantile-spread aware adaptive curriculum`을 제안 방법으로 만드는 것이 가장 자연스럽다

---

### 2.4 Replay buffer

파일:

- `ros2_ws/src/drl_agent/scripts/utils/buffer.py`
- `ros2_ws/src/drl_agent/scripts/policy/tqc_agent.py`

현재 상태:

- prioritized replay는 옵션으로만 존재한다
- priority는 mean TD error 기반이다
- importance sampling weight가 없다

논문용 수정 포인트:

1. Tail-TD priority
- 평균 TD error 대신 lower-tail quantile 오차로 priority를 계산

2. Uncertainty priority
- critic quantile spread가 큰 transition을 더 자주 뽑도록 수정

3. Importance sampling correction
- 논문용 prioritized replay라면 sampling bias 보정이 있어야 한다

4. Collision-aware replay mix
- collision 직전 transition 비율을 별도 관리하는 stratified replay도 가능하다

추천:

- 가장 현실적인 구현은 `tail-TD priority + importance sampling`이다

---

### 2.5 관측 상태 개선

파일:

- `ros2_ws/src/drl_agent/scripts/environment/environment.py`
- 관련 config 파일

현재 상태:

- LiDAR 전방 180도 요약값과 goal/state scalar를 사용한다
- 동적 장애물의 속도 정보는 직접 관측되지 않는다

논문용 수정 포인트:

1. Scan stacking
- 최근 2~4개 scan을 누적해 motion cue를 제공

2. Scan difference
- 현재 scan과 이전 scan 차이를 상태에 포함

3. Safety feature augmentation
- minimum front clearance
- left/right clearance asymmetry
- predicted collision margin

4. Recurrent policy는 2차 확장으로 고려
- 보행자 의도나 trajectory uncertainty를 직접 다루려면 RNN, GRU, LSTM 기반 policy가 강력할 수 있다
- 다만 1차 논문에서 `Risk-aware TQC` 자체를 제안하려면, RNN까지 동시에 도입하면 기여점이 섞일 위험이 있다
- 따라서 우선순위는 다음 순서가 적절하다
  - 1차: frame stacking 또는 scan difference
  - 2차: recurrent encoder 추가

정리하면:

- 시간성을 전혀 넣지 않는 것은 약하다
- 하지만 첫 논문 버전에서는 `frame stacking`이 가장 현실적이다
- 이후 확장판에서 `Recurrent Risk-aware TQC`로 발전시키는 것이 좋다

동적 환경 논문이라면 이 부분은 매우 중요하다.  
현재 stage 2 병목이 동적 객체 대응이라면, 상태 표현을 바꾸지 않고 알고리즘만 바꾸는 것은 설득력이 약할 수 있다.

---

### 2.6 로깅 및 관측 인프라

파일:

- `ros2_ws/src/drl_agent/scripts/policy/train_tqc_base.py`
- `ros2_ws/src/drl_agent/scripts/policy/train_tqc_curriculum_agent.py`

현재 상태:

- 기본 reward/성공 여부/일부 loss는 확인 가능하지만, 논문에서 내부 상태를 설득력 있게 보여주기엔 부족할 수 있다

논문용 수정 포인트:

1. TensorBoard 로깅 항목 대폭 확장
- critic이 추정한 Q-value의 평균/최대/최소
- 실제 rollout에서 얻은 discounted return
- `Q_est - G_true` bias
- policy entropy
- temperature parameter `alpha`
- critic loss / TD error
- quantile spread
- action jerk

2. episode-level CSV와 TensorBoard 역할 분리
- TensorBoard:
  - 학습 중 고주파 내부 상태 추적
  - Q-value, alpha, entropy, critic loss, quantile spread
- episode-level CSV:
  - 논문 표와 후처리용 지표 저장
  - success, collision, SPL, path length, q_bias, mean_alpha 등

3. "예측값 vs 실제값" 비교 가능 구조 만들기
- 평가 루프에서 true discounted return을 계산해 저장
- train loop에서 추정 Q값 통계와 같은 시점 기준으로 비교 가능하게 정렬

추천:

- 알고리즘 자체를 바꾸기 전에도 `train_tqc_base.py`의 로깅 항목부터 먼저 보강하는 것이 좋다

---

### 2.7 실험 제어 스위치와 baseline 실행 구조

파일:

- `ros2_ws/src/drl_agent/config/train_tqc_config.yaml`
- 관련 trainer/launcher 코드

현재 상태:

- baseline, curriculum, 알고리즘 변형을 실험별로 쉽게 켜고 끄는 구조가 문서상 충분히 명시되어 있지 않다

논문용 수정 포인트:

1. 알고리즘 선택 스위치
- 예:
  - `algo: tqc_vanilla`
  - `algo: tqc_improved`
  - `algo: sac`
  - `algo: td3`
- 동일한 학습/평가 파이프라인에서 알고리즘만 바꿔 비교할 수 있어야 한다

2. 커리큘럼 On/Off 스위치
- 예:
  - `use_curriculum: true/false`
- curriculum ablation을 위해 반드시 필요하다

3. stage 고정 스위치
- 예:
  - `fixed_stage: -1`
  - `fixed_stage: 3`
- 가장 어려운 stage에 바로 던져 학습 실패 데이터를 얻는 ablation에 유용하다

4. logging/profile 스위치
- 예:
  - `log_q_bias: true`
  - `log_action_jerk: true`
  - `log_spl: true`

추천:

- 논문 실험은 하드코딩이 아니라 config 스위치 중심으로 구성해야 재현성과 ablation 정리가 쉬워진다

---

## 3. 실험 설계에서 반드시 수정해야 할 점

### 3.0 외부 baseline 추가

현재 계획만으로는 TQC 내부 변형 비교가 중심이라서 논문 메시지가 약해질 수 있다.

반드시 포함할 baseline:

- SAC
- TD3
- TQC

권장 추가 baseline:

- TQC + IEQN
- TD7

공정 비교 원칙:

- 동일 state/action 정의
- 동일 curriculum 설정
- 동일 episode 길이
- 동일 evaluation protocol
- 동일 seed 집합

이 비교를 통해 다음 질문에 답해야 한다.

- 왜 평균 Q 기반 SAC/TD3보다 TQC가 적합한가?
- 왜 TQC 위에 risk-aware 설계를 얹을 필요가 있는가?

### 3.1 단일 seed 실험 금지

현재 학습 설정은 seed 하나 기준으로 운용되기 쉽다.

수정 필요:

- 최소 3 seeds
- 가능하면 5 seeds
- 평균과 표준편차 보고

대상 파일:

- `ros2_ws/src/drl_agent/config/train_tqc_config.yaml`

---

### 3.2 평가 지표 확장

현재 주요 지표:

- success rate
- collision rate
- timeout rate
- reward
- final goal distance

논문용으로 추가할 지표:

- Q-value estimation bias
- sample efficiency
- policy entropy / temperature alpha
- critic loss convergence stability
- action smoothness / action variance
- minimum clearance
- path length
- travel time
- average speed
- steering smoothness
- near-collision count
- SPL
- heading error / cross-track error
- stage별 성능
- obstacle type별 실패율

특히 안전성 논문이라면 `collision rate` 하나만으로는 부족하다.

저장 방식 원칙:

- 논문용 원본 로그는 `episode-level CSV`를 기본으로 한다
- step-level 전체 저장은 기본 정책으로 사용하지 않는다
- `success rate`, `collision rate`, `timeout rate`, `sample efficiency curve` 같은 집계 지표는 episode-level CSV를 후처리해서 만든다
- 즉, 저장은 에피소드마다 한 줄씩 하고, 논문 그래프는 여러 episode를 window 또는 evaluation split 단위로 묶어 계산한다

episode-level CSV에 넣을 대표 항목 예시:

- `episode_id`
- `global_step`
- `curriculum_stage`
- `success`
- `collision`
- `timeout`
- `total_reward`
- `episode_length`
- `final_goal_distance`
- `min_clearance`
- `near_collision_count`
- `path_length`
- `spl`
- `mean_speed`
- `mean_action_jerk`
- `mean_steering_change`
- `mean_heading_error`
- `mean_cross_track_error`
- `mean_q_est`
- `mean_discounted_return`
- `q_bias`
- `mean_alpha`
- `mean_entropy`
- `mean_critic_loss`
- `std_critic_loss`
- `mean_quantile_spread`

이렇게 해두면 별도 eval-level 파일이 없어도 논문 표와 그래프를 대부분 재구성할 수 있다.

평가지표는 아래 다섯 축으로 정리하는 것이 좋다.

#### A. 가치 추정 정확도와 과대평가 억제

TQC 계열 논문이라면 가장 중요한 축 중 하나다.

핵심 지표:

- estimated Q-value
- true discounted return
- estimation bias = `Q_est - G_true`
- absolute bias = `|Q_est - G_true|`
- overestimation ratio

권장 분석:

- 학습 중 critic이 예측한 `Q(s,a)`와 실제 rollout에서 얻은 discounted return을 비교
- episode마다 `mean_q_est`, `mean_discounted_return`, `q_bias`를 저장하고 후처리로 평균 bias를 계산
- 전체 평균 bias뿐 아니라 collision episode와 success episode를 분리해서 비교
- stage 2 이상 동적 환경에서 bias가 커지는지 확인
- TensorBoard에는 Q-value의 mean/max/min도 함께 남겨 과대평가 폭주를 시각적으로 확인

논문 메시지 예시:

- 기존 SAC 또는 baseline TQC는 동적 장애물 환경에서 Q-value를 과대평가하는 경향이 있다
- 제안 방법은 quantile tail을 더 보수적으로 사용해 추정 bias를 줄인다
- 그 결과 무리한 주행과 충돌이 감소한다

#### B. 샘플 효율성

모델 개선 효과를 가장 직관적으로 보여주는 지표다.

핵심 지표:

- reward vs timesteps
- success rate vs timesteps
- collision rate vs timesteps
- 특정 목표 성능에 도달하는 데 필요한 timesteps

권장 분석:

- 예: success rate 70%, 80%, 90%에 도달하는 step 수 비교
- episode CSV에서 누적 step 기준 rolling success rate를 계산
- curriculum stage별 승급까지 걸린 step 수 비교
- 동일 seed 집합에서 평균 곡선과 표준편차를 함께 제시

논문 메시지 예시:

- baseline은 100만 step 이후에야 안정 구간에 들어가지만 제안 방법은 더 이른 시점에 수렴한다
- 제안 방법은 동일 샘플 수에서 더 높은 success rate와 더 낮은 collision rate를 달성한다

#### C. Exploration-Exploitation 균형

TQC는 SAC 기반이므로 entropy와 temperature alpha의 동역학을 보여주는 것이 중요하다.

핵심 지표:

- policy entropy
- temperature parameter `alpha`
- curriculum stage 전환 시점 전후의 entropy 변화

권장 분석:

- warmup 종료 후 entropy 감소 속도
- stage 승급 직후 entropy 회복 여부
- local minima에 빠지는 실험에서 alpha가 너무 빨리 줄어드는지 여부
- episode마다 `mean_alpha`, `mean_entropy`를 저장하고 stage transition 기준으로 정렬해 비교

논문 메시지 예시:

- baseline은 환경 난이도 증가 시 entropy가 너무 빨리 죽어 탐험을 멈춘다
- 제안 방법은 동적 환경 변화 후에도 적절한 entropy를 유지해 더 강건한 정책을 학습한다

#### D. Critic 학습 안정성

훈련이 얼마나 안정적으로 이루어지는지 보여주는 축이다.

핵심 지표:

- critic loss
- TD error
- loss variance
- quantile spread variance

권장 분석:

- 평균 critic loss뿐 아니라 진동폭도 함께 비교
- moving average와 moving standard deviation을 같이 그리기
- dynamic obstacle stage에서 loss 폭주나 학습 붕괴가 줄어드는지 확인
- episode마다 `mean_critic_loss`, `std_critic_loss`를 저장하고 window 평균으로 비교

논문 메시지 예시:

- 제안 방법은 보상 분산이 큰 동적 환경에서도 critic loss 진동폭을 줄인다
- 더 안정적인 critic이 더 안정적인 actor update로 이어진다

#### E. 제어 출력 품질

현재 시스템은 RL이 직접 torque를 내는 구조가 아니라 waypoint/steering 계열 출력을 내므로, action 품질을 논문 지표로 삼을 수 있다.

핵심 지표:

- action jerk
- waypoint angle change
- action distribution variance
- steering smoothness
- speed smoothness

권장 분석:

- `|a_t - a_{t-1}|` 평균 및 분산
- waypoint angle의 연속 변화량
- 성공 episode와 collision episode에서 action jitter 비교
- episode마다 `mean_action_jerk`, `mean_steering_change`를 저장해 직접 비교

논문 메시지 예시:

- 제안 방법은 불필요한 steering oscillation을 줄인다
- 실제 Hunter SE에 적용할 때 물리적 부담이 적은 부드러운 주행을 유도한다

#### F. 국제 표준 네비게이션 지표

리뷰어 설득력을 높이려면 RL 지표 외에 navigation community에서 익숙한 지표를 같이 제시하는 편이 좋다.

핵심 지표:

- SPL (Success weighted by Path Length)
- heading error
- cross-track error (CTE)

권장 분석:

- 에피소드 시작 시 시작점과 목표점 사이의 기준 거리 기록
- 실제 이동 거리 누적
- 성공 시 `SPL = success * (shortest_path / max(actual_path, shortest_path))`
- 환경 또는 평가 루프에서 mean absolute heading error와 mean CTE를 계산해 episode-level로 저장

논문 메시지 예시:

- 제안 방법은 단순히 충돌을 줄이는 것뿐 아니라 더 짧고 효율적인 경로를 만든다
- heading error와 CTE 감소를 통해 경로 추종 품질도 개선되었다

정리하면, 논문 표와 그림은 최소한 아래 항목을 포함하는 것이 좋다.

- 성능: success rate, collision rate, timeout rate
- 안전성: minimum clearance, near-collision count
- 학습성: sample efficiency, critic loss stability
- 분포형 RL 특성: Q estimation bias, quantile spread
- SAC 기반 특성: entropy, alpha
- 제어 품질: action jerk, steering smoothness
- 네비게이션 품질: SPL, heading error, cross-track error

---

### 3.3 Ablation study 구성

최소 ablation 구성:

1. SAC
2. TD3
3. Baseline TQC
4. TQC + reward/safety shaping only
5. TQC + risk-aware actor only
6. TQC + adaptive curriculum only
7. Full proposed method

가능하면 추가:

8. TQC + IEQN
9. Proposed + IEQN

---

### 3.4 Generalization test

현재 프레임워크는 여러 world를 이미 포함하고 있다.

논문에서는 다음을 분리해야 한다.

- train world
- unseen test world
- obstacle density shift
- human motion randomness shift
- sensor noise shift

즉, 같은 맵에서만 잘 되는 모델이 아니라는 것을 보여줘야 한다.

---

## 4. 논문 작성을 위해 권장하는 실제 작업 순서

### Step 1. Baseline 고정

- 현재 TQC curriculum baseline을 재현 가능하게 정리
- 동시에 SAC, TD3 baseline의 동일 조건 실험 프로토콜을 고정
- seed, config, run_dir, eval protocol을 고정
- baseline 표와 학습 곡선 확보

### Step 2. 로깅 인프라 확장

- `train_tqc_base.py`에서 TensorBoard 항목 확장
- Q-value mean/max/min
- discounted return
- q_bias
- alpha / entropy
- critic loss / TD error
- action jerk

### Step 3. episode-level 논문 지표 확장

- 저장 단위는 `episode-level CSV`를 기본으로 통일
- near-collision
- minimum clearance
- steering smoothness
- path efficiency
- Q estimation bias 계산용 rollout return 기록
- entropy / alpha 기록
- critic loss 및 TD error 분산 기록
- SPL, heading error, cross-track error 기록

이 단계는 논문 그림과 표의 재료를 만드는 작업이다.

### Step 4. 실험 스위치 구조화

- `algo` 선택 스위치 추가
- `use_curriculum` on/off 스위치 추가
- 필요 시 `fixed_stage` 스위치 추가

### Step 5. 보상 검증 및 필요 시 최소 리밸런싱

- 현재 보상으로 학습이 충분히 잘 되면 그대로 유지
- timeout/freeze 또는 unsafe oscillation이 반복될 때만 progress reward, step penalty, dynamic obstacle penalty를 최소 범위에서 재조정

### Step 6. Ablation + seed 반복 실험

- 최소 3 seeds
- stage별 metric 집계
- reward보다 safety metric 중심으로 결과 정리

### Step 7. Generalization 실험

- unseen map
- 강화된 noise
- 더 빠른 동적 장애물

직접적인 `TQC 내부 알고리즘 수정`과 `제안 방법 구현`은 위 시스템 구축과 baseline 검증이 끝난 뒤의 후속 단계로 둔다.

---

## 5. 추천 논문 구조

### 5.1 Introduction

- 동적 환경 로봇 경로계획은 충돌 회피와 목표 도달을 동시에 만족해야 함
- 평균 Q 기반 RL은 위험 상태를 충분히 반영하지 못함
- 기존 curriculum은 고정 단계라서 불안정 구간을 세밀하게 다루기 어려움

### 5.2 Related Work

- SAC / TD3 / TQC 기반 navigation
- safe RL
- distributional RL
- curriculum learning for robot navigation

### 5.3 Method

- baseline TQC
- risk-aware objective
- quantile-spread based uncertainty metric
- adaptive curriculum
- safety-aware replay 또는 safety cost

### 5.4 Experimental Setup

- ROS2 + Gazebo + Hunter SE
- state/action 정의
- curriculum stage 정의
- training/test protocol

### 5.5 Results

- success/collision/timeout
- safety metric
- ablation
- generalization

### 5.6 Discussion

- 어떤 stage에서 가장 효과가 컸는지
- 성능 향상보다 collision reduction이 왜 중요한지
- 실제 로봇 적용 시 의미

---

## 6. 추천 1차 구현 목표

처음부터 너무 많이 바꾸지 말고, 우선은 `논문용 시스템 구축`에 필요한 조합으로 시작하는 것이 좋다.

### 제안 1차 버전

- `environment.py`
  - frame stacking 또는 scan difference 추가
  - minimum clearance logging 추가
  - 필요 시에만 near-collision penalty 보강
- `train_tqc_base.py`
  - TensorBoard Q/alpha/entropy/action jerk 로깅 확장
- `train_tqc_curriculum_agent.py`
  - episode-level metric 컬럼 확장
  - SPL, heading error, CTE 계산 추가
  - stage별 성능 기록 구조 정리
- `train_tqc_config.yaml`
  - `algo`, `use_curriculum`, `fixed_stage` 스위치 정리
- `실험군`
  - SAC, TD3, TQC, TQC+IEQN, Proposed 구성으로 정리

이 단계의 목적은 아래를 가능하게 만드는 것이다.

- SAC, TD3, TQC를 동일 조건에서 재현 가능하게 비교
- 안전성, 효율성, 가치 추정 안정성 지표를 논문용으로 저장
- 현재 프레임워크의 강점과 병목을 정량적으로 파악
- 이후 사용자가 직접 수행할 알고리즘 수정 단계의 기준 baseline 확보

---

## 7. 피해야 할 방향

- reward weight만 계속 조정하는 방식
- 단일 seed 결과만 보고 결론 내리는 방식
- success rate만 강조하고 collision severity를 무시하는 방식
- baseline 대비 비교 없이 제안 방법만 제시하는 방식
- 같은 맵에서만 평가하는 방식

---

## 8. 바로 다음 작업 추천

가장 먼저 할 일은 다음 셋이다.

1. `SAC`, `TD3`, `TQC` 공통 baseline 실험 프로토콜과 config 스위치 고정
2. `TensorBoard + episode-level CSV` 로깅 항목 확장
3. 현재 보상 함수의 재현성 검증 후, 필요할 때만 최소 reward rebalancing

그 다음에 baseline 결과를 보고 curriculum 또는 알고리즘 후속 연구로 넘어가는 순서가 가장 안정적이다.

원하면 다음 단계로 이어서 아래 문서도 만들 수 있다.

- 구현 체크리스트
- ablation 실험 계획표
- 논문 초록 초안
- method 수식 정리 문서
