# `phase3/speed_steering_risk_balanced` — 연속 speed/steering action + risk-balanced replay

`phase2/both`(TQC curriculum, waypoint+yield 3D action, risk_map_reward + action_risk_head 둘 다 ON)를
기반으로 연속 speed/steering action과 risk-balanced replay를 결합한 신규 profile.
현재 phase2 profile은 분리되어 있다. `phase2/both_legacy`가 예전 phase2/both 의미를 보존하고,
`phase2/both`와 `phase2/both_trajrisk_rbs`는 waypoint_yield 계약을 유지한 채 trajectory-risk target과
risk-balanced supervised loss를 켠다. `phase3/speed_steering_risk_balanced`는 여기에
`environment.action_mode: speed_steering`을 더해 action 계약 자체를 2D speed/steering으로 바꾼다.

## 1. 연속 speed/steering action mode

### 1.1 새 action 계약 (`environment.action_mode: speed_steering`)

| axis | 정규화 입력 | 물리 값 매핑 |
|---|---|---|
| `action[0]` | `[-1, 1]` | speed `[0.0, controller_cruise_speed_mps]` (선형, `-1 → 0 m/s`) |
| `action[1]` | `[-1, 1]` | center steering `[-vehicle_steering_limit_rad, +vehicle_steering_limit_rad]` |

- `action_dim: 2` (waypoint/yield 없음).
- Twist 계약은 기존과 동일: `linear.x = speed`, `angular.z = steering` — prefilter 쪽 변경 불필요.
- `-1` speed는 **연속적으로 완전 정지**를 의미한다 — waypoint 거리를 0으로 낮추거나 별도 yield 채널을
  거칠 필요가 없다(`pure_pursuit.speed_steering_action_to_command`).
- 이전 action이 들어가는 state 슬롯(`state[82]`, `state[83]` = `agent_state[2:4]`)은 원래부터
  `action[0]`/`action[1]`을 그대로 저장하는 코드였으므로 **코드 변경 없이** 자동으로 "이전 speed
  action, 이전 steering action"을 의미하게 된다.
- Actor/Critic/replay buffer/action-risk head/aux head는 전부 `action_dim`을 생성자 인자로 받는
  구조라 `action_dim=2`가 자동으로 전파된다(target_entropy도 `-action_dim`이 기본값이라 자동으로 `-2`).
  speed_steering의 action_dim 전파 자체는 생성자 계약을 그대로 사용한다. 별도 risk-balanced replay와
  weighted-loss 처리는 `rl/algorithms/tqc/agent.py`의 supervised aux/action-risk 학습 경로에 추가되어 있다.

### 1.2 기존 mode와의 차이

| | `waypoint_yield` (phase2 전체, 기존) | `waypoint` (legacy 2D ablation, 기존) | `speed_steering` (신규) |
|---|---|---|---|
| `action_dim` | 3 | 2 | 2 |
| 의미 | waypoint 거리·각도 + binary yield 스칼라 | waypoint 거리·각도만 | speed·steering 직접 |
| 정지 방법 | yield 채널로만(sealed 시 불가) | 불가(항상 `min_speed` 이상) | speed 축이 0이면 항상 가능 |
| yield reward / stage 5 봉인 해제 | 있음 | 없음(애초에 yield 없음) | **없음**(전 stage 동일 계약) |

`drl_agent/env/simulation/environment.py`의 `_step_callback_impl`은 `self.action_mode` 문자열로
분기한다(`self.action_dim >= 3`가 아니라). **기본값은 기존 `action_dim`에서 그대로 유추**하므로
(`action_dim>=3` → `waypoint_yield`, 아니면 `waypoint`) `action_mode` 키가 없는 기존 config는
byte-identical하게 동작한다. `training/train_tqc_curriculum.py`·`training/train_rl.py`의 CSV 텔레메트리
(`_motion_telemetry_sample`)와 `evaluation/real_policy_runner.py`(실기 추론)도 동일한
`pure_pursuit.speed_steering_action_to_command`를 호출하도록 같이 분기시켜, 시뮬레이션·실기가 같은
decode 함수를 공유한다.

### 1.3 Yield reward / Stage 5 계약

`speed_steering`에는 yield 채널 자체가 없으므로(action[2] 미존재) `_step_callback_impl`이 항상
`yielding=False`로 확정한다 — config가 뭐든 상관없이 binary yield 판정을 절대 거치지 않는다. 그 위에,
새 profile의 `environment_curriculum.yaml`은 `yield_reward.enabled: false`(마스터 스위치)까지 꺼서
`yield_bonus`/`idle_pen`(둘 다 `yield_action` bool이 아니라 `yield_enabled` 전역 스위치·속도 임계값으로
발동하는 블록)이 정당한 연속 정지를 "yield 선언 없음"으로 오인해 벌점을 주는 일을 원천 차단했다.
Stage별 `yield_reward.action_enabled` override는 전부 제거했고(전 stage 동일 의미),
`train_tqc_curriculum_config.yaml`의 `reset_buffer_on_promote_to`도 `[]`로 비웠다 — Stage 5에서
버퍼를 리셋할 계약 변경이 아예 없기 때문.

Human-risk / TTC / stall / anti-freeze reward는 유지된다 — 이들은 waypoint `action[0]`/`action[1]`
raw 값이 아니라 실제 GT odometry 속도(`latest_actual_signed_speed`)·commanded 속도(`v`)·progress
delta에서 계산되므로 action mode에 무관하게 그대로 동작한다. `env/rewards/reward_calculator.py`에는
speed_steering 전용 continuous-control shaping(`continuous_control_reward`)도 추가되어 있으며,
해당 config가 꺼져 있으면 기존 reward 항목은 그대로 유지된다.

### 1.4 Action-risk target: Ackermann swept-path

`_compute_directional_risk(theta, target_v, target_cmd_steering)`는 `self.action_mode`로
분기한다(`environment.py`):

- **`speed_steering`**: `pure_pursuit.ackermann_swept_path()`로 로봇의 실제 bicycle-model 궤적을
  `horizon_sec` 구간 내 여러 시각(`t`)에 샘플링하고, `aux_prediction_labels.
  compute_action_conditioned_risk()`가 그 궤적과 **모든 사람의 CV(등속) 궤적** 사이 전역(global)
  최소거리를 시각 매칭(같은 `t`끼리 비교)으로 계산한다. **sector로 다시 거르지 않는다** — 초기
  구현은 사람마다 bearing sector를 정한 뒤 action이 향하는 sector 하나만 조회했는데, 이러면 사람의
  bearing sector가 action의 heading sector와 다를 경우(예: 직진하는 로봇과 옆에서 가로지르는
  보행자) 실제로는 궤적이 거의 충돌해도 risk가 0으로 조회되는 회귀가 있었다(현재는
  `test_speed_steering_swept_path_catches_crossing_human`으로 회귀 테스트됨). 속도가 0이어도
  로봇은 매 샘플 시각에 원점에 머무는 것으로 취급되므로, 정지 상태에서 사람이 horizon 중간 시각에
  잠깐 가까워지는 경우도 놓치지 않는다(`test_speed_steering_stopped_robot_catches_intermediate_
  time_close_pass`).
  - **Prefilter 동역학 반영(가감속·조향각속도 제한)**: rollout은 선택한 `(target_v,
    target_cmd_steering)`이 **즉시** 적용된다고 가정하지 않는다 — `hunter_se_cmd_prefilter`의
    실제 제한(`accel_limit_mps2`/`brake_decel_mps2`/`steering_rate_deg_s`,
    `hunter_se_gazebo/config/hunter_se_cmd_prefilter.yaml`)을 반영해, 로봇의 **현재 실제
    속도/조향**(`self.latest_actual_signed_speed`/`self.latest_center_steering`, odometry·
    joint-state 기반)을 rollout의 초기 조건으로 삼아 목표값까지 rate-limited ramp로 적분한다
    (`pure_pursuit.ackermann_swept_path`, trapezoidal/중점-속도 적분으로 감속 구간의 이동거리를
    analytic 값에 가깝게 재현). 예: 2 m/s로 주행 중 정지를 선택해도 6 m/s² 제동으로 ~0.33초/~0.33m를
    더 진행한 뒤에야 실제로 멈춘다 — "정지 action = 즉시 정지"로 가정하면 이 잔여 이동 중 위험을
    과소평가한다(`test_speed_steering_braking_from_speed_reaches_true_stopping_point`로 회귀
    테스트됨). 프리필터의 속도 1차 지연(`speed_lag_tau_sec`)은 의도적으로 모델링하지 않았다 —
    이 차량의 튜닝값(`tau=0.05s`)에서는 유의미한 크기의 속도 변화(비상 정지 등, 바로 이 안전-critical
    영역)에서는 rate limit이 거의 항상 lag보다 먼저 지배적이 되므로, rate-limited ramp만으로도 그
    영역을 잘 근사한다(자세한 유도는 `pure_pursuit.ackermann_swept_path`의 docstring 참고). 두
    config가 어긋나면(prefilter yaml을 바꾸고 이쪽을 안 바꾸면) risk target이 실제와 다른 차량을
    모델링하게 되므로, `environment_curriculum.yaml`의 `directional_risk.rollout_accel_limit_mps2`
    /`rollout_brake_decel_mps2`/`rollout_steering_rate_deg_s`는 **`hunter_se_cmd_prefilter.yaml`과
    반드시 동일하게 유지**해야 한다(코드 쪽 기본값도 동일하게 맞춰 뒀지만, yaml에 명시적으로 적어
    두어 두 파일이 눈에 보이게 대응되도록 함). 이 불변식은 주석만으로는 지켜지지 않으므로(한쪽
    yaml만 바뀌어도 조용히 어긋날 수 있음) `test_phase3_rollout_dynamics_match_prefilter_config`가
    두 YAML의 값을 직접 비교하는 회귀 테스트로 강제한다 — source 트리에서 `hunter_se_gazebo`를
    찾지 못하면(source 경로 또는 `ament_index`의 installed share 둘 다 실패) skip되고, 반대로
    drl_agent의 sibling `src/` 트리는 있는데 그 안에 `hunter_se_gazebo/config/hunter_se_cmd_
    prefilter.yaml`이 없으면(이 저장소에서는 두 패키지가 항상 같은 checkout 안에 있으므로 비정상
    상태) skip이 아니라 `pytest.fail`로 처리한다.
  - **경로 샘플링 해상도**: `directional_risk.rollout_path_samples`(기본 15, 이전 고정값 5에서
    상향)가 `horizon_sec`를 몇 개의 시각으로 나눌지 결정한다. 샘플 사이 간격이 넓으면 빠르게
    가로지르는 사람의 실제 최소거리가 두 샘플 사이에서 발생할 때 과소평가(실제보다 안전하게 계산)될
    수 있다 — 즉각적인 차단 사항은 아니라고 판단해 기본값만 올렸고, analytic segment-minimum 비교로
    바꾸는 것은 후속 과제로 남겨뒀다.
  - **Midpoint-heading 위치 적분**: `pure_pursuit.ackermann_swept_path`는 속도/조향각은
    substep 시작·끝의 평균(`v_mid`/`steer_mid`)을 쓰면서도, 처음에는 위치 적분에 substep **종료
    후** heading(`heading += omega*dt` 다음에 `cos(heading)`/`sin(heading)`)을 사용해 2 m/s·최대
    조향·1초·15 samples 조건에서 analytic 원호 끝점과 ~88mm 차이가 나는 문제가 있었다. 지금은
    substep **중점** heading(`heading_mid = heading + 0.5*omega*dt`)으로 위치를 적분한 뒤에
    `heading`을 갱신하도록 고쳐, 같은 조건의 오차가 ~0.7mm로 줄었다(제동 이산화 잔여 오차, 예:
    1 m/s off-grid 제동거리 오차 ~3.3mm 수준은 Dc=3.0m 스케일에서 무시 가능). 고정 속도·고정
    조향각에 대한 analytic circular-arc 회귀 테스트
    (`test_swept_path_constant_arc_matches_analytic_rollout_endpoint`, `pure_pursuit.
    ackermann_rollout`의 닫힌형 원호 끝점과 비교)와 sample 경계에 걸치지 않는 초기 속도의 제동거리
    테스트(`test_swept_path_braking_from_off_grid_speed_still_close_to_analytic`)로 회귀
    테스트됨 — 둘 다 `tests/test_pure_pursuit_speed_steering.py`.
- **`waypoint_yield`/`waypoint` 기본 경로**: `directional_risk.waypoint_trajectory_risk_enabled=false`
  이면 `aux_prediction_labels.compute_directional_risk_map()`으로 로봇 **현재 위치**만 쓰는 기존
  per-sector 조회를 사용한다. `target_v`/`target_cmd_steering`에 의존하지 않으며, 이 의미는
  `phase2/both_legacy`, `phase2/baseline`, `phase2/reward_shaping_only`,
  `phase2/action_risk_head_only`, `phase2/obs_norm_optim_split`에 유지된다.
- **`waypoint_yield` trajectory-risk 경로**: `phase2/both`와 `phase2/both_trajrisk_rbs`는
  `directional_risk.waypoint_trajectory_risk_enabled=true`로 waypoint_yield action에도 Ackermann
  swept-path target을 적용한다. action 계약은 3D waypoint+yield 그대로지만, action-risk/aux target은
  선택된 waypoint가 유도하는 속도·조향 rollout을 반영한다.

### 1.5 Fresh-run 필요성

`action_dim`이 2로 바뀌면 actor/critic/action-risk-head 입력 폭이 phase2 checkpoint(action_dim=3)와
달라 어차피 로드가 깨진다. 이를 shape-mismatch crash로 우연히 발견하게 두지 않고,
`drl_agent/config/validation.py`의 `ConfigValidator._check_action_mode`가 `environment.action_mode
== "speed_steering"`이고 `resume=True`이면 재개 대상 체크포인트의 **action 계약을 검증한 뒤에만**
허용한다:

- `TrainTQCCurriculum._augment_profile_manifest_with_action_contract()`가 매 run 시작 시
  `configs/profile_manifest.json`에 `action_mode`/`action_dim`을 기록한다(기존 `_write_profile_
  manifest_if_requested`가 쓴 manifest에 추가로 덧붙임).
- resume 시 `_check_action_mode`가 재개 대상 run의 `configs/profile_manifest.json`을 읽어
  `action_mode`/`action_dim`이 **현재 profile과 정확히 일치**할 때만 통과시킨다. manifest가 없거나
  (legacy layout, 또는 이 기능 이전 run), 값이 다르면(예: phase2의 `waypoint_yield`/`action_dim=3`
  체크포인트) 하드 에러로 거부한다.
- 즉 같은 `speed_steering`(`action_dim=2`) 계약으로 만들어진 phase3 체크포인트끼리는 재개 가능하고,
  다른 계약의 체크포인트를 잘못 로드하는 경우만 차단된다 — "무조건 fresh-run 전용"이 아니다.

`train_node.py` / `run_profile.py --validate-only` 양쪽 경로 모두에서 학습 시작 전에 걸린다.
(`tests/test_config_validation.py`의 `test_speed_steering_resume_*` 케이스들이 accept/reject 양쪽을
회귀 테스트한다.)

## 2. Risk-balanced replay + weighted loss

### 2.1 Replay metadata

`rl/replay/buffer.py`의 `LAP`에 선택적 `risk_meta` 배열(고정 4열: `[stage, human_event,
risk_positive, collision_or_near]`, `RISK_META_COLUMNS`)을 추가했다. `store_risk_meta=False`(기본)면
배열 자체가 생성되지 않아 기존 buffer와 완전히 동일. 값은 `training/train_tqc_curriculum.py`/
`train_rl.py`의 `_compute_risk_meta()`가 매 step `replay_buffer.add()` 호출 시 계산한다:

- `human_event` — 이번 step의 privileged nearest-human 거리(`_human_min_dist_m_from_label`)가
  안전 sentinel(사람 없음, `== self._label_Dc`) **미만**인지 — 단순히 label이 존재하는지가 아니라,
  실제로 D_c 이내에 사람이 있는지로 판정한다(label은 사람이 없어도 항상 존재하고 sentinel 값 1.0을
  반환하므로, "label 존재 = event"로 판정하면 사람 없는 stage 0-2 transition까지 전부 human pool에
  들어가 버린다 — 이제는 수정됨).
- `risk_positive` — action_risk_head env target의 `risk_dir`이 임계값(`risk_meta.risk_positive_threshold`,
  기본 0.5) 초과인지. 헤드가 꺼져 있으면 human distance가 `risk_meta.human_risk_distance_m`(기본 2.0m)
  이내인지로 대체.
- `collision_or_near` — 환경이 `/step` 응답에 직접 실어보내는 **FULL-360 진짜 collision 판정 +
  최소거리**(`response.collision`/`response.min_obstacle_dist_m`, `EnvInterface.step()`이
  `self.last_collision`/`self.last_min_obstacle_dist_m`로 캐시)가 `near_collision_dist_m` ROS
  파라미터(paper-metric H-Coll 트래커와 동일 값, 기본 0.5m) 미만인지. `next_state[:environment_dim]`에서
  유추하지 **않는다** — 그 슬라이스는 RL 입력용 전방 180° `obs_state`이지 `check_collision()`이 실제로
  쓰는 전방위 360° `environment_state`가 아니어서(둘 다 길이가 `environment_dim`이라 우연히 타입은
  맞지만 값은 다른 배열이다), 측면·후방 충돌을 놓칠 수 있었다 — 이제는 수정됨. `environment_360.py`
  (Classic Gazebo 변형)도 동일하게 두 필드를 채운다.

체크포인트: `risk_meta`는 npz의 선택적 키(`aux_target`/`action_risk_target`과 동일 패턴)로
저장/복원된다. **의도적으로 fail-fast가 아니라 graceful fallback**을 택했다 — 이전 checkpoint에
`risk_meta` 키가 없으면 크래시하거나 값을 추측하지 않고 해당 슬롯을 0으로 남긴다(=모든 pool에서
"이벤트 없음"으로 취급되어 risk-balanced sampler가 자동으로 uniform pool로 fallback한다).

### 2.2 Risk-balanced sampling

`hyperparameters.replay_buffer.risk_balanced_sampling.enabled`(기본 `false`)로 켠다. **켜져도 TQC
critic은 항상 기존과 동일한 uniform/prioritized `sample()` 배치를 그대로 쓴다** — 별도의
`sample_risk_balanced()` 배치가 **aux_prediction/action_risk_head의 지도학습 loss에만** 쓰인다
(`rl/algorithms/tqc/agent.py::train()`). 인코더는 이 배치에 대해 별도 forward를 한 번 더 돌리므로
(risk-balanced 배치용 `z_rb`) 인코더도 이 신호로 학습되지만, critic loss 자체의 데이터 분포는
전혀 바뀌지 않는다.

Pool 구성(`replay_buffer.risk_balanced_sampling.ratio_*`, 초기값 uniform 0.5 / human_risk 0.25 /
collision 0.25):

- **uniform pool**: 버퍼 전체에서 균등 샘플.
- **human_risk pool**: `human_event` 또는 `risk_positive`가 1인 transition만.
- **collision pool**: `collision_or_near`가 1인 transition만.

특정 pool이 비어 있으면(예: 초기 warmup, 사람 없는 stage) 해당 지분은 **자동으로 uniform pool로
재분배**된다 — 에러도, 빈 batch도, 배치 크기 축소도 없다(`LAP.sample_risk_balanced`). Pool이
비어있지 않지만 요청 개수보다 작으면 replacement 샘플링으로 채운다(중복은 허용, 크래시는 없음).

Action-conditioned future-action lookup(`get_last_future_actions`)과 temporal state-history
lookup(`get_last_state_history`)은 기존에도 episode boundary(`traj_end`)를 넘지 않는 로직이었는데,
이번에 `indices=` 파라미터를 추가해 risk-balanced 배치에도 **동일한** boundary-safe 로직을 그대로
재사용한다(로직 자체를 복제하지 않음). `indices` 생략 시 기존처럼 `self.ind`(직전 `sample()`의
인덱스)를 쓰므로 기존 호출부는 무변경.

### 2.3 Weighted loss

- `hyperparameters.action_risk_head.pos_weight`/`pos_weight_cap`/`positive_threshold`/`loss_type`
  (`rl/networks/action_risk_head.py::weighted_action_risk_loss`) — action-risk head의 (risk_dir,
  min_dist_dir) 지도학습 loss에 양성(risk>threshold) element를 up-weight. `pos_weight==1.0`(기본)이면
  기존 `F.mse_loss`와 byte-identical.
- `hyperparameters.aux_prediction.risk_map_positive_weight`(기존 키, 이번에 `_cap` 추가)/
  `hazard_pos_weight`(기존 키) — aux head의 continuous risk-map / binary hazard-sector loss에 동일
  패턴 적용. `risk_map_loss_type: mse|smooth_l1`도 신규 추가(기본 `mse` = 기존과 동일).
- 두 가중치 모두 `*_cap`으로 상한을 둔다(raw class-imbalance ratio를 그대로 쓰지 않음). 신규
  profile은 `pos_weight=6.0`, `cap=10.0`(action_risk_head), `risk_map_positive_weight=6.0`,
  `hazard_pos_weight=6.0`(둘 다 `cap=10.0`) — 요청된 5~10 범위 내 보수적 값. 근거: phase2/both
  replay는 human-free stage 0-2 transition이 다수이고, human이 있는 stage에서도 "이번 step 임박
  위험 없음"이 절대다수라 raw inverse-frequency 비율은 20-50배 수준으로 추정된다. 6배는 그 불균형을
  일부 보정하면서도, 소수의 (잠재적으로 noisy한) 양성 라벨이 trunk loss를 지배하지 않도록 하는 절충값.

### 2.4 로깅

`Agent.train()`이 매 `scalar_log_interval`/`json_log_interval` step마다 다음을 TensorBoard +
`tqc_metrics.json`에 남긴다:

- `risk_balance/sampled_human_event_frac`, `risk_balance/sampled_risk_positive_frac`,
  `risk_balance/sampled_collision_frac` — 이번 step에 실제로 뽑힌 risk-balanced 배치의 이벤트 비율
  (목표 비율이 아니라 **실측값** — pool 부족으로 uniform에 재분배된 경우 자동으로 반영됨).
- `action_risk/loss`, `action_risk/positive_loss`, `action_risk/safe_loss`,
  `action_risk/positive_recall`, `action_risk/positive_f1`, `action_risk/pos_weight_applied`.
- `aux/risk_mse`, `aux/risk_positive_error`, `aux/risk_safe_error`, `aux/risk_positive_recall`,
  `aux/risk_positive_f1`.

`positive_recall`/`positive_f1`/`positive_error` breakdown은 "all-safe collapse"(전체 RMSE는
좋지만 실제 양성 이벤트에서는 거의 항상 틀리는 상태)를 탐지하기 위한 것 — 이 값들은 loss 계산에
관여하지 않는 순수 진단용 지표다.

## 3. 실행 명령

```bash
# validate만 (ROS 불필요)
python3 ros2_ws/src/drl_experiments/scripts/run_profile.py phase3/speed_steering_risk_balanced --validate-only

# 실제 학습 (최초 실행은 fresh run; 이후 재개는 §1.5의 action-contract 검증을 통과해야 함)
ros2 run drl_agent environment_curriculum_node.py --ros-args -p profile:=phase3/speed_steering_risk_balanced
ros2 run drl_agent train_node.py --ros-args -p profile:=phase3/speed_steering_risk_balanced -p seed:=0
```

`base_file_name`/`output_prefix`는 `tqc_phase3_speed_steering_risk_balanced` — `runtime/experiments/`
아래 run 디렉터리와 체크포인트 파일명이 `tqc_phase2_*`와 절대 충돌하지 않는다.

## 4. 검증 방법

profile/config 변경 뒤에는 아래 테스트 묶음으로 speed_steering action 계약, trajectory-risk target,
risk-balanced replay, resume validation을 확인한다.

```bash
cd ros2_ws/src/drl_agent
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 15 \
  python3 -m pytest -q -p no:cacheprovider \
  tests/test_pure_pursuit_speed_steering.py \
  tests/test_directional_risk_env.py \
  tests/test_risk_balanced_replay.py \
  tests/test_speed_steering_agent_dims.py \
  tests/test_phase3_speed_steering_profile.py \
  tests/test_config_validation.py

# 전체 회귀(패키지 전체)
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 15 \
  python3 -m pytest -q ros2_ws/src/drl_agent

# colcon 빌드 확인 (임시 build/install/log — 현재 사용 중인 build/install을 덮어쓰지 않음)
cd ros2_ws && source /opt/ros/humble/setup.bash
colcon build --packages-select drl_agent drl_agent_interfaces drl_experiments \
  --build-base /tmp/phase3_build --install-base /tmp/phase3_install --log-base /tmp/phase3_log \
  --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release

# Gazebo E2E 스모크 (별도 ROS_DOMAIN_ID, 현재 학습과 완전 분리된 세션에서만):
# 1) speed_steering이 실제로 정지/직진/좌우 회전을 만들어내는지 짧은 랜덤-액션 episode로 확인
# 2) risk_map_reward / action_risk_head env-side target이 예외 없이 나오는지 확인
# 3) [§1.4 prefilter 동역학 반영 검증 -- 필수] 제동/급가속 구간에서 CSV의 cmd_v(명령 속도),
#    filtered_cmd_v(prefilter 통과 후 실제 명령), latest_actual_signed_speed(odometry 실측),
#    action_risk_head/risk_map_reward의 risk target을 나란히 놓고 확인:
#    - 정지 action 직후에도 filtered_cmd_v/실측 속도가 즉시 0이 되지 않고 accel_limit_mps2/
#      brake_decel_mps2 근처 기울기로 감소하는지 (prefilter가 실제로 그렇게 동작하는지 자체 확인)
#    - 그 제동 구간 동안 risk target이 "이미 멈춘 것"처럼 과도하게 낮아지지 않는지 -- 사람이
#      근처에 있는 제동 상황에서 risk_dir이 급락하면 §1.4의 근사(특히 speed_lag_tau_sec 미모델링)가
#      이 설정에서는 부족하다는 신호이므로, rollout_path_samples 상향이나 lag 모델링 추가를 재검토
# (현재 세션에서는 실행하지 않음 — 진행 중인 phase2/both 학습과 Gazebo 인스턴스를 공유하지 않기 위해
#  반드시 별도 컨테이너/세션 + 별도 ROS_DOMAIN_ID로 실행할 것)
```
