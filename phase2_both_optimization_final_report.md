# `phase2/both` TQC 커리큘럼 학습 속도·성능 개선 — 최종 보고서

작업 범위: `environment_curriculum_node.py --ros-args -p profile:=phase2/both` +
`train_node.py --ros-args -p profile:=phase2/both -p seed:=0` 로 실행되는
`phase2/both` TQC 커리큘럼 학습 경로. `n_critics=5`, `n_quantiles=25`, 은닉 차원은
어디에서도 축소하지 않았습니다.

---

## 1. 검증된 원인과 병목

Stage 0 조사에서 **실측**(추정 아님)으로 확인한 병목:

| 항목 | 측정값 | 근거 |
|---|---|---|
| `/step` 평균 wall time | 203.6ms (steps/s ≈ 4.91) | `baseline_step_timing.py`, n=100+60, 반복 재현 |
| `propagate_state` 자체 비용(event-driven wait) | 100.8ms (unpause+sleep(0.1)+pause) | 격리된 프로브 |
| **신규 발견**: `_await_future`의 50ms 폴링 | 콜당 ~50.1ms의 "감지 지연"(실제 Gazebo 응답은 <1ms) | `await_future_poll_repro.py` |
| `use_sim_time` | `False` (확인됨, 사람 타이머가 wall-clock 기반) | `ros2 param get /gym_node use_sim_time` |
| Gazebo 물리 설정 | `max_step_size=0.001`, `real_time_factor=1.0` | `drl_arena.world` |
| 실제 replay checkpoint (180k/1M transitions) | 1032.5MB @ float64, `max_size=1e6`이면 eager 할당으로 ~5.7GB 추정 | 실제 체크포인트 `np.load` |
| stage 0–2 aux/action-risk 대상 | 사람 없음 → 항상 상수/degenerate (기존 9개 근거 중 하나로 사전 확인됨) | 커리큘럼 설정 + 기존 replay 검사 |
| actor 업데이트 시 critic 미고정 | `action_risk_head`는 이미 freeze되는데 critic은 안 됨 (9개 근거 중 하나) | 코드 검토로 확인 |
| per-step 동기 TensorBoard/JSON 로깅 | `.item()`이 sink당 중복 호출(같은 텐서 2회) | 코드 검토로 확인 |
| replay buffer dtype | `np.zeros(...)`에 dtype 미지정 → float64 | 코드 검토로 확인 |

---

## 2. 실제 반영한 변경 (Stage 1–8)

### Stage 1 — 사람 모션 결정론화 (`human_deterministic_stepping`, 기본 OFF)
사람 이동을 wall-clock 독립 타이머 대신 `propagate_state()`에서 정확한 정수 tick
수로 직접 구동. `compute_human_tick_plan()`이 `(duration, human_update_rate)`로부터
tick 수/dt를 계산(정수 배수 아니면 fail-fast). `_human_timer_callback`을
`_advance_humans_one_tick(dt)` 공유 바디로 리팩터링해 레거시·신규 경로가 완전히
같은 로직을 씀. **활성화 시 in-episode RNG draw 수가 wall-clock 속도와 완전히
무관해짐**(기존에는 그렇지 않았음 — 이것이 핵심 재현성 버그였음).

### Stage 2 — Gazebo 결정론적 스텝 (`gazebo_deterministic_stepping`, 기본 OFF)
`WorldControl(pause=True, multi_step=N)` 단일 호출로 물리를 정확히 N스텝 진행
(레거시는 unpause+pause 2회 호출). **실측**: N=1/50/100 → sim-time 델타가 정확히
0.001/0.05/0.1s. World-control 호출 수가 절반으로 줄어 **실측 24~25% 지연 감소**
(§10 표 참조). Sensor freshness는 `reset_callback`이 이미 쓰던 것과 동일한
scan/odom 카운트 기반 bounded wait로 검증(경고 0건, 매번 충분).

**제외된 하위 항목**: `/clock` 기반 정밀 완료 확인(원안 스펙) — §3 참조.

### Stage 3 — Stage 기반 aux/action-risk 연산 게이팅
- `TemporalFusionEncoder`: `temporal_gain==0`일 때 conv1d/GRU forward 자체를
  스킵(기존은 계산 후 0 곱셈). 출력/gradient가 기존과 수치적으로 동일함을 테스트로 증명.
- `Agent.train()`: `beta = _current_aux_beta()`를 먼저 계산해 `beta==0`이면
  aux_head forward(및 그 안의 temporal-context GRU, action-conditioned future-action
  lookup)를 통째로 스킵.
- 신규 `action_risk_head.enable_from_stage`(기본 0=항상 활성, 기존 동작 보존):
  stage 미만이면 forward 전체 스킵, `critic_risk_input` 활성 시 critic에는 고정
  all-zero (batch,2) 텐서를 대신 공급해 **critic 입력 shape가 stage에 따라 절대
  바뀌지 않음**을 보장.
- **`phase2/both`에 실제로 적용한 의도적 동작 변경**: `aux_prediction.
  stagewise_loss_schedule=[0,0,0,0.1,...]`, `action_risk_head.enable_from_stage=3`
  — stage 0–2(사람 없음)에서 aux/action-risk가 전혀 학습하지 않도록 설정.
  `temporal_actor_context.stage_enable_from`은 **변경하지 않음**(명시적 지시 준수).

### Stage 4 — 희소/degenerate risk target 처리
- `risk_map_positive_weight`(기본 1.0=균등, MSE와 byte-identical)로 risk-map의
  양(+) 셀을 더 무겁게 가중 가능.
- `hazard_pos_weight`(기본 None=기존 BCE와 동일, `hazard_pos_weight_cap=20.0`으로
  항상 상한).
- 공식 aux 평가(`aux_eval_metrics.py`)에 `aux_dynamic_sample_count/frac`,
  `aux_positive_event_count`를 추가 — RMSE가 "상수 0 구간을 재는 착시"인지
  구분 가능해짐. 이 모듈은 이전에 **단위 테스트가 전혀 없었음** — 8개 신규 테스트로
  기존 계약까지 함께 고정.
- **Critic 샘플링은 손대지 않음**(요청대로 균일 유지). 새 replay 메타데이터도
  추가하지 않음(가중치 방식이 target 값 자체로 판단하므로 불필요, checkpoint
  스키마 리스크 회피).

### Stage 5 — TQC 업데이트 불필요 연산 제거 (모두 수학적으로 동일, RNG/체크포인트 영향 없음)
- `_frozen_params()` exception-safe context manager 신설. actor 업데이트의
  `qf_pi = self.critic(...)` 호출을 critic 파라미터 freeze 상태로 실행(기존
  `action_risk_head`에 이미 있던 패턴과 동일 원리; critic_optimizer.zero_grad()가
  매 train() 시작 시 어차피 지우므로 **수치 결과에 영향 없음**, 낭비 backward만 제거).
  기존 `action_risk_head`의 bare set-False/call/set-True도 같은 context manager로
  교체(예외 안전성 버그 수정).
- `critic_target`/`encoder_target`/`checkpoint_encoder`/`action_risk_head_target`/
  `checkpoint_actor`에 생성 시점 영구 `requires_grad_(False)` (Polyak `.data.copy_()`
  는 영향 없음, 모두 no_grad 컨텍스트에서만 쓰이는 것 확인 후 적용).
- `_polyak_update_foreach()`: `torch._foreach_*`로 배치 처리(critic 5개 만큼의
  파라미터 텐서에 대해 CUDA kernel launch 횟수 감소). per-parameter loop와
  수치적으로 동일함을 직접 비교 테스트 + end-to-end train() 기반 비교 테스트로 증명.
- **이미 최적**이라 손대지 않은 것: `optimizer.zero_grad()`의 `set_to_none=True`는
  이 저장소의 PyTorch 2.4.1에서 이미 기본값(확인됨). Temporal feature 재사용은
  기존 코드가 이미 올바르게 하고 있었음.
- **제외**(요청대로 미적용): entropy-coefficient/actor forward 재사용(RNG 순서
  변경 위험), `torch.compile`/AMP/CUDA graphs(별도 검증 필요, 이번 범위 아님).

### Stage 6 — 로깅 interval화
`scalar_log_interval`/`json_log_interval`/`json_flush_interval`(모두 기본 1=기존과
byte-identical). 두 sink에 공통으로 필요한 `.item()`을 **딱 한 번만** 계산해
재사용(기존은 sink마다 중복 호출). JSON은 메모리에 버퍼링 후 `json_flush_interval`
개마다 한 번에 open/write/close(기존은 레코드마다 open/close). `flush_logs()`를
`train_tqc_curriculum.py`의 `finally:` 블록 한 곳에 연결 — 정상 종료/Ctrl+C/
EnvServiceError/기타 예외 4개 경로 모두 이 한 지점을 통과하므로 별도 처리 불필요.
조사 중 발견한 죽은 코드(`aux_loss_val` — 계산만 하고 아무 데도 안 씀, `compute_aux_loss`
자체가 이미 동일 값을 `aux/loss`로 반환) 제거. `phase2/both`에는
`scalar/json_log_interval=10, json_flush_interval=100`을 실제 적용(순수 로깅
빈도이므로 학습 수치에 영향 없음을 구조적으로 보장 — 로깅이 옵티마이저에 피드백되지
않음). 다운스트림 분석 도구(`aggregate_results.py` 등)는 `tqc_metrics.json`/
TensorBoard 스칼라를 전혀 읽지 않고 별도의 `eval_metrics_*.csv`(eval_freq 주기)를
쓴다는 것을 코드 검색으로 확인 — sparse 로깅이 논문 분석 파이프라인에 영향 없음.

### Stage 7 — Replay buffer float32 전환 + checkpoint 호환
`state/action/next_state/reward/not_done/aux_target/action_risk_target/traj_end`
모두 `dtype=np.float32`로 명시(기존 dtype 미지정 → float64). `sample()`/
`get_last_aux()`/`get_last_action_risk()`는 `torch.tensor(...)` 대신
`torch.from_numpy(...).to(device)`로 전환 — advanced-indexing이 만드는 복사본을
추가로 또 복사하지 않고 그대로 공유. `load()`는 새 `_cast_loaded()` 헬퍼로 **명시적,
로그를 남기는** cast를 수행하며, 기존 shape/schema 검증(모두 유지) **이후에** 실행.
**실제 기존 checkpoint로 read-only 검증**: `runtime/experiments/
20260726_182100_tqc_phase2_both_seed0/.../replay_buffer.npz`(180k transitions)를
로드 → 모든 필드가 float64→float32로 정확히 캐스팅·로그됨, 정밀도 rtol=atol=1e-6
이내 확인, **원본 파일 sha256/크기/mtime이 로드 전후 완전히 동일**(never overwritten
확인). **실측 메모리 절감**: 이 실제 체크포인트 기준 1032.5MB → 516.2MB(정확히 50%).

### Stage 8 — 관측 정규화 + 옵티마이저 분리 (고립된 실험 기능, `phase2/both`에 미적용)
- 신규 `obs_normalization.py`: 고정 물리 범위 정규화(러닝 정규화 아님). LiDAR/
  goal_dist/heading/prev_action/speed/yaw_rate/steering 스케일. 87-D(스태킹 없음)와
  327-D(observation_time_context, history_len=4) 레이아웃 모두에서 현재 프레임과
  모든 history 프레임의 LiDAR 블록이 **동일한** `lidar_scale`을 쓰도록 스케일
  벡터를 구성(요청 사항 그대로 충족). `Agent.train()`/`select_action()`에 배선.
  **명시적 범위 제외**: `get_last_state_history()` 기반 aux temporal-context 경로는
  이번 패스에서 정규화 미적용(별도 buffer fetch, 문서화된 제외).
- Checkpoint manifest: `tqc_io.save()`가 `<file>_obs_norm_manifest.json`을 항상
  기록. `load()`는 가중치 로드 **전**에 enabled 불일치/스케일 불일치/manifest
  없음(Stage 8 이전 체크포인트인데 정규화 활성)을 모두 fail-fast.
- `optimizer_groups`(기본 disabled=단일 flat Adam, 기존과 동일): 활성화 시 critic/
  encoder/aux_head/action_risk_head별로 다른 LR을 가진 **하나의** Adam 인스턴스
  (param_groups). encoder_lr을 critic_lr의 1/10로 설정한 실험에서 실제로 encoder
  파라미터 변화량이 더 작아짐을 controlled 비교 테스트로 증명.
- `updates_per_env_step`(UTD 비율)은 **이미 A1 실험으로 구현되어 있었음**(코드
  검색으로 확인) — 새로 만들 필요 없음.
- 신규 실험 프로파일 `drl_experiments/profiles/phase2/obs_norm_optim_split/`
  1개를 생성해 두 메커니즘을 함께 시연(environment_curriculum.yaml/
  train_tqc_config.yaml/train_tqc_curriculum_config.yaml은 `phase2/both`와
  byte-for-byte 동일 — 공정성 유지, `hyperparameters_tqc.yaml`만 새 블록 추가).
  `ConfigValidator`로 `validation OK` 확인 + 실제 config로 end-to-end 스모크
  테스트(정규화 활성 확인, 옵티마이저 group이 2개의 다른 LR 생성 확인, stage 0/3
  모두 clean).

---

## 3. 제외·보류한 제안과 이유

| 제안 | 상태 | 이유 |
|---|---|---|
| `_await_future`의 50ms 폴링 완화(event-driven callback, 또는 단순 간격 단축) | **제외** | 두 가지 방식 모두(event-driven `add_done_callback`+`threading.Event`, 그리고 같은 폴링 루프의 간격만 0.05→0.001/0.01로 단축) **실제 `/step` 호출을 재현 가능하게 행업(hang)**시킴. reset은 동일 경로로 매번 성공했는데 첫 실전 `/step`만 행업 — `_wait_for_sensor_freshness`(동일한 spin_once 패턴)는 같은 위치에서 문제없이 동작해 spin_once 자체의 문제는 아님을 확인. 원본 50ms 폴링 루프로 되돌리면 즉시 정상화됨(대조군으로 재확인). 근본 원인은 이 세션 예산 내에서 완전히 규명하지 못함(CPython GIL fairness/스레드 기아 가설). `gazebo_await_future_poll_interval_landmine` 메모리에 기록. |
| Stage 2의 `/clock` 기반 정밀 완료 확인(원안 스펙 그대로) | **제외**, 대체 설계로 구현 | `multi_step` 자체는 요청한 스텝 수만큼 정확히 진행함을 확인했으나(N=1/50/100→정확히 0.001/0.05/0.1s), 이를 `/clock` 구독+bounded wait로 확인하는 코드가 **역시 실전 `/step`만 행업**시킴(같은 실패 클래스, reset은 성공). 대신 sleep(duration) + 이미 검증된 sensor-freshness wait로 대체 — 실측 24~25% 개선은 그대로 확보. `gazebo_multi_step_clock_wait_landmine` 메모리에 기록. |
| Stage 4의 replay 메타데이터(`curriculum_stage`/`dynamic_present`/`risk_event`) 추가 | **제외** | 원안에 "필요시"로 조건부 명시. 가중치 방식(target 값 자체 기준)이 메타데이터 없이도 목적을 달성하므로 checkpoint 스키마 리스크를 추가로 감수할 이유가 없음. |
| Stage 4의 실제 가중치 값을 `phase2/both`에 적용 | **제외**(메커니즘만 구현) | 원안 표현이 "consider"(탐색적)이지 Stage 3의 "activate from stage 3"처럼 명확한 지시가 아님. 구체적 가중치 값은 연구 설계 판단이라 임의로 정하지 않음. |
| Stage 8의 7개 프로파일 전체(sweep family) | **제외**(1개만 생성) | 나머지 6개(특히 각 arm의 정확한 하이퍼파라미터)는 사용자의 연구 설계 결정 영역. 대표 1개(정규화+옵티마이저 분리 결합)로 메커니즘이 실제로 동작함을 증명. UTD2/UTD4는 기존 `updates_per_env_step` CLI 오버라이드로 이미 커버됨(별도 프로파일 불필요). |
| aux 공식 평가에 TTC/hazard prevalence 추가 | **제외** | `AuxEvalAccumulator`가 애초에 risk_map/min_dist만 다루고 TTC/hazard는 평가 대상이 아니었음(학습 loss 로깅만 존재) — "기존 지표에 prevalence 추가"보다 훨씬 큰 별도 기능(새 accumulator 필드, aux_eval.py 배선, CSV 스키마 변경)이라 이번 범위 밖으로 명확히 분리. |

---

## 4. 변경 파일 목록

**수정** (18개):
```
ros2_ws/src/drl_agent/config/environment.yaml
ros2_ws/src/drl_agent/config/environment_curriculum.yaml
ros2_ws/src/drl_agent/drl_agent/env/humans/human_motion_manager.py
ros2_ws/src/drl_agent/drl_agent/env/simulation/environment.py
ros2_ws/src/drl_agent/drl_agent/env/simulation/gazebo_service_wait.py
ros2_ws/src/drl_agent/drl_agent/rl/algorithms/tqc/agent.py
ros2_ws/src/drl_agent/drl_agent/rl/checkpointing/tqc_io.py
ros2_ws/src/drl_agent/drl_agent/rl/networks/action_risk_head.py
ros2_ws/src/drl_agent/drl_agent/rl/networks/aux_losses.py
ros2_ws/src/drl_agent/drl_agent/rl/networks/aux_prediction.py
ros2_ws/src/drl_agent/drl_agent/rl/networks/aux_temporal.py
ros2_ws/src/drl_agent/drl_agent/rl/replay/buffer.py
ros2_ws/src/drl_agent/drl_agent/training/aux_eval_metrics.py
ros2_ws/src/drl_agent/drl_agent/training/train_tqc_curriculum.py
ros2_ws/src/drl_agent/tests/test_action_risk_head.py
ros2_ws/src/drl_agent/tests/test_aux_prediction.py
ros2_ws/src/drl_agent/tests/test_profile_loader.py
ros2_ws/src/drl_agent/tests/test_temporal_fusion_encoder.py
ros2_ws/src/drl_experiments/profiles/phase2/both/hyperparameters_tqc.yaml
```

**신규** (11개):
```
ros2_ws/src/drl_agent/drl_agent/rl/networks/obs_normalization.py
ros2_ws/src/drl_agent/tests/test_agent_obs_normalization.py
ros2_ws/src/drl_agent/tests/test_aux_eval_metrics.py
ros2_ws/src/drl_agent/tests/test_gazebo_deterministic_stepping.py
ros2_ws/src/drl_agent/tests/test_human_deterministic_stepping.py
ros2_ws/src/drl_agent/tests/test_obs_normalization.py
ros2_ws/src/drl_agent/tests/test_optimizer_groups.py
ros2_ws/src/drl_agent/tests/test_replay_buffer_float32.py
ros2_ws/src/drl_agent/tests/test_tqc_agent_logging.py
ros2_ws/src/drl_agent/tests/test_tqc_update_optimizations.py
ros2_ws/src/drl_experiments/profiles/phase2/obs_norm_optim_split/ (5개 파일)
```

`package.xml`은 Stage 2 탐색 중 `rosgraph_msgs` 의존성을 추가했다가(제외된
`/clock` 실험용) 제거해 최종적으로 diff 없음(git status에 나타나지 않음).

---

## 5. Checkpoint·replay 호환 정책

| 변경 | Fresh-run 필요? | 정책 |
|---|---|---|
| Stage 1 (`human_deterministic_stepping`) | 아니오 | Agent 체크포인트/replay와 무관(환경 쪽 로직) |
| Stage 2 (`gazebo_deterministic_stepping`) | 아니오 | 동일 |
| Stage 3 (aux/action-risk stage gating) | 아니오 | 파라미터 shape 불변(스킵만 함), `set_curriculum_stage`가 이미 검증된 resume 경로로 게이트 자동 복원 |
| Stage 4 (loss weighting) | 아니오 | loss 계산 방식만 변경, 파라미터/체크포인트 영향 없음 |
| Stage 5 (compute 최적화) | 아니오 | 전부 수학적으로 동일한 연산 |
| Stage 6 (로깅 interval) | 아니오 | 로깅은 체크포인트와 무관 |
| Stage 7 (replay float32) | **아니오** — 명시적 호환 레이어 구현 | 기존 float64 체크포인트를 `_cast_loaded()`가 로드 시 자동 변환(로그 남김). 실제 체크포인트로 read-only 검증 완료, 원본 불변 확인 |
| Stage 8 정규화 | **활성화하면 fresh-run 전용** | Manifest 불일치 시 `RuntimeError`로 fail-fast(가중치 로드 전에 체크) — 실수로 다른 정규화 계약의 체크포인트를 이어받는 것을 원천 차단 |
| Stage 8 `critic_risk_input`(기존 기능, 이번에 건드리지 않음) | 기존 정책 그대로 | extra_dim 변경은 원래도 fresh-run 전용(기존 문서화됨) |

---

## 6. Fresh-run이 필요한 변경

`phase2/both`에 실제 적용한 변경 중 **fresh-run이 필요한 것은 없음** — Stage 3의
stagewise schedule/enable_from_stage 변경은 파라미터 shape를 바꾸지 않으므로 기존
체크포인트에서 resume 가능(다만 재개 시점부터 게이팅 동작이 달라짐은 당연히 의도된
효과). Stage 8(정규화)은 애초에 `phase2/both`에 비활성이므로 해당 없음.

---

## 7. Host 테스트 결과

```
python3 -m pytest -q   (cd ros2_ws/src/drl_agent, ROS/torch 미빌드 환경)
572 passed, 248 skipped, 0 failed   (총 820)
```
skip은 전부 rclpy/torch가 필요한 테스트(호스트에 미설치 — CLAUDE.md에 문서화된
정상 동작).

---

## 8. Docker clean build·테스트 결과

컨테이너 `7a2702b311a1`에서 **진짜 clean build**(`rm -rf build install log` 후
`colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release`, 전체 32개 패키지 대상):

```
Summary: 32 packages finished [1min 31s]
  5 packages had stderr output (모두 drl_agent와 무관한 기존 컴파일러 경고:
  hunav_agent_manager, hunav_gazebo_fortress_wrapper, lio_sam, octomap_server,
  pointcloud_to_laserscan)
```

`colcon test --packages-select drl_agent` (같은 clean build 위):
```
pytest: tests="820" errors="0" failures="0" skipped="0"
lint_cmake: Passed | xmllint: Passed
colcon test-result --verbose: 825 tests, 0 errors, 0 failures, 0 skipped
```
(825는 colcon 자체 집계 — pytest(820)+lint_cmake(1)+xmllint(1)+colcon 내부
가산(+3). 정확한 pytest 자체 수치는 820.)

---

## 9. ROS/Gazebo E2E 결과

- **실전 명령어 그대로 실행**: 새로 clean 재기동한 Gazebo 위에서
  `environment_curriculum_node.py --ros-args -p profile:=phase2/both` →
  `train_node.py --ros-args -p profile:=phase2/both -p seed:=0` 실행.
  Run dir 자동 생성, profile manifest 기록, 모든 CSV/TensorBoard 로그 정상 개설,
  `State dimension: 327 / Action dimension: 3` 확인(phase2/both의
  temporal_actor_context 설정과 일치), stage 0(사람 없음) 커리큘럼 정상 적용.
  **2개 에피소드 완주**(T:64 Ep:1, T:194 Ep:2, 둘 다 COLLISION으로 종료 — 워밍업
  중 랜덤 정책이므로 예상된 동작) — 에러/traceback 0건(grep 확인).
- **Ctrl+C(SIGINT) 인터럽트 경로**: 정상적으로 `[Curriculum] Training
  interrupted by user.`로 캐치되어 `curriculum_state.json` 저장, `flush_logs()`
  포함된 `finally:` 블록 정상 통과(이번 실행은 warmup 12000스텝 중 194스텝만
  진행해 `agent.train()`이 아직 호출 안 됐으므로 버퍼링된 JSON은 없었음 — 이
  케이스는 `test_flush_logs_is_idempotent_and_safe_with_no_buffered_records`로
  단위 테스트에서 이미 정밀 검증됨). rclpy 자체의 기존 SIGINT/rcl_shutdown 중복
  호출 경고(이번 변경과 무관, 이 세션 초반부터 알려진 무해한 현상)만 발생.
- **Legacy vs. 결정론적 스텝, 같은 최종 코드베이스로 재확인**(단일 인스턴스로
  정리 후 측정 — 앞서 프로세스 중복으로 한 번 잘못된 측정값이 나왔던 것을
  직접 재현·원인규명 후 정정):
  - Legacy: 204.0ms/step (steps/s≈4.90)
  - `gazebo_deterministic_stepping=true`: 152.8ms/step (steps/s≈6.54) — Stage 2
    당시 최초 측정(152.3~154.6ms)과 재현성 확인.
  - sim-time 정확성: `multi_step=100` → 정확히 +0.100000000000023s(부동소수점
    오차 이내) — 별도 프로브로 확인.
  - Sensor freshness 경고: 60~100 스텝 연속 실행 중 **0건** 발생(항상 충분히 fresh).
- **Stage 3 게이팅 실전 검증**: `ros2 param set /gym_node curriculum_stage 3`으로
  실제 커리큘럼을 stage 3('first_human_clean')으로 전환 → 로그에 `humans=1`
  명시적 확인, 실제 활성 pedestrian 존재 상태에서 10 스텝 연속 성공(Stage 1+2
  결정론 모드 동시 활성 조합도 별도로 10 스텝 성공 확인).
- **이전-episode 데이터 유입 없음**: reset 직후 관측이 이전 에피소드와 섞이지
  않는 것은 기존 freshness-wait 메커니즘(건드리지 않음)이 담당하며, 이번
  변경들이 그 경로를 우회하지 않음을 코드 추적으로 확인.

---

## 10. 변경 전후 성능 수치 표 (전부 실측, Docker/RTX 4080)

| 모드 | 평균 step 시간 | steps/s | 비고 |
|---|---|---|---|
| Legacy 기준선 (`gazebo_deterministic_stepping=false`) | 203.6–204.5ms | 4.89–4.91 | 여러 차례 반복 측정, 표준편차 작음(p95 ≤210ms) |
| `gazebo_deterministic_stepping=true`만 | **152.3–154.6ms** | **6.47–6.57** | Stage 2 최초 측정 + 최종 재확인(단일 인스턴스) 일관됨 |
| `human_deterministic_stepping=true`만(stage 0, 사람 0명) | 202.5–208.1ms | ≈4.9 | 구조적 확인용, 사람 tick이 no-op이라 legacy와 근사 |
| `human_deterministic_stepping`+legacy Gazebo, stage 3 실사람 | ≈306ms | ≈3.3 | world-control 호출 4회(2배)로 증가 — `_await_future` 폴링 병목이 그대로 남아있어 오히려 legacy보다 느림(§3에서 그 폴링 자체는 고치지 못함을 명시) |
| `human_deterministic_stepping`+`gazebo_deterministic_stepping` 동시, stage 3 실사람 | 202.5–211.7ms | ≈4.8 | world-control 호출이 2회(gazebo_deterministic이 호출 수를 다시 줄여줌)로 306ms 대비 개선, legacy와 비슷한 수준(재현성 확보가 주목적, 처리량은 legacy와 대등) |

**해석**: 이번 작업에서 가장 크고 확실한 처리량 개선은 `gazebo_deterministic_
stepping`(~25%). `human_deterministic_stepping`은 **재현성 버그 수정**이 주
목적이며, `_await_future` 폴링 병목(제외됨)과 결합될 때는 처리량이 오히려
나빠질 수 있음을 정직하게 측정해 표에 남김 — 두 스텝 모드를 함께 켜면 그 손실이
상쇄됨.

후속 정리에서 `phase2/both/environment_curriculum.yaml`에는
`gazebo_deterministic_stepping: true`를 명시해, 사용자가 쓰는 기본
`profile:=phase2/both` 학습 명령이 위의 빠른 Gazebo stepping 경로를 사용하도록
맞췄습니다. 패키지 공통 `config/environment_curriculum.yaml`의 기본값은 비교군과
레거시 실행을 위해 여전히 false입니다.

---

## 11. 메모리 사용량 비교 (실측)

| 항목 | float64(기존) | float32(변경 후) | 절감 |
|---|---|---|---|
| 실제 phase2/both replay checkpoint (180k transitions, state_dim=327) | 1032.5MB | **516.2MB**(실측 로드 후 측정) | 정확히 50.0% |
| `buffer_size=1,000,000`(설정값) 기준 추정 전체 buffer | ~5.7GB(추정) | ~2.87GB(추정) | 50%(비율은 실측 기반 외삽, 절대값은 추정) |

`priority` 텐서는 애초에 `torch.zeros(...)`(기본 float32)라 이번 변경과 무관.

---

## 12. Stage별 aux/risk target 통계

- **정성적 확인**(정량적 장기 학습 없이, 이 작업 예산 내 실전 확인): stage 0
  (`humans=0`)에서 `_current_aux_beta()==0.0`이고 `action_risk_head`
  forward가 스킵됨을 실제 phase2/both 설정으로 직접 확인(`agent._action_risk_
  active is False`, `agent._current_aux_beta() == 0.0`). Stage 3
  (`humans=1`, `first_human_clean`)에서는 `beta==0.1`, `action_risk_active
  is True`로 즉시 전환됨을 동일한 실제 설정 객체로 확인.
- **기존(pre-fix) 실측 근거**: 실제 checkpoint(180k transitions, 이 작업
  시작 이전 데이터)를 조사한 결과 이 표본 전체가 stage 0–2 구간 위주였고
  aux/action-risk 대상이 상수/degenerate였다는 것은 작업 시작 시점의 사전
  확인 사항으로, 이번 Stage 3 fix가 정확히 이 문제를 겨냥해 구현됨.
- **장기 학습 곡선(수천~수만 step) 비교는 이번 작업 범위에서 수행하지 않음**
  — 실제 curriculum 승급(`min_stage_steps=30000`)까지 도달하려면 상당한
  wall-clock 시간이 필요해 이 세션에서는 수행하지 않았고, 대신 stage
  전환을 직접 강제해 메커니즘 자체의 정확성을 검증하는 방식을 택함(§9).
  이는 후속 ablation의 첫 항목으로 남겨둠(§13).

---

## 13. 남은 위험과 후속 ablation 제안

1. **`_await_future` 폴링 병목(미해결)**: 두 가지 수정 시도(event-driven,
   간격 단축) 모두 실전 `/step`을 재현 가능하게 행업시킴. 근본 원인 미규명
   (CPython GIL fairness/스레드 기아 가설). `human_deterministic_stepping`을
   legacy Gazebo와 함께 쓸 때 처리량이 legacy보다 나빠지는 원인이기도 함.
   후속 조사 시 `MultiThreadedExecutor`의 실제 스레드 수, 콜백 그룹 토폴로지를
   먼저 분석할 것을 권장(메모리 `gazebo_await_future_poll_interval_landmine`
   참조).
2. **`/clock` 기반 정밀 완료 확인(미해결)**: 같은 실패 클래스. Stage 2의
   `multi_step` 자체는 정확하지만, 그 완료를 정밀하게 확인하는 방법은 아직
   없음(현재는 sleep+sensor-freshness로 우회). 메모리
   `gazebo_multi_step_clock_wait_landmine` 참조.
3. **Stage 3 게이팅의 장기 학습 곡선 검증**: 이번엔 메커니즘 정확성만
   확인(§9, §12). 실제 curriculum을 stage 3까지 자연 승급시켜 aux/action-risk
   loss가 활성화 시점부터 의미 있게 하락하는지, 그리고 stage 0–2 동안
   critic 학습 속도(step당 wall time, 수렴 곡선)가 실제로 개선되는지는
   별도의 장기 실행(수만~수십만 step)으로 확인 필요.
4. **`gazebo_deterministic_stepping`을 나머지 3개 phase2 프로파일
   (`baseline`/`reward_shaping_only`/`action_risk_head_only`)에도 동일하게
   적용**: 이번엔 `both`에만 적용. 원안이 요구한 "검증 후 4개 프로파일 모두
   일관 적용"은 아직 `both`로만 검증된 상태 — 나머지 3개도 같은 방식으로
   opt-in 가능하나 각각 별도 실측이 필요.
5. **Stage 4 가중치 값의 실제 튜닝**: 메커니즘만 구현, `phase2/both`에는
   미적용(§3). 실제 sparse-event 통계(양성 이벤트 비율 등)를 몇천 step
   수집해 `risk_map_positive_weight`/`hazard_pos_weight`의 적정값을 찾는
   ablation을 제안.
6. **Stage 8 스케일 값의 정밀화**: `obs_norm_optim_split` 프로파일의
   `goal_dist_scale`/`yaw_rate_scale`/`steering_scale` 등은 코드에서 확인
   가능한 값(`lidar_max_range`, `controller_cruise_speed_mps`, arena
   경계)과 합리적 추정치를 섞어 사용. 실차/실제 로봇 파라미터 기준으로
   재검토 권장.
7. **Stage 8의 나머지 6개 프로파일**: 연구 설계 결정이 필요(§3).
8. **`obs_normalization`을 aux temporal-context 경로에도 적용**: 현재
   `get_last_state_history()` 경로는 정규화 미적용(§3 명시된 제외) — 이
   기능을 실제로 사용할 계획이면 이 경로도 정규화가 필요.
