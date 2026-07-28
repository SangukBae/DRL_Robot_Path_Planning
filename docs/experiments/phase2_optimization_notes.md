# `phase2/both` Throughput Optimization — Summary

`environment_curriculum_node.py --ros-args -p profile:=phase2/both` +
`train_node.py --ros-args -p profile:=phase2/both -p seed:=0` 학습 경로의 속도 개선 작업 요약.
**모델 크기는 축소하지 않았다**(`n_critics=5`, `n_quantiles=25`, hidden dim 불변) — 개선은 전부
불필요한 연산/대기 제거·수치적으로 동일한 연산 재구성·메모리 표현 변경에서 나온다.

## 적용된 변경

| 항목 | 무엇을 줄였나 | 기본값 | 체크포인트 영향 |
|---|---|---|---|
| Gazebo deterministic stepping (`gazebo_deterministic_stepping`) | `unpause→sleep→pause` 2회 호출 → `WorldControl(multi_step=N)` 1회 호출 | OFF(패키지 공통), `phase2/both` profile에서 **ON** | 없음 |
| Human deterministic stepping (`human_deterministic_stepping`) | 보행자 이동 타이머를 wall-clock 독립 정수 tick으로 재구현 — **재현성 버그 수정**(기존엔 in-episode RNG draw 수가 wall-clock 속도에 의존) | OFF | 없음 |
| Stage 기반 aux/action-risk 게이팅 | stage 0–2(사람 없음)에서 aux head·temporal encoder forward/loss를 통째로 스킵 (`aux_prediction.stagewise_loss_schedule=[0,0,0,0.1,...]`, `action_risk_head.enable_from_stage=3`) | `phase2/both`에 적용 | 없음(파라미터 shape 불변) |
| TQC critic freeze + foreach polyak | actor 업데이트 시 critic gradient 계산 방지, Polyak 업데이트를 `torch._foreach_*`로 배치화 | 항상 적용(수학적으로 동일 연산) | 없음 |
| 로깅 interval화 | 매 step 동기 TensorBoard/JSON write → `scalar/json_log_interval`, `json_flush_interval`로 배치 | `phase2/both`: interval=10, flush=100 (기본은 1=byte-identical) | 없음 |
| Replay buffer float32 | `np.zeros(...)` dtype 미지정(float64) → 명시적 float32, 실제 checkpoint 기준 메모리 **50% 감소** | 항상 적용 | 없음 — 기존 float64 checkpoint는 로드 시 자동 cast(로그 남김) |
| 관측 정규화 + optimizer param groups | 고정 물리 범위 정규화 + critic/encoder/aux별 LR — **`phase2/both`에는 미적용**, 별도 실험 profile(`phase2/obs_norm_optim_split`)로만 시연 | OFF | 활성화 시 **fresh-run 전용**(manifest 불일치 시 fail-fast) |

## 측정치 (Docker, RTX 4080)

| 모드 | 평균 step 시간 | steps/s |
|---|---|---|
| Legacy (`gazebo_deterministic_stepping=false`) | 203.6–204.5 ms | 4.89–4.91 |
| `gazebo_deterministic_stepping=true` | **152.3–154.6 ms** | **6.47–6.57**(≈25% 개선) |
| `human_deterministic_stepping`만, stage 3(실제 보행자) | ≈306 ms | ≈3.3 (아래 미해결 항목 때문에 legacy보다 느림) |
| `human_deterministic_stepping` + `gazebo_deterministic_stepping`, stage 3 | 202.5–211.7 ms | ≈4.8 (legacy 수준 회복 — 재현성 확보가 주목적) |

replay checkpoint 메모리: 실측 180k-transition checkpoint 기준 1032.5MB(float64) → 516.2MB(float32).

## 미해결 — 다음에 시도하지 말 것

`env/simulation/gazebo_service_wait.py`의 `_await_future` 50ms 폴링을 event-driven callback으로
바꾸거나 폴링 간격만 줄이는 시도(0.05→0.001/0.01) **둘 다 실제 `/step` 호출을 재현 가능하게
행업(hang)시킨다**(reset은 매번 성공, 첫 실전 `/step`만 멈춤). 근본 원인은 미규명(CPython GIL
fairness/스레드 기아 가설). 마찬가지로 Gazebo `multi_step` 완료를 `/clock` 구독 기반으로 정밀
확인하려는 시도도 같은 방식으로 행업한다 — 현재는 sleep + 기존 sensor-freshness wait로 대체.
재시도 전에 `MultiThreadedExecutor` 스레드 수/콜백 그룹 토폴로지부터 분석할 것.

## 검증 상태

- Host: `pytest -q`(ROS/torch 미빌드 환경) 전부 통과 — skip은 전부 rclpy/torch가 필요한 테스트(호스트에
  미설치, CLAUDE.md에 문서화된 정상 동작).
- Docker clean build(`rm -rf build install log` 후 전체 패키지) + `colcon test --packages-select drl_agent`:
  pytest 전부 통과(0 errors, 0 failures, 0 skipped), lint_cmake/xmllint 통과. (정확한 케이스 수는
  suite가 계속 늘어나므로 여기 고정하지 않음 — `colcon test-result --verbose`로 항상 최신 값 확인.)
- 실전 `environment_curriculum_node.py` + `train_node.py`(`profile:=phase2/both`) E2E 실행,
  Ctrl+C 인터럽트 경로, stage 3 강제 전환(실제 보행자) 모두 에러 없이 확인.

## 남은 후속 작업

1. `gazebo_deterministic_stepping`을 나머지 3개 phase2 profile(`baseline`/`reward_shaping_only`/`action_risk_head_only`)에도 적용(현재 `both`에만 적용).
2. Stage 3 aux/action-risk 게이팅의 장기 학습 곡선 검증(이번엔 메커니즘 정확성만 확인).
3. `risk_map_positive_weight`/`hazard_pos_weight`(sparse risk label을 위한 loss weighting 메커니즘, stage gate와 무관) 실제 값 튜닝 — sparse-event 통계 수집 후 결정.
4. `obs_normalization`을 aux temporal-context 경로(`get_last_state_history()`)에도 적용할지 검토.
