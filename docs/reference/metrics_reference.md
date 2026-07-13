# Aux 실험 — 로깅 & 지표 스키마

auxiliary-prediction(aux) 실험에서 각 지표가 **어디서 생성되어 어디에 저장되는지** 정리한다.
두 계층으로 나뉜다: **학습 중 모니터링(training-time monitoring)** vs **정식(논문) 평가**.
모두 하위호환된다 — aux-off run은 이전과 완전히 동일하게 동작하고, 새 CSV 컬럼은 append-only이며,
aux 전용 컬럼은 aux가 꺼져 있으면 빈 칸이다.

## 1. 학습 중 모니터링

### A. 정책 성능 (학습 episode마다, 콘솔 + CSV)
한 줄 episode 요약(non-eval), aux on/off 동일 포맷:

```
T:{t} | Ep:{ep} | Steps:{n} | Reward:{r:.3f} | {GOAL/COLLISION/TIMEOUT/EVAL_CUT} | Stage:{s} | SPL:{..} | STL:{..} | PSC:{../n/a} | H-Coll:{0/1/n/a}
```

- `SPL/STL`은 `EpisodeMetrics`(utils/episode_metrics.py)에서 나온다.
- `PSC`(실제 human Personal-Space Compliance)와 `H-Coll`은 **label 기반**(privileged
  human-distance label)으로, trainer의 `_LabelProximity`가 계산한다. 이 둘은 **env가 label을
  내보내는지**(`aux_prediction.enabled`, env 측)에만 의존하고 agent의 aux head와는 무관하다 —
  따라서 **aux-OFF agent baseline도 실제 PSC/H-Coll을 보고**한다. env가 label을 내보내지 않으면
  `n/a`(빈 칸)이며, 오해를 부르는 0이 되지 않는다.
  - `PSC` = label이 있는 스텝 중 가장 가까운 human 거리 ≥ `psc_personal_space_m`인 비율.
  - `H-Coll` = 가장 가까운 human 거리가 `h_coll_radius_m` 아래로 떨어진 collision episode.
    step 이후 label을 함께 반영하므로 마지막 충돌 스텝까지 포함된다.
- `lidar_clearance_rate`(state-stream clearance **proxy**, human PSC가 **아님** — human을 벽/가구와
  구분하지 못함)는 `EpisodeMetrics`가 계산해 `episode_metrics_*.csv`에 기록한다. PSC로 오인되지
  않도록 별도로 분리했다.
- 저장 위치: `episode_metrics_*.csv`(SPL/STL/lidar_clearance_rate), `curriculum_episode_rewards_*.csv`
  (reward/result/stage/map_type).

### B. Aux 자기학습 (gradient 스텝마다, TensorBoard + tqc_metrics.json)
`tqc_agent.train()`(`aux_prediction_losses.compute_aux_loss`)에서 생성, 변경 없음:
`aux/loss`, `aux/risk_mse`, `aux/min_dist_mse`(v2), `aux/risk_quantile`(distributional 전용),
`aux/valid_len_mean`(action-conditioned 전용). aux-off이면 이 키들은 단순히 존재하지 않는다.

## 2. 정식 평가 (evaluate_and_print)

### A. 메인 정책 지표 (eval 콘솔 + CSV)
`Success / Collision / Timeout / SPL / STL / PSC / H-Coll / CTE`에 더해 map별 분해
(`curriculum_eval_per_map_*.csv`). `eval_metrics_*.csv`(논문)와 `eval_summary_*.csv`에 집계.

### B. 정식 aux 지표 (eval 콘솔 라인, **aux on일 때만**)
`utils/aux_eval_metrics.py`가 모든 eval 스텝에 대해 계산(single-step: `z_t`; action-conditioned:
`z_t` + boundary-safe `[a_t..a_{t+K-1}]`, 학습과 동일 정렬로 episode 경계를 넘지 않음):

| 지표 | 의미 |
|---|---|
| `aux_risk_rmse` | 예측 vs GT risk-map 셀 전체의 RMSE |
| `aux_min_dist_mae_m` | future min-distance MAE, **미터 단위**(정규화 오차 × D_c); min-dist head가 없으면 risk에서 유도 |
| `aux_peak_sector_acc` | (sample, horizon)별 argmax-sector 일치율; GT가 전부 0인 행 제외; 동점 → 최소 인덱스 |
| `aux_near_event_f1` | binary near-event = future min-dist < `aux_near_event_threshold_m`; zero-division-safe precision/recall/F1 |

콘솔(aux on):
```
Eval(aux) | AuxLossEval(RiskRMSE) {..} | MinDistMAE(m) {..} | PeakAcc {..} | EventF1 {..} (thr<{..}m, N={..})
```

## 3. 저장 위치 맵

| 지표 그룹 | TensorBoard | tqc_metrics.json | eval_summary_*.csv | curriculum_eval_per_map_*.csv | 콘솔 |
|---|---|---|---|---|---|
| aux/loss, aux/risk_mse, aux/min_dist_mse, aux/risk_quantile, aux/valid_len_mean | ✓ | ✓ | – | – | – |
| SPL / STL | – | – | ✓ (eval 평균) | ✓ (map별: base rate로 SPL) | ✓ (episode 라인) |
| PSC, H-Coll (label 기반) | – | – | ✓ (env label off면 빈 칸) | ✓ (off면 빈 칸) | ✓ (off면 n/a) |
| lidar_clearance_rate (proxy) | – | – | ✓ (항상) | – | ✓ (eval 라인) |
| aux_risk_rmse / aux_min_dist_mae_m / aux_peak_sector_acc / aux_near_event_f1 | – | – | ✓ (agent aux on) | ✓ (agent aux on) | ✓ (agent aux on) |

빈 칸 = 빈 CSV 셀 / 콘솔 `n/a`(지표 사용 불가), 절대 0이 아님.

## 4. Config (ROS params → run_manifest.json `aux_eval`)
- `aux_near_event_threshold_m` (기본 0.5) — aux F1의 near-event 거리
- `h_coll_radius_m` (기본 0.5) — human-collision 거리(label 기반)
- `psc_personal_space_m` (기본 0.5) — human personal-space 반경(label 기반 PSC)
- `stl_ref_speed_mps` (기본 1.0) — STL 최적 시간 = shortest_path / ref
- `lidar_clearance_radius_m` (기본 0.5) — LiDAR clearance-proxy 반경
- `risk_distance_scale`(D_c)는 **실행 중인 env** 노드에서 가져온다(단일 소스).

## 5. 참고 / 확장 지점
- **PSC / H-Coll은 label 기반**(env의 privileged human-distance label)이므로 **env가 label을 내보낼
  때에 한해** aux-OFF agent에서도 동작하고, 안 내보내면 빈 칸이다. `lidar_clearance_rate`는 별도의
  state-stream clearance proxy(human vs static 구분 못 함)로, Falcon/DiPCAN의 PSC와 직접 비교 불가라
  이름을 달리했다.
- `AuxEvalAccumulator`(utils/aux_eval_metrics.py)는 ESR / AD / ALV / encounter-count의 확장 지점이다.
  거기에 per-episode 배열을 추가로 모으고 `finalize()`를 확장하면 된다 — trainer 루프 변경 불필요.
- **여기서 실행하지 않음:** offline 검증 시 ROS/Gazebo 런타임 + torch를 쓸 수 없어, aux-head forward
  shape와 전체 eval 흐름은 live run이 아니라 contract/단위 테스트로 검증했다.

## 6. 동적 장애물 회피 진단 로그 (`dynamic_avoidance_metrics_*.csv`) — DYN_AVOID

`train_tqc_curriculum_agent.py`가 학습 중 자동 생성하는 **단일 파일** 진단 로그. 기존 CSV들에
흩어져 있던 "동적(보행자) 장애물 회피" 관련 지표를 한 곳에 모으고(재노출), 환경이 계산한
**privileged** human interaction / clutter / yield / near-event 지표를 추가한다. run tag 네이밍은
다른 CSV와 동일(`dynamic_avoidance_metrics_<run_tag>.csv`), episode 단위 1행. eval-cut(부분) episode는
paper CSV와 동일하게 건너뛴다.

**데이터 흐름:** 환경(`environment.py`)이 매 `/step`마다 `DynamicAvoidanceEpisodeDiag`
(`scripts/environment/dynamic_avoidance_telemetry.py`)에 privileged 로봇+보행자 ground truth를
누적 → episode 진단 dict를 read-only 파라미터 `episode_dynamic_diag`(JSON)로 노출 → 트레이너가
episode 종료 시(리셋 전, **eval 전**) `GetParameters`로 1회 읽어 `DynamicAvoidanceCSV`
(`scripts/utils/dynamic_avoidance_log.py`)로 기록. aux label과 무관하므로 **aux-OFF 베이스라인**에서도
보행자만 활성이면 동작한다.

- **읽기 타이밍이 매 step 누적을 요구하는 이유:** env는 trainer가 episode를 언제 끝낼지(특히 timeout,
  env엔 step 한도 없음) 알 수 없고, `evaluate_and_print`가 같은 env에서 eval episode를 돌려 누적기를
  덮어쓴다. 따라서 학습 episode의 진단은 **마지막 step 직후·eval 전**에 읽어야 하며, 파라미터는 매 step
  최신이어야 한다.
- **오버헤드 억제:** 실제 파라미터 write + `/parameter_events` 발행은 진단 **내용이 바뀐 step에만**
  수행한다(`state_key()` 비교). 보행자 없는 step/스테이지는 ROS 비용 0.
- **수집 실패 가시화:** env는 보행자가 없어도 항상 전체 dict를 publish하므로, trainer가 빈 `{}`를 받으면
  그것은 "사람 없음"이 아니라 **telemetry 경로 고장**(구버전 env·서비스/파싱 실패)이다. 이 경우 rate-limit
  WARN을 남겨 CSV의 unknown/NaN이 고장 때문임을 운영 중에 구분할 수 있게 한다.

**기존 파일에서 통합(재노출)한 동적 회피 관련 컬럼:**

| 원본 CSV | 통합한 컬럼 |
|---|---|
| `curriculum_episode_rewards_*` | episode, global_t, curriculum_stage, map_type, total_reward, steps, goal_reached→`success`, collision, timeout, final_goal_dist_m, mean_gazebo_rtf, seed, aux_enabled, aux_version |
| `episode_metrics_*` (paper) | near_collision_count, mean_speed_mps, lidar_clearance_rate |
| `episode_driving_*` | mean_cmd_v_mps, mean_cmd_steering_rad, low_obs_speed_frac |
| `curriculum_eval_per_map_*` / `eval_metrics_*` | 이들의 eval-time 집계(h_coll_rate/psc)와 **같은** per-episode human-proximity를 여기서는 episode 단위로 `min_human_distance_m` / `time_below_human_clearance_ratio` / `near_human_event`로 기록 |

> 기존 파일 포맷은 그대로 유지(하위호환)하고, 위 컬럼을 이 파일에 **중복 재노출**한다. 통합 매핑은
> `dynamic_avoidance_log.py` 상단 주석이 단일 소스다.

**새로 추가한 핵심 지표(privileged, 환경 계산):** `collision_object_type`(none/static/human/unknown),
`min_human_distance_m`, `time_below_human_clearance_ratio`, `human_ttc_min`, `near_human_event`,
`near_human_event_success`, `has_human_interaction`, `has_static_clutter_pressure`,
`static_clutter_steps`, `human_observed_steps`, `nearest_human_mode`, `human_modes_present`,
그리고 yield 진단 `yield_available` / `yield_used` / `yield_trigger_count` / `yield_steps` /
`yield_in_risk_steps` / `yield_no_risk_steps` / `risk_steps`. yield effectiveness는 후처리로
`yield_in_risk_steps / risk_steps`(risk 중 실제 yield 비율), false-positive는 `yield_no_risk_steps`로 계산.

**추가한 환경 신호(현재 신호만으로 계산 불가했던 값):** 위 privileged 지표들은 로봇↔보행자 정확 거리 /
closing speed(TTC) / 충돌 상대 종류 / 보행자 mode를 필요로 하는데, 이는 agent가 보지 못하는 sim 내부
상태다. 따라서 환경에 `DynamicAvoidanceEpisodeDiag` 누적기 + `episode_dynamic_diag` 파라미터를 새로
추가해 노출했다. Step 서비스(.srv) / aux label wire 포맷은 **건드리지 않았다**.

**NaN 규칙:** 정의 불가 수치는 `nan`(보행자가 episode 내내 없었으면 human 계열 전부 `nan`;
diag 자체가 없으면 충돌 시 `collision_object_type=unknown`, 그 외 수치 `nan`), 문자열 미정은 `""`.
`near_human_event`는 보행자가 관측된 episode에서만 0/1, 관측 0이면 `nan`.

**환경 측 임계값(ROS params, 기본값):** `dyn_diag_near_human_dist_m`(1.0, near/risk & personal-space
ratio), `dyn_diag_interaction_radius_m`(2.0, has_human_interaction), `dyn_diag_collision_attrib_radius_m`
(0.7, human vs static 충돌 귀속), `dyn_diag_ttc_collision_radius_m`(0.5), `dyn_diag_static_clutter_lidar_m`
(0.6, static clutter 판정 LiDAR 임계).
