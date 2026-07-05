# Configuration

모든 설정 파일은 `drl_agent/config/`에 위치한다.

---

## 파일 목록

| 파일 | 설명 |
|------|------|
| `environment.yaml` | 상태/액션 차원, 충돌 임계값, 장애물 카탈로그, height filter |
| `environment_curriculum.yaml` | 커리큘럼 환경 설정 (10단계 정의, 3D 하이브리드 액션, 맵별 장애물/휴먼 구성) |
| `train_tqc_curriculum_config.yaml` | 커리큘럼 진급 규칙 (임계값, 연속 통과 횟수) |
| `train_tqc_config.yaml` | TQC 단일 학습 파라미터 |
| `train_td7_config.yaml` | TD7 학습 파라미터 |
| `train_sac_curriculum_config.yaml` | SAC 커리큘럼 학습 파라미터 |
| `train_a3c_curriculum_config.yaml` | A3C 커리큘럼 학습 파라미터 |
| `hyperparameters_tqc.yaml` | TQC 네트워크 구조, 학습률 |
| `hyperparameters_td7.yaml` | TD7 네트워크 구조, 학습률 |
| `test_tqc_config.yaml` | TQC 테스트 파라미터 (시작/목표 쌍, 에피소드 수) |

---

## 주요 학습 파라미터

`train_tqc_config.yaml` (`train_settings` 블록):

| 파라미터 | 현재 값 | 설명 |
|---------|--------|------|
| `max_timesteps` | 2,000,000 | 최대 학습 타임스텝 |
| `timesteps_before_training` | 12,000 | 랜덤 액션 워밍업 스텝 (10 Hz 기준) |
| `eval_freq` | 12,000 | 평가 주기 (스텝) |
| `load_model` | false | 기본 fresh start. `true` 로 두면 같은 seed의 최근 체크포인트에서 모델·리플레이 버퍼 복원 (커리큘럼 상태는 `curriculum_state.json` 별도 필요) |

`train_tqc_curriculum_config.yaml` (`curriculum_settings` 블록):

| 파라미터 | 현재 값 | 설명 |
|---------|--------|------|
| `min_stage_steps` | 30,000 | 스테이지 진급 전 최소 타임스텝 |
| `min_stage_episodes` | 20 | 스테이지 진급 전 최소 에피소드 수 |

---

## 충돌/목표 임계값

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `goal_threshold` | 0.42 m | 목표 도달 판정 거리 |
| `d_front` | 0.476 m | 충돌 안전 영역 — 전방 |
| `d_rear` | 0.410 m | 충돌 안전 영역 — 후방 |
| `d_left` / `d_right` | 0.322 m | 충돌 안전 영역 — 좌우 |

---

## 커리큘럼 stage 필드 (`environment_curriculum.yaml`의 `curriculum.stages[]`)

10단계(0~9). 각 stage가 가질 수 있는 활성-개수 필드:

| 필드 | 예 | 설명 |
|------|-----|------|
| `active_static` | `6` | 맵 무관 단일 정적 장애물 수 (fallback) |
| `active_humans` | `1` | 맵 무관 단일 휴먼 수 (fallback) |
| `active_static_by_map` | `{corridor: 5, intersection: 7}` | **map_type별** 정적 수 override |
| `active_humans_by_map` | `{corridor: 1, intersection: 1}` | **map_type별** 휴먼 수 override |

에피소드마다 `map_type` 확정 후 우선순위 **`*_by_map[map_type]` → 단일 `active_*` → base**로
개수가 정해지고, `0` 미만 방지 + `obstacle_pool_static_size` / `obstacle_pool_human_size`
상한 클램프된다. 좁은 corridor에 더 적게, 넓은 intersection/clutter/lobby에 더 많이 줄 때 사용.
허용 map key: `lobby/corridor/intersection/clutter` (그 외는 경고 후 무시). 단일값만 쓰는 기존
stage는 그대로 동작(하위호환). 자세한 설계는 `docs/experiments/map_curriculum_plan.md` §3(Stage별 설정)·§7(커리큘럼 stage).

## 커리큘럼 진급 설정

`train_tqc_curriculum_config.yaml`:

| 파라미터 | 설명 |
|---------|------|
| `min_stage_steps` | 진급 검토 전 현재 스테이지에서 소비해야 할 최소 타임스텝 |
| `min_stage_episodes` | 진급 검토 전 현재 스테이지에서 완료해야 할 최소 에피소드 수 |
| `pass_eval_success_rate` | 진급 요건: 스테이지별 최소 성공률 임계값 리스트 |
| `pass_eval_collision_rate` | 진급 요건: 스테이지별 최대 충돌률 임계값 리스트 |
| `pass_eval_spl` | 진급 요건: 스테이지별 최소 SPL(경로효율) 임계값 리스트 (빈 리스트=비활성) |
| `consecutive_eval_passes` | 임계값을 연속으로 통과해야 하는 평가 횟수 |

진급 임계값 리스트 길이는 `(스테이지 수 − 1)` = **9**(10-stage)이며, 인덱스는 스테이지 번호다.
범위를 벗어난 인덱스는 마지막 항목으로 클램프되므로 리스트가 짧아도 동작은 한다.
`min_stage_steps` / `min_stage_episodes` 조건을 먼저 충족한 뒤, 성공률/충돌률(/SPL) 임계값을 `consecutive_eval_passes`회 연속 통과해야 다음 스테이지로 진급한다.

### 하이브리드 액션 / yield 관련 진급 필드
| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `reset_buffer_on_promote_to` | `[5]` | 지정 stage로 진급 시 리플레이 버퍼 초기화. Stage 5에서 yield 축(`action[2]`)이 봉인 해제되어 컨트롤 컨트랙트가 바뀌므로, off-contract 경험이 critic을 오염시키지 않도록 리셋한다. |
| `rewarmup_steps` | `5000` | 버퍼 리셋 직후 gradient 없이 랜덤 액션으로 재워밍업하는 스텝 수(on-contract 데이터 재적재). |

---

## LiDAR Height Filter

`environment.yaml`:

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `obs_z_min_sensor_m` | -0.455 m | 포인트클라우드 필터 하한 (센서 프레임) |
| `obs_z_max_sensor_m` | 0.250 m | 포인트클라우드 필터 상한 (센서 프레임) |

센서가 지상 ~0.60 m에 장착되어 있으므로 실제 지상 기준으로는 약 0.045–0.850 m 범위를 관측한다.

---

## 장애물 카탈로그

`environment.yaml`에서 `obstacle_catalog: "drl_obstacle_assets"`로 지정한다. 카탈로그 파일 위치: `drl_obstacle_assets/config/obstacle_catalog.yaml`

각 장애물은 다음 필드를 가진다:

| 필드 | 설명 |
|------|------|
| `motion_type` | `dynamic` 또는 `static` |
| `radius` | 장애물 반경 [m] |
| `yaw_random` | 초기 방향 랜덤화 여부 |

총 38종의 장애물 모델이 `drl_obstacle_assets/models/`에 있다.
