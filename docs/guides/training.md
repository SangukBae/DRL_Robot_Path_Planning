# Training

## 커리큘럼 학습 (권장)

10단계 자동 진급 방식. 빈 환경 → corridor 정적 → +intersection(구조) → +사람(동적 회피) → +위치추정 노이즈 → +clutter(지형, yield 해제) → +proprio 노이즈 → +lobby·군중 → 군중 확대 → final(통합) 순으로 **한 단계에 한 축씩** 난이도를 높인다. 혼합맵 단계는 좁은 corridor에 더 적은 장애물/사람을 주도록 맵별 개수(`*_by_map`)를 쓴다. (동적 장애물은 제거되어, 움직이는 장애물은 사람뿐이다.)

### 실행 순서

```bash
# [터미널 1] Gazebo 시뮬레이션
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false

# [터미널 2] 커리큘럼 환경 노드 (config_file은 기본값 자동 주입, 명시 override 시에만 필요)
ros2 run drl_agent environment_curriculum.py

# [터미널 3] TQC 커리큘럼 학습
ros2 run drl_agent train_tqc_curriculum_agent.py
```

### 출력 파일

| 경로 | 내용 |
|------|------|
| `<run_dir>/logs/curriculum_episode_rewards_<run>.csv` | 에피소드별 스테이지·보상·성공 여부 |
| `<run_dir>/logs/curriculum_state.json` | 현재 스테이지·글로벌 스텝 (재개용) |
| `<run_dir>/pytorch_models/` | 학습 중 체크포인트 모델 |
| `<run_dir>/final_models/` | 학습 종료 시 최종 모델 |
| `<run_dir>/results/` | 평가 결과 (.npy) |

> `run_dir` 기본값: `runtime/tqc`

### TensorBoard 모니터링

```bash
tensorboard --logdir <run_dir>/logs
```

---

## 학습 재개

`train_tqc_config.yaml`의 `train_settings` 블록 아래 `load_model: true`로 설정한다 (`train_tqc_curriculum_config.yaml`이 아님).

```yaml
# train_tqc_config.yaml
train_settings:
  load_model: true
  base_file_name: "tqc_agent"  # 체크포인트 탐색 시 base_file_name + seed 조합으로 찾음
  seed: 0
  # ...
```

복원 범위:

| 조건 | 복원 내용 |
|------|---------|
| `load_model: true` | 모델 가중치 + 리플레이 버퍼 |
| `load_model: true` + `curriculum_state.json` 존재 | 위 항목 + 커리큘럼 스테이지·글로벌 스텝 |

`curriculum_state.json`이 없으면 가중치는 로드되지만 커리큘럼 스테이지는 0부터 재시작된다.

> **주의**: 체크포인트 자동 탐색은 `pytorch_models_dir` 안에서 `base_file_name` + `seed`가 일치하는 가장 최근 파일을 찾는다. `base_file_name`이나 `seed`가 이전 학습과 달라지면 같은 `run_dir`라도 체크포인트를 찾지 못한다.

---

## 커리큘럼 스테이지

10단계(0~9). **한 스테이지에 한 축만 크게 변경**(구조 → 사람 → 위치추정 노이즈 → 지형 →
proprio 노이즈 → 새 맵·군중 → 통합)하여 학습을 안정화한다. 정적/사람 수는 좁은 corridor와
넓은 맵에 다르게 주기 위해 스테이지별로 `active_static_by_map` / `active_humans_by_map`(맵별 개수)를
쓴다 — 표의 `C/I/Cl/L`은 corridor / intersection / clutter / lobby 값이다.

| 스테이지 | 이름 | 맵 | 정적 (맵별) | 사람 (맵별) | 노이즈 |
|---------|------|----|------------|------------|--------|
| 0 | empty | lobby | 0 | 0 | clean |
| 1 | corridor_static | corridor | 3 | 0 | clean |
| 2 | add_intersection | corridor, intersection | C3 / I6 | 0 | clean |
| 3 | first_human_clean | corridor, intersection | C4 / I6 | C1 / I1 | clean |
| 4 | first_human_noisy | corridor, intersection | C4 / I6 | C1 / I1 | weak loc |
| 5 | add_clutter_clean | + clutter | C4 / I6 / Cl8 | C1 / I1 / Cl1 | weak loc (+ yield 해제) |
| 6 | add_clutter_noisy | corridor, intersection, clutter | C4 / I6 / Cl8 | C1 / I1 / Cl1 | weak loc + light proprio |
| 7 | add_lobby | + lobby (4종) | C4 / I6 / Cl8 / L8 | C1 / I2 / Cl2 / L3 | weak loc + light proprio + human-scan |
| 8 | scale_crowd | 4종 | C5 / I7 / Cl8 / L8 | C2 / I3 / Cl4 / L5 | drift loc + light proprio |
| 9 | full_complexity | 4종 | C5 / I7 / Cl8 / L9 | C3 / I4 / Cl4 / L6 | robustness_train + medium proprio |

> 움직이는 장애물 = 사람뿐(동적 장애물 제거됨). 스테이지 이름은 `environment_curriculum.yaml`의
> `name:` 값이다. corridor는 5.2 m 차선이라 배치 상한(활성화 후보 ~10)이 빡빡해 모든
> 스테이지에서 정적·사람이 가장 적다 — `*_by_map`이 같은 스테이지 안에서 이를 조절한다.
> 맵별 개수를 생략한 스테이지(0,1)는 단일 `active_static`/`active_humans`를 그대로 쓴다(하위호환).
> stop/yield 축은 Stage 0–4에서 봉인되고 Stage 5에서 해제된다(진급 시 버퍼 리셋 + 재워밍업).
> → 필드 사용법: [config_reference](../reference/config_reference.md#커리큘럼-stage-필드-environment_curriculumyaml의-curriculumstages)

진급 조건 (모두 만족해야 함):

1. `min_stage_steps` 이상 현재 스테이지에서 학습 (타임스텝 기준)
2. `min_stage_episodes` 이상 에피소드 완료
3. 위 조건 충족 후, `pass_eval_success_rate` / `pass_eval_collision_rate` 임계값을 `consecutive_eval_passes`회 연속 통과

→ [설정 파라미터](../reference/config_reference.md#커리큘럼-진급-설정)

---

## Gazebo 월드 옵션

```bash
# DRL Arena (기본, 25×25m 밀폐 환경, 외벽 ±12.5)
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false

# Hospital world
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py world:=hospital rviz:=false

# 키보드 teleop (검증용)
ros2 launch hunter_se_gazebo hunter_se_validation_empty.launch.py
```

---

## LIO-SAM (선택 — SLAM 매핑)

```bash
# [터미널 1] Gazebo 시뮬레이션
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false

# [터미널 2] LIO-SAM
ros2 launch lio_sam run_scout_ignition.launch.py
```
