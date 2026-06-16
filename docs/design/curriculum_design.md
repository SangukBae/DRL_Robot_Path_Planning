# Curriculum Design

이 문서는 **커리큘럼 학습**(난이도를 단계적으로 올리는 방식)이 어떻게 동작하는지 설명한다.

## What
빈 맵에서 시작해 장애물·사람·맵 복잡도·localization noise를 **stage 별로 점점 키운다**. 각 stage에서 일정 성능을 내면 자동으로 다음 stage로 진급한다. **한 stage에 한 축만 크게 바꾸도록** 7단계로 구성한다(구조→사람→지형→일반화→노이즈).

## Why
- 처음부터 복잡한 환경을 주면 정책이 충돌만 반복하며 학습이 안 된다.
- 쉬운 과제부터 성공 경험을 쌓게 하면 더 안정적으로, 더 빨리 수렴한다.
- 한 번에 여러 축(맵 구조+사람+노이즈)을 동시에 올리면 분포 전환이 커져 불안정 → 축을 분리한다.

## How — 7단계
혼합맵 stage는 **맵별 활성 개수**(`active_static_by_map` / `active_humans_by_map`)로 좁은
corridor에 적게, 넓은 intersection/clutter/lobby에 많이 준다. 표의 `C/I/Cl/L` =
corridor / intersection / clutter / lobby 값.

| stage | 이름 | 정적 (맵별) | 사람 (맵별) | 맵 | 노이즈 | 핵심 축 |
|:--:|--|:--:|:--:|--|--|--|
| 0 | empty | 0 | 0 | lobby | clean | 기본 goal 도달 |
| 1 | corridor_static | 3 | 0 | corridor | clean | corridor 정적 |
| 2 | add_intersection | C3 / I6 | 0 | corridor+intersection | clean | **구조** |
| 3 | first_human | C4 / I6 | C1 / I1 | corridor+intersection | weak_goal_noise | **사람** |
| 4 | add_clutter | C4 / I7 / Cl8 | C1 / I2 / Cl3 | +clutter | weak + light proprio | **지형** |
| 5 | generalize | C5 / I7 / Cl9 / L8 | C2 / I3 / Cl5 / L5 | 4종 전부 | drift + light proprio | **일반화** |
| 6 | full_complexity | C5 / I8 / Cl9 / L9 | C3 / I5 / Cl6 / L6 | 4종 전부 | robustness_train + medium proprio | **final** |

> 움직이는 장애물 = 사람뿐(동적 장애물은 코드에서 제거). 맵 종류는 [map_curriculum_design](map_curriculum_design.md) 참고.
> localization noise는 Stage 0~2 clean, **Stage 3부터** ramp-up. corridor는 5.2 m 차선의
> 배치 상한(활성화 후보 ~10) 때문에 전 stage에서 정적·사람이 가장 적다. 맵별 개수를 생략한
> stage(0,1)는 단일 `active_static`/`active_humans`를 그대로 쓴다(하위호환).

### 맵별 활성 개수 (`*_by_map`) 우선순위
에피소드마다 `map_type`이 정해진 뒤: **① `active_*_by_map[map_type]` → ② stage 단일
`active_*` → ③ base 값**. `0` 미만 방지 + pool 상한 클램프. 좁은 corridor만 따로 줄이고 싶을
때 같은 stage 안에서 `{corridor: 5, intersection: 7}`처럼 쓴다. (필드 사용법:
[config_reference](../reference/config_reference.md#커리큘럼-stage-필드-environment_curriculumyaml의-curriculumstages))

## 진급 규칙 (모두 충족해야 진급)
1. 현재 stage에서 `min_stage_steps` 이상 학습
2. `min_stage_episodes` 이상 에피소드 완료
3. 평가에서 `pass_eval_success_rate`(성공률↑) / `pass_eval_collision_rate`(충돌률↓) / `pass_eval_spl`(경로효율↑) 기준을 **`consecutive_eval_passes`회 연속** 통과
   (7-stage → 임계값 리스트는 6-entry, 인덱스 = stage 번호; 범위 초과 시 마지막 항목으로 클램프)

## 동작 방식 (selector 구조)
- `environment_curriculum.py`가 `curriculum_stage`라는 ROS 파라미터를 읽는다.
- trainer는 평가 통과 시 `set_parameters`로 이 값을 올린다.
- 매 reset마다 `_apply_curriculum_stage(idx)`가 해당 stage 설정을 적용한다:
  - **base 값을 먼저 복원한 뒤 stage override를 deep-merge** → stage 간 설정이 새어나가지 않음(no leakage).
  - 바꾸는 것: 활성 장애물/사람 수(단일 + **맵별 `*_by_map`**), 허용 맵 종류, 장애물 그룹, noise 프로파일.
- 단, **맵별 활성 개수는 stage 적용 시점만으로 부족**하다(이번 episode의 map_type이 아직 미정).
  실제 개수는 reset 중 `_select_episode_layout()`로 map_type이 정해진 **직후** base
  `environment.py::_apply_episode_active_counts()`가 확정한다(장애물 활성화 직전). 순수 결정
  로직은 ROS-free `map_catalog.resolve_active_count` / `clamp_active_by_map`.
- pool(장애물/사람 엔티티)은 시작 시 최대 크기로 1회 생성하고, stage는 **활성 개수만** 바꾼다(런타임에 create/remove 안 함 → reset 빠름).

## 평가는 stage와 분리
- "어떤 stage에서 어떤 프로파일을 쓰나"는 커리큘럼이 정한다.
- "무엇을 로깅하나 / 어떤 noise로 평가하나"는 run-level 설정(`localization_logging`, `evaluation`)으로 분리한다.

## Where in code
- stage 적용: `environment/environment_curriculum.py::_apply_curriculum_stage`, `_parse_active_by_map`, `_resolve_noise_override`, `_deep_merge`
- 맵별 활성 개수 결정: `environment/environment.py::_apply_episode_active_counts` (reset 중 map_type 확정 직후) + ROS-free `environment/map_catalog.py::resolve_active_count`, `clamp_active_by_map`
- 진급 판정: `policy/train_tqc_curriculum_agent.py::_check_stage_advance`, `evaluate_and_print`
- stage 정의: `config/environment_curriculum.yaml` (`curriculum.stages`)
- 진급 파라미터: `config/train_tqc_curriculum_config.yaml` → [config_reference](../reference/config_reference.md)
