# Curriculum Design

이 문서는 **커리큘럼 학습**(난이도를 단계적으로 올리는 방식)이 어떻게 동작하는지 설명한다.

## What
빈 맵에서 시작해 장애물·사람·맵 복잡도·localization noise를 **stage 별로 점점 키운다**. 각 stage에서 일정 성능을 내면 자동으로 다음 stage로 진급한다.

## Why
- 처음부터 복잡한 환경을 주면 정책이 충돌만 반복하며 학습이 안 된다.
- 쉬운 과제부터 성공 경험을 쌓게 하면 더 안정적으로, 더 빨리 수렴한다.

## How — 5단계
| stage | 이름 | 정적 장애물 | 사람 | 맵 | localization |
|:--:|--|:--:|:--:|--|--|
| 0 | empty | 0 | 0 | lobby | clean |
| 1 | static_only | 3 | 0 | corridor | clean |
| 2 | slow_dynamic | 5 | 1 | corridor+intersection | weak_goal_noise |
| 3 | mixed_medium | 6 | 4 | +clutter | drift_goal_noise (+약한 proprio) |
| 4 | full_complexity | 9 | 5 | 4종 전부 | robustness_train (+proprio) |

> 움직이는 장애물 = 사람뿐(동적 장애물은 코드에서 제거). 맵 종류는 [map_curriculum_design](map_curriculum_design.md) 참고.

## 진급 규칙 (모두 충족해야 진급)
1. 현재 stage에서 `min_stage_steps` 이상 학습
2. `min_stage_episodes` 이상 에피소드 완료
3. 평가에서 `pass_eval_success_rate`(성공률↑) / `pass_eval_collision_rate`(충돌률↓) 기준을 **`consecutive_eval_passes`회 연속** 통과

## 동작 방식 (selector 구조)
- `environment_curriculum.py`가 `curriculum_stage`라는 ROS 파라미터를 읽는다.
- trainer는 평가 통과 시 `set_parameters`로 이 값을 올린다.
- 매 reset마다 `_apply_curriculum_stage(idx)`가 해당 stage 설정을 적용한다:
  - **base 값을 먼저 복원한 뒤 stage override를 deep-merge** → stage 간 설정이 새어나가지 않음(no leakage).
  - 바꾸는 것: 활성 장애물/사람 수, 허용 맵 종류, 장애물 그룹, noise 프로파일.
- pool(장애물/사람 엔티티)은 시작 시 최대 크기로 1회 생성하고, stage는 **활성 개수만** 바꾼다(런타임에 create/remove 안 함 → reset 빠름).

## 평가는 stage와 분리
- "어떤 stage에서 어떤 프로파일을 쓰나"는 커리큘럼이 정한다.
- "무엇을 로깅하나 / 어떤 noise로 평가하나"는 run-level 설정(`localization_logging`, `evaluation`)으로 분리한다.

## Where in code
- stage 적용: `environment/environment_curriculum.py::_apply_curriculum_stage`, `_resolve_noise_override`, `_deep_merge`
- 진급 판정: `policy/train_tqc_curriculum_agent.py::_check_stage_advance`, `evaluate_and_print`
- stage 정의: `config/environment_curriculum.yaml` (`curriculum.stages`)
- 진급 파라미터: `config/train_tqc_curriculum_config.yaml` → [config_reference](../reference/config_reference.md)
