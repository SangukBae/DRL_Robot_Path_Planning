# Curriculum Design

이 문서는 **커리큘럼 학습**(난이도를 단계적으로 올리는 방식)이 어떻게 동작하는지 설명한다.

## What
빈 맵에서 시작해 장애물·사람·맵 복잡도·localization noise를 **stage 별로 점점 키운다**. 각 stage에서 일정 성능을 내면 자동으로 다음 stage로 진급한다. **한 stage에 한 축만 크게 바꾸도록** 10단계로 구성한다(구조 → 사람 → 위치추정 노이즈 → 지형 → 자기수용(proprio) 노이즈 → 새 맵·군중 → 통합). Stage 3–6은 **엄격한 단일 축**(직전 대비 새 변수 정확히 하나), Stage 7–9는 여러 축을 함께 키우는 **통합(generalization) 단계**다.

## Why
- 처음부터 복잡한 환경을 주면 정책이 충돌만 반복하며 학습이 안 된다.
- 쉬운 과제부터 성공 경험을 쌓게 하면 더 안정적으로, 더 빨리 수렴한다.
- 한 번에 여러 축(맵 구조+사람+노이즈)을 동시에 올리면 분포 전환이 커져 불안정 → 축을 분리한다.

## How — 10단계
혼합맵 stage는 **맵별 활성 개수**(`active_static_by_map` / `active_humans_by_map`)로 좁은
corridor에 적게, 넓은 intersection/clutter/lobby에 많이 준다. 표의 `C/I/Cl/L` =
corridor / intersection / clutter / lobby 값.

| stage | 이름 | 정적 (맵별) | 사람 (맵별) | 맵 | 노이즈 | 핵심 축 |
|:--:|--|:--:|:--:|--|--|--|
| 0 | empty | 0 | 0 | lobby | clean | 기본 goal 도달 |
| 1 | corridor_static | 3 | 0 | corridor | clean | corridor 정적 주행 |
| 2 | add_intersection | C3 / I6 | 0 | corridor+intersection | clean | **구조** |
| 3 | first_human_clean | C4 / I6 | C1 / I1 | corridor+intersection | clean | **사람(동적 회피)** |
| 4 | first_human_noisy | C4 / I6 | C1 / I1 | corridor+intersection | weak loc | **위치추정 노이즈** |
| 5 | add_clutter_clean | C4 / I6 / Cl8 | C1 / I1 / Cl1 | +clutter | weak loc | **지형** (+yield 해제) |
| 6 | add_clutter_noisy | C4 / I6 / Cl8 | C1 / I1 / Cl1 | corridor+intersection+clutter | weak loc + light proprio | **proprio 노이즈** |
| 7 | add_lobby | C4 / I6 / Cl8 / L8 | C1 / I2 / Cl2 / L3 | 4종 전부 | weak loc + light proprio | **새 맵** + human-scan 노이즈 + 군중↑(1→3) |
| 8 | scale_crowd | C5 / I7 / Cl8 / L8 | C2 / I3 / Cl4 / L5 | 4종 전부 | drift loc + light proprio | **군중 확대**(통합) |
| 9 | full_complexity | C5 / I7 / Cl8 / L9 | C3 / I4 / Cl4 / L6 | 4종 전부 | robustness_train + medium proprio | **final 통합** |

> 움직이는 장애물 = 사람뿐(동적 장애물은 코드에서 제거). 맵 종류는 [map_curriculum_design](map_curriculum_design.md) 참고.
> localization noise는 Stage 0~3 clean, **Stage 4부터** ramp-up. corridor는 5.2 m 차선의
> 배치 상한(활성화 후보 ~10) 때문에 전 stage에서 정적·사람이 가장 적다. 맵별 개수를 생략한
> stage(0,1)는 단일 `active_static`/`active_humans`를 그대로 쓴다(하위호환).

### stop/yield 액션의 봉인·해제
yield 축(`action[2]`)은 **Stage 0–4 봉인**(`yield_reward.action_enabled=false`, 순수 회피 주행부터),
**Stage 5에서 해제**되어 정지/creep이 허용된다. 이 경계에서 컨트롤 컨트랙트가 바뀌므로 진급 시 버퍼를
리셋(`reset_buffer_on_promote_to:[5]`)하고 `rewarmup_steps`만큼 재워밍업해 off-contract 경험이 critic을
오염시키지 않게 한다 — **6개 baseline 공통**. 주행 2축 의미는 전 stage 동일 → 그 외 경계엔 리셋 없음.

> 맵별 개수는 episode마다 `map_type` 확정 후 **① `active_*_by_map[map_type]` → ② stage 단일 `active_*`
> → ③ base 값** 순으로 정해진다(pool 상한 클램프). 필드 사용법은
> [config_reference](../reference/config_reference.md#커리큘럼-stage-필드-environment_curriculumyaml의-curriculumstages).

## 진급 규칙 (모두 충족해야 진급)
1. 현재 stage에서 `min_stage_steps` 이상 학습
2. `min_stage_episodes` 이상 에피소드 완료
3. 평가에서 `pass_eval_success_rate`(성공률↑) / `pass_eval_collision_rate`(충돌률↓) / `pass_eval_spl`(경로효율↑) 기준을 **`consecutive_eval_passes`회 연속** 통과
   (10-stage → 임계값 리스트는 9-entry, 인덱스 = stage 번호; 범위 초과 시 마지막 항목으로 클램프)

## 동작 방식 (selector 구조)
- trainer가 평가 통과 시 `curriculum_stage` ROS 파라미터를 올리면, `environment_curriculum.py`가 매 reset마다
  해당 stage 설정을 적용한다: **base 복원 후 stage override를 deep-merge**(stage 간 누수 방지).
- 맵별 활성 개수만은 이번 episode의 map_type이 정해진 **직후** 확정한다(장애물 활성화 직전).
- pool(장애물/사람)은 시작 시 최대 크기로 1회 생성하고 stage는 **활성 개수만** 바꾼다(런타임 create/remove
  없음 → reset 빠름).
- "어떤 stage에서 어떤 프로파일을 쓰나"(커리큘럼)와 "무엇을/어떤 noise로 평가하나"(run-level 설정)는 분리돼 있다.

## Where in code
- stage 적용: `environment/environment_curriculum.py::_apply_curriculum_stage`, `_parse_active_by_map`, `_resolve_noise_override`, `_deep_merge`
- 맵별 활성 개수 결정: `environment/environment.py::_apply_episode_active_counts` (reset 중 map_type 확정 직후) + ROS-free `environment/map_catalog.py::resolve_active_count`, `clamp_active_by_map`
- 진급 판정: `policy/train_tqc_curriculum_agent.py::_check_stage_advance`, `evaluate_and_print`
- stage 정의: `config/environment_curriculum.yaml` (`curriculum.stages`)
- 진급 파라미터: `config/train_tqc_curriculum_config.yaml` → [config_reference](../reference/config_reference.md)
