# Map Curriculum Design

이 문서는 **구조화된 맵 4종**과 그 위에서 장애물·시작/목표·사람을 배치하는 방식을 짧게 설명한다. 전체 설계 원안과 상세 분석은 [../experiments/map_curriculum_plan.md](../experiments/map_curriculum_plan.md)에 있다.

## What
완전 무작위 scatter 맵 대신, 의미가 있는 **4종 구조 맵**을 stage별로 샘플링한다.
- **lobby**: 넓은 개방 공간
- **corridor**: 긴 단일 통로 (world-x 방향)
- **intersection**: 십자 교차 통로
- **clutter**: 짧은 내부 벽 여러 개 + 장애물

## Why
- 재현·설명 가능한 환경(논문에서 "맵 타입별 성능 / 구조 일반화"를 말할 수 있음).
- 기존 19×19 `drl_arena.world`를 그대로 쓰면서 **내부 벽만** 바꿔 구조를 만든다(외벽·world 파일 미변경, "안 B").

## How — 핵심 규칙
- **내부 벽**은 박스 엔티티 pool로 startup에 1회 생성 → episode마다 set_pose로 활성/지하 주차(런타임 create/remove 없음).
- **장애물**은 맵 타입별 **허용 키 그룹**으로 제한(예: corridor는 통로를 막는 큰 가구 제외). 풀은 (map, size group)별 커버리지를 보장하도록 group-aware로 미리 spawn하고, episode마다 허용 subset만 활성화.
- **start/goal/사람**은 맵의 free region 안에서만 샘플링(벽·dead-zone 회피, goal은 reachable 휴리스틱).
- stage가 올라갈수록 맵 종류와 복잡도가 함께 증가. → [curriculum_design](curriculum_design.md)

## localization과의 연결
corridor는 길이방향 위치추정이 가장 부정확하므로, localization noise의 `map_type_multipliers`가 corridor에서 sigma/drift를 증폭(이방성). → [localization_noise_design](localization_noise_design.md)

## Where in code
- 맵 레이아웃/벽/배치: `environment/environment.py` (`_build_map_layouts`, `_select_episode_layout`, `_sample_goal_layout`, 장애물 풀 커버리지)
- stage별 허용 맵/그룹: `config/environment_curriculum.yaml` (`map_layout_*`, stage의 `allowed_map_types`/`allowed_static_groups`)
- 상세 설계/검증: [../experiments/map_curriculum_plan.md](../experiments/map_curriculum_plan.md)
