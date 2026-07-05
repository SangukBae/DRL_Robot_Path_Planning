# Structured Map Curriculum — 설계 원안 & 구현 기록

무작위 scatter 맵을 **4종 구조 맵**(lobby / corridor / intersection / clutter)으로 바꾸고,
curriculum stage가 오를수록 맵 종류·구조 복잡도까지 함께 키우는 확장의 설계 문서다.
목표는 재현·설명 가능한 환경(논문에서 "맵 타입별 성능 / 구조 일반화"를 말할 수 있게)이다.

> 이 문서는 **설계 원안 + 구현 기록**이다. 실제 stage 정의는 `environment_curriculum.yaml`,
> 요약 설계는 [map_curriculum_design](../design/map_curriculum_design.md) /
> [curriculum_design](../design/curriculum_design.md)이 source of truth다.

---

## 1. 좌표계 (현재 구현 = 균일 25×25)

- `drl_arena.world` 외벽을 **±12.5(25×25)**로 확장. 4종 맵이 모두 이 외곽을 공유한다.
- **외곽 vs navigable**: 외벽 중심선 ±12.5, 내부 충돌 face ±12.35(벽 두께 0.3), 실제 샘플링/주행 가능한
  **navigable extent = face − `map_wall_clearance`(0.55) ≈ ±11.8**.
- `lobby`는 외곽 전체를, `corridor/intersection`은 폭(5.2 m) 고정·길이만 증가, `clutter`는 내부 벽이
  span 비례.
- 좌표 동기화 지점은 두 곳뿐: `environment_curriculum.yaml`(+`environment.yaml`)와 `drl_arena.world`
  (`goal_obstacle ±11.5 + obstacle_wall_margin 1.0 = ±12.5 = 물리 벽 중심선`).

원안에는 "lobby 25 / 나머지 21" 차등안(안 A)과 "world 미수정 19×19"(안 B)도 있었으나, 최종적으로
**균일 25×25(안 A 변형)**를 채택했다. 맵 타입별 inner extent가 필요해지면 per-map `inner` 파라미터 +
perimeter wall pool을 추가하면 된다.

> **버그 수정 기록:** structured start band(예: corridor 우측 `x∈[9.3, 11.8]`)가 legacy scatter box
> `±9.0`으로 dead-zone 평가돼 100% 탈락하던 문제를 수정 — `_build_map_layouts`가 navigable extent를
> `map_navigable_lower/upper`로 저장하고 start 경로의 dead-zone 검사가 이 값을 쓰게 함
> (`tests/test_start_pose_feasibility.py`로 잠금).

---

## 2. 4종 맵 개념

- **lobby** — 중앙 open space가 넓고 주변부에만 짧은 벽/island. 자유로운 사람 교차가 핵심.
- **corridor** — 길고 단순한 주 통로(가로/세로 하나). 좁은 길에서 마주침·추월·양보 학습.
- **intersection** — 십자/T자 교차 통로. 중심 교차부의 사람-로봇 상호작용.
- **clutter** — 짧은 내부 벽 + 장애물 다수, 여러 경로 후보. 국소 우회·시야 가림·좁은 통과.

---

## 3. Stage별 설정 필드

각 stage는 활성 개수 외에 맵 관련 필드를 가진다:

```yaml
- name: stage_x
  active_static: 6 / active_humans: 1          # 단일값(맵 무관) — fallback
  active_static_by_map: {corridor: 5, intersection: 7}   # 맵별 override (선택)
  active_humans_by_map: {corridor: 1, intersection: 1}
  allowed_map_types: ["corridor", "intersection"]
  map_type_probs: [0.5, 0.5]
  allowed_static_groups: ["small", "medium"]   # 크기 그룹 필터
  eval_map_types: ["corridor", "intersection"] # 평가 시 순회할 맵
```

**활성 개수 우선순위**(episode마다 map_type 확정 후): ① `active_*_by_map[map_type]` → ② stage 단일
`active_*` → ③ base 값. 정수화 + `max(0,·)` + pool 상한 클램프. corridor처럼 좁은 맵에 적게, 넓은 맵에
많이 줄 때 사용. 순수 결정 로직은 ROS-free `map_catalog`에 있고 `tests/test_active_by_map.py`로 잠겨 있다.
(자세한 필드 설명은 [config_reference](../reference/config_reference.md).)

---

## 4. 장애물 그룹

- **크기 그룹**: `small`(radius ≤ 0.40) / `medium`(≤ 0.65) / `large`(> 0.65). 단, catalog radius와 실제
  footprint 차이가 큰 항목(긴 desk/shelf 등)이 있어, 1차 필터는 radius로 하되 **맵별 허용/금지는 obstacle
  key 단위**로 관리한다.
- **맵별 허용 방침**: corridor/intersection은 통로를 막지 않게 `small/medium` 위주(큰 가구 제외 또는 외곽
  anchor 한정), clutter는 복잡도가 목적이라 `large`까지 허용, lobby는 open-space 유지를 위해 큰 가구를 중앙
  밖에서만. 센서 관측성이 낮은 `house_cooking_bench`(높이 0.10 m)는 제외, 지나치게 긴 `bookstore_desk_b`는
  lobby 전용/제외.
- **pool 제약**: obstacle pool은 첫 reset에 모델 정체성을 고정하고 이후 `set_pose`만 쓴다(정상 학습 중
  create/remove 회피). 따라서 `allowed_static_groups`는 "episode마다 새로 spawn"이 아니라 **"(map, size
  group) 커버리지를 보장하도록 미리 spawn한 pool에서 subset을 활성화"**하는 의미다.

실제 허용/금지 key 목록은 코드 상수(`environment.py`의 `MAP_TYPE_ALLOWED_STATIC_KEYS` /
`STATIC_GLOBALLY_BANNED_KEYS`)와 `obstacle_catalog.yaml`이 source of truth다.

---

## 5. 내부 벽

작은 물체를 줄 세우는 대신 **box wall entity**를 별도로 둔다(틈·구조 흔들림 방지). 벽 pool은 startup에
1회 spawn하고, 미사용 맵의 벽은 지하 parking, episode마다 `set_pose`로만 활성화한다. 맵별로:
lobby=주변부 짧은 벽, corridor=긴 평행 벽 2개, intersection=`+` 통로용 코너/긴 벽, clutter=짧은 세로/가로 벽 여러 개.

---

## 6. 배치 · spawn 로직

- **정적 장애물**: episode의 `map_type` + stage `allowed_static_groups` + 맵의 free/forbidden region을 보고
  배치. **큰 것부터**(large→medium→small) 놓아 통로/교차부/경로가 끊기지 않게 한다.
- **start/goal/human**: 정사각 random이 아니라 맵의 free region 안에서만 샘플링(corridor는 통로 양끝 성향,
  intersection은 다른 팔, lobby는 open area 양측). 맵이 커지면 spawn bounds와 parking slot도 함께 확대.
  `_sample_train_start_pose` / `change_goal` / `_build_human_spawn_regions` 등이 `current_layout_spec`를 읽는다.
- clutter 외에는 구조상 연결성이 자동 보장되므로, 명시적 그래프 경로 검사는 1차에서 생략했다.

---

## 7. 커리큘럼 stage (현재 10-stage)

> **현재 구현(10-stage).** 원칙: 한 stage에 한 축만 크게 변경(구조 → 사람 → 위치추정 노이즈 → 지형 →
> proprio 노이즈 → 새 맵·군중 → 통합). Stage 3–6은 엄격한 단일 축, 7–9는 통합 단계.
>
> | stage | maps | static (by_map) | humans (by_map) | loc noise | proprio | 핵심 축 |
> |---|---|---|---|---|---|---|
> | 0 empty | lobby | 0 | 0 | clean | – | 기본 goal 도달 |
> | 1 corridor_static | corridor | 3 | 0 | clean | – | corridor 정적 |
> | 2 add_intersection | corridor,intersection | C3 / I6 | 0 | clean | – | **구조** |
> | 3 first_human_clean | corridor,intersection | C4 / I6 | C1 / I1 | clean | – | **사람**(동적 회피) |
> | 4 first_human_noisy | corridor,intersection | C4 / I6 | C1 / I1 | weak | – | **위치추정 노이즈** |
> | 5 add_clutter_clean | +clutter | C4 / I6 / Cl8 | C1 / I1 / Cl1 | weak | – | **지형** (+yield 해제) |
> | 6 add_clutter_noisy | corridor,intersection,clutter | C4 / I6 / Cl8 | C1 / I1 / Cl1 | weak | light | **proprio 노이즈** |
> | 7 add_lobby | 4종 | C4 / I6 / Cl8 / L8 | C1 / I2 / Cl2 / L3 | weak | light | **새 맵**+scan 노이즈+군중↑ |
> | 8 scale_crowd | 4종 | C5 / I7 / Cl8 / L8 | C2 / I3 / Cl4 / L5 | drift | light | **군중 확대**(통합) |
> | 9 full_complexity | 4종 | C5 / I7 / Cl8 / L9 | C3 / I4 / Cl4 / L6 | robustness_train | medium | **final 통합** |
>
> (C/I/Cl/L = corridor/intersection/clutter/lobby. corridor가 항상 가장 적음 — 5.2 m 차선의 배치 상한.)
> 승급 게이트는 9-entry 리스트. yield 축은 Stage 0–4 봉인·Stage 5 해제(진급 시 버퍼 리셋 + 재워밍업).

논문용으로는 stage 평균만 보지 말고 **맵 타입별**(corridor/intersection/clutter/lobby) 성능을 따로
집계해야 한다 — 섞인 stage에서 특정 맵만 무너지는 것을 놓치지 않도록. (trainer가 `map_type`을 로깅하고
`curriculum_eval_per_map_*.csv`를 출력한다.)

*(초기 5-stage 제안은 역사적 기록으로만 남겨두었으며, 위 10-stage 표가 현재 기준이다.)*

---

## 8. 구현 현황 & 검증

**바뀐 파일:**
- `environment_curriculum.yaml` — `map_layout_enabled`, `map_inner_*`/`map_wall_*`/통로 폭, stage별
  `allowed_map_types`/`map_type_probs`/`allowed_static_groups`/`eval_map_types`, parking slot 재배치.
- `environment.py` — 맵 정책 상수, `_build_map_layouts`(벽/free/human region), 그룹-aware pool 커버리지,
  벽 box pool(startup spawn + set_pose), layout-aware 샘플링, read-only `current_map_type` 파라미터.
- `environment_curriculum.py` — 맵 관련 값도 base snapshot/restore에 포함(stage 누수 방지).
- `train_tqc_curriculum_agent.py` — `map_type` 로깅, per-map 평가 집계.

**핵심 설계 결정:**
- 25×25는 world 외벽 + `map_inner_*`만 바꾸면 되도록 모든 extent를 파생시킴.
- 고정 정체성 pool을 유지하려고, `(map_type, size_group)` 커버리지를 보장하는 합집합(15키: small/medium/large
  각 5)을 startup에 1회 spawn하고 episode마다 허용 subset만 활성화.
- `eval_map_types`가 평가를 실제 제어: trainer가 eval 전후로 `curriculum_eval_mode`를 토글하고 env가
  `eval_map_types`를 round-robin. `try/finally` + 실제 적용값 재확인으로 eval 고착/desync 방지.
- lobby large 배치는 open-area를 `radius + margin`만큼 팽창해 회피, start는 전방 내부 벽을 ray-march로 검사.

**검증(정적/오프라인):** `py_compile` 통과, YAML 파싱 OK, 허용키 전부 카탈로그 존재·banned 누수 없음,
15키로 4맵×3그룹 커버리지 충족.

**남은 리스크(런타임 미실측):** 벽 box의 실제 spawn/teleport, 좁은 corridor + large 동시 배치 시 reset
실패율, per-episode/eval get/set_parameters 호출의 wall-clock 영향은 사용자가 직접 빌드/실행해 확인 필요.
명시적 connectivity heuristic은 1차 생략(구조상 보장 + clutter는 넓은 gap으로 완화).
