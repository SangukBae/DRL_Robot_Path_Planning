# Structured Map Curriculum Expansion Plan

이 문서는 현재 `drl_agent` 커리큘럼 환경을 다음 방향으로 확장하기 위한 구체 설계안이다.

- 맵 타입을 `lobby / corridor / intersection / clutter` 4종으로 확장
- `25 x 25` 마스터 맵을 기준으로 사용
- `lobby`는 전체 `25 x 25` 사용
- `corridor / intersection / clutter`는 내부 외벽을 당겨 `21 x 21` 유효 영역처럼 사용
- 맵 타입에 따라 내부 벽 배치, 정적 장애물 후보군, 장애물 배치 가능 영역, 사람 spawn 제약을 다르게 적용
- 커리큘럼 stage가 올라갈수록 장애물 수만이 아니라 맵 종류와 맵 구조 복잡도도 함께 증가

이 설계의 목표는 다음과 같다.

- 완전 무작위 scatter 맵보다 재현 가능하고 설명 가능한 구조적 환경 제공
- 논문에서 `맵 타입별 성능`과 `구조 일반화`를 명확히 설명 가능하게 만들기
- 현재 코드베이스의 reset / pool / curriculum 구조를 크게 깨지 않고 확장하기

---

## 1. 현재 구조 요약

현재 커리큘럼 환경은 다음 구조다.

- Gazebo world 실행: `ros2_ws/src/hunter_se_gazebo/launch/simulate_hunter_se_ignition.launch.py`
- 환경 설정: `ros2_ws/src/drl_agent/config/environment_curriculum.yaml`
- 커리큘럼 stage 적용: `ros2_ws/src/drl_agent/scripts/environment/environment_curriculum.py`
- 실제 reset / start / goal / obstacle 배치: `ros2_ws/src/drl_agent/scripts/environment/environment.py`
- 학습 루프 / 평가 / stage 승급: `ros2_ws/src/drl_agent/scripts/policy/train_tqc_curriculum_agent.py`
- 정적 장애물 카탈로그: `ros2_ws/src/drl_obstacle_assets/config/obstacle_catalog.yaml`

현재 정적 장애물 배치는 사실상 아래 함수들에 의해 결정된다.

- `Environment._sample_free_pose(...)`
- `Environment._sample_train_start_pose(...)`
- `Environment.change_goal(...)`
- `Environment._build_human_spawn_regions(...)`

즉, 현재는

- 정사각형 범위에서
- start / goal / 기존 장애물과의 거리만 보고
- 아무 구조 없이 무작위 배치

가 기본 동작이다.

그리고 이 무작위 배치의 기준 범위가 현재 코드에 하드코딩 성격으로 퍼져 있다.

- `lower / upper`: 로봇 start 샘플링 범위
- `goal_obstacle_lower / goal_obstacle_upper`: goal, static obstacle, human 샘플링 기준 범위
- `parking_slot_xs / parking_slot_ys`: 비활성 obstacle 주차 위치

즉 맵을 키우고 4종으로 늘리려면, 단순히 벽만 추가하는 것으로 끝나지 않고
`샘플링 bounds 자체를 layout-aware하게 바꾸는 작업`이 반드시 같이 들어가야 한다.

### 1.1 런치 파일 기준 제약

`simulate_hunter_se_ignition.launch.py`를 기준으로 보면 현재 런타임 제약은 아래와 같다.

- `world` 인자는 현재 `drl_arena`, `hospital`만 alias 지원
- Gazebo service bridge가 `/world/default/...`를 하드코딩
- 따라서 별도 `.world` 파일을 늘릴 경우 SDF world name이 반드시 `default`여야 함
- 학습 reset은 launch 시점 spawn 위치가 아니라 `environment.py`의 `/set_pose` 호출로 다시 배치됨

추가로 현재 기본 world인 `drl_arena.world` 자체에도 중요한 물리 제약이 있다.

- 바닥 plane은 `38 x 38`이지만
- 실제 충돌 외벽은 `x, y = ±9.5`에 놓인 `19 x 19` 아레나

따라서 맵 4종 확장은 두 방식 중 하나로 정리해야 한다.

- 권장안 A: 기존 `drl_arena.world` 하나를 유지하고 episode마다 내부 벽과 장애물을 바꿔 맵 타입을 만들기
- 대안: world 파일을 여러 개로 늘리되, launch alias / resource path / world name `default` 제약까지 같이 관리하기

중요:

- `drl_arena.world`를 그대로 유지하면 실제 usable arena는 `19 x 19`가 상한이다
- 따라서 `25 x 25` lobby를 쓰려면 `master world 유지`만으로는 불가능하고, world 파일의 외벽 4개를 바꾸는 수정이 필수다

즉 현재 계획은 아래 두 선택지 중 하나를 먼저 확정해야 한다.

- 선택지 1: `25 x 25`를 유지하고 `drl_arena.world` 외벽도 함께 확장한다
- 선택지 2: world 수정 없이 전체 structured map을 `19 x 19` 안에서 다시 스케일링한다

논문용 커리큘럼을 빠르게 안정화하려면, "world 수정 없는 master world 유지"와 "`25 x 25` lobby"를 동시에 주장하면 안 된다.

이번 수정은 이 구조를 완전히 갈아엎는 것이 아니라,

- `맵 타입`이라는 상위 개념을 추가하고
- 그 맵 타입에 따라
  - 내부 벽
  - 유효 외벽
  - 정적 장애물 허용 후보
  - 장애물 허용 구역
  - 사람 / start / goal spawn 구역

을 바꾸는 방식으로 확장한다.

---

## 2. 목표 맵 정의

### 2.1 마스터 맵 좌표계

기준 맵은 현재 두 안 중 하나로 확정해야 한다.

#### 안 A. world까지 수정하는 경우

- 기준 맵을 `25 x 25`로 둔다
- 전체 외곽 경계: `x, y in [-12.5, 12.5]`
- `lobby`는 이 전체 범위를 그대로 사용
- 나머지 세 맵은 내부 외벽을 당겨 유효 경계를 `x, y in [-10.5, 10.5]`로 사용

즉

- `lobby`: 실 사용 크기 `25 x 25`
- `corridor / intersection / clutter`: 실 사용 크기 `21 x 21`

이다.

#### 안 B. world를 그대로 두는 경우

- 기준 맵을 현재 외벽 안쪽 `19 x 19`로 둔다
- 전체 외곽 경계: `x, y in [-9.5, 9.5]`
- `lobby`는 이 범위 전체를 사용
- 나머지 세 맵은 내부 외벽을 당겨 `15~17 m`급 유효 영역으로 재설계한다

즉 structured map 자체는 만들 수 있지만,
초기 문서의 `25 x 25 / 21 x 21` 숫자는 이 안에서는 그대로 사용할 수 없다.

권장:

- 논문에서 꼭 `25 x 25`가 필요하면 world 수정부터 먼저 한다
- 아니면 1차 구현은 `19 x 19` 기준으로 structured curriculum을 먼저 완성한다

### 2.2 4종 맵의 개념

#### lobby

- 중앙 open space가 넓다
- 외곽 또는 주변부에만 짧은 벽 / island 구조를 둔다
- 사람과의 자유로운 교차가 핵심

#### corridor

- 길고 단순한 주 통로가 존재한다
- 복도 방향은 가로 또는 세로 중 하나
- 벽과 좁은 길에서 마주침 / 추월 / 양보 학습이 핵심

#### intersection

- 십자형 또는 T자형 교차 통로가 존재한다
- 중심 교차부에서 사람-로봇 상호작용이 일어난다

#### clutter

- 완전 미로는 아니지만 짧은 내부 벽과 장애물이 여러 개 놓인다
- 여러 경로 후보가 존재한다
- 국소 우회, 시야 가림, 좁은 통과가 핵심

---

## 3. 새 설정 파라미터 설계

다음 설정을 `environment_curriculum.yaml`의 `environment:` 또는 `curriculum.stages:`에 추가한다.

### 3.1 전역 base 설정

아래 숫자 예시는 `안 A (world까지 확장)` 기준이다.
`안 B (기존 19 x 19 유지)`를 택하면 같은 필드 구조를 쓰되 값만 축소해서 다시 잡아야 한다.

```yaml
environment:
  map_master_lower: -12.5
  map_master_upper: 12.5
  map_inner_lower: -10.5
  map_inner_upper: 10.5
  map_wall_thickness: 0.5
  map_corridor_width: 3.5
  map_intersection_width: 3.5
  map_lobby_open_half_extent: 4.0
  map_layout_mode: "random_from_stage"
  map_use_master_world: true
  parking_outer_margin: 2.5
```

역할:

- `map_master_*`: `25 x 25` 전체 맵 기준
- `map_inner_*`: `21 x 21` 유효 내부 맵 기준
- `map_wall_thickness`: 내부 벽 두께
- `map_corridor_width`, `map_intersection_width`: corridor / intersection 통로 폭
- `map_lobby_open_half_extent`: lobby 중앙 개방 영역 반폭
- `map_use_master_world`: world 파일을 유지하고 내부 구조만 바꿀지 여부
- `parking_outer_margin`: 비활성 obstacle 주차 슬롯을 실제 arena 밖으로 얼마나 더 밀어낼지

추가로 아래처럼 `spawn bounds`도 분리하는 편이 좋다.

```yaml
environment:
  map_master_start_lower: -11.5
  map_master_start_upper: 11.5
  map_inner_start_lower: -9.5
  map_inner_start_upper: 9.5
  map_master_goal_lower: -11.0
  map_master_goal_upper: 11.0
  map_inner_goal_lower: -9.0
  map_inner_goal_upper: 9.0
  parking_slot_xs: [-15.0, -17.0, 15.0, 17.0]
  parking_slot_ys: [-15.0, -17.0, 15.0, 17.0]
```

이유는 간단하다.

- 맵이 `25 x 25`로 커지면 로봇 spawn 영역도 넓어져야 함
- static obstacle spawn 영역도 넓어져야 함
- human spawn 영역도 비례해서 넓어져야 함
- 하지만 벽 근처 margin까지 그대로 넓히면 reset 실패율이 늘 수 있으므로 `map 크기`와 `실제 spawn 가능 범위`는 분리하는 편이 안전함
- 동시에 `parking_slot_xs / parking_slot_ys`도 새 arena 바깥으로 다시 밀어내야 함

그리고 parking slot 개수도 pool 크기와 함께 다시 잡아야 한다.

- 현재 기본 설정은 `obstacle_pool_static_size: 9`, `obstacle_pool_human_size: 5`라 총 14 슬롯 수준이면 운영 가능
- 하지만 `group 합집합 사전 spawn` 전략을 쓰면 static pool이 20~30개 이상으로 커질 수 있다
- 그러면 `4 x 4 = 16` 슬롯 예시는 더 이상 충분하지 않다
- 즉 `parking_slot_xs / parking_slot_ys`는 위치뿐 아니라 `총 개수`도 새 pool 크기에 맞춰 함께 확장해야 한다

권장 규칙:

- `len(parking_slots) >= obstacle_pool_static_size + obstacle_pool_human_size`
- 가능하면 정확히 맞추지 말고 `10~20%` 여유를 둔다

주의:

- 현재 slot 조합에는 `(12, 12)` 같은 점이 포함된다
- arena를 `±12.5`로 키우면 이런 슬롯은 아레나 내부로 들어와 비활성 장애물이 학습 영역에 남게 된다
- `±14` 중심 슬롯은 모두 자동으로 내부가 되는 것은 아니지만, 새 외벽과의 간격이 충분한지 별도 검토가 필요하다

### 3.2 stage별 설정

각 stage는 기존 `active_static`, `active_humans` 외에 다음 필드를 가진다.

```yaml
curriculum:
  stages:
    - name: stage_x
      active_static: 5
      active_humans: 1
      allowed_map_types: ["corridor", "intersection"]
      map_type_probs: [0.7, 0.3]
      allowed_static_groups: ["corridor_small", "corridor_medium"]
      eval_map_types: ["corridor", "intersection"]
```

역할:

- `allowed_map_types`: 이 stage에서 에피소드마다 선택 가능한 맵 타입 목록
- `map_type_probs`: 맵 샘플링 확률
- `allowed_static_groups`: 이 stage에서 사용할 정적 장애물 후보군
- `eval_map_types`: 평가 때 고정 또는 순회할 맵 타입 목록

주의:

- `map_type_probs` 길이는 `allowed_map_types`와 같아야 한다
- omitted 시 균등 분포를 기본값으로 둔다
- stage가 맵 타입을 바꾼다면, `environment_curriculum.py`의 base snapshot / restore에 이 새 필드들도 포함해야 함
- 그렇지 않으면 이전 stage 설정이 다음 stage로 누수된다

---

## 4. 장애물 그룹 설계

맵 구조를 유지하려면 장애물을 맵 타입별로 다르게 골라야 한다.

### 4.1 크기 그룹

현재 `radius` 기준으로 우선 3그룹으로 나눈다.

- `small`: `radius <= 0.40`
- `medium`: `0.40 < radius <= 0.65`
- `large`: `radius > 0.65`

다만 첨부 분석 기준으로는 `catalog radius`와 실제 XY footprint 차이가 꽤 크다.

- `bookstore_desk_b`: radius는 `0.60`인데 실제 길이는 약 `5.99 m`
- `warehouse_shelf / warehouse_shelf_e`: radius는 `0.70`인데 실제 길이는 약 `3.92 m`
- `bookstore_desk_a`: radius는 `0.60`인데 실제 길이는 약 `3.62 m`
- `house_kitchen_cabinet`: radius는 `0.65`인데 실제 길이는 약 `3.22 m`

즉 1차 필터는 `radius`로 하더라도, 실제 맵별 허용/금지 결정은
`obstacle key 단위`로 별도 관리하는 것이 맞다.

### 4.2 맵 타입별 기본 그룹

#### corridor

- 기본 허용: `small`, `medium`
- 제한: `large` 대부분
- 이유: 통로 구조를 쉽게 망가뜨리기 때문

추천 허용 obstacle key:

- `warehouse_bucket`
- `warehouse_trash_can`
- `hospital_chair`
- `bookstore_chair`
- `house_chair_a`
- `bookstore_column_a`
- `hospital_bedside_table`
- `hospital_drawer`
- `hospital_instrument_cart1`
- `hospital_mop_cart`
- `hospital_surgical_trolley`
- `warehouse_cluttering_c`
- `warehouse_cluttering_d`
- `house_kitchen_table`
- `house_refrigerator`

기본 제한 obstacle key:

- `warehouse_shelf`
- `warehouse_shelf_e`
- `bookstore_desk_a`
- `bookstore_desk_b`
- `bookstore_shelf_a`
- `bookstore_shelf_b`
- `bookstore_shelf_c`
- `bookstore_info_desk`
- `hospital_trolley_bed`
- `hospital_xray_machine`
- `house_bed`
- `house_wardrobe`
- `house_sofa`
- `house_fitness_equipment`
- `house_kitchen_cabinet`

#### intersection

- 기본 허용: `small`, `medium`
- 일부 `large`는 외곽 anchor에서만 허용
- 이유: 교차부를 너무 쉽게 막지 않도록 하기 위해

추천 허용 obstacle key:

- `corridor` 허용 목록 전체
- `hospital_metal_cabinet`
- `hospital_vending_machine`
- `hospital_parking_trolley_max`
- `hospital_wheelchair`
- `warehouse_cluttering_a`
- `bookstore_column_a`
- `house_fitness_equipment`

외곽 anchor에서만 제한적으로 허용:

- `bookstore_desk_a`
- `bookstore_shelf_a`
- `bookstore_shelf_b`
- `bookstore_shelf_c`
- `house_kitchen_cabinet`
- `hospital_xray_machine`
- `warehouse_shelf`
- `warehouse_shelf_e`

기본 제한 obstacle key:

- `bookstore_desk_b`
- `bookstore_info_desk`
- `hospital_trolley_bed`
- `house_bed`
- `house_sofa`
- `house_wardrobe`

#### clutter

- 기본 허용: `small`, `medium`, `large`
- 이유: 복잡도 자체가 목적이기 때문

추천 허용 obstacle key:

- `warehouse_bucket`
- `warehouse_cluttering_a`
- `warehouse_cluttering_c`
- `warehouse_cluttering_d`
- `warehouse_trash_can`
- `warehouse_shelf`
- `warehouse_shelf_e`
- `hospital_instrument_cart1`
- `hospital_drawer`
- `hospital_bedside_table`
- `hospital_metal_cabinet`
- `hospital_vending_machine`
- `hospital_chair`
- `hospital_mop_cart`
- `hospital_parking_trolley_max`
- `hospital_surgical_trolley`
- `hospital_wheelchair`
- `hospital_trolley_bed`
- `hospital_xray_machine`
- `bookstore_desk_a`
- `bookstore_shelf_a`
- `bookstore_shelf_b`
- `bookstore_shelf_c`
- `bookstore_info_desk`
- `bookstore_chair`
- `bookstore_column_a`
- `house_kitchen_table`
- `house_bed`
- `house_kitchen_cabinet`
- `house_refrigerator`
- `house_wardrobe`
- `house_chair_a`
- `house_fitness_equipment`
- `house_sofa`

제한 또는 별도 검토 대상:

- `bookstore_desk_b`
  너무 길어서 clutter에서도 통로 단절 위험이 큼
- `house_cooking_bench`
  높이 `0.10 m`라 현재 LiDAR height filter에서 거의 안 보일 수 있음

#### lobby

- 기본 허용: `small`, `medium`
- 일부 `large`는 중앙 개방 영역 밖에서만 허용
- 이유: open-space 성질을 유지해야 하기 때문

추천 허용 obstacle key:

- `corridor` 허용 목록 전체
- `hospital_metal_cabinet`
- `hospital_vending_machine`
- `hospital_parking_trolley_max`
- `hospital_xray_machine`
- `bookstore_desk_a`
- `bookstore_shelf_a`
- `bookstore_shelf_b`
- `bookstore_shelf_c`
- `bookstore_info_desk`
- `house_kitchen_cabinet`
- `house_wardrobe`
- `house_sofa`
- `warehouse_shelf`
- `warehouse_shelf_e`

중앙 open area 밖에서만 허용:

- `hospital_trolley_bed`
- `house_bed`
- `bookstore_desk_b`

기본 제한 obstacle key:

- `house_cooking_bench`
  센서 관측성이 낮아 학습용 장애물로 부적합

### 4.3 공통 제외 / 주의 obstacle

맵 타입과 무관하게 아래 항목은 기본적으로 주의가 필요하다.

- `house_cooking_bench`
  높이가 너무 낮아 현재 LiDAR 필터에서 거의 사라질 수 있음
- `bookstore_desk_b`
  실제 길이가 매우 길어서 radius 기반 배치와 잘 맞지 않음
- `hospital_wheelchair`
  visual/collision 차이가 있어 shape 해석에 주의 필요

즉 1차 구현에서는 `house_cooking_bench`는 아예 제외하고,
`bookstore_desk_b`는 lobby 전용 또는 완전 제외가 안전하다.

### 4.4 설정 형태

코드에 하드코딩하거나 YAML에서 별도 블록으로 둘 수 있다.

추천은 코드에 그룹 사전을 두고, stage는 그룹 이름만 지정하는 방식이다.

예:

```python
STATIC_GROUPS = {
    "corridor_compact": {
        "warehouse_bucket",
        "warehouse_trash_can",
        "hospital_chair",
        "bookstore_chair",
        "house_chair_a",
        ...
    },
    "intersection_edge_large": {
        "bookstore_desk_a",
        "warehouse_shelf",
        "hospital_xray_machine",
        ...
    },
    "clutter_dense": {...},
    "lobby_perimeter_large": {...},
}
```

추천 구조는 아래처럼 두 단계로 나누는 것이다.

- 1차: `globally_banned_keys`
- 2차: `map_type -> allowed_keys` 또는 `map_type -> allowed_groups`

이렇게 해야 stage가 많아져도 규칙이 단순하게 유지된다.

다만 현재 코드에는 구조적 제약이 하나 더 있다.

- obstacle pool은 첫 reset 때 모델 URI를 고정해서 spawn한다
- 이후에는 `set_pose`만 사용하고 `create/remove`는 정상 학습 중 피한다
- 즉 각 pool slot의 "모델 정체성"은 startup 이후 고정이다

그래서 `에피소드마다 allowed_static_groups를 바꿔 다른 obstacle key를 고른다`는 계획은
현재 pool 구조와 그대로는 맞지 않는다.

따라서 1차 구현 권장안은 아래다.

- `create/remove` 기반의 에피소드별 모델 교체는 피한다
- 대신 `group별 합집합`을 기준으로 static pool을 미리 넉넉히 spawn한다
- 에피소드마다 그 중에서 `map_type`과 `stage`에 맞는 subset만 활성화한다
- pool 채우기 자체도 group-aware하게 한다

즉 단순한 `catalog cycling`으로 size-9 풀을 채우는 방식은 부족하다.

- 특정 맵의 allowed group 멤버가 pool 안에 0개일 수 있기 때문이다
- 따라서 pool 초기화 단계에서부터 `key/group coverage`를 보장하도록 엔트리를 구성해야 한다

즉 문서의 `allowed_static_groups`는
`에피소드마다 새 모델을 spawn한다`가 아니라
`사전 spawn된 pool subset을 선택한다`는 의미로 해석해야 한다.

---

## 5. 내부 벽 설계

핵심은 작은 장애물을 벽처럼 억지로 줄 세우는 것이 아니라,
`내부 벽 전용 모델` 또는 `단순 box wall entity`를 별도 생성하는 것이다.

### 5.1 왜 별도 벽이 필요한가

작은 물체를 일렬로 놓는 방식은 다음 문제가 있다.

- 틈이 생길 수 있음
- 벽처럼 보이지 않음
- obstacle 종류 랜덤성 때문에 구조가 흔들림
- 교차로 / 복도 의미가 약해짐

따라서 `corridor`와 `intersection`은 내부 벽 기반으로 만드는 것이 맞다.

### 5.2 추천 구현 방식

`drl_obstacle_assets`에 단순한 긴 벽 모델 또는 box wall 모델을 추가하고,
맵 타입별로 그 벽들을 고정 좌표에 set_pose 하는 방식을 사용한다.

아래 크기 예시는 `안 A` 기준이다.
`안 B`를 선택하면 같은 topology를 유지하되 전체 좌표를 `19 x 19` 안으로 재스케일링한다.

런치 파일 기준으로 보면 이 부분은 다음처럼 나뉜다.

- 기존 `drl_arena.world` 안에서 런타임 spawn / set_pose만 할 경우: launch 수정 거의 불필요
- 새로운 wall 모델을 쓸 경우: Gazebo가 찾을 수 있게 `GZ_SIM_RESOURCE_PATH`에서 package share가 보이는지 확인 필요
- 아예 별도 world 파일을 늘릴 경우: `simulate_hunter_se_ignition.launch.py`의 known world alias와 `/world/default/...` 제약까지 같이 수정 필요

맵 타입별 예시:

#### lobby

- 전체 외곽은 `25 x 25`
- 중앙 open space 유지
- 주변부에 짧은 벽 4개 정도

#### corridor

- 내부 외벽으로 `21 x 21` 유효 영역 축소
- 긴 평행 벽 2개로 주 통로 생성

#### intersection

- 내부 외벽으로 `21 x 21` 축소
- 모서리 벽 4개 또는 긴 벽 조합으로 `+` 통로 생성

#### clutter

- 내부 외벽으로 `21 x 21` 축소
- 짧은 세로/가로 벽 여러 개 배치

### 5.3 새 코드 단위

`environment.py`에 다음 helper를 추가한다.

- `_sample_episode_map_type()`
- `_build_map_layout(map_type)`
- `_activate_layout_walls(layout_spec)`
- `_get_layout_free_regions(map_type)`
- `_get_layout_human_regions(map_type)`

`layout_spec`는 다음 정보를 갖는다.

- effective bounds
- wall list
- forbidden regions
- preferred start/goal regions
- preferred human regions

---

## 6. 정적 장애물 배치 로직 변경

기존:

- `_place_pool_group()`가 바로 `_sample_free_pose()` 사용

변경 후:

- 현재 episode의 `map_type`
- 현재 stage의 `allowed_static_groups`
- 현재 map의 `free regions / forbidden regions`

을 보고 배치

추가로 현재 pool 구조와의 절충안을 명확히 정해야 한다.

- 현재 `_initialize_obstacle_pool()`은 startup 시 모델 종류를 슬롯에 고정한다
- `_activate_random_obstacles()`는 그 슬롯들을 shuffle해서 일부만 arena에 올린다
- 따라서 `map별 허용 obstacle group`은 "배치 필터"일 뿐 아니라 "pool 구성 전략" 문제이기도 하다

1차 구현 권장 방식:

1. 사용할 obstacle key 합집합을 먼저 확정
2. 그 합집합으로 static pool을 startup 시 미리 spawn
3. 각 pool entry에 `group_tags` 또는 `key`를 저장
4. reset 때 `map_type`에 맞는 entry만 activation 후보로 사용
5. parking slot 수도 이 새 pool 크기에 맞춰 함께 늘린다

비권장 방식:

- stage/map 전환마다 create/remove로 모델 정체성 자체를 교체하는 방식

이 방식은 reset wall-clock과 failure surface를 크게 늘릴 가능성이 높다.

여기서 중요한 점은 `정적 장애물 배치 범위도 맵 크기에 비례해서 커져야 한다`는 것이다.

현재 코드는 사실상 아래 범위를 공유한다.

- start: `lower ~ upper`
- goal: `goal_obstacle_lower ~ goal_obstacle_upper`
- static obstacle: `goal_obstacle_lower + wall_margin ~ goal_obstacle_upper - wall_margin`
- human spawn: 동일 계열 bounds

새 구조에서는 episode별 layout spec이 최소한 아래를 제공해야 한다.

- `effective_start_bounds`
- `effective_goal_bounds`
- `effective_static_bounds`
- `human_spawn_regions`
- `forbidden_regions`

즉 맵이 커지면 start와 장애물도 같이 넓게 퍼져야 하지만,
`corridor / intersection`처럼 구조가 있는 맵에서는 단순 bounding box 확대가 아니라
`허용 구역 기반 샘플링`으로 바뀌어야 한다.

### 6.1 새 배치 순서

1. map type 선택
2. 내부 벽 활성화
3. 해당 map type의 유효 영역 결정
4. stage에 허용된 static group으로 후보군 제한
5. `large -> medium -> small` 순서로 샘플링
6. 통로 폭 / 중앙 open area 침범 여부 검사

### 6.2 왜 큰 장애물 먼저인가

큰 장애물을 나중에 놓으면:

- corridor가 막히거나
- intersection 중심이 사라지거나
- clutter에서 path가 완전히 끊길 수 있다

그래서 먼저 큰 것부터 놓고,
남은 자리에 작은 것을 채워야 한다.

### 6.3 필요한 새 검사

기존 거리 기반 collision 검사 외에 다음이 필요하다.

- `forbidden region overlap`
- `reserved corridor width violation`
- `central open area violation`
- `minimum connectivity heuristic`

완전한 그래프 기반 path check까지는 1차 구현에서 과할 수 있으므로,
초기 버전은 corridor/intersection/lobby는 구조상 connectivity가 자동 보장되게 만들고,
clutter에서만 최소 경로 폭 heuristic을 적용하는 것이 현실적이다.

---

## 7. start / goal / human spawn 제약

현재는 start/goal도 거의 정사각 random sampling이다.
맵 타입이 들어오면 이것도 바꿔야 한다.

특히 이번 변경에서 가장 중요한 것은 아래다.

- 맵이 커지면 로봇 스폰 가능 영역도 비례해서 커져야 함
- 장애물 스폰 가능 영역도 비례해서 커져야 함
- 사람 스폰 가능 영역도 비례해서 커져야 함
- 하지만 모든 맵이 동일한 사각형 샘플링을 쓰면 corridor / intersection 구조가 바로 무너짐

### 7.1 start / goal

새 원칙:

- map type의 free region 안에서만 샘플링
- corridor에서는 통로 양끝 성향이 강하게 나오게 유도
- intersection에서는 다른 팔에 goal이 잡히도록 유도 가능
- lobby에서는 넓은 open area 양측 또는 대각 방향 샘플링

추천 helper:

- `_sample_start_pose_for_map(map_type, layout_spec)`
- `_sample_goal_for_map(map_type, layout_spec, start_pose)`

실제 수정 대상은 아래 기존 함수들이다.

- `_sample_train_start_pose()`
- `change_goal()`
- `_sample_free_pose()`
- `_sample_free_pose_in_region()`
- `_build_human_spawn_regions()`

즉 이 다섯 곳이 모두 `current_layout_spec`를 읽도록 맞춰야 한다.

### 7.2 humans

새 원칙:

- `corridor`: 통로 위 spawn
- `intersection`: 각 팔에서 spawn
- `clutter`: free region 중 obstacle 밀집을 피한 곳
- `lobby`: open area 주변 또는 가로지르는 방향

기존 `human_placement_mode`를 확장한다.

예:

- `quadrants`
- `global_random`
- `corridor_lanes`
- `intersection_arms`
- `lobby_crossings`

---

## 8. 커리큘럼 stage 재설계

기존 stage는 장애물 수 / 사람 수 중심이다.
새 구조에서는 맵 종류까지 함께 올린다.

추천 예시는 아래와 같다.

### Stage 0

- `allowed_map_types: ["lobby"]`
- 정적 `0`
- 사람 `0`

목표:

- 기본 goal reaching

### Stage 1

- `allowed_map_types: ["corridor"]`
- 정적 `3~4`
- 사람 `0`

목표:

- 벽/통로 기반 주행

### Stage 2

- `allowed_map_types: ["corridor", "intersection"]`
- `map_type_probs: [0.7, 0.3]`
- 정적 `4~5`
- 사람 `1`

목표:

- 단일 사람 + 교차 구조

### Stage 3

- `allowed_map_types: ["corridor", "intersection", "clutter"]`
- 정적 `6~8`
- 사람 `2~3`

목표:

- 구조 다양화 + 복수 경로 학습

### Stage 4

- `allowed_map_types: ["corridor", "intersection", "clutter", "lobby"]`
- 정적 `8~10`
- 사람 `4~5`

목표:

- 최종 일반화

핵심 원칙:

- 쉬운 맵을 제거하기보다 비율을 낮춘다
- 즉 stage가 올라갈수록 새 맵을 추가한다
- 다만 승급 판단은 stage 전체 평균만 보지 말고, 논문용 평가는 맵 타입별로 따로 본다

### 8.1 trainer 관점에서 추가로 바꿔야 할 것

`train_tqc_curriculum_agent.py`는 현재 `curriculum_stage`만 기록하고 `map_type`은 기록하지 않는다.

따라서 논문 작성용으로는 최소 아래 수정이 필요하다.

- episode CSV에 `map_type` 컬럼 추가
- eval CSV 또는 paper metrics CSV에 `map_type`별 집계 추가
- `evaluate_and_print()`에서 mixed-map 평균만 내지 말고 `corridor / intersection / clutter / lobby`별 성능도 별도 저장

이유는 단순하다.

- stage 안에서 맵이 섞이면 평균 성공률은 좋아 보여도 특정 맵에서만 무너질 수 있음
- 지금 구조 그대로면 승급은 가능하지만 논문 표와 그림을 만들 정보가 부족함

---

## 9. 코드 수정 순서

추천 순서는 아래다.

### Step 1. 설정과 stage schema 확장

수정 파일:

- `ros2_ws/src/drl_agent/config/environment_curriculum.yaml`
- `ros2_ws/src/drl_agent/scripts/environment/environment_curriculum.py`

작업:

- `allowed_map_types`
- `map_type_probs`
- `allowed_static_groups`
- map 관련 base bounds
- map 관련 eval 설정
- parking slot 재배치 규칙

파싱 및 stage 적용 경로 추가

추가 주의:

- `environment_curriculum.py`의 `_snapshot_human_base()`와 `_apply_curriculum_stage()`는 지금 human/noise/count 중심이다
- 맵 타입 관련 값도 같은 방식으로 `base -> stage override` 복원 경로를 넣어야 한다
- 아니면 stage 전환 때 map 설정이 누적 오염된다

### Step 2. obstacle 그룹과 pool 전략 먼저 확정

수정 파일:

- `ros2_ws/src/drl_obstacle_assets/config/obstacle_catalog.yaml`
- `ros2_ws/src/drl_agent/scripts/environment/environment.py`

작업:

- `globally_banned_keys`
- `map_type별 allowed key/group`
- pool entry가 어떤 key/group에 속하는지 저장
- "group 합집합을 미리 spawn하고 subset만 활성화" 방식으로 풀 설계 확정
- `obstacle_pool_static_size`와 `parking_slot` 개수를 이 전략에 맞춰 함께 상향
- pool 초기화가 `allowed group coverage`를 보장하도록 key 배치를 설계

이 단계가 먼저 필요한 이유:

- 현재 obstacle pool은 모델 정체성이 startup에 고정되기 때문
- 이 충돌을 먼저 해결하지 않으면 뒤 단계의 `allowed_static_groups`가 구현 불가능하다

### Step 3. episode-level map type 상태 추가

수정 파일:

- `ros2_ws/src/drl_agent/scripts/environment/environment.py`

작업:

- `self.current_map_type`
- `self.current_layout_spec`

를 episode reset 시점에 결정하는 로직 추가

추가 항목:

- `self.current_map_effective_bounds`
- `self.current_start_bounds`
- `self.current_goal_bounds`
- `self.current_static_regions`
- `self.current_human_regions`

### Step 4. 내부 벽 전용 layout 시스템 추가

수정 파일:

- `ros2_ws/src/drl_agent/scripts/environment/environment.py`
- 필요 시 `drl_obstacle_assets`

작업:

- 내부 벽 모델 생성 / 주차 / 활성화 로직
- map type -> wall spec 변환

런치 영향:

- 같은 world 안에서 wall entity만 추가하면 `simulate_hunter_se_ignition.launch.py`는 거의 그대로 둬도 됨
- 새 world 파일 방식이면 `world` alias와 `/world/default/...` 호환성까지 재정의해야 함

중요:

- `25 x 25` lobby를 유지하려면 이 단계 전에 `drl_arena.world` 외벽 수정 여부를 먼저 확정해야 함
- world를 안 고치면 이 단계의 숫자도 `19 x 19` 기준으로 다시 잡아야 함

### Step 5. Step 2에서 정의한 그룹을 배치 로직에 연결

수정 파일:

- `ros2_ws/src/drl_agent/scripts/environment/environment.py`
- 필요 시 `ros2_ws/src/drl_obstacle_assets/config/obstacle_catalog.yaml`

작업:

- Step 2에서 정의한 `allowed key/group` 메타데이터를 `_place_pool_group_for_layout(...)` 경로에 연결
- 배치 시점에 `map_type`과 `stage`에 맞지 않는 pool entry를 필터링
- size group / semantic group은 여기서 새로 설계하지 않고, Step 2의 정의를 재사용

### Step 6. 배치 함수 교체

수정 파일:

- `ros2_ws/src/drl_agent/scripts/environment/environment.py`

작업:

- `_sample_free_pose()`를 직접 없애기보다
- `_sample_free_pose_for_layout(...)`
- `_place_pool_group_for_layout(...)`

형태로 확장

### Step 7. start / goal / human spawn 제약 강화

수정 파일:

- `ros2_ws/src/drl_agent/scripts/environment/environment.py`

작업:

- `_sample_train_start_pose()`
- `change_goal()`
- `_sample_human_spawn_pose()`

를 layout-aware하게 수정

여기에 반드시 포함할 것:

- 맵 크기 증가에 맞는 spawn bounds 확대
- map type별 시작/목표 선호 영역
- corridor/intersection에서 통로 밖 샘플링 금지
- clutter/lobby에서 너무 외곽 corner에만 몰리지 않도록 분포 보정
- parking slot을 새 arena 바깥으로 재배치

참고:

- 현재 launch의 초기 robot spawn 좌표는 학습 reset 때 덮어써지므로 커리큘럼 설계의 핵심 수정 포인트는 아님

### Step 8. 검증 및 시각화

작업:

- stage별 20~50 reset 반복
- start-goal reachable 여부 확인
- 사람과 정적 장애물이 corridor/intersection 구조를 깨지 않는지 확인
- RViz / Gazebo screenshot 수집
- map 크기 변경 후 spawn 분포 heatmap 확인
- map_type별 success/collision/timeout 로그 확인
- parking 상태의 비활성 obstacle이 arena 안에 남지 않는지 확인

---

## 10. 검증 체크리스트

필수 검증은 아래다.

### 구조 검증

- `lobby`가 실제로 open-space인지
- `corridor`가 실제로 단일 주 통로를 유지하는지
- `intersection`이 실제로 교차 구조를 가지는지
- `clutter`가 지나치게 미로형으로 막히지 않는지

### spawn 검증

- start가 벽/장애물 안에 걸리지 않는지
- goal이 unreachable한 곳에 찍히지 않는지
- humans가 벽 안 또는 blocked region에 생성되지 않는지

### 학습 검증

- reset 실패율
- stage 승급 안정성
- aux enabled 상태에서 wall-clock slowdown
- corridor와 lobby 간 성공률 분포 차이
- 맵 크기 증가 후 평균 episode 길이 변화
- map_type 섞인 stage에서 특정 맵만 과도하게 실패하지 않는지
- clutter에서 reachable 실패로 reset 재시도가 폭증하지 않는지

### 논문 검증

- 맵 타입별 성공률 / collision rate / timeout rate를 따로 집계할 수 있는지
- generalization 평가에서 map type을 로그로 남길 수 있는지
- 같은 stage 안에서도 map_type별 SPL / CTE / jerk 분해가 가능한지

---

## 11. 구현 시 주의할 점

### 11.1 wall과 static obstacle을 섞지 말 것

내부 벽은 구조 정의용이다.
랜덤 정적 장애물은 구조 내부의 변동성 부여용이다.

이 둘을 같은 개념으로 취급하면 corridor/intersection 의미가 금방 무너진다.

### 11.2 병렬 환경이 없으므로 reset 안정성이 매우 중요

현재 Gazebo 기반 단일 환경 학습은 reset 실패나 spawn 실패가 wall-clock에 바로 타격을 준다.
따라서 path planner 수준의 정교한 map generation보다,
deterministic하고 실패율이 낮은 layout system이 우선이다.

특히 아래 두 방식은 1차 구현에서 피하는 편이 좋다.

- 에피소드마다 obstacle model을 create/remove로 갈아끼우는 방식
- clutter에서 매 reset마다 reachable 여부를 여러 번 재샘플링으로 맞추는 방식

clutter도 1차에는 `결정론적 벽 배치 + 검증된 free region` 위주로 가는 것이 안전하다.

### 11.3 1차 구현은 “맵 타입 4개 + stage별 샘플링”까지만

처음부터 너무 많은 랜덤성을 넣지 않는 것이 좋다.

1차 구현 권장 범위:

- 맵 타입 4개
- 내부 벽 고정 배치
- map type별 static group 제한
- start/goal/human spawn 제약

후속 확장:

- 맵 타입별 jitter
- corridor orientation randomization
- intersection arms variation
- clutter wall count randomization

### 11.4 현재 코드 기준으로 꼭 바뀌는 세 파일의 역할 정리

`simulate_hunter_se_ignition.launch.py`

- master world 유지 전략이면 큰 수정은 불필요
- 새 world alias 또는 wall resource path가 필요할 때만 수정

`environment_curriculum.py`

- stage schema 확장
- map 관련 base snapshot / restore
- stage별 map sampling rule 적용

`train_tqc_curriculum_agent.py`

- `map_type` 로깅
- per-map evaluation 집계
- 논문용 metrics 출력 보강

---

## 12. 최종 한 줄 요약

이번 수정은 `완전 무작위 정사각 scatter 환경`을
`25 x 25 마스터 맵 위에서 4종 구조 맵을 stage별로 샘플링하는 structured curriculum`으로 바꾸는 작업이다.

실제 구현 순서는 아래로 고정한다.

1. stage schema에 `맵 타입 / 확률 / 장애물 그룹` 추가
2. `25 x 25`를 쓸지 `19 x 19` 기준으로 갈지 world 전략 먼저 확정
3. obstacle pool을 `group 합집합 사전 spawn + subset activation` 방식으로 정리
4. episode마다 현재 `map_type` 결정
5. map type별 내부 벽 활성화
6. map type별 허용 장애물 후보군만 사용
7. 맵 크기에 맞춰 `start / goal / static / human` 샘플링 bounds와 parking slot도 같이 재설계
8. layout-aware 정적 장애물 배치
9. layout-aware start / goal / human spawn
10. trainer에 `map_type` 로깅과 per-map 평가 추가
11. reset / 통로 / reachable 검증

---

## 13. 구현 현황 (안 B, 1차)

이 문서를 source of truth로 삼아 `안 B`(기존 `drl_arena.world` 19×19 유지) 기준으로 1차 구현을 완료했다.

### 13.1 바뀐 파일과 역할

- `config/environment_curriculum.yaml`
  - `map_layout_enabled: true` + `map_inner_*` / `map_wall_*` / `map_corridor_width` /
    `map_intersection_width` / `map_lobby_open_half_extent` / `map_wall_clearance` /
    `map_static_coverage_per_group` 추가
  - stage별 `allowed_map_types` / `map_type_probs` / `allowed_static_groups` /
    `eval_map_types` 추가 (§8 Stage 0~4 설계 반영)
  - `obstacle_pool_static_size`(fallback)와 `parking_slot_*`를 19×19 바깥(±11~±17)으로 재설계
- `scripts/environment/environment.py`
  - 맵 정책 상수: `MAP_TYPE_ALLOWED_STATIC_KEYS`, `STATIC_GLOBALLY_BANNED_KEYS`,
    `static_size_group()` (§4)
  - `_build_map_layouts()` (벽/free_region/human_region/open_area), `_build_static_pool_coverage()`
    (그룹-aware 합집합 사전 spawn), `_ensure_parking_slots()`,
    `_choose_map_type()` / `_select_episode_layout()` / `_activate_layout_walls()`
  - 벽 box pool: `_make_box_sdf()` / `_spawn_box_entity()` / `_initialize_wall_pool()`
    (startup 1회 spawn, 미사용 맵 벽은 지하 parking, episode마다 set_pose만)
  - layout-aware 샘플링: `_sample_free_pose()`/`_sample_free_pose_in_region()`/
    `_sample_train_start_pose()`/`change_goal()`/`_build_human_spawn_regions()`가
    `current_layout_spec`를 읽음. static 배치(`_place_pool_group`)는 map_type 허용키
    + size group 필터 + lobby large open-area 회피로 동작
  - read-only ROS 파라미터 `current_map_type` 추가
- `scripts/environment/environment_curriculum.py`
  - human base snapshot/restore 패턴에 `allowed_map_types`/`map_type_probs`/
    `allowed_static_groups`/`eval_map_types` 추가 → stage 누수 방지
- `scripts/policy/train_tqc_curriculum_agent.py`
  - `_fetch_current_map_type()`로 episode마다 map_type 읽기
  - curriculum episode CSV에 `map_type` 컬럼 추가
  - `evaluate_and_print()`에 per-map 집계 + `curriculum_eval_per_map_*.csv` 출력 추가

### 13.2 핵심 제약 반영 방식 (Step 2/5 분리)

1. **25×25 vs 19×19 충돌** → `안 B`로 19×19 안에서 구현. 모든 extent가
   `map_inner_*`에서 파생되므로, 나중에 `안 A`는 world 외벽 + `map_inner_*` 값만 바꾸면 됨.
2. **parking slot 재배치** → config를 ±11~±17로 옮기고, 코드가 풀 크기에 맞춰
   부족하면 arena 바깥 ring grid로 자동 확장(`_ensure_parking_slots`).
3. **고정 정체성 pool** → create/remove 없이, `(map_type, size_group)`별 커버리지를
   보장하는 합집합(검증 결과 15키: small 5 / medium 5 / large 5)을 startup에 1회 spawn하고,
   episode마다 허용 subset만 set_pose로 활성화. (Step 2 = 그룹/풀 전략 정의:
   `_build_static_pool_coverage`; Step 5 = 배치 로직 연결: `_place_pool_group` 필터)
4. **pool↔parking 정합** → 코드가 `len(parking_slots) >= static+human`(+15% 여유)를 강제.

### 13.3 검증 결과 (정적/오프라인)

- `py_compile`: 3개 .py 모두 통과
- YAML 로드: 정상, 5개 stage 필드 파싱 확인
- 맵 정책: 허용키 전부 카탈로그에 존재, banned 누수 없음, `bookstore_desk_b`는 lobby 전용
- 커버리지: 15키로 4개 맵 × (small/medium/large) 타깃(=5) 전부 충족, 모든 stage의
  active_static을 worst-map 기준으로도 만족(최소 corridor 10 후보 ≥ active 9)
- 레이아웃 기하: 모든 내부 벽이 ±9.5 외벽 안, free-region 사용가능 비율
  lobby 1.00 / corridor 0.96 / intersection 0.97 / clutter 0.87

### 13.4 리뷰 피드백 반영 (eval/large/start)

- **eval_map_types가 평가를 실제 제어** → trainer가 평가 루프 전후로 writable 파라미터
  `curriculum_eval_mode`를 raise/lower하고, env는 이 플래그가 켜지면 (train 모드를 유지한 채 =
  장애물/사람 활성화 유지) `eval_map_types` round-robin으로 맵을 고른다. 플래그 토글 시
  `_eval_map_cursor`를 리셋해 매 평가가 eval 맵을 처음부터 고르게 순회. 이전엔 평가 루프가
  훈련 분포를 그대로 써서 per-map CSV가 "우연히 나온 맵" 로그였던 문제 해결.
  - 견고성: eval 루프를 `try/finally`로 감싸 예외가 나도 `curriculum_eval_mode`를 항상 내려
    이후 학습 reset이 eval round-robin에 고착되지 않게 함. `_set_eval_mode(True)` 실패 시
    조용히 진행하지 않고 경고 + `metrics["eval_map_applied"]=False` + per-map 로그에 fallback
    표시(훈련 분포로 평가됨을 명시). 복구(False) 실패도 경고.
  - `_set_eval_mode()`는 SetParameters 성공 여부가 아니라, 토글 직후 `get_parameters`로
    `curriculum_eval_mode`의 **실제 적용값**을 재확인(`_fetch_eval_mode`)해 그 값이 목표와
    같을 때만 True를 반환. set 타임아웃-실제적용 / 이미 목표상태였던 경우에도
    `eval_map_applied`가 ground truth와 일치.
- **lobby large open-area 침범** → `_in_open_area(x, y, margin)`이 open-area 사각형을 `margin`만큼
  팽창. large 배치 시 `margin = radius + map_large_footprint_margin`(기본 2.0 m)을 적용해
  카탈로그 radius가 과소평가하는 실제 footprint를 보수적으로 덮음. 결과: large는 중앙(±4)을
  크게 벗어난 둘레 band(≈6.7~7.8 m)에만 배치(오프라인 acceptance 0.26 × 200 tries → 사실상 항상 성공).
- **start 전방 내부 벽 미검사** → `_front_hits_internal_wall()`로 heading 방향을 ray-march(6 step,
  거리 `clearance + robot_radius`)해 내부 벽을 향한 시작을 reject. 기존 외벽/lingering 전용
  front cone 검사를 보완.

### 13.5 미검증 / 남은 리스크

- **ROS/Gazebo 런타임 미실행**: 단일 환경이라 사용자가 직접 빌드/실행해야 검증 가능.
  특히 (a) 벽 box의 실제 spawn/teleport, (b) 19×19에서 large 장애물 + 좁은 corridor(폭 3.5m)
  동시 배치 시 reset 실패율, (c) per-episode/eval get_parameters·set_parameters 추가 호출의
  wall-clock 영향은 실측 필요.
- **corridor 폭 3.5m vs large 장애물**: corridor/intersection은 stage에서 size group을
  small/medium로 제한해 통로 막힘을 회피. clutter/lobby만 large 허용.
- **intersection 코너 블록(6.75×6.75)**: 큰 단일 충돌 박스라 LiDAR에 벽으로 보임(의도된 동작).
- **start 전방 내부 벽**: ray-march로 직선 전방만 검사(6 step). 측면 근접 벽은 wall_clearance가
  거름. 곡선 경로까지는 보지 않음(1차 단순화).
- **connectivity heuristic 미구현**: §6.3의 명시적 경로폭/연결성 검사는 1차 생략.
  corridor/intersection은 구조상 연결 보장, clutter는 짧은 벽+넓은 gap으로 완화.
