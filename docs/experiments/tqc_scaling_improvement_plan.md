# TQC Scaling Improvement Plan

`train_tqc_curriculum_agent.py`의 현재 학습 구조에서 **"GPU 사용률이 낮으니 모델을 키우자"**는 판단을
어떤 순서와 기준으로 적용할지 정리한 문서다. 목표는 단순한 GPU 점유율 상승이 아니라, **동적 장애물
회피 성능을 해치지 않으면서 sample efficiency와 representation quality를 올리는 것**이다.

## 1. 현재 진단

현재 GPU 사용률이 낮은 1차 원인은 모델이 너무 작아서라기보다 다음 구조에 있을 가능성이 크다.

- `ROS2/Gazebo` 환경 step이 느리다.
- 학습 루프가 사실상 `1 env step -> 1 train update` 구조다.
- batch size도 `256`이라 4080급 GPU에 들어가는 계산량이 크지 않다.

따라서 **GPU 사용률 20% = 모델이 너무 작다**로 바로 해석하면 안 된다. 우선은
`update-to-data ratio (UTD)`와 batch size를 통해 GPU에 더 많은 학습 일을 넣는 쪽이 자연스럽다.

## 2. 현재 모델 요약

현재 TQC 계열 학습 모델은 이미 단순한 MLP policy가 아니다.

- `TemporalFusionEncoder`
  - 현재 state와 scan history를 분리해 인코딩
  - temporal feature는 curriculum stage에 따라 gain이 켜짐
- `Gaussian Actor`
  - latent를 입력으로 하는 SAC-style tanh Gaussian policy
- `TQC Critic`
  - critic 5개, critic당 quantile 25개
- `Action-conditioned Auxiliary Head`
  - 미래 action sequence를 GRU + self-attention으로 요약
  - risk map / min distance / TTC / hazard를 예측

또한 shared encoder는 actor loss가 아니라 **critic loss + aux loss**로 학습되고, actor는
`z.detach()`를 읽는다. 그래서 구조 변경의 초점은 actor보다 **critic/encoder와 aux shaping**에 두는 것이 맞다.

## 3. 설계 원칙

개선 방향은 아래 원칙을 따른다.

1. GPU를 더 쓰기 위한 첫 수단은 모델 폭 확장보다 **UTD 증가**다.
2. actor보다 **critic/encoder** scaling을 우선한다.
3. aux head는 "더 크게"보다 **encoder를 더 잘 shaping**하도록 개선한다.
4. 한 번에 많은 축을 바꾸지 않는다.
5. aux loss 개선 여부는 aux metric이 아니라 **회피 성능 지표**로 판단한다.

## 4. 우선순위

### 4.1 1순위: UTD ratio 증가 + batch size 확대

현재 기본 구조:

```text
1 env step -> 1 train update
batch_size = 256
```

우선 실험:

- `updates_per_env_step = 2`
- `batch_size = 512`

그 다음 단계:

- `updates_per_env_step = 4`
- `batch_size = 512` 또는 `1024`

이 단계의 목적:

- GPU 사용률 상승
- 동일 env data에서 더 많은 gradient update 수행
- sample efficiency 개선 가능성 확인

주의:

- UTD를 너무 높이면 오래된 replay data에 과적합할 수 있다.
- critic 불안정성이 생기면 바로 낮춰야 한다.
- 시작은 `2 -> 4 -> 8` 순서가 맞다.

## 4.2 2순위: aux beta stage schedule

현재 aux는 shared encoder를 critic과 함께 학습시킨다. 따라서 stage 초반부터 aux가 너무 강하면
기본 주행 학습을 방해할 수 있다.

권장 방향:

- 초기 stage에서는 aux를 약하게 또는 꺼둔다
- dynamic obstacle이 본격적으로 들어오는 stage부터 점진적으로 키운다

예시:

```yaml
stagewise_loss_schedule: [0.0, 0.02, 0.05, 0.10, 0.15]
```

정확한 값은 stage 개수와 커리큘럼 설계에 맞게 조정한다. 핵심은 **초기 clean stage에서 encoder를 aux로 과하게 끌지 않는 것**이다.

현재 코드 구조상 이 변경은 구현 비용이 작고, curriculum과도 직접 맞물린다. 따라서 critic scaling보다
앞에 두는 편이 더 안전하다.

## 4.3 3순위: critic/encoder scaling

현재 critic은 이미 강한 편이지만, off-policy continuous control에서 병목은 actor보다 critic/value 쪽인 경우가 많다.

권장 방향:

- critic hidden: `256 -> 384`를 먼저 시도하고, 그 다음 `512`
- plain MLP -> `Residual MLP`
- LayerNorm 또는 유사 정규화 추가

권장 순서:

1. `384` 규모 실험
2. 안정적이면 `512` 실험

actor는 초기에는 그대로 유지한다.

- `actor_hdim = 256` 유지
- critic/encoder 변화의 효과를 먼저 본다

이 단계는 **GPU 사용률을 높이는 목적**도 있지만, 더 중요한 목적은 **value estimation capacity**를 늘리는 것이다.

다만 1차 실험으로 바로 넣기보다는, `UTD + batch + aux beta schedule`이 실제로 성능에 기여하는지 본 뒤
2차 실험으로 넣는 편이 더 좋다.

## 4.4 4순위: FiLM/gated aux fusion

현재 action-conditioned aux head는 `z_t`와 미래 action context를 단순 concat한다.

```text
fused = concat(z_t, action_context)
```

이건 안정적이지만, **"이 state에서 이 미래 action이 위험에 어떤 의미를 갖는가"**를 반영하기엔 표현력이 제한적일 수 있다.

권장 변경:

- `FiLM fusion`
- 또는 `gated fusion`

예시:

```text
gamma, beta = MLP(action_context)
z_mod = gamma * z + beta
fused = concat(z_mod, action_context)
```

이 변경은 aux head를 크게 키우는 것보다 **encoder shaping quality**를 높일 가능성이 있다.

다만 코드 변경이 들어가므로, 1차 실험보다는 2차 실험에 두는 것이 맞다.

## 4.5 5순위: temporal encoder ablation

현재 temporal actor context는 이미 들어가 있다.

- encoder type: `conv1d`
- temporal feature dim: `32`
- stage 2부터 temporal gain 활성

동적 장애물 회피 관점에서 temporal branch는 충분히 중요하므로, 다음 순서로 ablation하는 것이 좋다.

| 실험 | 구조 |
|------|------|
| `T0` | 현재 `Conv1D`, `32D` |
| `T1` | `Conv1D`, `64D` |
| `T2` | `GRU-lite`, `64D` |
| `T3` | small Transformer, `64D` |

권장 순서:

1. `temporal_feature_dim 32 -> 64`
2. Conv1D depth/width 증가
3. GRU-lite 비교
4. Transformer는 마지막

history length가 짧기 때문에 Transformer는 우선순위가 높지 않다.

## 4.6 6순위: risk-balanced replay

현재 prioritized replay는 꺼져 있다. 동적 장애물 회피에서는 중요한 transition 비율이 낮을 수 있으므로,
위험 상황을 조금 더 자주 학습하는 샘플링 전략은 가치가 있다.

예시 배치 구성:

- `50%` random sample
- `25%` high-risk sample
- `25%` failure-near sample

다만 이 단계는 구현 난이도와 분포 편향 리스크가 있으므로 1차 실험 범위에는 넣지 않는다.

## 4.7 7순위: latent dynamics auxiliary

장기적으로는 작은 world-model 성격의 보조 loss도 후보가 된다.

예시:

- `pred(z_{t+1})` from `(z_t, a_t)`
- next goal distance 예측
- next heading error 예측
- next min scan 예측

이 단계는 full Dreamer/TD-MPC2보다 훨씬 가볍지만, 여전히 구현 비용이 크므로 후순위다.

## 5. 1차 실험 세트

처음부터 많은 축을 바꾸지 말고 아래 정도만 비교한다.

| 실험 | 변경점 |
|------|--------|
| `A0` | 현재 baseline |
| `A1` | `UTD=2` 또는 `4`, `batch=512` |
| `A2` | `A1 + aux beta stage schedule` |

이 단계에서는 **critic residual scaling과 FiLM fusion을 일부러 보류**한다. 목적은
`UTD + batch` 자체의 효과와 `aux beta schedule`의 효과를 먼저 분리해서 보는 데 있다.

## 5.1 2차 실험 세트

1차 실험이 개선을 보일 때만 구조 변경을 올린다.

| 실험 | 변경점 |
|------|--------|
| `A3` | `A2 + critic hidden 384 residual` |
| `A4` | `A3 + FiLM aux fusion` |

이 순서를 권장한다.

- `384 residual`은 `512 residual`보다 안전한 첫 scaling step이다.
- `FiLM fusion`은 유망하지만 코드 변경이 들어가므로, 먼저 UTD/schedule 효과를 확인한 뒤 넣는 편이 해석이 쉽다.

## 6. 평가 지표

aux loss만 보고 판단하면 안 된다. 반드시 아래를 같이 본다.

- success rate
- collision rate
- timeout rate
- SPL
- PSC
- H-Coll
- per-map success / collision
- success episode mean speed
- timeout episode mean speed
- timeout low-speed ratio
- critic loss stability
- `beta * aux_loss / critic_loss`

특히 이 문제에서는 **timeout + 저속 정체**가 핵심 실패 형태이므로 아래 두 개를 꼭 봐야 한다.

- timeout episode mean speed
- timeout low-speed ratio

## 7. 리스크

가장 중요한 리스크는 **aux branch capacity가 너무 커져서 shared encoder를 덜 shaping하게 되는 것**이다.

현재 구조에서는:

- encoder는 `critic_loss + beta * aux_loss`로 학습
- actor는 encoder를 직접 업데이트하지 않음

따라서 aux head를 과하게 키우면 aux task를 head가 자기 파라미터로 처리하고, encoder latent 품질은
오히려 덜 좋아질 수 있다. 그래서 aux 쪽은 "더 크게"보다 **schedule, fusion, small structured change**가 우선이다.

## 8. 결론

현재 구조는 이미 `TQC + temporal fusion + action-conditioned aux`가 결합된 꽤 강한 baseline이다.
따라서 첫 개선은 "모델을 무작정 크게"가 아니라 아래 순서가 맞다.

1. `UTD ratio`와 `batch size`를 올려 GPU에 더 많은 학습 일을 준다.
2. `aux beta stage schedule`로 stage별 encoder shaping 강도를 조절한다.
3. 그 다음에 critic/encoder scaling과 FiLM/gated fusion을 올린다.
4. 위험 샘플링과 latent dynamics aux는 후순위 실험으로 둔다.

가장 추천하는 첫 수정은 다음 조합이다.

- `updates_per_env_step = 2` 또는 `4`
- `batch_size = 512`
- `aux beta stage schedule` 적용

그 다음 단계로:

- critic hidden `384` residual scaling
- FiLM aux fusion
- temporal feature `32 -> 64`

순으로 진행한다.

## 9. 구현 및 실행 (A0–A4)

A1–A4는 **코드 복제 없이** 기존 코드에 config/flag 분기를 추가하는 방식으로 구현했다.
모든 옵션의 **기본값은 baseline(A0)과 동일**하므로, 아무 override 없이 실행하면 이전과
byte 수준으로 같은 학습이 돈다. 실험은 `config/experiments/A{1..4}/` 아래의 override
`hyperparameters_tqc.yaml` 하나로 선택하며(그 외 파일은 기본 config로 fallback),
UTD 비율만 CLI 파라미터로 준다.

### 공통 사전 조건

두 개의 터미널이 먼저 떠 있어야 한다(기존과 동일).

```bash
# 1) Gazebo + Hunter SE
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false

# 2) 커리큘럼 환경 노드 (aux 라벨을 붙이려면 aux_prediction.enabled=true 상태여야 함)
ros2 run drl_agent environment_curriculum.py
```

아래 학습 명령의 `train_config_file`은 **디렉터리**를 가리킨다(파일이 아니라).
그 디렉터리에 있는 `hyperparameters_tqc.yaml`만 override로 쓰이고, `train_tqc_config.yaml`은
없으므로 기본값으로 fallback된다. 절대경로 대신 아래처럼 `$PWD` 기반 경로를 써도 된다.

```bash
CFG=$PWD/ros2_ws/src/drl_agent/config/experiments   # 실험 config 루트
```

### A0 — baseline (변경 없음)

```bash
ros2 run drl_agent train_tqc_curriculum_agent.py
# (seed 지정: --ros-args -p seed:=0)
```

### A1 — UTD ratio 증가 + batch size 확대

- **구현**: 학습 루프의 non-checkpoint 경로에서 `1 env step -> N train update`가
  가능하도록 옵션화. `train_tqc_curriculum_agent.py`가 `updates_per_env_step`만큼
  `rl_agent.train()`을 반복한다. checkpoint 경로(`train_and_checkpoint`)는 손대지 않았다.
- **config key**:
  - `train_tqc_config.yaml: updates_per_env_step`(기본 1) 또는 CLI `-p updates_per_env_step:=N`(N>0이면 우선).
  - `hyperparameters_tqc.yaml: batch_size`(실험 파일에서 256 → 512).
- **바뀐 파일**: `scripts/policy/train_tqc_base.py`(설정/파라미터 로드),
  `scripts/policy/train_tqc_curriculum_agent.py`(train 반복), `config/train_tqc_config.yaml`,
  `config/experiments/A1/hyperparameters_tqc.yaml`.

```bash
ros2 run drl_agent train_tqc_curriculum_agent.py --ros-args \
  -p updates_per_env_step:=4 \
  -p train_config_file:=$PWD/ros2_ws/src/drl_agent/config/experiments/A1
```

### A2 — aux beta stage schedule

- **구현**: `aux_prediction.stagewise_loss_schedule`는 이미 코드에서 지원됨
  (`tqc_agent._current_aux_beta`가 비어있지 않으면 stage별 beta를 씀). A2는 이 값을
  실제로 채운 실험 config와, 메인 config 주석/문서를 보완한 것이다. 커리큘럼 10-stage에
  맞춘 10-entry 리스트를 쓴다.
- **config key**: `hyperparameters_tqc.yaml: aux_prediction.stagewise_loss_schedule`
  (기본 `[]` = baseline 글로벌 beta 경로). 비어있지 않으면 `loss_weight + aux_beta_warmup_steps`를 **대체**한다.
- **바뀐 파일**: `config/hyperparameters_tqc.yaml`(주석/문서),
  `config/experiments/A2/hyperparameters_tqc.yaml`(A1 + schedule).

```bash
ros2 run drl_agent train_tqc_curriculum_agent.py --ros-args \
  -p updates_per_env_step:=4 \
  -p train_config_file:=$PWD/ros2_ws/src/drl_agent/config/experiments/A2
```

### A3 — critic hidden 384 residual scaling

- **구현**: `tqc_networks.Critic`에 `residual`/`layernorm` 옵션 추가. 둘 다 false면
  기존 plain `nn.Sequential` critic과 파라미터 이름·수치까지 동일(baseline 불변).
  true면 `_ResidualCriticBody`(`in_proj -> [LN?→Linear→act→Linear + skip] × n_blocks -> out`)를
  쓰고 `critic_hdim`을 384로 올린다.
- **config key**: `hyperparameters_tqc.yaml`의 `critic_residual`(기본 false),
  `critic_layernorm`(기본 false), `critic_residual_blocks`(기본 2), `critic_hdim`(실험 384).
- **바뀐 파일**: `scripts/policy/tqc_networks.py`(residual body),
  `scripts/policy/tqc_agent.py`(config → Critic 전달), `config/hyperparameters_tqc.yaml`(키 추가),
  `config/experiments/A3/hyperparameters_tqc.yaml`(A2 + residual 384).
- **주의**: residual critic은 critic state_dict가 바뀌므로 **fresh run 전용**이다
  (baseline plain-critic checkpoint를 strict load할 수 없음). trainer가 이를
  **강제**한다 — `critic_residual=true`인데 `load_model:=true`(또는
  `resume_weight_prefix`)이면 hybrid resume를 막기 위해 즉시 에러로 중단한다.

```bash
ros2 run drl_agent train_tqc_curriculum_agent.py --ros-args \
  -p updates_per_env_step:=4 \
  -p train_config_file:=$PWD/ros2_ws/src/drl_agent/config/experiments/A3
```

### A4 — FiLM aux fusion

- **구현**: `aux_prediction.ActionConditionedAuxHead`에 `fusion_type` 옵션 추가. 기본
  `concat`은 기존 `[z_t, action_ctx]` 그대로(byte-identical). `film`이면 action context에서
  `(gamma, beta)`를 만들어 `z_mod = (1+gamma)*z_t + beta`로 latent를 modulation한 뒤
  `[z_mod, action_ctx]`를 trunk로 보낸다. **identity 초기화**(zero-init generator)라 학습
  시작 시점 출력은 concat과 완전히 동일하고(검증 max|Δ|=0), trunk 입력 폭도 그대로라
  actor/critic·trunk는 손대지 않는다. FiLM generator는 trunk/output head를 만든
  **뒤에 마지막으로** 생성하므로, 같은 seed에서 trunk/head 초기 가중치가 concat(A3)과
  **동일**하다 — A4는 초기조건이 같은 상태에서 fusion만 다른 순수 ablation이다.
- **config key**: `hyperparameters_tqc.yaml: aux_prediction.fusion_type`(기본 `concat`; `concat`|`film`).
- **바뀐 파일**: `scripts/policy/aux_prediction.py`(fusion_type + FiLM),
  `config/hyperparameters_tqc.yaml`(키 추가),
  `config/experiments/A4/hyperparameters_tqc.yaml`(A3 + film).
- **주의**: FiLM generator가 추가되어 aux head state_dict가 바뀌므로 **fresh run 전용**이며,
  A3와 마찬가지로 `fusion_type=film`이면 trainer가 `load_model:=true` /
  `resume_weight_prefix`를 거부한다.

```bash
ros2 run drl_agent train_tqc_curriculum_agent.py --ros-args \
  -p updates_per_env_step:=4 \
  -p train_config_file:=$PWD/ros2_ws/src/drl_agent/config/experiments/A4
```

### 실험 config 유지보수

`config/experiments/A{1..4}/hyperparameters_tqc.yaml`은 baseline `config/hyperparameters_tqc.yaml`을
복사해 실험 키만 바꾼 파일이다(각 파일 상단 헤더에 명시). baseline이 바뀌면 이 파일들도
재생성해야 drift가 없다. 각 실험이 baseline과 다른 키만 요약하면:

| 실험 | updates_per_env_step (CLI) | batch_size | stagewise_loss_schedule | critic_residual / hdim | fusion_type |
|------|------|------|------|------|------|
| A0 | 1 | 256 | `[]` | false / 256 | concat |
| A1 | 4 | 512 | `[]` | false / 256 | concat |
| A2 | 4 | 512 | 10-entry | false / 256 | concat |
| A3 | 4 | 512 | 10-entry | true / 384 | concat |
| A4 | 4 | 512 | 10-entry | true / 384 | film |
