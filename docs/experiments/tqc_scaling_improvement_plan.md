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

## 4.2 2순위: critic/encoder scaling

현재 critic은 이미 강한 편이지만, off-policy continuous control에서 병목은 actor보다 critic/value 쪽인 경우가 많다.

권장 방향:

- critic hidden: `256 -> 384` 또는 `512`
- plain MLP -> `Residual MLP`
- LayerNorm 또는 유사 정규화 추가

권장 순서:

1. `384` 규모 실험
2. 안정적이면 `512` 실험

actor는 초기에는 그대로 유지한다.

- `actor_hdim = 256` 유지
- critic/encoder 변화의 효과를 먼저 본다

이 단계는 **GPU 사용률을 높이는 목적**도 있지만, 더 중요한 목적은 **value estimation capacity**를 늘리는 것이다.

## 4.3 3순위: aux beta stage schedule

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

## 4.4 4순위: temporal encoder ablation

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

## 4.5 5순위: aux fusion 개선

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
| `A1` | `UTD=4`, `batch=512` |
| `A2` | `A1 + critic hidden 384/512 residual` |
| `A3` | `A2 + aux beta stage schedule` |

권장 순서는 `A0 -> A1 -> A3 -> A2`도 가능하다. 현재 구조에서 **aux schedule**은 코드 변경 범위가 작고,
critic scaling보다 먼저 볼 가치가 있다.

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
2. actor보다 critic/encoder를 먼저 scaling한다.
3. aux는 head 확장보다 `beta stage schedule`과 `FiLM/gated fusion`으로 encoder shaping을 개선한다.
4. 위험 샘플링과 latent dynamics aux는 후순위 실험으로 둔다.

가장 추천하는 첫 수정은 다음 조합이다.

- `updates_per_env_step = 4`
- `batch_size = 512`
- `aux beta stage schedule` 적용

그 다음 단계로:

- critic residual scaling
- temporal feature `32 -> 64`
- FiLM aux fusion

순으로 진행한다.
