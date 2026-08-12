# TQC용 Auxiliary Prediction Network (AUX_PRED)

TQC curriculum agent 위에 얹은 future-aware auxiliary prediction.
87-D state와 actor/critic 사이에 **shared encoder**를 두고, auxiliary head가 그 latent에서
**동적 장애물의 future risk**를 예측하게 해 encoder가 motion-aware feature를 학습하도록 유도한다.
모든 관련 코드에는 `AUX_PRED` 태그가 붙어 있어 찾기/제거가 쉽다.

기본값(양쪽 config `enabled: false`)에서는 baseline TQC와 byte 단위로 동일하다.

## 아키텍처 (블록 다이어그램)

```
                             s_t  (87-D state)
                                   │
                                   ▼
                     ┌─────────────────────────┐
                     │   SharedEncoder  E      │   87 → 256 → 128
                     └────────────┬────────────┘
                                  │ z_t (latent 128)
        ┌─────────────────────────┼──────────────────────────────┐
        │ z_t.detach()            │ (z_t, a_t)                    │ z_t
        ▼                         ▼                               ▼
  ┌───────────┐            ┌────────────┐             ┌───────────────────────┐
  │  Actor π  │──► a_t ──► │   Critic   │             │        Aux Head       │◄─ action_ctx   (v2)
  └───────────┘            └─────┬──────┘             │   trunk (MLP + LN)    │◄─ temporal_ctx (v2)
    (→ 환경)                     │ critic_loss        │   → risk map  H×K     │
                                 │                    │     [+min-dist][+distr]│
                                 │                    └───────────┬───────────┘
                                 │                                │ aux_loss
                                 │                                ▼
                                 │                       privileged label
                                 │                    (human_states CV rollout)
                                 ▼
                   (critic_loss + β·aux_loss).backward()
                     → E + Critic + Aux(head·GRU) 갱신
                       Actor는 z_t.detach() → encoder로 grad 안 감

  v2 context (aux head 입력에만 concat, 정책은 비순환 유지):
    action_ctx   : [a_t .. a_{t+K-1}] → action embed → GRU (+선택 self-attention)
    temporal_ctx : [s_t .. s_{t-N+1}] → SharedEncoder E → GRU

  배포/추론 : s_t → E → Actor 만 사용 (aux head·label 경로 전부 drop)
```

## 1. 차용한 논문 아이디어

| 논문 (`paper/`) | 재사용한 아이디어 |
|---|---|
| **Falcon** | shared latent이 future-prediction head를 먹이고, aux loss는 shared representation으로만 back-prop (`L = L_main + beta·L_aux`); privileged sim label은 학습 시에만 사용 |
| **Proximity-Aware** | human별 trajectory 대신 **고정 크기 egocentric risk map**을 예측: sector별 `risk = clamp(1 - d/D_c, 0, 1)`. masking/data association 없음 |
| **DiPCAN** | privileged ground-truth는 학습 label 생성에만; 배포 시 정책은 `encoder → actor`뿐 |

## 2. 이 프로젝트용 단순화

- **Single-step (off-policy).** TQC는 replay buffer에서 i.i.d. transition을 뽑으므로, 한 state의
  latent에서 future risk map을 바로 예측한다(recurrent encoder는 actor/critic 경로를 바꾸고 off-policy
  안정성을 해쳐 배제).
- **고정 크기 label.** 가변 pedestrian 수/ID 매칭 대신 `H x K` risk map → buffer가 상수 폭 target을
  저장하고 head는 MLP.
- **Constant-velocity rollout.** future position은 짧은 CV 외삽. 학습 중 pedestrian이 예측 가능/비반응적
  이라 정확하다.

## 3. v1 (기본) vs v2 (opt-in)

| | v1 (주 경로) | v2 (옵션) |
|---|---|---|
| Encoder | `87 → 256 → 128`, shared | 동일 (**불변** — actor/critic을 먹임) |
| Aux head | risk map `H x K` | + future min-distance `H`; + distributional risk; 더 깊은 trunk(LayerNorm) |
| Action context | 없음 | action 시퀀스 GRU (+ 선택적 self-attention) → aux head 입력에 concat |
| Temporal | 없음 | 최근 state history GRU → aux head 입력에 concat |
| Loss | risk map MSE | + min-distance/distributional 항, beta warmup |

**핵심 원칙 — aux-branch 전용.** 추가 capacity는 전부 **aux head**에만 들어간다. shared encoder(`256/128`)와
actor/critic은 절대 바뀌지 않는다(`latent_dim`이 actor/critic 입력 폭이라, 이를 키우면 그 net들의 checkpoint
resume이 깨진다). 그래서 off-policy actor/critic 경로가 안정적으로 유지되고, v2 옵션은 config 플래그로 켜는
additive head일 뿐이다. temporal/action context도 오직 aux head만 먹여 정책은 비순환으로 남는다.

**프로필별 설정:** future min-distance head, 더 깊은 LayerNorm aux trunk,
action-conditioned 변형과 beta warmup은 phase2 확장 프로필에서 사용할 수 있다. aux-only temporal
context는 actor-visible frame stack과 중복될 수 있어 프로필별로 on/off하며,
`phase2/both_trajrisk_rbs_cf_st`에서는 `temporal_enabled: false`다. distributional risk와 encoder
LayerNorm은 연결돼 있지만 기본 off다.

### 3b. Action-conditioned aux
v1은 **한** state `s_t`에서 미래를 예측하지만, 같은 `s_t`도 다음 action에 따라 다른 미래로 이어진다.
action-conditioned 변형은 `z_t` **와** 다가올 action 시퀀스 `[a_t..a_{t+K-1}]`로부터 같은 target을 예측해
action-aware future representation을 학습한다(target/loss는 v1과 동일, 예측 소스만 다름). off-policy라
action 시퀀스는 replay buffer의 `traj_end`(episode 경계) 플래그를 이용한 boundary-safe 인덱스 walk로
재구성하며, episode 밖 action은 loss에 도달하지 않는다.

### 3c. Temporal context (최근 state history)
단일 state는 장애물이 *어디로 향하는지*를 못 담지만 짧은 history는 담는다. `temporal_enabled: true`면
최근 `history_len`개 state(3b와 같은 boundary-safe backward walk)를 shared encoder로 인코딩한 뒤 작은 GRU로
요약해 aux head 입력에 concat한다. 정책은 여전히 비순환.

### 3d. Loss balancing
`aux_beta_warmup_steps`(기본 5000)가 `beta_aux`를 0→목표값으로 선형 ramp해, 초반 noisy aux head가 critic
학습을 교란하지 않게 한다. distributional risk는 대체로 MSE를 되풀이해 가치가 낮아 기본 off.

### Resume / ablation 호환성
aux head·temporal encoder는 **학습 전용**이며, checkpoint 구조가 현재 config와 안 맞으면
`rl/algorithms/tqc/agent.py`의 `Agent.load()`(구현은 `rl/checkpointing/tqc_io.py`)가
그 부분만 새로 초기화(경고 로그)하고 encoder/actor/critic은 깨끗이 resume한다. replay buffer는 옛 buffer가
episode 경계를 가졌을 때만 이월되고, 아니면 fresh buffer로 degrade(warmup이 다시 채움). config가 맞으면
이전과 동일하게 로드된다.

## 4. 파일 구조

신규(모두 `AUX_PRED` 태그):
- `rl/networks/aux_prediction.py` — `SharedEncoder`, `AuxiliaryHead`, `ActionConditionedAuxHead`
- `rl/networks/aux_losses.py` — `compute_aux_loss`
- `rl/networks/aux_temporal.py` — `TemporalContextEncoder`
- `env/observation/aux_prediction_labels.py` — privileged label 생성 + wire header

연결 지점: `rl/algorithms/tqc/agent.py`(구성·추론·save/load),
`rl/algorithms/tqc/update.py`(aux loss를 포함한 update), `rl/replay/buffer.py`(aux target 저장 +
boundary-safe walk), `rl/checkpointing/tqc_io.py`(temporal encoder save/load), `env/environment_interface.py`
(모든 클라이언트에서 label 분리), `training/train_tqc_curriculum.py`,
`env/simulation/risk_targets.py`(label wire 조립), 관련 config.

## 5. 학습 데이터 흐름

```
env step/reset (privileged human_states + GT robot pose)
  → compute_future_risk_labels → label [H*K risk | H min-dist]
  → 87-D state 뒤에 append
trainer
  → 87-D RL state + aux label 분리 → replay_buffer.add(..., aux_target)
agent.train()
  → z = encoder(s)
  → critic(z, a) → critic_loss ;  aux_head(z) → aux_loss
  → (critic_loss + beta·aux_loss).backward()   # encoder+critic+aux 갱신
  → actor는 z.detach()를 읽음                   # actor는 encoder를 갱신 안 함
```

Gradient 규칙: encoder는 `critic_loss + beta·aux_loss`로만 갱신, actor gradient는 `z.detach()`로 encoder에
도달하지 않음, target 경로는 Polyak-synced `encoder_target` 사용.

## 6. 추론 / 배포

평가/테스트/실로봇에서 정책은 `encoder → actor`뿐이다. aux head와 label 경로는 학습 전용이라 쓰이지 않고,
env 측 플래그가 off면 서비스는 순수 87-D state를 반환한다. baseline TQC checkpoint는 변화 없이 로드된다.

## 7. 켜는 방법

v1(risk map)은 양쪽 config에서 켜고 `num_sectors`/`horizons_sec`를 일치시킨다(불일치 시 trainer fail-fast):
- `hyperparameters_tqc.yaml` / `environment_curriculum.yaml` → `aux_prediction.enabled: true`

v2 add-on(부분집합 가능): `min_distance_loss_weight > 0`, `use_distributional_aux`,
`action_condition_attention`+`aux_trunk_*`, `temporal_enabled`(+`history_len`),
`aux_beta_warmup_steps`/`encoder_layernorm`.

## 8. Safety guard (fail-fast)

- **Config 일관성.** label 기하가 두 config에 나뉘어 있어, env가 모든 label 앞에
  `[VERSION, num_sectors, num_horizons, horizons...]` **wire header**를 붙인다. 첫 reset에서 trainer가 이를
  agent config와 필드별로 비교해, version/sector/horizon 값·개수·label 길이 중 하나라도 다르면 raise한다.
  "총 길이는 같지만 구조가 다른" 경우까지 잡고, env가 실제로 내보낸 wire를 기준으로 하므로 권위 있다.
- **Non-curriculum 차단.** aux는 `training/train_tqc_curriculum.py`에만 연결된다. 지원하지 않는 TQC trainer가
  aux-enabled agent를 만들면 `__init__`에서 즉시 raise(`AUX_SUPPORTED` 플래그). IEQN trainer는 aux 없는 별도
  agent라 영향 없다.
