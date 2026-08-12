# TQC Curriculum 학습 모델 구조 (block diagram)

`train_tqc_curriculum.py`(`TrainTQCCurriculum`)가 학습하는 강화학습 모델의 전체 구조를
정리한 문서다. 이 trainer는 **TQC + Temporal Fusion Encoder + Action-conditioned Auxiliary
Prediction**을 기본 확장 경로로 사용하고, profile에 따라 Action-Risk Head,
spatiotemporal LiDAR(ST), counterfactual multi-horizon risk(CF)를 추가해 ROS2 service 기반
환경과 주고받으며 커리큘럼(10-stage)으로 학습한다.

- 정책은 **비순환(non-recurrent)** 이다. 시간 정보는 (1) actor/critic이 보는 압축 temporal
  feature와 (2) aux head 전용 context로만 들어가고, actor/critic body 자체는 순환하지 않는다.
- **shared encoder는 actor loss로 학습되지 않는다.** encoder는 `critic_loss + β·aux_loss`로만
  갱신되고, actor는 `z.detach()`를 읽는다. 그래서 구조의 초점은 actor보다 **encoder/critic + aux
  shaping**에 있다.
- 관련 코드 태그: `AUX_PRED`(aux), `TEMPORAL_ACTOR`(temporal fusion), 실험 플래그 `A1~A4`.

관련 모듈: `rl/algorithms/tqc/{agent,update,networks,metrics}.py`, `rl/networks/tqc.py`,
`rl/networks/action_risk_head.py`, `rl/networks/aux_prediction.py`,
`rl/networks/aux_temporal.py`, `rl/networks/aux_losses.py`, `rl/replay/buffer.py`(LAP),
`env/environment_interface.py`, `env/simulation/risk_targets.py`.

---

## 1. 전체 데이터 흐름 (환경 ↔ trainer ↔ 모델)

```
      ┌──────────────────────────── ROS2 서비스 경계 ────────────────────────────┐
      │                                                                          │
  ┌───────────────────────────┐        /reset, /step, /seed, ...        ┌────────────────────┐
  │ EnvironmentCurriculum      │◄───────────────────────────────────────│ TrainTQCCurriculum │
  │ (gym_node, 별도 프로세스)  │                                         │  (train_online 루프)│
  │                            │  state(+aux label 부착) , reward, done  │                    │
  │  Gazebo Ignition + Hunter  │───────────────────────────────────────►│                    │
  │  LiDAR→80bin, 보행자 CV    │        /gym_node/set curriculum_stage   │                    │
  └───────────────────────────┘◄───────────────────────────────────────└─────────┬──────────┘
                                                                                  │
   EnvInterface가 env가 붙인 privileged aux label을 잘라내                        │ s_t, a_t, r, s_{t+1}, aux_label
   self.last_aux_label로 노출 (state는 순수 관측만 남음)                          ▼
                                                                        ┌──────────────────────┐
                                                                        │  LAP Replay Buffer    │
                                                                        │  (state,action,next,  │
                                                                        │   reward,done,aux) +  │
                                                                        │   trajectory 경계     │
                                                                        └──────────┬───────────┘
                                                                                   │ minibatch
                                                                                   ▼
                                                                        ┌──────────────────────┐
                                                                        │  RL 모델 (아래 2절)   │
                                                                        └──────────────────────┘
```

- 학습 루프(`train_online`)는 매 env step마다 `select_action → step → buffer.add`를 돌리고,
  warmup 이후 **A1** `updates_per_env_step` 만큼 `rl_agent.train()`을 호출한다(기본 1).
- 평가 주기마다 `evaluate_and_print`로 success/collision/timeout·SPL·PSC·H-Coll 등을 재고,
  승급 gate를 만족하면 `set_curriculum_stage(stage+1)`로 환경과 모델 stage를 함께 올린다.

---

## 2. 모델 내부 구조 (핵심 block diagram)

```
                         s_t  (stacked state, 예: 327-D = 현재87 + scan history 80×3)
                                          │
             ┌────────────────────────────┴────────────────────────────┐
             │  TemporalFusionEncoder  E_ψ   (TEMPORAL_ACTOR)           │
             │  split(state):                                          │
             │    current [obs80 + agent7] = 87 ─► SharedEncoder ─► z_cur (128) │
             │    scan history [obs_{t..t-3}] 80×4 ─► ScanTemporalEncoder(conv1d 또는 ST Conv2d) │
             │    fuse: Linear([z_cur ; gain·z_tmp]) ─► ELU ─► z_t (latent 128)  │
             │    (gain = 0  stage<2  /  1  stage≥2  : 커리큘럼 stage로 게이팅)   │
             └───────────────────────────┬────────────────────────────┘
                                         │ z_t (128)
        ┌────────────────────────────────┼───────────────────────────────────────────┐
        │ z_t.detach()                   │ (z_t, a)                                    │ z_t (graph 유지)
        ▼                                ▼                                             ▼
 ┌──────────────────┐           ┌───────────────────────┐          ┌──────────────────────────────────────┐
 │  Actor  π_φ      │           │  Critic  Q_θ  (TQC)   │          │  ActionConditionedAuxHead  (AUX_PRED) │
 │  MLP 128→256→256 │           │  n_critics=5,          │          │  a_seq=[a_t..a_{t+K-1}] (K=4)        │
 │      →256        │           │  각 head 25 quantiles │          │    → embed Lin(3→64) → GRU(→256)     │
 │  → mean,log_std  │  a ─────► │  (A3: residual+LN,    │          │    → self-attn(4h,경계 마스크)       │
 │  tanh Gaussian   │           │   hidden 384 옵션)     │          │    → valid_len-1 gather → ctx(256)   │
 │  → a∈[-1,1]^3    │           │  target: 상위 2/critic │          │  fuse(z_t, ctx):                     │
 └────────┬─────────┘           │  절삭(truncation)     │          │    concat  = [z_t ; ctx]             │
          │ (r,θ,yield)         └──────────┬────────────┘          │    film(A4)= [(1+γ)z_t+β ; ctx]      │
          ▼                                │ current vs target      │      (γ,β = zero-init Lin(ctx))      │
   Pure Pursuit → cmd_vel                  │ quantile Huber         │  (+temporal_ctx concat, 기본 off)   │
   → 환경(정책 배포시 여기까지만 사용)     │                        │  → aux trunk (2×[Lin→LN→ELU],256)   │
                                           │                        │  → heads:                            │
                                           │                        │     risk_map (H·K=48) sigmoid       │
                                           │                        │     min_dist (H=3)    sigmoid       │
                                           │                        │     ttc (1) sigmoid / hazard(3) logit│
                                           │                        │     [risk_quant] 옵션                │
                                           │                        └──────────────┬───────────────────────┘
                                           │                                       │ pred
                                           ▼                                       ▼
                                     critic_loss                             aux_loss = compute_aux_loss(
                                                                               pred, privileged label,  ...)
                                                                             (risk MSE + min-dist + ttc + hazard)
```

### Gradient / optimizer 규칙

```
 z   = E_ψ(state)                # graph 유지 (critic·aux가 encoder를 shaping)
 z_a = z.detach()                # actor/temperature는 encoder로 grad 안 보냄

 critic_optimizer.step( (critic_loss + β·aux_loss + β_risk·risk_loss + β_cf·cf_loss).backward() )
     └─ 파라미터 그룹 = Critic ∪ Encoder(E_ψ) ∪ AuxHead ∪ ActionRiskHead ∪ CFHead
 actor_optimizer.step( (actor_loss + λ_cf·predicted_cf_risk).backward() ) # z_a 위에서만
 ent_coef_optimizer.step( ... )                        # 온도 α auto
 target: Critic·Encoder는 Polyak(τ)로 target 네트워크 동기화
```

- **β (aux 가중치)** = `_current_aux_beta()`:
  - **A2** `stagewise_loss_schedule`가 있으면 `β = schedule[stage]` (stage별 encoder shaping 강도).
  - 없으면 `aux_beta_warmup_steps` 동안 `0 → loss_weight` 선형 ramp, 이후 상수 `loss_weight`.
- aux label은 환경이 붙인 **privileged** 정보(보행자 CV rollout 기반 future risk)이며 학습에만
  쓰이고 **정책 배포 경로(actor + encoder)에는 들어가지 않는다** (DiPCAN식 training-only branch).

---

## 3. 커리큘럼 stage에 따라 바뀌는 것 (텐서 shape 불변)

```
 set_curriculum_stage(stage)  ─┬─►  TemporalFusionEncoder.temporal_gain = (stage≥2 ? 1 : 0)
                               │        (초기 stage: 현재 상태만; temporal encoder는 grad 미수신)
                               └─►  _current_aux_beta() 가 참조하는 current_stage 갱신
                                        (A2 schedule 사용 시 stage별 β)
```

stage가 바뀌어도 **네트워크 텐서 모양은 그대로**이고, gain/β 같은 스칼라만 켜지고 램프된다.
`reset_buffer_on_promote_to`가 설정된 프로필은 action 계약이 바뀌는 stage(현재 CF/ST 프로필은
Stage 5의 yield 해제)에서 replay를 비우고 재워밍업한다. 이때 CF head 파라미터는 보존하되
supervised-update counter와 actor-penalty warm-up/ramp를 0부터 다시 시작한다.

---

## 4. 텐서 차원 요약 (shipped config 기준)

| 구성요소 | 입력 → 출력 | 비고 |
|---|---|---|
| stacked state | 327-D | 현재 87 + scan history 80×3 (obs stacking N=4) |
| current frame | 87-D | obs 80(front 180° LiDAR) + agent 7 |
| SharedEncoder | 87 → 256 → 128 | ELU, latent=128 |
| ScanTemporalEncoder | 80×4 → 32 | 기본 conv1d; ST ON이면 시간×각도 Conv2d + ordered angular tokens + 선택 range-rate, stage2부터 gain=1 |
| Fusion | 128+32 → 128 | actor/critic 입력 폭 고정 |
| Actor | 128 → 256³ → (mean,log_std)×3 | tanh Gaussian, a∈[-1,1]³ = (r, θ, yield) |
| Critic (TQC) | (128+3) → 256³ → 25 | n_critics=5, target 상위 2/critic 절삭 → 115 target quantile |
| Aux action GRU | embed 3→64, GRU→256, attn 4head | K=4 future action, 경계 마스크 |
| Aux trunk | (128+256) → 256 ×2 | Linear→LN→ELU |
| Aux heads | risk 48 / min_dist 3 / ttc 1 / hazard 3 | label_dim = 55 |

action 3축: `a[0]=r`(waypoint 거리), `a[1]=θ`(waypoint 각), `a[2]=yield`(정지/양보 스칼라).
정책 출력(정규화 [-1,1])은 Pure Pursuit(`hybrid_action_to_command`)로 물리 `cmd_vel`로 변환된다.

---

## 5. ST/CF 선택 기능 (`phase2/both_trajrisk_rbs_cf_st`)

- **ST (`spatiotemporal_lidar`)**: 4-frame LiDAR의 시간과 scan 각도를 별도 축으로 유지하는
  Conv2d encoder다. `angular_tokens`로 좌우 순서를 보존하고 `use_range_rate=true`면 frame 차분을
  접근/이탈 단서로 추가한다. 기존 conv1d encoder와 독립 토글이며 활성화 시 fresh run이 필요하다.
- **CF (`counterfactual_multi_horizon_risk`)**: 환경의 `risk_targets.py`가 동일한 pre-step human
  snapshot에서 7개 고정 후보와 실제 실행 action을 0.5/1.0/1.5/2.0초 horizon으로 rollout한다.
  replay에 저장된 target으로 CF head를 지도학습하고, 충분히 학습된 뒤 예측 위험을 actor loss에
  직접 더한다. 기본 프로필 값은 5,000 supervised update warm-up, 10,000 update 선형 ramp,
  `weighted_mean` horizon 가중치 `[0.4, 0.3, 0.2, 0.1]`이다.
- 두 기능은 독립적으로 on/off 가능하다. 둘 다 켠 `both_trajrisk_rbs_cf_st`는 네트워크와 replay
  schema가 바뀌므로 기존 checkpoint/replay를 재사용하지 않는 fresh-run 전용이다.

## 6. 실험 플래그(A1~A4)가 이 구조에서 바꾸는 지점

| 실험 | 다이어그램상 위치 | 변경 | 기본값(baseline) |
|---|---|---|---|
| **A1** | 학습 루프(모델 밖) | `updates_per_env_step` 만큼 `train()` 반복 + batch 512 | 1, batch 256 |
| **A2** | β (aux loss weight) | `stagewise_loss_schedule`로 stage별 β | `[]`(warmup ramp) |
| **A3** | Critic body | residual MLP + LayerNorm, hidden 384 | plain MLP 256 |
| **A4** | Aux fuse(z_t, ctx) | `concat` → `film`(identity-init) | concat |

- A3/A4는 critic/aux state_dict를 바꾸므로 **fresh run 전용**이며, trainer가 `load_model` /
  `resume_weight_prefix`를 거부한다(단, A4 guard는 aux가 실제 enabled일 때만).
- A4 FiLM은 trunk/head를 먼저 만들고 generator를 마지막에 zero-init하므로, 같은 seed에서
  concat과 **초기 trunk가 동일**한 순수 fusion ablation이다.

자세한 실험 실행 명령은 `docs/experiments/tqc_scaling_improvement_plan.md` 참고.

---

## 7. 관련 문서

- `docs/design/aux_prediction_design.md` — auxiliary prediction 상세 설계
- `docs/design/curriculum_design.md` — 10-stage 커리큘럼 승급 로직
- `docs/design/environment_design.md` — 상태/행동/보상 및 환경 인터페이스
- `docs/experiments/tqc_scaling_improvement_plan.md` — A1~A4 scaling 실험
