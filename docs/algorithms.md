# Algorithms

모든 알고리즘 구현은 `drl_agent/scripts/policy/`에 위치한다.

---

## TQC — Truncated Quantile Critics (주력)

**파일**: `tqc_agent.py`, `train_tqc_curriculum_agent.py`

분포형 RL. 다수의 Critic 네트워크가 각각 분위수(quantile) 분포를 추정하고, 상위 분위수를 잘라내(truncate) 과대추정 편향을 줄인다. 연속 액션 공간에서 SAC보다 안정적으로 수렴하는 경향이 있다.

- 리플레이 버퍼: LAP (Loss-Adjusted Priorities) — 선택적 우선순위 경험 재생
- 커리큘럼 학습과 결합한 `train_tqc_curriculum_agent.py`가 이 프로젝트의 주력 스크립트

---

## TQC + IEQn — IEQn 부등식 제약 변형

**파일**: `tqc_ieqn_agent.py`, `train_tqc_ieqn_agent.py`

TQC에 부등식 제약(Inequality Constraint)을 추가해 안전 제약 조건을 보상 함수 없이 직접 강제하는 변형. 충돌 회피를 제약으로 정식화할 때 유용하다.

---

## TD7 — Twin Delayed DDPG v7

**파일**: `td7_agent.py`, `train_td7_agent.py`

TD3의 개선 버전. 상태-액션 임베딩을 통한 표현 학습과 체크포인트 기반 정책 선택을 도입해 과대추정 및 과적합을 완화한다.

- 테스트 스크립트: `test_td7.launch.py`

---

## SAC — Soft Actor-Critic

**파일**: `sac_agent.py`, `train_sac_agent.py`

최대 엔트로피 RL. 보상 극대화와 정책 엔트로피 극대화를 동시에 수행해 탐험과 활용의 균형을 자동으로 조절한다. 온도 파라미터 α를 자동 조정한다.

---

## A3C — Asynchronous Advantage Actor-Critic

**파일**: `a3c_agent.py`, `train_a3c_agent.py`

Policy Gradient 계열. 여러 워커가 비동기적으로 그레이디언트를 누적해 글로벌 네트워크를 업데이트한다. Off-policy 알고리즘(TQC, TD7, SAC) 대비 샘플 효율이 낮으나 구현이 단순하다.

---

## 알고리즘 비교

| 알고리즘 | 계열 | 리플레이 버퍼 | 특징 |
|---------|------|------------|------|
| TQC | Off-policy Actor-Critic | ✅ LAP | 분위수 추정, 과대추정 완화 |
| TQC + IEQn | Off-policy Actor-Critic | ✅ LAP | TQC + 안전 제약 |
| TD7 | Off-policy Actor-Critic | ✅ | 표현 학습, 체크포인트 선택 |
| SAC | Off-policy Actor-Critic | ✅ | 최대 엔트로피, 자동 온도 조정 |
| A3C | On-policy Policy Gradient | ❌ | 비동기 업데이트, 낮은 샘플 효율 |

---

## 유틸리티

`drl_agent/scripts/utils/`:

| 파일 | 설명 |
|------|------|
| `buffer.py` | LAP 리플레이 버퍼 |
| `file_manager.py` | 모델 체크포인트, YAML 로드 |
| `plot_metrics.py` | 학습 지표 시각화 |
| `plot_reward.py` | 보상 곡선 시각화 |
| `plot_trajectories_on_map.py` | 궤적 분석 |
