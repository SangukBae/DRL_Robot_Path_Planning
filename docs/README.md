# Documentation Hub

DRL 로봇 경로계획 시스템 문서 인덱스. **무엇부터 읽을지**는 아래 "독자별 추천 경로"를 따라가면 된다.

문서는 5가지 역할로 나뉜다: **개요(overview) · 가이드(guides) · 설계(design) · 레퍼런스(reference) · 실험(experiments)**.

## 독자별 추천 경로
- **처음 보는 사람** → [overview/system_overview](overview/system_overview.md) → [overview/training_pipeline](overview/training_pipeline.md) → [overview/repository_map](overview/repository_map.md)
- **학습만 돌리려는 사람** → [guides/installation](guides/installation.md) → [guides/training](guides/training.md) → [guides/evaluation](guides/evaluation.md)
- **내부 구조를 이해하려는 사람** → [overview/training_pipeline](overview/training_pipeline.md) → [design/environment_design](design/environment_design.md) → [design/curriculum_design](design/curriculum_design.md) → [reference/state_action_reference](reference/state_action_reference.md)
- **aux network만 보고 싶은 사람** → [design/aux_prediction_design](design/aux_prediction_design.md) → [reference/metrics_reference](reference/metrics_reference.md) → [experiments/aux_ablation_logging](experiments/aux_ablation_logging.md)
- **localization noise만 보고 싶은 사람** → [design/localization_noise_design](design/localization_noise_design.md) → [guides/real_robot_deployment](guides/real_robot_deployment.md) → [experiments/simulation_validation](experiments/simulation_validation.md)
- **논문 실험을 하려는 사람** → [experiments/experiment_protocol](experiments/experiment_protocol.md) → [experiments/paper_preparation_guide](experiments/paper_preparation_guide.md)

## 강화학습 시스템 이해의 핵심 문서 (이 5개면 큰 그림이 잡힌다)
1. [overview/system_overview.md](overview/system_overview.md) — 시스템 전체가 무엇인지
2. [overview/training_pipeline.md](overview/training_pipeline.md) — reset→state→action→step→train→eval 흐름
3. [design/environment_design.md](design/environment_design.md) — environment.py vs environment_curriculum.py 역할 분리
4. [reference/state_action_reference.md](reference/state_action_reference.md) — state 87D / action 2D 정확한 정의
5. [design/curriculum_design.md](design/curriculum_design.md) + [design/localization_noise_design.md](design/localization_noise_design.md) + [design/aux_prediction_design.md](design/aux_prediction_design.md) — 학습에 끼어드는 3가지 메커니즘

## 전체 목록
### 개요 (overview/)
| 문서 | 내용 |
|--|--|
| [system_overview](overview/system_overview.md) | 시스템이 무엇인지 한 장 요약 |
| [training_pipeline](overview/training_pipeline.md) | 학습 루프 데이터 흐름 |
| [repository_map](overview/repository_map.md) | 패키지/폴더/알고리즘 지도 |

### 가이드 (guides/)
| 문서 | 내용 |
|--|--|
| [installation](guides/installation.md) | 설치·빌드·Docker·RGL LiDAR |
| [training](guides/training.md) | 커리큘럼 학습 실행, 재개, 월드 옵션 |
| [evaluation](guides/evaluation.md) | 평가, 결과 집계, 일반화 평가 |
| [real_robot_deployment](guides/real_robot_deployment.md) | 실로봇 배포 + 위치오차 강건성 |
| [troubleshooting](guides/troubleshooting.md) | 환경 변수, 흔한 오류 |

### 설계 (design/)
| 문서 | 내용 |
|--|--|
| [environment_design](design/environment_design.md) | 환경 노드 동작, 두 파일 역할 분리 |
| [curriculum_design](design/curriculum_design.md) | 7단계 커리큘럼·맵별 활성 개수(`*_by_map`)·진급 규칙 |
| [aux_prediction_design](design/aux_prediction_design.md) | 공유 인코더 + 미래 위험 aux head |
| [localization_noise_design](design/localization_noise_design.md) | 위치추정 오차 모사 모델 |
| [map_curriculum_design](design/map_curriculum_design.md) | 구조화 맵 4종 |

### 레퍼런스 (reference/)
| 문서 | 내용 |
|--|--|
| [state_action_reference](reference/state_action_reference.md) | state/action 차원 표 |
| [config_reference](reference/config_reference.md) | config 파일·파라미터 |
| [metrics_reference](reference/metrics_reference.md) | 학습/평가 지표·CSV 컬럼 |
| [ros_interface_reference](reference/ros_interface_reference.md) | 서비스/토픽/브릿지 |

### 실험 (experiments/)
| 문서 | 내용 |
|--|--|
| [experiment_protocol](experiments/experiment_protocol.md) | 6개 알고리즘 비교 프로토콜 |
| [aux_ablation_logging](experiments/aux_ablation_logging.md) | aux on/off 비교 로깅 |
| [simulation_validation](experiments/simulation_validation.md) | 시뮬레이션 검증 절차 |
| [map_curriculum_plan](experiments/map_curriculum_plan.md) | 구조화 맵 상세 설계 원안 |
| [paper_preparation_guide](experiments/paper_preparation_guide.md) | 논문화 작업 가이드(원 README) |
