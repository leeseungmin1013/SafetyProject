# SafetyProject

**MomentGuard Full-Parameter Safety Pilot**

Qwen2.5-7B, Llama-3.1-8B, Gemma-3-12B에서 **초기 체크포인트와 full-parameter AdamW 미세조정**을 비교해 안전성, 유용성, 과잉 거부, 시간, VRAM을 계측하는 공동 실험 저장소다.

## 이 저장소가 답하려는 질문

1. 일반적인 양성 instruction tuning만으로 각 모델의 jailbreak ASR이 얼마나 변하는가?
2. 작업 성능 개선과 기존 일반 능력 손실은 어느 정도인가?
3. 안전성 변화가 실제 유해 응답 감소인지, 모든 요청을 거절하는 과잉 거부인지 구분되는가?
4. 7B·8B·12B full-parameter AdamW 실행에 필요한 시간, GPU-hours, peak VRAM은 얼마인가?
5. 후속 MomentGuard 실험을 어떤 모델·학습률·예산으로 확장해야 하는가?

## 범위

포함:

- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-3-12b-it`
- 학습 전 Initial 평가
- BF16 full-parameter SFT + AdamW
- SAMSum, AG News, GSM8K, OpenBookQA, IFEval
- JailbreakBench, WildJailbreak, StrongREJECT
- XSTest 기반 과잉 거부 평가
- wall time, GPU-hours, throughput, peak VRAM 계측

제외:

- LoRA, QLoRA, SafeLoRA, SaLoRA
- 양자화 학습
- Qwen3-30B MoE
- 파일럿 결과만으로 최종 우열 또는 통계적 유의성을 주장하는 것
- 유해 프롬프트와 모델 원문 응답을 공개 Git 이력에 저장하는 것

GuardSpace와 SPA는 full-parameter 공식 구현 및 공정한 계산 예산을 확인한 뒤 후속 비교에 포함할지 결정한다. 파일럿의 필수 조건은 아니다.

## 시작점

- 전체 설계: [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md)
- 평가 규약: [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md)
- 실행 절차: [`docs/RUNBOOK.md`](docs/RUNBOOK.md)
- 팀 협업: [`docs/TEAM_ONBOARDING.md`](docs/TEAM_ONBOARDING.md)
- 결정 기록: [`docs/DECISIONS.md`](docs/DECISIONS.md)
- 공통 설정 명세: [`configs/pilot.yaml`](configs/pilot.yaml)
- 실행 행렬: [`experiments/pilot_matrix.csv`](experiments/pilot_matrix.csv)

## 파일럿 완료 조건

- 세 모델 모두 Initial 평가 완료
- 세 모델 모두 full AdamW 1회 이상 완료 또는 재현 가능한 OOM/실행 불가 근거 확보
- 선택 모델에서 독립 seed 2개 완료
- 모든 run에 config, Git SHA, 데이터 manifest hash, 모델 revision, GPU 정보 기록
- ASR, 과잉 거부, task metric, task CE 변화, 시간, GPU-hours, peak VRAM 산출
- judge 불일치와 무작위 사례를 합쳐 최소 100건의 blinded audit 계획 확정
- 후속 실험의 모델, 학습률, GPU 예산에 대한 go/no-go 결정 기록

## 저장소 운영 원칙

- `main` 직접 push 금지, pull request와 최소 1인 review 사용
- 실험 조건 변경은 먼저 `docs/DECISIONS.md`에 기록
- 최종 test 결과는 학습률과 방법 선택이 끝날 때까지 봉인
- 원시 유해 데이터와 원문 응답은 접근 제한 artifact storage에 저장
- 결과가 좋지 않거나 가설과 반대여도 run 삭제 금지

본 저장소의 설정 파일은 실험 계약의 단일 출처다. 실제 LLaMA-Factory 실행 YAML은 고정한 버전과 GPU 환경을 확인한 뒤 이 명세에서 생성한다.
