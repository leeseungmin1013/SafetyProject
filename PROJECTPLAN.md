# MomentGuard 통합 프로젝트 계획

**문서 상태:** 실행 전 통합 계획서

**기준일:** 2026-11-10

**적용 범위:** 영어 단일 턴, instruction-tuned 7B-12B 모델, BF16 full-parameter fine-tuning

**핵심 비교:** Initial / AdamW / SPF / cache-SPF / One-shot / MomentGuard

**원본 연구 계획:** [leeseungmin1013/MomentGuard](https://github.com/leeseungmin1013/MomentGuard), commit `835669b85013c3a2b188f55d6799e749d57c6d52`

**입력 자료:** `MomentGuard_연구_실험계획서.md`, SPF 중심 수정본, `MomentGuard_SPF_수식개선_실험계획.pdf`

## 0. 문서의 역할과 우선순위

이 문서는 SafetyProject의 전체 연구 범위, 단계별 실험, 진입·중단 기준, 실행량, 산출물과 열린 결정을 한곳에서 관리하는 상위 계획서다.

문서 간 우선순위는 다음과 같다.

1. 확정된 결정은 `docs/DECISIONS.md`가 우선한다.
2. 전체 범위와 단계 간 관계는 이 문서를 따른다.
3. 평가의 세부 분모와 오류 처리는 `docs/EVALUATION_PROTOCOL.md`를 따른다.
4. 실행·재시도·중단 절차는 `docs/RUNBOOK.md`를 따른다.
5. 파일럿의 현재 설정값은 `configs/pilot.yaml`을 따른다.
6. 설정 파일과 이 문서가 충돌하면 장시간 실행을 시작하지 않고 결정 기록과 설정을 먼저 동기화한다.

현재 `docs/EXPERIMENT_PLAN.md`는 Initial 대 full AdamW 실행 파일럿의 상세 계획으로 유지한다. 이 문서는 해당 파일럿을 0단계로 포함하고, 그 결과를 바탕으로 SPF 구조, 방향 노후화, AdamW 모멘트 원인, MomentGuard와 최종 비교까지 확장한다.

## 1. 프로젝트 목적

일반적인 양성 instruction tuning에서도 모델의 안전성 정렬이 약해질 수 있다. SPF는 작업 그래디언트와 충돌하는 저랭크 안전성 방향을 제거하지만, 실제 AdamW 업데이트에는 과거 1차 모멘트와 2차 모멘트의 좌표별 스케일이 반영된다. 따라서 SPF 직후의 그래디언트가 안전해 보여도 실제 파라미터 이동은 안전성 대리 손실을 증가시킬 수 있다.

이 프로젝트는 다음을 순서대로 검증한다.

1. 현재 모델·데이터·full-parameter 조건에서 일반 AdamW가 안전성, 유용성, 과잉 거부를 실제로 변화시키는가?
2. 안전성 그래디언트가 저랭크 구조를 가지며 단일 또는 소수 참조 샘플로 재현되는가?
3. 안전성 방향을 재사용하면 언제 판정 오류가 생기고, 어느 갱신 주기까지 비용을 절약할 수 있는가?
4. 같은 모델·배치·2차 모멘트에서 과거 1차 모멘트만 바꾸면 실제 업데이트 방향과 독립 안전성 손실이 달라지는가?
5. 현재 업데이트를 보정하는 것과 보정된 1차 모멘트를 다음 스텝에 저장하는 것은 각각 어떤 효과를 가지는가?
6. MomentGuard가 One-shot, SPF, AdamW보다 안전성·유용성·과잉 거부·비용의 Pareto 관점에서 실용적인가?

연구의 핵심 결과는 다음 두 가지다.

- 동일 체크포인트에서 AdamW 모멘트 이력이 만드는 방향 역전과 실제 Audit CE 변화
- 첫 업데이트를 같게 맞춘 One-shot과 Persistent/MomentGuard의 후속 학습 경로 및 최종 ASR 차이

## 2. 범위

### 2.1 포함

- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-3-12b-it`
- 모든 언어 모델 파라미터를 학습하는 full-parameter BF16 SFT
- AdamW의 1차·2차 모멘트와 실제 적용 델타 계측
- SPF의 블록별 truncated SVD와 그래디언트 투영
- Dynamic, Periodic, Static safety-direction cache
- One-shot update correction과 Persistent state correction
- 안전성, 유용성, 과잉 거부, 실행 시간, GPU-hours, VRAM, 저장 비용 평가
- 동일 상태 반사실적 진단과 실제 전체 학습 경로 비교
- 실패·음성 결과·capacity failure 보존

### 2.2 제외

- LoRA, QLoRA, SafeLoRA, SaLoRA
- 양자화 학습
- Qwen3-30B MoE
- ASCENT 직접 재현 및 최종 비교군 편입
- 유해 프롬프트와 모델 원문 응답의 공개 Git 저장
- 파일럿 subset만으로 최종 모델 우열 또는 모집단 수준의 통계적 유의성 주장
- Final test를 보고 rank, K, 학습률, 보정 강도를 재선택하는 행위

ASCENT는 방향 노후화와 주기적 재추정의 관련 연구로만 사용한다. GuardSpace와 SPA는 full-parameter 공식 구현과 공정한 계산 예산이 확인된 뒤 별도 확장 여부를 결정한다.

## 3. 위협 모델과 해석 범위

- 입력 데이터는 양성 instruction tuning 데이터다. 유해 답변을 학습시키는 공격형 fine-tuning은 핵심 범위가 아니다.
- 안전성 참조 데이터는 유해 요청과 안전한 거부 또는 안전한 대안 응답의 쌍이다.
- 안전성 CE/NLL은 개입과 메커니즘 진단을 위한 대리 지표다.
- 행동 안전성은 학습과 분리된 유해 요청에 대한 실제 생성 응답의 ASR로 평가한다.
- 낮은 ASR이 무차별 거절 때문인지 확인하기 위해 XSTest 과잉 거부를 필수로 평가한다.
- 단일 스텝 진단은 해당 체크포인트의 국소적 인과 효과로 해석한다.
- 전체 학습 결과는 서로 다른 학습 경로의 결과이며 단일 스텝 원인 분석과 구분한다.

## 4. 연구 가설

| ID | 가설 | 조작 | 필요한 증거 |
| --- | --- | --- | --- |
| H0 | 양성 full AdamW SFT가 일부 모델의 기존 안전성을 변화시킨다. | Initial 대 AdamW | 동일 요청의 paired ASR 변화와 Audit/Task 변화 |
| H1 | 안전성 그래디언트는 작업 그래디언트보다 낮은 유효 rank를 가진다. | 안전성/작업 gradient의 rank 1·4·8 분석 | CER 분포와 독립 행동 평가 |
| H2 | 단일 안전성 샘플의 주요 부분공간은 별도 안전성 배치와 겹친다. | 단일 후보 5개 대 배치 16개 | 부분공간 겹침, 충돌 판정 일치율, 샘플 편차 |
| H3 | 오래된 safety cache는 최신 방향과 다른 충돌 판정을 만들 수 있다. | 같은 `theta`, `g_u`, optimizer state에 cache/fresh 적용 | false-safe 비율과 Audit CE 증가 |
| H4 | 과거 1차 모멘트는 SPF 이후에도 실제 업데이트의 안전성 방향을 역전시킬 수 있다. | `m_prev` 유지 대 제거 | 실제 델타 부호 역전과 Audit CE 증가 |
| H5 | 2차 모멘트의 좌표별 스케일도 잔여 충돌에 기여할 수 있다. | 항등·fresh·reset·누적 `v` 스케일 비교 | 방향 및 노름을 통제한 Audit CE 차이 |
| H6 | 현재 업데이트 보정과 보정된 상태 저장은 분리 가능한 효과를 가진다. | 첫 이동이 같은 One-shot 대 Persistent | 첫 스텝 동일성, 이후 모멘트·충돌·CE·ASR 분기 |
| H7 | MomentGuard는 같은 유틸리티와 과잉 거부 범위에서 One-shot보다 ASR을 낮출 수 있다. | 같은 seed와 데이터 순서의 paired 전체 학습 | 최종 ASR, Task Test, XSTest, 비용 |

## 5. 모델 전략

### 5.1 모델과 역할

| ID | Repository | 역할 | 확장 규칙 |
| --- | --- | --- | --- |
| `qwen25-7b` | `Qwen/Qwen2.5-7B-Instruct` | anchor, LR 탐색, A-D 전체 메커니즘 실험 | 모든 단계의 첫 구현과 진단을 수행 |
| `llama31-8b` | `meta-llama/Llama-3.1-8B-Instruct` | 아키텍처 전이 확인 | anchor에서 동결한 핵심 방법만 실행 |
| `gemma3-12b` | `google/gemma-3-12b-it` | 대형·상이한 아키텍처 전이 확인 | anchor에서 동결한 핵심 방법만 실행 |

Qwen에서 모든 rank, K, 대조군의 Cartesian product를 실행한 뒤 다른 모델에서 반복하지 않는다. A-D에서 선택한 rank, cache 주기, 보정식과 허용오차를 Final test를 열기 전에 동결한다. Llama와 Gemma는 Initial, AdamW, Dynamic SPF, 선택 cache-SPF, One-shot, MomentGuard의 핵심 비교만 수행한다.

### 5.2 고정 조건

- 각 모델의 immutable model revision과 tokenizer revision을 manifest에 기록한다.
- 모델별 공식 chat template을 사용한다.
- 의미가 같은 system policy를 사용하되 직렬화 결과와 template hash를 기록한다.
- Initial과 학습 후 모델의 decoding 설정, prompt ID, stop condition을 일치시킨다.
- Gemma는 text-only 입력만 사용하고 vision component가 메모리에 포함되는지 별도 기록한다.
- 모델별 전체 LM 파라미터 수와 실제 trainable parameter 수가 일치하는지 preflight에서 검증한다.

## 6. 데이터 설계

### 6.1 작업 데이터

| Split | 규모 | 출처 | 용도 |
| --- | ---: | --- | --- |
| Task Train | 2,000 | `databricks/databricks-dolly-15k` | full-parameter 학습 |
| Task Dev | 250 | 동일 revision | LR, checkpoint, rank, K, 보정 강도 선택 |
| Task Test | 250 | 동일 revision | 선택 완료 후 봉인된 task CE 및 생성 평가 |

- 카테고리 비율을 유지한다.
- 동일·유사 instruction과 공유 지문을 그룹으로 묶어 한 split에만 배치한다.
- answer가 max length 때문에 잘리는 예시는 split 전에 제외한다.
- prompt token과 answer token 수를 각각 기록한다.
- Task Test는 모든 방법 선택이 끝날 때까지 접근 권한을 제한한다.

### 6.2 안전성 참조·개발·감사 데이터

| Split | 목표 규모 | 출처와 조건 | 용도 |
| --- | ---: | --- | --- |
| Safety Reference | 256 | 고정 revision의 `PKU-Alignment/PKU-SafeRLHF`에서 안전 라벨 응답을 검수 | 개입용 gradient와 cache 구성 |
| Single Reference Candidates | 5 | Reference에서 위험 범주가 겹치지 않게 선택 | 단일 샘플 대표성 A2 |
| Batch Reference | 16 | 위 5개 및 평가 데이터와 분리 | 단일 샘플 대 배치 기저 비교 |
| Safety Pilot | 64 | Reference와 다른 prompt group | 수치 허용오차, 개발용 생성 평가 |
| Safety Audit | 64 | Reference/Pilot과 다른 prompt group | 독립 CE, 방향, 실제 손실 검증 |

모든 안전성 응답은 적절한 거부 또는 안전한 대안을 제공하는지 사람이 검수한다. 같은 원 prompt에는 응답 하나만 남긴다. 위험 범주, 의도, 응답 유형, 검수 상태를 manifest에 포함한다.

Reference는 개입 데이터이며 Final 행동 평가에 포함하지 않는다. Pilot은 설정 개발에 사용할 수 있지만 Audit은 확인 실험의 독립 패널로 유지한다.

### 6.3 행동 안전성과 과잉 거부 평가

| 데이터 | 파일럿/최종 용도 |
| --- | --- |
| HarmBench DirectRequest | SPF 원 계획과의 연결 및 고정 행동 기준 ASR |
| JailbreakBench | 표준 직접 공격 비교 |
| WildJailbreak | 위험 범주 층화 500개 파일럿, 최종 확장 |
| StrongREJECT | graded harmfulness와 사전 고정 threshold ASR |
| XSTest | benign 250개 과잉 거부, 위험 contrast 별도 보고 |

공격 축은 direct, PAP, AutoDAN-Turbo를 파일럿에 사용한다. TAP과 RAINBOW는 최종 확장에만 사용한다. 모든 behavior set과 모든 attack의 Cartesian product는 실행하지 않는다. 구체적인 조합은 평가 manifest에서 Final 출력 생성 전에 동결한다.

### 6.4 외부 유용성 평가

- SAMSum: ROUGE-L, BERTScore
- AG News: Accuracy, Macro-F1
- GSM8K: Exact Match
- OpenBookQA: Accuracy
- IFEval: strict/loose instruction accuracy

GSM8K는 이 프로젝트의 주 학습 데이터가 아니라 외부 유용성 평가다. 모델 간 절대 CE는 비교하지 않고 각 모델 내부의 `CE_after - CE_initial`만 해석한다.

### 6.5 중복과 누수 방지

1. 정규화 exact match를 수행한다.
2. character n-gram 후보를 생성한다.
3. 고정 embedding model로 의미 유사 후보를 검색한다.
4. 후보를 사람이 판정한다.
5. 겹치면 학습·개발·개입 측 예시를 제거하고 Final 평가는 변경하지 않는다.

검사 대상은 Dolly, PKU-SafeRLHF 파생 split, HarmBench, JailbreakBench, WildJailbreak, StrongREJECT, XSTest와 모든 외부 유용성 데이터다. manifest에는 dataset revision, row ID, group ID, 정규화 hash, 제외 이유, split seed와 최종 hash를 기록한다.

## 7. 공통 학습 설정

| 항목 | 값 |
| --- | --- |
| finetuning type | full parameter |
| precision | BF16 |
| quantization | 없음 |
| optimizer | AdamW |
| primary `beta1` | 0.9 |
| `beta2` | 0.999 |
| epsilon | `1e-8` |
| weight decay | 0.0 |
| max sequence length | 1,024 |
| packing | 사용하지 않음 |
| global batch | 16 examples |
| train budget | Task Train 2,000개 1 epoch, 125 optimizer steps |
| scheduler | 5 optimizer step linear warm-up 후 constant |
| gradient clipping | 기본 비활성 |
| activation checkpointing | 활성 |
| checkpoint | 0, 25, 50, 75, 100, 125 |
| diagnostic cadence | 실행 중 매 5 optimizer step, full checkpoint 저장 없이 스칼라·필요 통계만 기록 |
| sharding | ZeRO-3 또는 FSDP full sharding 중 P0에서 하나로 고정 |

`beta1=0.9`는 모든 주 비교의 공통 AdamW 기준이다. SPF 논문의 `beta1=0.5`는 Qwen anchor에서 민감도/재현 진단으로만 추가하고 주 비교와 섞지 않는다. `beta1=0`은 모멘트 이력 제거 대조군이다.

microbatch는 GPU 용량에 맞게 줄일 수 있지만 gradient accumulation으로 global batch 16을 유지한다. 정밀도, 길이, trainable parameter 범위, optimizer step 수, 데이터 수를 특정 군에만 바꾸지 않는다.

### 7.1 학습률 선택

Qwen anchor에서 seed 42로 다음을 실행한다.

- `5e-6`
- `1e-5`
- `2e-5`

Task Dev answer-token CE가 유한하고 가장 낮은 값을 공통 LR로 선택한다. NaN/Inf가 없고, checkpoint 저장·재개가 성공하며, Task Dev CE가 Initial보다 악화되지 않아야 한다. 동률이면 작은 LR을 선택한다. 안전성 Final 결과를 LR 선택에 사용하지 않는다.

선택 LR은 세 모델의 주 비교에 공통 적용한다. 모델별 LR 최적화는 전체 핵심 비교가 끝난 뒤 별도 확장으로만 허용한다.

### 7.2 seed

- LR 파일럿: 42
- 실행 가능성: 11
- 확인 실험: 11, 23, 37
- 데이터 split: 2026

같은 seed의 비교군은 초기 모델, 작업 배치 순서, 안전성 참조 순서, 평가 ID와 가능한 RNG 상태를 공유한다.

## 8. 수학적 정의

### 8.1 손실과 그래디언트

- `L_u(theta)`: Dolly assistant answer token의 평균 작업 CE
- `g_u = grad L_u(theta)`: accumulation이 끝난 실제 작업 그래디언트
- `L_s(theta)`: 유해 요청에 대한 안전한 응답 token의 평균 안전성 CE
- `s_cache`: 알고리즘이 실제 사용하는 저장된 안전성 gradient 또는 기저
- `s_fresh`: 같은 참조 데이터와 현재 `theta`에서 다시 계산한 최신 gradient
- `s_audit`: 학습·개입과 분리된 Safety Audit 패널의 gradient

모든 내적과 노름은 FP64 누산을 사용한다. 학습과 gradient 자체의 저장 정밀도는 구현 설정에 따르되 manifest에 기록한다.

### 8.2 AdamW

```text
m_t = beta1 * m_prev + (1 - beta1) * x_t
v_t = beta2 * v_prev + (1 - beta2) * x_t^2
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
D_t = diag(1 / (sqrt(v_hat) + epsilon))
delta_theta = -eta_t * D_t * m_hat
R(s, delta_theta) = s dot delta_theta
delta_L = L(theta + delta_theta) - L(theta)
```

`R > 0`이면 1차 근사에서 안전성 CE가 증가한다. 실제 판정은 내적뿐 아니라 `delta_L_audit`를 함께 사용한다.

### 8.3 SPF

전역 충돌은 다음으로 판정한다.

```text
s_fresh dot g_u < -tau(s_fresh, g_u)
```

충돌이면 보정 대상 2차원 가중치 행렬별로 안전성 gradient `G_s`의 rank-k truncated SVD를 계산한다.

```text
G_s ~= U_k Sigma_k V_k^T
G_u_prime = G_u - U_k (U_k^T G_u)
```

초기 보정 대상은 attention과 MLP의 2차원 weight matrix다. embedding, norm, bias, lm_head 등 미보정 파라미터도 전역 충돌 내적과 실제 델타 계측에는 포함한다. 모델별 parameter-name allowlist와 제외 목록을 manifest로 남긴다.

### 8.4 허용오차와 분류

```text
tau(s, u) = max(1e-12, 1e-6 * ||s||_2 * ||u||_2)
```

- `R > tau`: 증가/충돌
- `R < -tau`: 감소/안전 방향
- 그 사이: 중립

직접 CE의 중립 범위 `delta_CE`는 no-op 반복 평가에서 다음으로 고정한다.

```text
delta_CE = max(1e-5 nats/token, 5 * max_absolute_noop_difference)
```

0 norm, 비유한 gradient, 비유한 update는 방향 분모에서 제외하고 별도 실패 유형으로 보고한다.

### 8.5 MomentGuard 후보식

현재 safety direction `s`와 Adam preconditioner `D`를 고정한다. `p = D s`이면 실제 이동의 안전성 조건은 `p dot z >= 0`이다. 기존 bias-corrected 1차 모멘트와 가장 가까운 반공간 투영 후보를 다음과 같이 사전 등록한다.

```text
p = D * s
if p dot m_hat < -tau(p, m_hat) and ||p||_2 > 1e-12:
    alpha = -(p dot m_hat) / ||p||_2^2
    z = m_hat + alpha * p
else:
    z = m_hat

delta_theta_corrected = -eta_t * D * z
m_raw_corrected = (1 - beta1^t) * z
```

이 식은 실험 C가 1차 모멘트 원인 근거를 통과했을 때 주 구현으로 활성화한다. 보정 후 실제 적용 델타에서 `R(s_cache, delta_theta) <= tau`인지 재검사한다. 실패하면 해당 step을 residual-conflict로 기록하며 조용히 반복 보정하지 않는다.

## 9. 전체 실행 흐름과 단계 게이트

```text
P0 환경·데이터 고정
  -> P1 Initial/AdamW 실행 파일럿
  -> A 안전성 저랭크 구조·대표성·잔여 충돌
  -> B 방향 cache 노후화
  -> C AdamW 모멘트 원인 분리
  -> D One-shot/MomentGuard 상태 저장 검증
  -> E 동결된 핵심 방법의 최종 비교
```

각 단계는 이전 단계에서 기대한 현상이 없더라도 결과를 삭제하지 않는다. 게이트를 통과하지 못하면 다음 알고리즘 개발을 강행하지 않고 원인 분석, 범위 축소 또는 음성 결과 보고로 전환한다.

## 10. P0-P1: 실행 파일럿

### 10.1 P0 완료 조건

- 세 모델 immutable revision과 라이선스 확인
- LLaMA-Factory 또는 선택 학습 코드 commit 고정
- Dolly, PKU-SafeRLHF, 모든 평가 데이터 revision 고정
- split manifest와 leakage report 생성
- CPU 소형 텐서에서 수식 AdamW와 PyTorch AdamW 일치 검증
- 10-step 연속 실행 대 저장·재개 결과 허용오차 내 일치
- run manifest schema 검증
- private artifact storage 읽기·쓰기와 접근자 확인
- 예상 GPU-hours와 예산 승인

### 10.2 P1 실행

1. 세 모델 Initial 평가
2. Qwen LR sweep
3. 선택 LR, seed 11의 세 모델 full AdamW 125-step 실행
4. 기준에 따라 seed 23·37 확인 실행

Initial과 AdamW에서 Task Dev, utility pilot, safety pilot, XSTest, 시간, throughput과 VRAM을 측정한다. Task Test와 Final behavior 결과는 선택 완료 전 열지 않는다.

### 10.3 P1 게이트

다음 중 하나 이상이면 A-D 메커니즘 실험의 필요 신호로 본다.

- Initial 대비 ASR 절대 변화 5%p 이상
- Initial 대비 ASR 상대 변화 20% 이상
- 외부 utility 평균 변화 3% 이상
- XSTest benign refusal 변화 3%p 이상
- step-level safety proxy와 실제 업데이트 사이의 반복 가능한 충돌 신호

ASR 변화가 작아도 step-level 충돌이 있으면 메커니즘 연구는 진행할 수 있다. 세 모델 모두 행동·대리 지표 변화가 없고 충돌도 희소하면 A의 구조 진단까지만 수행한 뒤 범위를 재검토한다.

## 11. 실험 A: 안전성 구조와 잔여 충돌

### 11.1 A1 저랭크 구조

- Qwen의 Initial, step 50, step 125에서 측정한다.
- attention/MLP weight matrix별 `k in {1,4,8}`의 CER을 기록한다.
- 안전성 gradient와 작업 gradient의 CER 분포를 같은 block에서 paired 비교한다.
- 층, block type, parameter shape별 median, IQR, bootstrap interval을 보고한다.

```text
CER(G, k) = sum_{i=1..k}(sigma_i^2) / ||G||_F^2
```

rank 후보는 Dev에서 다음 순서로 고른다.

1. eligible block의 median safety CER이 0.80 이상인 가장 작은 k
2. 없으면 k=8을 진단 상한으로 사용하되 저랭크 충분성 미확인으로 표시
3. utility CER 대비 safety CER 우위와 독립 Audit 효과가 없으면 구조 근거 미충족

### 11.2 A2 단일 샘플 대표성

- Single Reference 5개를 각각 같은 `theta`, 같은 k에서 계산한다.
- 별도 Batch Reference 16개의 평균 gradient 기저와 비교한다.
- Frobenius subspace overlap, 충돌 판정 일치율, sample별 편차를 기록한다.

```text
overlap(U_single, U_batch)
  = ||U_single^T U_batch||_F^2 / min(rank(U_single), rank(U_batch))
```

median overlap 0.80 이상과 충돌 판정 일치율 80% 이상을 대표성의 유망 신호로 사용한다. 미달하면 단일 샘플을 주 알고리즘 입력으로 사용하지 않고 Reference batch 8을 기본값으로 한다.

### 11.3 A3 구조의 행동 유효성

- 같은 rank의 안전성 기저와 Haar/random orthonormal 기저를 비교한다.
- 보정 전, 안전성 기저 보정, 무작위 기저 보정의 delta norm을 맞춘다.
- Safety Audit CE, Task Dev CE, Safety Pilot 생성 ASR을 평가한다.
- 안전성 기저의 이득이 2개 이상 seed에서 무작위 기저보다 일관되어야 구조의 행동 유효성 근거로 인정한다.

### 11.4 A4 잔여 충돌 위치

각 진단 step에서 다음을 순서대로 기록한다.

1. 원래 `g_u`
2. SPF 후 `g_u_prime`
3. AdamW `m_hat`
4. preconditioned update `D m_hat`
5. 실제 적용 `delta_theta`

각 단계에서 `s_fresh` 내적, cosine, norm, Audit CE 변화를 계산한다. 잔여 충돌을 truncated-rank 잔차, 미보정 파라미터, 과거 `m`, 누적 `v`, 수치오차로 분류한다.

### 11.5 A 게이트

- A1 저랭크 집중과 A3 독립 안전성 이점이 모두 있으면 선택 rank로 B-D를 진행한다.
- 구조는 있으나 A3 이점이 없으면 SPF/MomentGuard의 행동 성능 주장을 보류하고 메커니즘 진단만 진행한다.
- 구조가 약하면 Reference 구성, 보정 block, rank 상한을 한 번만 사전 정의된 Dev 절차로 재검토한다.
- 재검토 후에도 약하면 음성 결과를 보존하고 전체 방법 비교를 축소한다.

## 12. 실험 B: 방향 노후화와 cache 주기

### 12.1 비교군

| 이름 | 갱신 주기 | 의미 |
| --- | ---: | --- |
| Dynamic SPF | K=1 | 매 optimizer step 최신 gradient와 기저 |
| Periodic SPF-short | K=8 | 짧은 재사용 |
| Periodic SPF-long | K=32 | 긴 재사용 |
| Static SPF | step 0 한 번 | 전체 학습 동안 초기 방향 재사용 |

갱신 시 safety gradient와 SVD 기저를 함께 저장한다. 모든 군은 매 step 작업 gradient에 보정을 시도한다. K는 microbatch가 아니라 optimizer step 기준이다.

### 12.2 동일 상태 진단

저장된 같은 `theta_t`, 작업 gradient, `m`, `v`, 학습률에서 cache와 fresh 정보만 바꾼다.

- cache/fresh gradient cosine
- block별 subspace overlap
- cache-safe/fresh-conflict false-safe 비율
- cache-conflict/fresh-safe false-positive 비율
- 보정 델타 간 cosine과 norm 차이
- 각 후보의 Safety Audit CE 변화

서로 다른 학습 경로의 파라미터를 동일 상태 진단에 사용하지 않는다.

### 12.3 실제 학습 비교

Qwen anchor에서 K=1, 8, 32, Static을 동일 seed와 데이터 순서로 실행한다. ASR, utility, over-refusal, 안전성 계산 횟수, SVD 횟수, GPU-hours를 비교한다.

재사용 가능 후보는 다음을 모두 만족하는 가장 큰 K다.

- K=1 대비 평균 ASR 악화 2%p 이내
- cache-safe/fresh-conflict 비율 2% 이하 또는 해당 사례의 Audit CE 증가가 `delta_CE` 이하
- Task Dev/Test 허용 범위 충족
- XSTest benign refusal 악화 2%p 이내

어떤 K에서도 오류가 늘지 않으면 안전성 정보가 해당 125-step 범위에서 재사용 가능하다는 결과로 보고한다. 오류가 크면 Dynamic SPF를 D-E의 기준으로 사용한다.

## 13. 실험 C: AdamW 모멘트 원인 분리

### 13.1 진단 지점

- Qwen AdamW/SPF 궤적에서 실행 중 `t=1,5,10,...,125`의 스칼라·후보 통계를 기록한다.
- full optimizer checkpoint는 0,25,50,75,100,125에만 저장한다.
- 대형 full checkpoint 저장 비용 때문에 매 5 step 상태 저장이 필요하면 별도 저장 예산 승인을 받는다.
- 주 반사실적 재실행은 full checkpoint 지점에서 수행하고, 매 5 step 결과는 on-the-fly 진단으로 사용한다.

### 13.2 반사실적 후보

같은 `theta`, 현재 보정 gradient `x`, 학습률, step과 `v`를 고정한다.

```text
M0 = x
M1 = (beta1 * m_prev + (1 - beta1) * x) / (1 - beta1^t)
D0 = I
D1 = diag(1 / (sqrt((beta2 * v_prev + (1 - beta2) * x^2) /
                     (1 - beta2^t)) + epsilon))
D_reset = diag(1 / (sqrt(((1 - beta2) * x^2) /
                          (1 - beta2^t)) + epsilon))
D_fresh = diag(1 / (abs(x) + epsilon))
u_ij = -D_j * M_i
```

주 비교는 다음이다.

- `u_01 -> u_11`: 실제 Adam scale에서 과거 1차 모멘트의 효과
- `u_00 -> u_01`: 현재 gradient에서 2차 모멘트 scale의 효과
- `D_reset -> D1`: 과거 제곱 gradient의 효과
- norm-matched 후보: 단순 업데이트 축소와 방향 변화 분리

### 13.3 실제 손실 검증

각 후보를 동일 사전 상태의 임시 모델 복제본에 한 번 적용한다.

- Safety Audit CE 전후 차이
- 현재 task batch CE 전후 차이
- 고정 Task Dev panel CE 전후 차이
- 자연 norm과 actual AdamW delta norm에 맞춘 조건 모두 평가

역전 사례는 source 후보가 비증가이고 target 후보가 증가인 경우다. 실제 CE 증가 확인은 다음을 모두 만족해야 한다.

```text
delta_L_audit(target) > delta_CE
delta_L_audit(target) - delta_L_audit(source) > delta_CE
```

### 13.4 C 게이트

H4와 H5를 각각 판정한다.

- seed별 적격 진단 지점 10개 이상
- 최소 2개 seed에서 역전률 10% 이상 및 역전 2건 이상
- 합친 역전 사례의 절반 이상에서 실제 Audit CE 증가 확인

H4 통과 시 Persistent state correction을 주 실험으로 진행한다. H5만 통과하면 update-space 또는 preconditioner-aware correction을 우선하고 Persistent는 탐색으로 둔다. 둘 다 미통과면 MomentGuard 전체 학습을 축소하고 원인 분석 결과만 보고한다.

`beta1=0.5` 민감도와 `beta1=0` 학습은 원인 해석의 보조 대조군이며 주 `beta1=0.9` 결과와 별도 표로 보고한다.

## 14. 실험 D: MomentGuard와 상태 저장

### 14.1 실험군

| 군 | 현재 step의 적용 델타 | 다음 step에 저장하는 `m` |
| --- | --- | --- |
| AdamW/SPF | 원래 AdamW 델타 | 원래 `m_t` |
| One-shot | 보정된 `-eta D z` | 원래 `m_t` |
| Persistent/MomentGuard | One-shot과 동일한 보정 델타 | 보정된 raw `m_raw_corrected` |
| Full `m` reset | 현재 gradient 반영 전 `m_prev=0` | reset 후 일반 규칙의 `m_t` |
| Update projection | 실제 델타를 안전 반공간에 투영 | 원래 `m_t` |

One-shot과 Persistent는 첫 업데이트 직후 다음이 같아야 한다.

- 모델 파라미터
- `v`
- optimizer step counter
- scheduler state
- RNG와 다음 batch 위치
- safety cache

`m`만 저장 규칙에 따라 달라야 한다. 첫 step 파라미터의 최대 절대 차이가 구현 허용오차를 넘으면 해당 분기를 무효 처리한다.

### 14.2 단기 분기

- 시작점: step 25, 50, 75, 100
- 길이: 각 10 optimizer step
- seed: 11, 23, 37
- 공통: 같은 이후 task batch, 같은 safety reference, 같은 보정 규칙

첫 보정 후 두 번째 step에서 양쪽 모두 보정을 끈 추가 진단을 한 번 수행해 상태 전파식을 검증한다. 이후 주 분기는 양쪽 모두 동일한 보정 규칙을 사용한다.

기록 항목:

- Audit CE와 Task Dev CE
- `||m||_2`, `||z-m_hat||_2 / max(||m_hat||_2, 1e-12)`
- 실제 update norm과 누적 path length
- 충돌·보정 발동 횟수
- 첫 분기 후 파라미터·gradient·cache 차이

### 14.3 D 게이트

Persistent의 유망 신호는 최소 2개 seed에서 다음을 만족하는 것이다.

- 분기 종료 Audit CE가 One-shot보다 `delta_CE` 이상 낮음
- Task Dev CE 상대 악화가 1% 이내
- 과잉 거부 개발 지표가 2%p 이상 악화되지 않음

미충족이면 E에서 One-shot과 Persistent의 최소 대조는 유지하되, K와 보정 강도의 추가 탐색은 중단한다.

## 15. 실험 E: 최종 전체 학습 비교

### 15.1 Qwen anchor 비교군

| 우선순위 | 군 | 목적 |
| ---: | --- | --- |
| 1 | Initial | 미세조정 전 기준 |
| 1 | AdamW | 일반 full SFT 기준 |
| 1 | Dynamic SPF | 최신 safety gradient 기준 |
| 1 | 선택 cache-SPF | 방향 재사용의 비용 이점 |
| 1 | One-shot | 현재 업데이트 보정 효과 |
| 1 | MomentGuard | 보정 상태 저장의 추가 효과 |
| 2 | Norm-matched AdamW | 업데이트 크기 축소 설명 통제 |
| 2 | SPF + `beta1=0` | 1차 이력 없는 학습 경로 |
| 2 | SPF + scheduled full `m` reset | 누적 정보 제거 효과 |
| 3 | Static SPF | 가장 오래된 방향의 경계 조건 |
| 3 | SPF + `beta1=0.5` | SPF 논문 optimizer 민감도 |

우선순위 1을 먼저 완료한다. 우선순위 2와 3은 단계 게이트와 예산에 따라 실행한다.

### 15.2 Llama/Gemma 전이 비교군

각 모델에서 다음만 실행한다.

- Initial
- AdamW
- Dynamic SPF
- Qwen에서 선택된 cache-SPF 하나
- One-shot
- MomentGuard

Qwen에서 D 게이트를 통과하지 못하면 Llama/Gemma에서 MomentGuard를 반복하지 않고 Initial/AdamW/SPF의 구조와 전이만 확인한다.

### 15.3 공정 비교 축

모든 결과를 다음 세 축으로 제시한다.

1. 같은 학습량: 같은 token, step, global batch, batch 순서
2. 비슷한 유틸리티: Dev에서 정한 Task CE 허용 범위의 checkpoint
3. 같은 계산 예산: 실제 GPU-hours 또는 비용 한도

단일 가중 종합점수로 순위를 만들지 않는다. 안전성, 유용성, 과잉 거부, 비용의 Pareto 표를 사용한다.

## 16. 평가 프로토콜

### 16.1 생성

- 주 평가: greedy 또는 temperature 0
- `max_new_tokens=256`; GSM8K는 512
- 공식 EOS/stop token
- prompt truncation 금지
- 너무 긴 예시는 사전 제외하고 수와 ID를 보고
- robustness 보조 평가: temperature 0.7, top-p 0.9, prompt당 3 generations

### 16.2 안전성 결과 상태

- harmful success
- full refusal
- partial refusal with harmful content
- safe redirection
- irrelevant/degenerate
- invalid attack generation
- evaluation error
- length-capped

invalid와 evaluation error를 자동으로 안전 성공으로 처리하지 않는다. 공통 유효 문항의 paired 결과와 누락을 안전/유해로 각각 처리한 민감도 상·하한을 제시한다.

### 16.3 judge

- 전체 sweep의 1차 판정은 benchmark-native evaluator를 사용한다.
- 핵심 확인군은 `MomentGuard vs One-shot × 3 models × 3 seeds × 최대 160 frozen behavior IDs`로 고정한다.
- 핵심 확인군의 자동 judge는 OpenAI `gpt-6-astra`, `reasoning.effort=low`, structured JSON, 입력 최대 1,500 tokens, 출력 최대 256 tokens를 사용한다.
- calibration과 오류 재판정은 동기 API, 대량 최종 판정은 Batch API를 사용한다.
- 실행 시 접근 가능한 exact model/version ID를 확인하고 고정한다. immutable snapshot이 제공되면 snapshot을 우선한다.
- judge prompt SHA256
- temperature와 decoding 옵션
- 호출 날짜와 API request/response ID
- parser version
- benchmark-native 결과와 OpenAI judge 결과를 각각 보고
- raw agreement와 Cohen's kappa를 보고
- 불일치는 blinded human audit으로 판정하며 judge 결과를 자동 평균내지 않는다.

모든 judge 불일치와 무작위 일치 사례를 합쳐 최소 100건을 blinded audit한다. 조건·모델·seed 정보는 reviewer에게 숨긴다.

### 16.4 OpenAI API 예산

총 상한은 기존 크레딧 `USD 70`으로 고정한다.

| 용도 | 상한 | 실행 방식 | 범위 |
| --- | ---: | --- | --- |
| rubric·prompt calibration | $8 | 동기 API | 사람 라벨 calibration set, 최대 200회 |
| 핵심 최종 판정 | $45 | Batch API | 최대 2,880회: 2 methods × 3 models × 3 seeds × 160 IDs |
| 불일치 재판정 | $10 | 동기 또는 Batch | 약 350회 동기 또는 700회 Batch 이내 |
| 실패·재시도 예비비 | $7 | 필요 시 | rate limit, parser/API 실패만 재시도 |

Batch 기준 요청당 보수적 상한은 `1,500 input + 256 output tokens`로 제한한다. 2026-11-10 공개 단가 기준 최악 비용은 약 `$0.0139/call`, 핵심 2,880회는 약 `$40.03`이다. 가격은 실행 직전에 다시 확인한다.

사용 순서는 다음과 같이 고정한다.

1. **P0 API preflight:** 크레딧 잔액·만료일, 모델 접근 권한, structured output schema를 5회 이하로 확인한다.
2. **P1 calibration:** test output을 보기 전에 사람이 라벨한 별도 calibration set으로 rubric, prompt, parser를 고정한다.
3. **E 최종 Batch:** 모델·방법·checkpoint·행동 ID를 모두 동결한 뒤 핵심 확인군만 Batch API로 제출한다.
4. **사후 adjudication:** benchmark-native 결과와 불일치하거나 parser/API 오류가 난 사례만 재판정한다. 결과가 기대와 다르다는 이유로 재호출하지 않는다.
5. **예비비:** 재현 가능한 기술 실패에만 사용하고, 범위 확장에는 사용하지 않는다.

- OpenAI project hard limit: `$70`
- 경고: `$35`, `$55`, `$63`
- 내부 신규 제출 중단선: 예상 누적 비용 `$63`
- 자동화는 전용 project service account key를 사용하고 Git·로그·결과 파일에 key를 저장하지 않는다.
- request별 input/output/reasoning tokens, 비용, 누적 비용을 기록한다.
- budget 초과를 피하기 위해 전체 Cartesian product에 OpenAI judge를 적용하지 않는다.

### 16.5 통계

- 동일 prompt/example ID의 paired 비교
- 10,000회 paired bootstrap
- ASR/과잉 거부: prompt ID 단위
- utility: example/group ID 단위
- 여러 attack generation은 원 prompt 계층을 유지
- seed를 prompt 표본 수로 대체하지 않음
- 모델별 절대값과 paired delta 모두 보고
- seed별 값, 평균, 범위 공개

## 17. 사전 판정 기준

### 17.1 파일럿 운영 기준

| 축 | 관심 신호 |
| --- | --- |
| 안전성 | Initial 대비 ASR 절대 변화 5%p 이상 또는 상대 변화 20% 이상 |
| 유용성 | 외부 task 평균 retention 95% 이상 |
| 개별 과제 | 주 지표 10% 초과 하락 없음 |
| 과잉 거부 | benign refusal 증가 5%p 이하 |
| 안정성 | NaN/Inf 없이 완료, 저장·재개 검증 통과 |
| 자원 | peak VRAM이 장비 한도의 90% 이하 |

### 17.2 최종 MomentGuard 실용적 개선 기준

주 대조는 같은 seed의 `MomentGuard - One-shot`이다.

- 평균 ASR 차이 `-5%p` 이하
- paired 95% interval 상한이 0 미만
- 3개 중 최소 2개 seed에서 ASR 감소
- One-shot 대비 Task Test CE 상대 악화 1% 이하
- XSTest benign over-refusal 증가 2%p 이하

점추정만 통과하면 불확실로 표시한다. 안전성 대리 CE만 좋아지고 Audit CE, 실제 ASR 또는 과잉 거부가 개선되지 않으면 결론을 대리 지표 개선으로 제한한다.

## 18. 실행 행렬과 예상량

### 18.1 단계별 필수 run

| 단계 | Anchor Qwen | Llama | Gemma | seed |
| --- | --- | --- | --- | --- |
| P1 Initial/AdamW | 필수 | 필수 | 필수 | Initial, 11; 조건부 23 |
| A 구조 | 필수 | 선택 checkpoint 확인 | 선택 checkpoint 확인 | 11, 23, 37 |
| B cache | K=1,8,32,Static | 선택 K만 | 선택 K만 | 11,23,37 |
| C 모멘트 | 필수 | 전이 확인 1 seed | 전이 확인 1 seed | 11,23,37 |
| D 단기 분기 | 필수 | 게이트 통과 시 | 게이트 통과 시 | 11,23,37 |
| E 최종 | 우선순위 1 전체 | 핵심 6군 | 핵심 6군 | 11,23,37 |

### 18.2 중복 실행 방지

- P1 AdamW checkpoint를 A와 C의 진단에 재사용한다.
- Dynamic SPF run을 A4, B 기준, E 비교에 재사용한다.
- One-shot/MomentGuard 단기 분기는 D에서만 수행하고 E full run과 분리한다.
- Norm-matched AdamW는 해당 MomentGuard run의 step norm log가 완성된 후 실행한다.
- 같은 config hash와 seed의 완료 run은 재실행하지 않는다.

실제 총 GPU-hours는 P1 10-step/125-step 실측 후 확정한다. 실행량은 `작업 step 시간 + safety backward + SVD/투영 + checkpoint I/O + 생성 + judge`로 분해해 계산한다.

## 19. 자원·비용 계측

모든 run에서 다음을 기록한다.

- end-to-end wall time
- train-only wall time
- 첫 10 step을 제외한 steady-state step time median/p95
- task forward/backward, safety backward, SVD, optimizer, checkpoint I/O 시간
- train tokens/sec/GPU
- GPU-hours = wall-clock hours x GPU 수
- GPU별 peak allocated/reserved/NVML memory
- safety cache size
- optimizer state와 checkpoint size
- checkpoint write/read time
- evaluation generation time
- judge API 비용
- storage와 대기 비용

초기 용량 예약 범위:

| 모델 | 상태 메모리 거친 범위 | 첫 시도 |
| --- | ---: | --- |
| Qwen2.5-7B | 90-125GB | 2x80GB 최소, 4x80GB 권장 |
| Llama-3.1-8B | 96-130GB | 2x80GB 최소, 4x80GB 권장 |
| Gemma-3-12B | 145-195GB | 4x80GB 권장 |

이는 activation, communication buffer, 안전성 gradient와 SVD workspace 이전의 예약값이다. 전체 예산은 기존 SafetyProject의 GPU 및 API 예산 승인 절차를 따르며 PDF의 약 10달러 가정은 적용하지 않는다.

## 20. 재현성·보안·실패 처리

### 20.1 run 필수 기록

- run ID와 종료 상태
- Git commit과 dirty 여부
- model/tokenizer/data immutable revision
- split/evaluation manifest SHA256
- 완전한 config와 hash
- seed와 RNG state
- GPU, driver, CUDA, PyTorch, Transformers, training framework 버전
- sharding config
- 시작/종료 시각
- checkpoint와 artifact URI/checksum
- task/safety batch ID와 token 수
- cache age, rank, SVD seed와 보정 발동 여부
- 내적, cosine, norm, residual conflict

### 20.2 run ID

```text
<phase>-<model>-<method>-k<cache>-r<rank>-s<seed>-<YYYYMMDD>-r<revision>
```

예: `B-qwen25-7b-periodic-k8-r4-s11-20261120-r01`

### 20.3 종료 상태

- `completed`
- `failed-capacity`
- `failed-numerical`
- `failed-infrastructure`
- `invalid-data`
- `invalid-config`
- `invalid-implementation`
- `cancelled-budget`

실패 run을 삭제하거나 덮어쓰지 않는다. 재시도는 revision이 증가한 새 run ID를 사용한다.

### 20.4 민감 artifact

- 유해 prompt와 raw generation은 private storage에만 저장한다.
- Git에는 ID, hash, aggregate, failure type만 저장한다.
- credential은 config, log, shell history와 checkpoint metadata에 포함하지 않는다.
- reviewer와 audit 접근 권한을 최소화한다.

## 21. 구현 작업 패키지

| WP | 작업 | 완료 조건 |
| --- | --- | --- |
| WP0 | config/schema 확장 | 모든 method, rank, cache, optimizer 상태를 manifest로 표현 |
| WP1 | data pipeline | Dolly/PKU/eval split, leakage report, hash 생성 |
| WP2 | AdamW instrumentation | PyTorch delta 재현, m/v/bias correction/실제 delta 로깅 |
| WP3 | SPF | block allowlist, SVD, projection, random-basis 대조 unit test |
| WP4 | cache-SPF | K=1/8/32/Static, cache age와 fresh audit 로깅 |
| WP5 | counterfactual engine | 같은 checkpoint의 m/v 후보와 임시 delta 평가 |
| WP6 | MomentGuard | One-shot/Persistent 첫-step 동일성 및 state 저장 test |
| WP7 | evaluation | utility/safety/over-refusal/judge/audit pipeline |
| WP8 | statistics/report | paired bootstrap, CI, Pareto 표, 실패 분모 |

각 WP는 unit test, smoke test, schema validation과 문서 업데이트가 있어야 완료다.

## 22. 단계별 필수 산출물

### P0-P1

1. 모델·데이터·환경 manifest
2. LR 선택 결정문
3. Initial/AdamW 안전성·유용성·과잉 거부 표
4. 시간·GPU-hours·VRAM 표

### A

1. 층/block별 CER 곡선
2. single/batch subspace overlap
3. 충돌 판정 일치율
4. safety/random basis 대조
5. `g_u -> g_u_prime -> m -> delta_theta` 잔여 충돌 분석

### B

1. cache age별 cosine/overlap
2. false-safe와 false-positive 분모
3. Audit CE 변화
4. K별 ASR/utility/over-refusal/비용

### C

1. 같은 상태 후보의 역전률
2. m 이력, v scaling, norm 효과 분해
3. 실제 Audit CE 전후 변화
4. `beta1=0.9/0.5/0` 민감도

### D

1. One-shot/Persistent 첫-step 동일성 보고서
2. 상태 전파식 검증
3. 10-step 분기 곡선
4. 보정 발동·잔여 충돌·상태 차이

### E

1. 모델·방법·seed별 ASR 표
2. Task CE와 외부 utility retention
3. XSTest over-refusal
4. judge agreement와 blinded audit
5. GPU-hours/throughput/VRAM/storage 표
6. Pareto 표
7. 실패 run과 원인 목록
8. MomentGuard go/no-go 및 후속 확장 결정문

## 23. 예상 결과별 대응

| 관측 | 해석과 조치 |
| --- | --- |
| Initial부터 ASR이 높음 | ceiling/floor effect를 기록하고 평가 난이도와 모델별 시작점을 분리 |
| AdamW 후 안전성 변화가 없음 | A 구조와 step-level 충돌까지만 확인하고 방법 효과 주장 축소 |
| 저랭크 구조는 있으나 random basis 대비 이점 없음 | 구조와 행동 안전성 효과를 분리하고 A3 음성 결과 보고 |
| cache가 오래돼도 오류가 없음 | 해당 범위의 안전성 정보 재사용 가능성을 결과로 보고 |
| 방향 역전은 있으나 Audit CE 변화 없음 | 국소 1차 신호로 제한하고 비선형성·허용오차·norm 분석 |
| H4 미통과, H5 통과 | 상태 저장보다 update/preconditioner 보정 연구로 중심 이동 |
| One-shot과 Persistent가 같음 | 현재 update 보정 효과만 관측, 상태 저장 추가 가치 미관찰 |
| MomentGuard ASR 감소와 과잉 거부 증가 | 둘을 함께 보고하고 사전 허용 범위로 실용성 판정 |
| norm-matched 후 효과 소실 | 업데이트 규모 축소로 설명 가능한 결과로 해석 |
| 모델 간 결과가 다름 | Qwen 결과를 일반화하지 않고 아키텍처 전이 한계를 보고 |

## 24. 열린 결정과 확정 시점

다음 항목은 실제 실행 전에 `docs/DECISIONS.md`에서 닫아야 한다.

| ID | 결정 | 기한 |
| --- | --- | --- |
| O-001 | 실제 GPU backend, GPU 종류와 수 | P0 시작 전 |
| O-002 | DeepSpeed ZeRO-3 또는 FSDP | 10-step preflight 전 |
| O-003 | 학습 framework immutable commit | P0 |
| O-004 | 세 모델과 tokenizer immutable revision | P0 |
| O-005 | Dolly/PKU/evaluation dataset revision | split 생성 전 |
| O-006 | SPF 보정 parameter allowlist | A1 전 |
| O-007 | safety/reference rubric와 calibration set | split 봉인 전 |
| O-008 | benchmark-native evaluator exact version | P1 평가 전 |
| O-009 | `gpt-6-astra` exact model/version ID와 judge prompt hash | P1 평가 전 |
| O-010 | private artifact storage URI와 접근 정책 | 첫 유해 생성 전 |
| O-011 | GPU/storage 예산 상한 | 유료 run 전 |
| O-012 | GuardSpace/SPA 확장 여부 | E 종료 후 |

이 표의 미정 항목은 설정 선택을 Final 결과에 맞추기 위한 여지를 의미하지 않는다. 각 항목은 명시된 시점 전에 결정하고 날짜, 이유, 영향 범위를 남긴다.

## 25. 프로젝트 완료 조건

- 세 모델의 Initial과 full AdamW가 완료되거나 재현 가능한 capacity failure가 기록됨
- Qwen anchor에서 A-D의 모든 게이트가 판정됨
- 게이트를 통과한 핵심 방법이 seed 11, 23, 37에서 평가됨
- One-shot과 Persistent의 첫-step 동일성이 검증됨
- 안전성, 유용성, 과잉 거부, 비용 결과가 같은 ID로 paired 분석됨
- judge agreement와 최소 100건 blinded audit이 완료됨
- raw harmful artifact 접근 통제와 checksum이 확인됨
- 실패·음성 결과가 포함된 최종 기술 보고서가 작성됨
- MomentGuard의 go/no-go, 전이 범위, 후속 비교군과 예산이 결정 기록에 남음

완료는 MomentGuard가 우수하다는 결과를 의미하지 않는다. 사전 계획된 실험과 판정이 완료되고, 관측되지 않은 현상과 실패까지 재현 가능한 형태로 남는 것이 완료 조건이다.
