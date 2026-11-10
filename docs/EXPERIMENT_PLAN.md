# Full-Parameter AdamW 안전성 파일럿 계획서

**상태:** 실행 전 초안

**작성일:** 2026-11-10

**핵심 범위:** 영어 단일 턴, instruction-tuned 7B–12B 모델, full-parameter BF16 AdamW
**명시적 제외:** LoRA, QLoRA, SafeLoRA, SaLoRA

## 1. 목적과 연구 질문

이 파일럿은 새로운 방어법의 최종 순위를 정하기 위한 실험이 아니다. 이후 전체 실험을 설계하기 전에 다음 불확실성을 제거하는 것이 목적이다.

1. 일반 양성 instruction tuning 후 모델의 jailbreak ASR이 실제로 증가하는가?
2. 그 변화가 모델 계열에 공통적인가, 특정 아키텍처에만 나타나는가?
3. AdamW 학습으로 task CE는 개선되면서 외부 유용성은 얼마나 보존되는가?
4. ASR 감소가 과잉 거부 증가로 설명되지는 않는가?
5. 7B, 8B, 12B full training의 실제 GPU 요구량과 후속 전체 실험 비용은 얼마인가?
6. MomentGuard 및 full-parameter 비교법을 투입할 가치가 있는 충분한 안전성 변화가 관측되는가?

## 2. 사전 가설

| ID | 가설 | 파일럿 관측 | 해석 제한 |
| --- | --- | --- | --- |
| H1 | 양성 SFT만으로도 일부 모델의 안전성이 저하될 수 있다. | Initial 대비 AdamW의 ASR 변화 | 인과 원인은 이 파일럿만으로 특정하지 않는다. |
| H2 | task CE 개선과 외부 유용성 보존은 동일하지 않다. | Dolly CE와 5개 외부 과제의 task metric | CE는 토크나이저 간 직접 비교하지 않는다. |
| H3 | 모델별 안전성 변화량과 자원 비용이 다르다. | 세 모델의 paired delta 및 GPU-hours | 파라미터 수 외 아키텍처 차이가 포함된다. |
| H4 | 일부 안전성 개선은 과잉 거부로 설명될 수 있다. | 유해 ASR과 XSTest benign refusal을 함께 비교 | 자동 refusal 판정은 audit으로 검증한다. |

파일럿은 탐색적이다. `p < 0.05` 여부보다 효과 크기, 신뢰구간, 모델·seed 간 일관성, 실행 가능성을 우선한다.

## 3. 모델

| ID | Hugging Face repository | 역할 | 주의점 |
| --- | --- | --- | --- |
| qwen25-7b | `Qwen/Qwen2.5-7B-Instruct` | anchor 및 LR 탐색 | Qwen native chat template 고정 |
| llama31-8b | `meta-llama/Llama-3.1-8B-Instruct` | 아키텍처 전이 | gated access 및 라이선스 확인 |
| gemma3-12b | `google/gemma-3-12b-it` | 대형·상이한 아키텍처 전이 | text-only 입력, vision component 메모리 별도 기록 |

모든 모델은 실행 전에 immutable model revision을 manifest에 기록한다. `main`, `latest` 같은 이동 참조는 최종 manifest에서 허용하지 않는다.

비교의 의미를 유지하기 위해 다음을 통일한다.

- 각 모델의 공식 chat template 사용
- 의미가 동일한 공통 system policy 사용
- max input length와 max new tokens 통일
- 동일 example ID와 동일 prompt content 사용
- generation stop 조건을 기록
- Initial과 학습 후 모델의 decoding 조건 일치

## 4. 학습 데이터와 분할

### 4.1 주 학습 데이터

기본 후보는 `databricks/databricks-dolly-15k`다. 양성 instruction tuning이 기존 안전성에 미치는 영향을 보기 위한 데이터이며 안전 거절 응답을 학습시키는 데이터가 아니다.

고정 분할 목표:

| Split | 규모 | 용도 |
| --- | ---: | --- |
| Task Train | 2,000 | full AdamW 학습 |
| Task Dev | 250 | LR 선택, checkpoint 진단 |
| Task Test | 250 | 봉인된 task CE 최종 평가 |

분할은 카테고리 비율을 유지하고, 동일·유사 instruction 또는 공유 지문을 그룹으로 묶어 한 split에만 배치한다. 데이터 revision, row ID, 정규화 hash, split seed를 manifest로 저장한다.

길이 제한 때문에 정답이 잘리는 예시는 split 전에 제외한다. 프롬프트와 assistant 응답 토큰 수를 따로 기록한다.

### 4.2 평가 데이터와의 중복 방지

Task Train/Dev와 다음 평가셋 사이에 정규화 exact match, character n-gram, embedding retrieval 기반 중복 검사를 수행한다.

- SAMSum
- AG News
- GSM8K
- OpenBookQA
- IFEval
- JailbreakBench
- WildJailbreak
- StrongREJECT
- XSTest

의미 중복 후보는 사람이 판정한다. 겹치면 학습 또는 개발 측 예시를 제거하고 최종 평가셋은 변경하지 않는다.

## 5. 학습 조건

### 5.1 공통 조건

| 항목 | 파일럿 값 |
| --- | --- |
| finetuning type | full parameter |
| precision | BF16 |
| quantization | 없음 |
| optimizer | AdamW |
| β₁, β₂ | 0.9, 0.999 |
| ε | 1e-8 |
| weight decay | 0.0 |
| max sequence length | 1,024 |
| packing | 사용하지 않음 |
| global batch | 16 examples |
| scheduler | linear warm-up 후 constant |
| warm-up | 첫 5 optimizer steps |
| gradient clipping | 기본 비활성; 필요 시 모든 군을 다시 실행 |
| train budget | Task Train 2,000개 1 epoch, 약 125 optimizer steps |
| checkpoints | step 0, 25, 50, 75, 100, 125 |
| distributed | DeepSpeed ZeRO-3 또는 FSDP full sharding 중 하나를 사전 고정 |

메모리 때문에 microbatch를 바꾸더라도 gradient accumulation으로 global batch 16을 유지한다. 학습 예시 순서, seed, optimizer step 수를 비교군에서 맞춘다.

### 5.2 학습률 탐색

anchor인 Qwen2.5-7B에서 다음을 seed 42로 실행한다.

- `5e-6`
- `1e-5`
- `2e-5`

Task Dev CE가 유한하고 가장 낮은 값을 공통 파일럿 LR로 선택한다. 다음 조건도 동시에 충족해야 한다.

- NaN/Inf 없음
- gradient norm 폭주 없음
- checkpoint 저장·재개 성공
- Task Dev CE가 Initial보다 악화되지 않음

동률이면 더 작은 LR을 사용한다. 안전성 test 결과를 LR 선택에 사용하지 않는다. 모델별 LR 최적화는 후속 실험으로 미룬다.

### 5.3 seed

- LR 탐색: 42
- 1차 실행 가능성: 11
- 확인 실행: 23
- 최종 확장 권장: 11, 23, 37

Initial checkpoint는 학습 seed가 없지만 stochastic decoding을 사용하는 보조 평가에서는 generation seed를 별도 기록한다.

## 6. 실행 단계와 중단 규칙

### P0. 환경 및 데이터 검증

완료 조건:

- 세 모델 immutable revision과 라이선스 확인
- LLaMA-Factory commit 고정
- 학습·평가 dataset revision 고정
- split manifest와 leakage report 생성
- 10-step 저장·재개 결과가 연속 실행과 허용오차 내 일치
- run manifest schema 검증

### P1. Initial 평가

세 모델에 학습 없이 동일 평가 파이프라인을 적용한다.

- Task Test CE는 아직 열지 않고 Task Dev CE만 계산
- utility pilot subset
- safety pilot subset
- XSTest
- inference peak VRAM과 throughput

Initial 결과로 모델을 제외하지 않는다. ceiling/floor effect만 기록한다.

### P2. Qwen LR 탐색

Qwen2.5-7B에서 세 LR을 실행한다. 매 checkpoint에 Task Dev CE와 학습 안정성만 본다. 선택 규칙에 따라 LR을 동결하고 결정 로그에 기록한다.

### P3. 세 모델 AdamW 실행 가능성

선택 LR, seed 11로 세 모델을 125 step 실행한다. 25 step 이전 OOM/NaN이면 다음 순서로 대응한다.

1. microbatch를 줄이고 accumulation을 늘려 global batch 유지
2. activation checkpointing 설정 점검
3. ZeRO-3/FSDP shard 설정 점검
4. 더 많은 동일 GPU로 재시도
5. 그래도 실패하면 `failed-capacity`로 기록

정밀도 변경, 양자화, LoRA 전환으로 통과시키지 않는다.

### P4. 확인 seed

세 모델 전체를 무조건 반복하지 않는다. 다음 기준으로 seed 23 실행 대상을 정한다.

- Initial 대비 ASR 절대 변화가 5%p 이상이거나
- utility 평균 변화가 3% 이상이거나
- 과잉 거부 변화가 3%p 이상이거나
- 후속 MomentGuard의 anchor 후보인 모델

최소한 Qwen anchor는 seed 23을 실행한다.

### P5. 파일럿 종합 및 후속 결정

후속 full experiment 진입 조건:

- 한 모델 이상에서 재현 가능한 ASR 변화 신호
- task metric 보존 또는 해석 가능한 degradation curve
- full AdamW 자원 요구량 산출
- 평가 judge agreement와 audit 절차 검증
- raw artifact 접근 통제 확립

효과가 작아도 실행 결과를 폐기하지 않는다. 이 경우 후속 연구 질문을 “안전성 악화 방지”에서 “모멘트 수준의 국소 충돌 검증”으로 제한할 수 있다.

## 7. 평가 설계 요약

상세 규칙은 `EVALUATION_PROTOCOL.md`를 따른다.

### 7.1 유용성

| Benchmark | 주 지표 | 보조 지표 |
| --- | --- | --- |
| Dolly Task Dev/Test | answer-token CE | category CE |
| SAMSum | ROUGE-L, BERTScore | answer-token CE |
| AG News | Accuracy, Macro-F1 | label-token CE |
| GSM8K | Exact Match | final-answer CE |
| OpenBookQA | Accuracy | option NLL |
| IFEval | strict/loose instruction accuracy | failure category |

모델 간 절대 CE는 비교하지 않는다. 각 모델에서 `CE_after − CE_initial`만 해석한다.

### 7.2 안전성

- JailbreakBench: 표준 비교 및 직접 공격
- WildJailbreak: 자연스럽고 다양한 공격 분포
- StrongREJECT: graded harmfulness score와 thresholded ASR
- 공격: direct + PAP + AutoDAN-Turbo를 파일럿에 사용
- TAP, RAINBOW는 최종 확장에 사용
- 전체 sweep 판정기: benchmark-native evaluator의 고정 version
- 핵심 확인군 판정기: OpenAI `gpt-6-astra`의 실행 시 고정한 exact version

benchmark-native 결과와 OpenAI 결과는 각각 보고한다. OpenAI judge는 `MomentGuard vs One-shot × 3 models × 3 seeds × 최대 160 frozen behavior IDs`에 집중하고, 불일치는 blinded human audit으로 해결한다.

### 7.3 과잉 거부

XSTest benign 250개를 주 지표로 사용한다. 완전 거절과 부분 거절을 구분하고, 위험 contrast set의 수락률도 별도 보고한다.

## 8. 비용 및 메모리 측정

### 8.1 기록 항목

- end-to-end wall time
- 첫 10 step을 제외한 steady-state step time 중앙값과 p95
- train tokens/sec/GPU
- GPU-hours
- GPU별 peak allocated VRAM
- GPU별 peak reserved VRAM
- NVML peak device memory
- checkpoint 크기와 저장 시간
- 평가 generation 시간과 judge API 비용

GPU-hours는 `wall-clock hours × 사용 GPU 수`로 계산한다. 서로 다른 GPU 수의 방법을 wall time만으로 비교하지 않는다.

### 8.2 용량 예상과 실측 목적

BF16 full AdamW의 parameter, gradient, master/optimizer states는 구현에 따라 대략 파라미터당 12–16 bytes가 필요하다. activation 이전의 거친 범위는 다음과 같다.

| 모델 | 상태 메모리 예상 | 권장 첫 시도 |
| --- | ---: | --- |
| Qwen2.5-7B | 약 90–125GB | 2×80GB 최소, 4×80GB 권장 |
| Llama-3.1-8B | 약 96–130GB | 2×80GB 최소, 4×80GB 권장 |
| Gemma-3-12B | 약 145–195GB | 4×80GB 권장 |

이는 예약값이지 결과가 아니다. activation, communication buffer, vision component 때문에 실제치는 달라지며 P3 실측값으로 전체 예산을 갱신한다.

## 9. 통계와 보고

- 동일 prompt ID의 Initial/AdamW 결과를 paired 비교한다.
- ASR과 refusal rate는 prompt ID 단위 paired bootstrap 10,000회로 95% 구간을 구한다.
- utility는 example ID 단위 paired bootstrap을 사용한다.
- 여러 attack generation을 같은 독립 표본처럼 합치지 않는다.
- 모델별 절대값과 paired delta를 모두 보고한다.
- seed별 값, 평균, 범위를 숨기지 않는다.
- 파일럿 subset 결과에 최종 성능이라는 표현을 사용하지 않는다.

단일 가중 종합점수로 순위를 만들지 않는다. 안전성, 유용성, 과잉 거부, 비용의 Pareto 표를 제시한다.

## 10. 파일럿 판정 기준

아래 값은 후속 실행 우선순위를 위한 실용 기준이며 논문상의 성공 기준이 아니다.

| 축 | 관심 신호 |
| --- | --- |
| 안전성 | Initial 대비 ASR 절대 변화 ≥5%p 또는 상대 변화 ≥20% |
| 유용성 | 외부 task 평균 retention ≥95% |
| 개별 과제 | 어느 과제도 주 지표 10% 초과 하락 없음 |
| 과잉 거부 | benign refusal 증가 ≤5%p |
| 안정성 | NaN/Inf 없이 완료, 재개 검증 통과 |
| 자원 | peak VRAM이 장비 한도의 90% 이하 |

관심 신호가 없다는 이유로 run을 삭제하거나 다른 test subset을 탐색하지 않는다.

## 11. 전체 실험으로의 확장

파일럿 후 계획하는 전체 비교군의 후보는 다음과 같다.

- Initial
- Full AdamW
- One-shot moment correction
- MomentGuard persistent-state correction
- Norm-matched AdamW
- Full-parameter 재현이 가능한 GuardSpace
- Full-parameter 재현이 가능한 SPA

SafeLoRA와 SaLoRA는 본 저장소의 no-LoRA 범위에서 제외한다. 논문 수치를 참고할 수는 있지만 서로 다른 모델·데이터·평가 설정의 수치를 동일 표의 직접 비교 결과처럼 사용하지 않는다.

전체 실험은 3 models × 선택한 training methods × 3 seeds로 확장한다. OpenAI judge 예산은 총 `$70`이며 calibration `$8`, 핵심 Batch 판정 `$45`, 불일치 재판정 `$10`, 예비비 `$7`로 제한한다. 예상 누적 비용이 `$63`에 도달하면 신규 제출을 중단한다.

## 12. 예상 산출물

1. 세 모델 Initial/AdamW 비교표
2. 모델별 학습 곡선과 CE 변화
3. benchmark·attack·judge별 ASR 표
4. XSTest 과잉 거부 표
5. utility retention 표
6. runtime, GPU-hours, throughput, VRAM 표
7. judge agreement 및 audit 보고서
8. 실패 run과 원인 목록
9. 후속 전체 실험 go/no-go 결정문

## 13. 열려 있는 결정

다음 항목은 실행 전 `DECISIONS.md`에서 닫아야 한다.

- [ ] 실제 GPU 종류와 수
- [ ] DeepSpeed ZeRO-3 또는 FSDP 선택
- [ ] LLaMA-Factory commit
- [ ] 모델 immutable revisions
- [ ] Dolly revision 및 split manifest
- [ ] benchmark-native evaluator exact version
- [ ] `gpt-6-astra` exact version ID와 prompt hash
- [ ] judge calibration set과 audit rubric
- [ ] private artifact storage 위치와 접근 정책
- [ ] GPU 및 storage 예산 상한
- [ ] GuardSpace/SPA full-parameter 호환성
