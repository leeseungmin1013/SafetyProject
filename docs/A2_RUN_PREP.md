# A2 실험 설계 · Single Reference 대표성

## 연구 질문

Qwen2.5-7B-Instruct의 Initial checkpoint에서 개별 Single Reference가 Mixed Batch Reference와 비슷한 rank-10 safety subspace를 만들고, 동일한 utility gradient에 대해 같은 SPF conflict 판정을 내리는지 평가한다.

Frobenius overlap은 [SPF 논문 §4.1 Figure 4 / Eq. 5](https://arxiv.org/html/2601.10141v1#S4.SS1), conflict gate는 [§4.2 Algorithm 1](https://arxiv.org/html/2601.10141v1#S4.SS2)을 따른다. Single과 Batch의 conflict decision agreement는 [SafetyProject A2 계획](https://github.com/leeseungmin1013/SafetyProject/blob/main/PROJECTPLAN.md#112-a2-단일-샘플-대표성)의 평가 항목이다.

## 확정된 범위

- `Qwen/Qwen2.5-7B-Instruct`, Initial checkpoint, BF16, rank 10.
- `data/a2_reference_manifest.json`의 Single Reference 5개와 Mixed Batch Reference 16개. Batch gradient는 16개 sample gradient의 산술평균으로 계산한다.
- GSM8K `main/train`의 고정 probe 순서에서 첫 8개를 본실험에 사용한다. Effective batch size는 64, microbatch size는 1이다. 동일한 utility gradient를 모든 conflict gate 비교에 재사용한다.
- 모델 파라미터는 고정한다. Optimizer step, intermediate checkpoint, category별 gradient 비교는 실행 범위에서 제외한다.

## 실행 전 확인

- Model/tokenizer와 GSM8K revision을 고정하고 probe sampling seed와 본실험에 사용하는 512개 row ID를 기록한다.
- PKU-SafeRLHF 고정 revision에서 선택한 prompt 및 response를 viewer에서 검수한 항목과 대조하고 hash를 기록한다. Audit panel 중복 여부는 미확인이다.
- Chat template, assistant-only loss, per-example mean, max sequence length와 truncation 정책을 runner에서 명시한다. 길이 제한으로 응답 토큰이 모두 사라진 샘플은 오류로 처리한다.
- Overlap의 2D parameter block 범위와 raw gradient conflict gate의 trainable parameter 범위를 기록한다.

## 지표와 판정 기준

1. **Frobenius overlap.** 각 attention/MLP 2D weight matrix에서 Single 및 Mixed Batch의 safety gradient를 각각 randomized truncated SVD로 분해한다. Top-10 left singular vectors를 `U_single`, `U_batch`로 두고 `||U_singleᵀ U_batch||_F² / 10`을 계산한다. Matrix별 overlap의 median을 Single마다 구하고, 5개 Single median의 median을 primary overlap으로 보고한다.
2. **Conflict gate.** 동일한 GSM8K probe의 utility gradient `g_u`와 각 Reference의 safety gradient `g_s`를 전체 trainable parameter에 걸쳐 비교한다. `⟨g_s, g_u⟩ < 0`이면 `conflict`, 0 이상이면 `no conflict`다. 음수는 utility loss를 줄이는 작은 gradient-descent update에서 safety response loss가 1차 근사상 증가할 수 있음을 뜻한다. 실제 parameter update나 생성 응답 평가를 수행하지 않는다.
3. **Conflict decision agreement.** 같은 probe에서 각 Single의 gate 결과를 Mixed Batch의 gate 결과와 비교한다. Single별 일치 횟수는 `x/8`, 전체는 `x/40`으로 보고한다. Batch의 conflict 횟수와 두 class별 agreement도 기록한다. Batch에서 해당 class가 없으면 `N/A`다.
4. **Decision rule.** median overlap ≥ 0.80과 overall agreement ≥ 80%를 모두 충족하면 single-sample 대표성의 유망 신호로 해석한다. 기준 미달 시 팀 계획의 후속 기본 입력은 Reference batch 8이다. 여기서 8은 safety sample 수다.

## 실행 결과

| 항목 | 관측값 |
|---|---:|
| 완료한 utility probes | 8/8 |
| Median Frobenius overlap | 0.245579 |
| Overall conflict decision agreement | 40/40 (100%) |
| Mixed Batch conflict | 0/8 |
| Agreement on conflict probes | N/A |
| 두 기준 충족 | 미달 |

5개 Single은 각각 8/8로 일치했다. 모든 관측은 `no conflict`였으므로, 실제 conflict가 발생하는 probe에서의 agreement는 이 run으로 평가할 수 없다. Overlap 0.245579는 이번 checkpoint와 Reference 구성에서 관측된 결과다.

## Runner 실행

DGX Spark의 CUDA PyTorch 환경에 `datasets`와 `transformers`를 설치한 뒤 실행한다. Scratch에는 전체 FP32 safety gradient 6개를 저장할 여유 공간이 필요하다. 7B 기준 약 180 GB에 파일 오버헤드가 추가된다. Smoke는 두 gradient만 저장한다. 출력 디렉터리는 run마다 새 경로를 지정한다.

```bash
python a2_runner.py --model-revision "$MODEL_SHA" --gsm-revision "$GSM_SHA" \
  --scratch "$A2_SCRATCH" --output results/a2-smoke \
  --smoke --smoke-samples 4 --accumulation-device cuda
python a2_runner.py --model-revision "$MODEL_SHA" --gsm-revision "$GSM_SHA" \
  --scratch "$A2_SCRATCH" --output results/a2-initial --probe-count 8 --accumulation-device cuda
```

`MODEL_SHA`와 `GSM_SHA`는 각각 Qwen2.5-7B-Instruct와 GSM8K의 고정 commit SHA다. 기본 seed는 `20261125`, 기본 sequence limit은 `2048`이며 길이가 초과되면 prompt의 앞부분을 자른다. Overlap은 attention/MLP 2D parameter matrix에 대해 randomized truncated SVD로 계산한다. Conflict gate는 전체 trainable parameter의 raw gradient inner product를 사용한다. `report.json`으로 진행 상태와 metrics를 저장한다.
