# Qwen2.5-7B-Instruct 기반 SPF 재현 및 A4 잔여 충돌 검증

**연구 실행 명세서 · v0.9 (실행 전 결정 대기)**  
**작성 기준:** 2026-11-21  
**연구 트랙:** R — SPF 재현 / A4 — 단일 step 내 잔여 충돌 전이  
**모델:** `Qwen/Qwen2.5-7B-Instruct` 단일 모델  
**환경:** 단일 NVIDIA DGX Spark 사용 예정, **실기기 접근·성능 검증 아직 미실시**  
**학습:** 전체 파라미터 Full Fine-tuning; LoRA/QLoRA 및 이전 소형 모델 파일럿 제외  
**Utility:** GSM8K (`openai/gsm8k`, `main`)  
**상태:** 연구 질문·주요 비교 설계 확정 / 완전한 학습 스크립트 및 장비 검증 미완료

> **실행 준비도 판단**: 연구 명세의 큰 구조는 정해졌으나, 현재는 **실험 실행 가능 단계가 아니라 구현 및 검증 직전 단계**이다. 지금 즉시 수행 가능한 것은 장비 점검, 모델·데이터 준비, optimizer 수치 단위 테스트, 3–5 step 메모리 게이트다. SPF 재현 및 A4 본 실험을 재현성 있게 시작하려면 §11의 필수 결정을 완료하고 §9의 게이트를 통과해야 한다.

## 1. 결정된 사항과 제외 범위

| 구분 | 확정 |
|---|---|
| 연구 모델 | Qwen2.5-7B-Instruct **1종**부터 시작 |
| 학습 파라미터 | **전체 파라미터 업데이트** (Full Fine-tuning) |
| 장비 | 기존 연결한 DGX Spark 1대 기준; 실제 RAM SKU·OS·드라이버 미확인 |
| Utility 데이터 | GSM8K Math |
| 트랙 R | Standard SFT(AdamW) vs SPF 재현 |
| 트랙 A4 | `g_u → g'_u → m → Δθ` 안전성 충돌 전이 측정; 재현과 연구 질문 분리 |
| 확장 | 파일럿에서 현상을 확인한 뒤 epoch·seed·모델 순차 확장 |
| 명시적으로 폐기 | 더 작은 모델 파일럿, 처음부터 3개 모델 병렬, LoRA/QLoRA를 주 방법으로 쓰는 계획 |

**주의:** DGX Spark의 일반적인 공식 사양은 CPU/GPU **128GB 통합 메모리**이며 H100/A800의 *80GB GPU와 동일 장비가 아니다*. 2026년 기준 일부 OEM에는 64GB 버전도 있으므로 실물 장비를 확인한다. PyTorch 공식 기술 사례는 DGX Spark에서 **Llama-3.1-8B Full FT**를 시연했다. 따라서 Qwen2.5-7B Full FT는 시도할 타당한 근거가 있지만, **추가 safety backward와 gradient/SVD가 필요한 SPF·A4까지 동작한다는 검증은 아니다**.

## 2. 연구 질문 및 통과 기준

### Track R — 재현
- RQ-R1: GSM8K로 무해한 SFT를 진행할 때 Qwen2.5-7B의 안전성 평가지표가 어떻게 변하는가?
- RQ-R2: 원 논문의 SPF가 같은 데이터·학습 조건에서 ASR을 낮추면서 GSM8K 정확도 손실을 제한하는가?
- 핵심 비교: 동일 초기 checkpoint, 동일 batch 순서, 동일 평가 파이프라인의 **SFT vs SPF**.
- 원 논문은 Qwen2.5-7B-Instruct도 사용한다. 다만 원 논문 수치와 불일치하면 **학습 템플릿, safety sample, 판정기, optimizer, 분산 방식**을 먼저 비교해야 한다.

### Track A4 — 잔여 충돌
- RQ-A4-1: SPF로 현재 gradient의 안전성 충돌을 줄인 뒤에도 **모멘트 누적** 때문에 충돌이 재발하는가?
- RQ-A4-2: **AdamW 좌표별 사전조건화**가 비충돌 모멘트를 충돌 업데이트로 바꿀 수 있는가?
- RQ-A4-3: 이러한 내적 충돌이 독립 safety probe의 실제 loss 변화와 관련 있는가?
- A4는 MomentGuard 성능 주장이 아니라 **MomentGuard가 필요한지 판단하기 위한 기전 검증**이다.

## 3. 검증된 논문 기본값과 제안 기본값 구분

**원 논문 명시값**: SPF 논문은 4×NVIDIA A800 80GB, AdamW β₁=0.5/β₂=0.999, learning rate `2e-5`, batch size `64`, 5 epochs로 서술한다. Safety gradient는 매 step 1개 safety sample로 구하며 충돌 시 block-wise truncated SVD의 상위 `k` left singular vectors를 사용한다. `k≈10`은 논문의 저랭크 관찰 근거이고, **코드 구현의 정확한 rank 및 예외 처리 조건은 확인 필요**하다.

**이번 실험 잠정 기본값 (확정 아님)**:

| 항목 | 값 | 근거 / 조치 |
|---|---|---|
| model ID | `Qwen/Qwen2.5-7B-Instruct` | revision(commit hash) pin 필요 |
| dataset | `openai/gsm8k`, config `main` | train=7,473, test=1,319 |
| optimizer | AdamW | β=(0.5,0.999), lr=2e-5 |
| full-parameter | `requires_grad=True` 전체 | 학습 가능한 총 파라미터 수 기록 |
| effective batch | 64 examples | 제안: micro batch=1, gradient accumulation=64; **global batch 64와 연산상 차이 검증** |
| epoch | Pilot 20→50 optimizer steps; 본 1→최대 5 epochs | 1 epoch≈ceil(7473/64)=117 steps, 5 epoch≈585 steps |
| model compute | BF16 | optimizer-state dtype과 별개로 기록 |
| max_seq_length | 임시 512 tokens | 데이터 길이 분포·절단율 보고 후 1024 고려 |
| grad checkpointing | ON | 실제 메모리 측정 전제 |
| gradient clipping | OFF (A4 원인 분리용 초안) | 재현 조건은 논문 코드/설정에서 확인 후 고정 |
| weight decay | 0.0 (A4 초기 clean condition) | PyTorch `AdamW` 기본은 0.01이므로 **명시 지정 필수**; 별도 nonzero 비교 |
| epsilon | 1e-8 | optimizer default 명시 기록, 원 논문과 비교 |
| lr scheduler | constant 우선 | 원 논문/공식 코드에서 확인하면 맞출 것 |
| SPF safety sample | 동일한 고정 1개/step | **어떤 샘플인지 미결정** |
| SPF rank | k=10 제안 | 구조별 shape 정의·rank 제한 규칙 확인 |
| seed | 42 (Pilot), 본 주요 조건 3 seeds | 초기 파일럿에서 1개만 |
| evaluation | GSM8K EM + 독립 safety benchmark | 판단기·프롬프트 및 출력 형식 미결정 |

**중요한 실험 단위:** SPF 보정은 gradient accumulation의 **마지막, 실제 AdamW optimizer step 직전**, 누적·평균된 `g_u`를 대상으로 해야 한다. micro-step마다 별도 투영하면 동일한 알고리즘이라고 보기 어렵다.

## 4. 학습 및 평가 데이터

1. **학습:** `openai/gsm8k`, `main/train` 7,473개. 질문→user, 전체 풀이→assistant. Qwen 공식 `apply_chat_template` 사용; assistant 토큰만 cross entropy. `<<calculator>>`와 `####` 최종 답 표기는 변환 규칙을 정하고 고정한다.
2. **파일럿 validation:** train에서 256개를 고정 holdout하면 본 학습과 표본이 달라진다. 따라서 pilot-only 검증에 사용하고 원 논문 본 재현에서는 원래 train 전체를 사용한다. **테스트 1,319개를 하이퍼파라미터 선택용으로 반복 조회하지 않는다**.
3. **SPF projection sample:** 유해 지시문 + 안전한 거부 답변의 고정 단일 쌍. 논문 Appendix B와 일치 여부 확인. safety loss는 거부 답변에 대한 teacher-forced NLL.
4. **A4 독립 probes:** projection sample과 겹치지 않는 16~32개 거부 응답 쌍. 각 probe에 대한 `∇L_s`와 loss drift를 별도 확인. **동일 projection sample에 대한 충돌률은 독립 일반화 성능이 아니다**.
5. **Safety 평가:** 원 논문과 직접 비교하려면 **HEx-PHI와 판정 규칙**을 우선 확보한다. 별도 외부 검증은 HarmBench, over-refusal은 XSTest safe subset(선택). 서로 다른 ASR을 하나로 합치지 않는다.
6. **평가 출력:** GSM8K는 최종 숫자 정답의 정규화된 exact match, Safety는 프롬프트·생성 hyperparameters·judge 버전·프롬프트 별 판정 원본 저장. seed와 generation 설정 고정.

## 5. A4의 정확한 수학적 정의

현재 파라미터 θₜ에서 `sₜ = ∇θ L_s(θₜ)`와 `gᵤ = ∇θ L_u(θₜ)`를 구한다. SPF의 조건부 투영 결과를 `gᵤ′ = P_SPF(gᵤ, sₜ)`라고 한다.

실제 update `d`가 safety loss를 1차에서 증가시키는 조건은 **`sₜ · d > 0`**이다. SGD 기준으로 `d=-ηg`이므로 `sₜ · g < 0`이 충돌과 같다. 이 부호 규칙을 전 지표에 통일한다.

A4 측정 파이프라인(같은 θₜ와 이전 optimizer state에서):

```
[1] g_u  = utility gradient
[2] g'_u = SPF(g_u, s_t)
[3] m_t  = β1*m_{t-1} + (1-β1)*g'_u
    v_t  = β2*v_{t-1} + (1-β2)*(g'_u)^2
[4] m_hat, v_hat = bias-corrected moments
[5] d_raw    = -η * g'_u
    d_m      = -η * m_hat
    d_pre    = -η * m_hat / (sqrt(v_hat) + eps)
    d_adamw  = d_pre - η*weight_decay*θ_t     # decoupled weight decay
[6] d_actual = θ_after_step - θ_before_step   # realized finite-precision update
```

> 이 분해에서 `d_m`은 실제 AdamW가 아니라 **1차 모멘트 효과만 분리하기 위한 가상 업데이트**다. `d_pre`는 상태와 정밀도까지 동일하게 구현한 가상 AdamW gradient update, `d_adamw`에는 decay까지 포함한다. 그 이후 실제 파라미터 변화 `d_actual`과 차이를 따로 기록한다.

각 단계에서 `cos(sₜ, d)` 및 `sₜ·d`, `||d||`를 기록한다. `cos(sₜ,d)>δ`를 conflict로 정의하고 초기 허용오차 `δ=1e-3`을 제안한다. `δ=1e-4`/`1e-2`에 대한 민감도 분석도 수행한다. Zero norm은 결측/무효로 처리한다.

**부호 역전(event) 정의**:

- `Residual`: `cos(s,d_raw)≤δ` 이나 `cos(s,d_adamw)>δ`
- `Momentum-induced`: `cos(s,d_raw)≤δ`, `cos(s,d_m)>δ`
- `Preconditioner-induced`: `cos(s,d_m)≤δ`, `cos(s,d_pre)>δ`
- `Decay-induced`: `cos(s,d_pre)≤δ`, `cos(s,d_adamw)>δ`
- `Actual-parameter mismatch`: `cos(s,d_adamw)` 와 `cos(s,d_actual)`이 다른 경우 별도 기록

실제 safety drift는 독립 probe에 대해 `ΔL_s=L_s(θₜ+d_actual)-L_s(θₜ)`로 측정한다. **1차 내적만으로 실제 유해 응답률(ASR) 증가를 주장하지 않는다.**

### 5.1 비교군/관찰 실험

| 조건 | 무엇을 고정/바꾸나 | 해석 |
|---|---|---|
| A4-0 No SPF | 같은 θₜ·mₜ₋₁·vₜ₋₁에서 `g_u` 직접 적용 | 원본 baseline |
| A4-1 SPF + AdamW | 같은 상태, `g'_u` 적용 | **핵심 효과** |
| A4-2 SPF + β₁=0 | 동일 상태의 **한 step**에서 β₁만 0으로 계산 | 과거 모멘트 제거의 반사실적 효과 |
| A4-3 SPF + identity D | 같은 `m_hat`, `D=I` | preconditioning 원인 분리 |
| A4-4 SPF + β₁=0.9 | 같은 과거 상태 한 step에서 branch | **역사 일치하지 않으므로 탐색적**; 정식 결과는 β₁=0.9로 별도 학습 필요 |
| A4-5 SPF + decay | decay 0 vs 0.01 등 | weight decay 분리 |

**공정한 비교 핵심:** 모든 A4 branch는 optimizer.step() 전에 같은 model checkpoint, 동일 `g_u`, 동일 safety probe 및 (가능하면) 동일 이전 optimizer state에서 계산하며, **branch가 실험 원본 state를 mutate하지 않도록** 한다. 실경로 차이는 별도 학습 경로에서 추가 검증한다. 모델 파라미터를 직접 저장/복제하는 방식 대신 shape별 streaming inner-product를 사용한다.

### 5.2 측정 시점과 결과

Pilot: 10~20개 optimizer step에서 A4 probe logging 정확성 검증.  
본 계측: 초기·중기·후기 checkpoint별 16~32개, 총 **48~96개 step**(가용성에 따라 조정). 각 step에서 projection sample 기준과 독립 probe 기준을 구분하여 저장. 다수 sample gradient 계산이 메모리를 초과하면 독립 probe는 별도 replay pass에서 순차 계산.

출력: (a) 단계별 conflict rate, (b) 부호 역전 전이행렬, (c) cosine score 분포, (d) block/layer별 기여도, (e) loss drift와 1차 근사 상관관계, (f) FP32 vs BF16 moments 정밀도 민감도(가능할 때). Step 표본 상관관계를 반영해 seed/학습 구간 별로 불확실성을 표시한다.

## 6. 구현 아키텍처

```
qwen-spf-a4/
  configs/                # pilot.yaml, reproduce.yaml, a4.yaml
  src/
    data.py               # GSM8K tokenize, label masking, deterministic ordering
    model.py              # Qwen load, full parameters, dtype and revision check
    safety_grad.py        # one-shot safety / independent probe gradient
    spf.py                # conflict condition, blockwise SVD_k, projection
    optimizer_probe.py    # functional AdamW, non-mutating state-aware replay
    train.py              # accumulation -> SPF -> step -> checkpoint
    evaluate.py           # GSM8K, Safety ASR, over-refusal
    log.py                # csv/parquet/jsonl metadata
  tests/
    test_projection.py    # projection idempotence / norm / shapes
    test_adamw.py         # torch.optim.AdamW and functional probe equivalence
    test_replay.py        # replay state nonmutation + sign metrics
  artifacts/              # only logs, figures, minimal checkpoints
  README.md
  environment.lock
```

**SPF 구현 메모:** 원 논문의 `Unflatten`은 파라미터 블록 구조를 요구한다. Linear weight는 원 shape 행렬, embedding/LM head는 별도 정책을 고정하고, bias/RMSNorm 1D 파라미터는 reshape 또는 skip 규칙을 **정식 결정**해야 한다. 상위 k 추출 시 `k ≤ min(matrix shape)`과 SVD 알고리즘(정확/randomized), seed, dtype을 로그에 남긴다. 단일 7B 전체 safety gradient와 utility gradient를 매 step 무분별하게 두 벌 복사하면 통합 메모리 부족 가능성이 크다.

**AdamW 측정 메모:** 기본 `torch.optim.AdamW`는 입력 파라미터 dtype에 따라 state dtype이 달라질 수 있다. 따라서 **FP32 moment를 원한다면 이를 보장하는 구현을 명시적으로 선택**하고 state dtype을 assert한다. `foreach`/`fused` 등 구현 경로도 기록한다. BF16 파라미터에는 수치상 작은 update가 반올림으로 사라질 수 있으므로 actual vs predicted Δθ의 차이를 계산한다.

**필수 단위 테스트:** 작은 인공 텐서로 `(1)` SPF 비충돌 no-op/충돌 projection, `(2)` AdamW 1·2회 step의 bias correction·decay 재현, `(3)` 각 branch가 원 state를 변경하지 않음, `(4)` 부호 판정 및 epsilon edge case, `(5)` BF16 실현 업데이트 차이 확인.

## 7. Full Fine-tuning 메모리 및 실현 가능성

7.62B 파라미터 근사, **10진 GB** 기준:

| 메모리 구성 요소 | 대략적 크기 |
|---|---:|
| BF16 모델 가중치 | 15.2GB |
| BF16 gradient 1벌 | 15.2GB |
| AdamW m,v를 FP32로 유지 | 61.0GB |
| 합계 (추가 safety gradient 및 activation 제외) | **91.4GB** |
| FP32 master weights 추가 시 | +30.5GB → **121.9GB** |
| Safety gradient 한 벌을 추가 보유하면 | +15.2GB |

이 추정에는 embedding 등 실제 파라미터 수 오차, optimizer temporaries, activation, 메모리 allocator, 시스템 상주 메모리, SVD 버퍼가 빠져 있다. 128GB 통합 메모리의 128GB를 전부 학습에 할당할 수 없다. **Full FT의 가능성 ≠ Full-precision moments + 동시에 보유한 2개의 full gradient + SPF의 가능성**이다.

**권장 Gate 순서:**
1. 일반 Full AdamW BF16로 3–5 step 수행: 정상 업데이트, max RSS·CUDA allocated/reserved, wall-time, swap 확인.
2. SPF safety backward 추가: 메모리를 해제하는 sequential pass/streamed gradients부터 시험.
3. A4 snapshot + functional replay 실행: state 복사 금지, layerwise streaming scalar metric.
4. 모두 통과하면 20→50 optimizer steps. 불안정하면 Full trainable 유지하되 moment dtype 및 offload/gradient storage의 선택을 **새 실험 조건**으로 기록한다. 단일 Spark의 CPU offload는 통합 메모리 총량을 늘리지 않는다. NVMe offload는 기능과 속도를 실제 검증해야 한다.

공식 PyTorch 사례에서 Llama 3.1 8B Full FT를 Spark로 수행했으나 TorchTune 설정과 이 연구의 AdamW state dtype/추가 safety backward는 다를 수 있다. **실측 전 실행시간을 확정하면 안 된다**.

## 8. 재현과 A4의 실행 순서

| Phase | 실행 | 완료 조건 |
|---|---|---|
| D0 | 원 논문/공식 구현 대조 및 §11 결정 | 선택값·버전 freeze |
| E0 | Spark 실기기 점검 + Full FT 3–5 step | OOM/NaN/심각한 swap 없음; 실제 Δθ 검증 |
| P1 | Standard AdamW 20→50 steps | loss·체크포인트·빠른 평가 동작 |
| P2 | SPF 20→50 steps | projection test 및 baseline 일치 비교 |
| A4-P | replay 10~20 steps | optimizer functional parity, branch nonmutation |
| R1 | SFT/SPF 각각 1 epoch | 동일 평가 프로토콜의 최초 재현 결과 |
| A4-M | 초기·중기·후기 48~96 step probes | 단계별 residual/momentum/precondition conflict 보고 |
| R2 | 필요 시 5 epochs, 주요 조건 3 seeds | Safety–Utility 차이·재현성 검증 |
| A4-EXT | 실제 학습 β₁ 실경로 실험 | 반사실적 1step 관찰의 일반화 확인 |

**순서 정책:** 코드 구현은 Track R와 A4가 병렬 가능하지만, 단일 Spark에서는 큰 학습 잡은 순차 실행. A4 replay는 SPF 학습 checkpoint를 공유한다. 재현 실험 중 A4가 필요 이상의 memory/throughput을 요구하면 replay를 별도 프로세스로 분리한다.

## 9. Day-0 즉시 실행 체크리스트

### 9.1 장비 정보 수집 (DGX Spark 실제 터미널에서)

```bash
uname -m
cat /etc/os-release | head -8
nvidia-smi
free -h
df -h
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT
swapon --show
docker --version
python3 - <<'PY'
import platform
print('architecture:',platform.machine())
try:
 import torch
 print('torch:',torch.__version__,'cuda:',torch.version.cuda,'available:',torch.cuda.is_available())
 if torch.cuda.is_available(): print('device:',torch.cuda.get_device_name(0))
except Exception as e: print('torch import error:',repr(e))
PY
```

기록할 것: GPU 이름, RAM 실제 총량, nvidia-smi 메모리, 디스크 여유, 컨테이너/드라이버/PyTorch 버전, 재시작·원격접속 여부, 외부 네트워크, 현재 공유 사용 여부. DGX Spark에서는 NVIDIA가 제공하는 **ARM64 호환 PyTorch 컨테이너** 사용을 권장한다. NVIDIA playbook의 예시 이미지 `nvcr.io/nvidia/pytorch:25.11-py3`는 **재현용 후보**이며 실제 드라이버와의 호환성 확인이 필요하다.

### 9.2 실험 무결성 체크

- [ ] `model revision`, `dataset revision`, training config, container digest 고정
- [ ] projection safety sample, 독립 probe, 평가 dataset 분리
- [ ] optimizer **state tensor dtype과 device를 직접 출력하고 검증**
- [ ] microbatch 1 × accumulation 64에서 scale/gradient 평균 확인
- [ ] SPF 구현의 shape, rank, 1D parameter 처리, 매 step recalculation 정의
- [ ] functional AdamW vs 기준 optimizer unit test 통과
- [ ] 3~5 step 동안 CPU RAM, GPU/Unified memory, swap, peak time 수집
- [ ] 모델 파라미터가 실제로 변경되는지 수치 검사
- [ ] 원본 상태를 변형하지 않는 A4 single-step replay 검증
- [ ] checkpoint 저장/복구 후 **동일 다음 step** 재현 시험

**중단 규칙:** OOM, NaN/Inf, checkpoint restore mismatch, safety sample leakage, optimizer state dtype 미기록, 모든 update가 BF16 반올림으로 0처럼 되는 현상, GPU가 지속적으로 swap thrash일 경우 본 실험에 진입하지 않는다.

## 10. 평가·시간·비용 산정

- **표본 수:** GSM8K `train=7473`, `test=1319`. `global_batch=64`이면 epoch당 `ceil(7473/64)=117` optimizer steps, 5 epochs 585 steps/조건.
- **Track R 학습 시간:** `T_R = 585 × (t_AdamW + t_SPF)` for 5 epochs each (`t` in seconds per optimizer step). 1-epoch 파일럿 후 실측으로 대체.
- **A4 비용:** `T_A4 = N_probe × (t_replay + t_safety_probe_forward_backward)`; 독립 probe 16~32개를 매 step 별도로 gradient 계산하면 비용이 커지므로 순차/샘플링.
- **평가 비용:** `N_generation × average generation seconds + judge_calls × unit_API_price` (판정기 종류가 정해져야 계산 가능).
- **전력:** `E_kWh = mean_power_kW × wall_time_h`; `cost = E_kWh × actual_KRW_per_kWh`. DGX Spark 240W는 전원 공급 장치 정격이지 측정된 평균 전력은 아님.
- **저장:** BF16 model checkpoint 약 15.2GB + optimizer states(수십 GB 이상) + RNG/scheduler metadata. 전체 optimizer checkpoint를 3개만 저장해도 수백 GB가 될 수 있어 보관 정책 필요.

**성능 성공 기준:** SPF의 Safety-ASR 및 GSM8K 정확도를 Standard SFT와 paired 설정에서 비교. A4는 residual conflict 및 원인별 부호 역전의 빈도·크기와 safety loss drift를 보고. 유의미한 잔여 충돌이 확인될 때에만 MomentGuard 구현으로 확장. 재현 실패도 실험 조건·원인과 함께 결과로 기록한다.

## 11. 반드시 결정해야 하는 것 — 우선순위 순

### Blocking (정해야 본 SPF/A4 실험이 재현 가능)

**D1. Safety direction에 쓸 정확한 샘플**  
A: 원 논문 one-shot safety sample을 공식 자료에서 추적해 동일하게 사용 (**재현 우선 추천**) / B: 독립적으로 고정한 안전한 거부 쌍 1개. 평가 집합과 중복 금지.

**D2. FP32 moment + Full FT를 위한 실행 경로**  
A: TorchTune/NVIDIA 컨테이너를 기본으로 하되 custom FP32 AdamW와 projection을 결합하는 구현 (**정밀도 우선**) / B: 실제 실행 가능한 BF16 moment 및 모멘트 정밀도 ablation을 먼저 수행 (**메모리 우선**). 어떤 경우든 전체 파라미터 학습을 유지하고 state dtype과 optimizer 구현 경로 기록. Spark에서 E0 실행 후 최종 확정.

**D3. 원 논문 재현 정확도 수준**  
A: 논문 수준 재현(원본 safety sample, format, judge, optimizer·SVD 구현 최대한 동일) / B: 개념적 재현(표준화한 자체 평가 및 구현). 초기에는 **A를 목표로 하되, 원본 구현이 불완전하면 차이를 명시**할 것을 추천.

**D4. 평가 판정기 및 예산**  
A: HEx-PHI 원 논문 방식 + 가능하면 판정기 유료 API / B: HarmBench 오픈 판정기 + 독립 ASR / C: 파일럿 최소 safety subset, 본 실험에서 확장. 유료 판정기 사용 가능 여부·예산 확정 필요.

**D5. SPF 세부규칙 freeze**  
원 논문 코드 우선: global dot conflict gating, block-wise SVD 상위 k(초기 10), weight/embedding/1D 규칙, truncation, precision, scheduler, grad clipping, weight decay. 소스 없이 독자 구현한 변형을 논문 완전 재현이라고 부르지 않는다.

### Non-blocking (초기 권장값으로 진행 가능)

- **D6. max_seq_length:** 512 우선, GSM8K 절단율 0에 가깝지 않으면 1024/동적 배치 조정.
- **D7. A4 probe 수:** 파일럿 10–20, 본 48–96 측정, 안전성 독립 probe 16–32.
- **D8. checkpoint 정책:** 초기/중기/후기와 1 epoch + 5 epoch, optimizer state 포함 checkpoint 수 최소화.
- **D9. 실행 시간·장비 사용 가능 시간:** 현장 평균 step time을 재고 실제 일정 산정.
- **D10. 통계 반복:** 예비 1 seed; 주요 성공 조건 3 seeds.

## 12. 사용자에게 확인할 핵심 질문

1. **Safety 샘플·평가:** SPF 원 논문의 one-shot safety sample과 HEx-PHI 평가 방식을 **최대한 그대로 재현**하는 게 최우선인가? 아니면 먼저 자체 안전성 샘플/간이 평가로 파일럿 진행해도 되는가?
2. **실행 방식:** DGX Spark에서 공식 TorchTune 기반 Full FT를 최대한 재사용하고 **FP32 모멘트 구현을 시험**하는 방향에 동의하는가? (실측 실패 시 BF16 moment 변형을 별도 조건으로 기록)
3. **평가 비용:** 유료 API 판정기 비용이 허용되는가, 아니면 **로컬 오픈 평가만** 진행할 것인가?
4. **Spark 사용:** 장비는 1대가 전용으로 사용 가능하며 실험 데이터·체크포인트를 저장할 여유 공간이 얼마나 되는가? `free -h`, `nvidia-smi`, `df -h` 출력이 있으면 정확하게 판단 가능.

이 네 가지에 답하면 blocking 선택을 구체화할 수 있다. 그러나 **실제 실행 가능 판정은 E0 결과까지 필요**하다. 아직 완성된 GPU 학습 코드가 제공된 것은 아니며, 이 문서는 그것을 구현하고 검증하기 위한 계약서이다.

## 참고자료 (2026-11-21 확인)

1. SPF 원문 (algorithm, experimental setup, appendix): https://arxiv.org/html/2601.10141
2. Qwen2.5-7B-Instruct 공식 모델 카드: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
3. GSM8K 원 데이터: https://huggingface.co/datasets/openai/gsm8k
4. NVIDIA DGX Spark 기술 사양: https://docs.nvidia.com/dgx/dgx-spark/hardware.html
5. PyTorch DGX Spark 8B Full Fine-tuning 사례: https://pytorch.org/blog/unlock-reasoning-in-llama-3-1-8b-via-full-fine-tuning-on-nvidia-dgx-spark/
6. NVIDIA DGX Spark PyTorch Fine-tune Playbook: https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-pytorch-fine-tune/README.md
7. SPF 소프트웨어 아카이브 후보 (실제 코드 내용 검증 전): https://zenodo.org/records/21289041

---
**문서 버전 상태:** 실행 설계 v0.9. 원 논문 미명시 설정은 '제안/확인 필요'로 표시. 실제 장비 접근, 완성 학습 코드 구현 및 3–5 step 테스트 이전에는 '지금 즉시 전체 실험 가능'이라고 판단하지 않는다.