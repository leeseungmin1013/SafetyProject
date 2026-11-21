# Qwen2.5-7B-Instruct 기반 A4 잔여 충돌 검증

**연구 실행 명세서 · v0.9 통합 실행판 (A4 실행 계약)**  
**작성 기준:** 2026-11-21  
**연구 트랙:** A4 — 단일 step 내 잔여 충돌 전이  
**모델:** `Qwen/Qwen2.5-7B-Instruct` 단일 모델  
**환경:** 단일 NVIDIA DGX Spark; CUDA 할당 상한 55 GiB(약 59.1 GB), FP32 optimizer state는 NVMe  
**학습:** 전체 파라미터 Full Fine-tuning; LoRA/QLoRA 및 이전 소형 모델 파일럿 제외  
**Utility:** GSM8K (`openai/gsm8k`, `main`)  
**상태:** 구현·로컬 단위 테스트·고정 데이터 전수 token scan 완료 / DGX smoke 미실행

> **실행 준비도 판단**: A4 코드는 55 GiB(약 59.1 GB) CUDA budget을 목표로 FP32 AdamW 상태를 NVMe에서 조각별 처리한다. 비공개 one-shot/audit 파일을 지정하고 DGX preflight를 통과해야 학습이 시작된다. 60 GB 이내 실기기 성공은 아직 smoke로 검증되지 않았다. A1–A3 및 행동 평가는 다른 팀원의 병렬 트랙이며 A4 시작의 선행조건이 아니다.

## 1. 결정된 사항과 제외 범위

| 구분 | 확정 |
|---|---|
| 연구 모델 | Qwen2.5-7B-Instruct **1종**부터 시작 |
| 학습 파라미터 | **전체 파라미터 업데이트** (Full Fine-tuning) |
| 장비 | 기존 연결한 DGX Spark 1대 기준; 실제 RAM SKU·OS·드라이버 미확인 |
| Utility 데이터 | GSM8K Math |
| 병렬 트랙 | A1–A3 및 전체 SPF 재현은 다른 팀원이 수행; 본 runner와 독립 |
| 트랙 A4 | `g_u → g'_u → m → Δθ` 안전성 충돌 전이 측정 |
| 확장 | 파일럿에서 현상을 확인한 뒤 epoch·seed·모델 순차 확장 |
| 명시적으로 폐기 | 더 작은 모델 파일럿, 처음부터 3개 모델 병렬, LoRA/QLoRA를 주 방법으로 쓰는 계획 |

**주의:** DGX Spark의 일반적인 공식 사양은 CPU/GPU **128GB 통합 메모리**이며 H100/A800의 *80GB GPU와 동일 장비가 아니다*. 2026년 기준 일부 OEM에는 64GB 버전도 있으므로 실물 장비를 확인한다. PyTorch 공식 기술 사례는 DGX Spark에서 **Llama-3.1-8B Full FT**를 시연했다. 따라서 Qwen2.5-7B Full FT는 시도할 타당한 근거가 있지만, **추가 safety backward와 gradient/SVD가 필요한 SPF·A4까지 동작한다는 검증은 아니다**.

## 2. 연구 질문 및 통과 기준

### 병렬 Track R — 본 실행 범위 밖
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

**이번 A4 실행 고정값**:

| 항목 | 값 | 근거 / 조치 |
|---|---|---|
| model ID | `Qwen/Qwen2.5-7B-Instruct` | model/tokenizer revision `a09a35458c702b33eeacc393d103063234e8bc28` |
| dataset | `openai/gsm8k`, config `main` | train=7,473, test=1,319 |
| optimizer | AdamW | β=(0.5,0.999), lr=2e-5 |
| full-parameter | `requires_grad=True` 전체 | 학습 가능한 총 파라미터 수 기록 |
| effective batch | 64 examples | micro batch=1, gradient accumulation=64, 단일 CUDA device |
| epoch | smoke 5 steps; pilot 20 steps; 본 1 epoch | 1 epoch=`ceil(7473/64)=117` steps |
| model compute | BF16 | optimizer-state dtype과 별개로 기록 |
| max_seq_length | **576 tokens (확정)** | 고정 tokenizer/revision으로 GSM8K train 7,473건 전수 검사: 최대 567, 512 초과 5건, truncation 0 |
| grad checkpointing | ON | 실제 메모리 측정 전제 |
| gradient clipping | OFF | A4 원인 분리를 위해 고정 |
| weight decay | 0.0 (A4 초기 clean condition) | PyTorch `AdamW` 기본은 0.01이므로 **명시 지정 필수**; 별도 nonzero 비교 |
| epsilon | 1e-8 | optimizer default 명시 기록, 원 논문과 비교 |
| lr scheduler | constant | 117-step A4 동안 고정 |
| SPF safety sample | 동일한 비공개 고정 1개/step | 저장소 밖 파일, SHA256만 manifest 기록 |
| SPF rank | k=10 | 모든 2D trainable weight, `k≤min(shape)`, FP32 randomized SVD |
| seed | 42 | A4 main 1회; 후속 반복 seed 11/23/37은 별도 확장 |
| A4 evaluation | 16개 독립 audit gradient + panel CE | projection sample과 중복 금지; 행동 ASR은 병렬 평가 트랙 |

**중요한 실험 단위:** SPF 보정은 gradient accumulation의 **마지막, 실제 AdamW optimizer step 직전**, 누적·평균된 `g_u`를 대상으로 해야 한다. micro-step마다 별도 투영하면 동일한 알고리즘이라고 보기 어렵다.

## 4. 학습 및 평가 데이터

1. **학습:** `openai/gsm8k`, `main/train` 7,473개. 질문→user, 전체 풀이→assistant. Qwen 공식 `apply_chat_template` 사용; assistant 토큰만 cross entropy. `<<calculator>>`와 `####` 최종 답 표기는 변환 규칙을 정하고 고정한다.
2. **파일럿 validation:** train에서 256개를 고정 holdout하면 본 학습과 표본이 달라진다. 따라서 pilot-only 검증에 사용하고 원 논문 본 재현에서는 원래 train 전체를 사용한다. **테스트 1,319개를 하이퍼파라미터 선택용으로 반복 조회하지 않는다**.
3. **SPF projection sample:** 유해 지시문 + 안전한 거부 답변의 고정 단일 쌍. 논문 Appendix B와 일치 여부 확인. safety loss는 거부 답변에 대한 teacher-forced NLL.
4. **A4 독립 probes:** projection sample과 겹치지 않는 거부 응답 쌍 16개. 각 probe에 대한 `∇L_s`와 loss drift를 별도 확인. **동일 projection sample에 대한 충돌률은 독립 일반화 성능이 아니다**.
5. **Safety 평가:** 원 논문과 직접 비교하려면 **HEx-PHI와 판정 규칙**을 우선 확보한다. 별도 외부 검증은 HarmBench를 사용할 수 있고, 기존 SafetyProject 규약에 따라 **XSTest safe subset의 over-refusal 평가는 필수로 유지**한다. 서로 다른 ASR을 하나로 합치지 않는다.
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

각 단계에서 `cos(sₜ, d)` 및 `sₜ·d`, `||d||`를 기록한다. SafetyProject의 공통 규칙에 맞춰 주 conflict 판정은 `sₜ·d > τ(sₜ,d)`로 정의한다.

```text
τ(s, d) = max(1e-12, 1e-6 * ||s||₂ * ||d||₂)
```

`s·d < -τ`는 안전 방향, 그 사이는 중립이다. `cos(s,d)>1e-3`과 `1e-4`/`1e-2` 비교는 주 판정이 아니라 민감도 분석으로만 보고한다. Zero norm과 비유한 gradient/update는 결측으로 숨기지 않고 별도 실패 유형으로 기록한다.

**부호 역전(event) 정의**:

- `SPF residual`: SPF 직후 `s·d_raw > τ`; truncated rank, 제외 block 또는 condition gating의 잔차
- `Momentum-induced`: `s·d_raw ≤ τ`이고 `s·d_m > τ`
- `Preconditioner-induced`: `s·d_m ≤ τ`이고 `s·d_pre > τ`
- `Decay-induced`: `s·d_pre ≤ τ`이고 `s·d_adamw > τ`
- `Realized-update residual`: 예측 `d_adamw`는 비충돌이나 `s·d_actual > τ`; BF16 저장·반올림 등 실제 파라미터 변화의 잔차

실제 safety drift는 독립 probe에 대해 `ΔL_s=L_s(θₜ+d_actual)-L_s(θₜ)`로 측정한다. **1차 내적만으로 실제 유해 응답률(ASR) 증가를 주장하지 않는다.**

### 5.1 비교군/관찰 실험

| 조건 | 무엇을 고정/바꾸나 | 해석 |
|---|---|---|
| A4-0 No SPF | 같은 θₜ·mₜ₋₁·vₜ₋₁에서 `g_u` 직접 적용 | 원본 baseline |
| A4-1 SPF + AdamW | 같은 상태, `g'_u` 적용 | **핵심 효과** |
| A4-2 SPF + `m_prev` reset | β₁·현재 gradient·`v_prev`·step·θ를 고정하고 `m_prev` 유지 vs 0 | 과거 1차 모멘트의 주 인과 진단 |
| A4-3 SPF + identity D | 같은 `m_hat`, `D=I` | preconditioning 원인 분리 |
| A4-4 SPF + β₁={0,0.9} | 각각 별도 전체 학습 궤적 | β₁ 민감도; 같은 `m_prev`에서 한 step만 β₁을 바꾼 결과를 정식 비교로 사용하지 않음 |
| A4-5 SPF + decay | decay 0 vs 0.01 등 | weight decay 분리 |

**공정한 비교 핵심:** 모든 A4 branch는 optimizer.step() 전에 같은 model checkpoint, 동일 `g_u`, 동일 safety probe 및 (가능하면) 동일 이전 optimizer state에서 계산하며, **branch가 실험 원본 state를 mutate하지 않도록** 한다. 실경로 차이는 별도 학습 경로에서 추가 검증한다. 모델 파라미터를 직접 저장/복제하는 방식 대신 shape별 streaming inner-product를 사용한다.

### 5.2 측정 시점과 결과

Smoke는 1–5 step을 모두 probe한다. 본 실행은 한 epoch의 사전등록 step
`[1,2,3,4,5,10,20,30,40,50,60,70,80,90,100,117]`에서 총 **16회** 계측한다.
각 시점에 projection sample의 `s_fresh`, 독립 audit 레코드의 `s_audit`, 16개 전체 panel의
step 전후 CE를 구분해 저장한다. 전체 gradient는 probe 동안만 private NVMe에 순차 저장한 뒤 삭제한다.

출력은 단계별 signed dot/cosine/update norm/conflict 판정, momentum·preconditioner·decay·실현
업데이트 전이 event, No-SPF 반사실적, `m_prev`/`v_prev` reset 반사실적, Audit `ΔCE`다.
블록별 기여도, 행동 ASR, 다중 seed와 BF16-moment 민감도는 이번 A4 main의 완료조건이 아니라
후속 확장이다.

## 6. 구현 아키텍처

**현재 A4 실행 구현:** `src/safety_project_a4/`, `configs/qwen25_spf_a4.yaml`,
`scripts/a4_*.sh`, `docs/A4_RUNBOOK.md`. 이 구현은 A1–A3와 독립된 단일-device streamed
gradient 경로이며, 평상시에는 safety gradient의 저랭크 기저만 보유하고 probe step에서만
원 gradient를 private NVMe spool에 임시 저장해 실제 AdamW update를 계측한다. FP32
`m/v`는 NVMe 파일로 유지하며 32 MiB 상태 조각을 순차 처리한다. CUDA 할당 상한은 55 GiB다.

| 실제 경로 | 역할 |
|---|---|
| `configs/qwen25_spf_a4.yaml` | 단일 실행 계약과 revision/step/probe 고정값 |
| `src/safety_project_a4/data.py` | Qwen chat template, assistant-only label, truncation 금지 |
| `src/safety_project_a4/gradient_capture.py` | reference gradient hook, SVD 기저, private spool |
| `src/safety_project_a4/nvme_optimizer.py` | NVMe FP32-state AdamW 및 조각별 반사실적 stage |
| `src/safety_project_a4/train.py` | accumulation → SPF → A4 probe → actual step |
| `src/safety_project_a4/preflight.py` | 데이터·메모리·디스크·CUDA·token/model gate |
| `src/safety_project_a4/analyze.py` | conflict/event/Audit `ΔCE` 집계 |
| `tests/` | AdamW parity, state nonmutation, projection, metric, 분석, encoding 테스트 |
| `scripts/a4_*.sh` | preflight, smoke, main, analyze 진입점 |

**SPF 구현 메모:** 원 논문의 `Unflatten`은 파라미터 블록 구조를 요구한다. Linear weight는 원 shape 행렬, embedding/LM head는 별도 정책을 고정하고, bias/RMSNorm 1D 파라미터는 reshape 또는 skip 규칙을 **정식 결정**해야 한다. 상위 k 추출 시 `k ≤ min(matrix shape)`과 SVD 알고리즘(정확/randomized), seed, dtype을 로그에 남긴다. 단일 7B 전체 safety gradient와 utility gradient를 매 step 무분별하게 두 벌 복사하면 통합 메모리 부족 가능성이 크다.

**AdamW 측정 메모:** 기본 `torch.optim.AdamW`는 입력 파라미터 dtype에 따라 state dtype이 달라질 수 있다. 따라서 **FP32 moment를 원한다면 이를 보장하는 구현을 명시적으로 선택**하고 state dtype을 assert한다. `foreach`/`fused` 등 구현 경로도 기록한다. BF16 파라미터에는 수치상 작은 update가 반올림으로 사라질 수 있으므로 actual vs predicted Δθ의 차이를 계산한다.

**필수 단위 테스트:** 작은 인공 텐서로 `(1)` reference capture가 utility gradient를 보존하고
projection하는지, `(2)` AdamW 1·2회 step이 `torch.optim.AdamW`와 일치하는지, `(3)` No-SPF
반사실적이 원 state를 변경하지 않는지, `(4)` norm-scaled 부호 판정과 전이 event, `(5)`
assistant-only masking과 무절단 규칙, `(6)` 결과 집계를 검증한다. NVMe 상태 파일, chunk 경계,
gradient slice 및 실제 변화량도 테스트한다.

## 7. Full Fine-tuning 메모리 및 실현 가능성

7.62B 파라미터 근사, **10진 GB** 기준:

| 메모리 구성 요소 | 대략적 크기 |
|---|---:|
| BF16 모델 가중치 | 15.2GB |
| BF16 gradient 1벌 | 15.2GB |
| AdamW m,v를 FP32로 유지 | 61.0GB |
| 전부 GPU 상주 시 합계 (activation 제외) | **91.4GB** |
| 현재 구현의 GPU 상주 기본량 (가중치+gradient) | **30.4GB** |
| 현재 구현의 NVMe FP32 m/v | **61.0GB** |
| probe 시 NVMe gradient 3벌 최대 | **45.6GB** |
| FP32 master weights 추가 시 | +30.5GB → **121.9GB** |
| Safety gradient 한 벌을 추가 보유하면 | +15.2GB |

이 추정에는 optimizer temporaries, activation, 메모리 allocator, 시스템 상주 메모리, SVD
버퍼가 빠져 있다. 위 GB는 10진 단위이며 약 60.8 GiB 프로세스 제한과 직접 비교할 때는
GiB로 환산해야 한다. 현재 구현은 CUDA 55 GiB budget, CPU 컨테이너 최소 32 GiB, NVMe
여유 최소 140 GiB를 사전점검한다. 이 기준 통과는 성공 보장이 아니며 smoke의 peak 측정이
실행 가능 여부를 결정한다.

**권장 Gate 순서:**
1. private data와 CUDA/CPU/NVMe 자원, 모델 로드를 preflight에서 확인한다.
2. NVMe FP32 AdamW + SPF + A4 probe가 포함된 5-step smoke에서 peak reserved, 시간,
   실제 `d_actual`, 상태·gradient 임시 파일 삭제를 확인한다.
3. 통과 시 20-step pilot, 이어서 117-step main을 수행한다. 실패하면 peak·I/O·수치 상태를
   기록하고 조건 변경은 새 run으로 취급한다.

공식 PyTorch 사례에서 Llama 3.1 8B Full FT를 Spark로 수행했으나 TorchTune 설정과 이 연구의 AdamW state dtype/추가 safety backward는 다를 수 있다. **실측 전 실행시간을 확정하면 안 된다**.

## 8. 재현과 A4의 실행 순서

| Phase | 실행 | 완료 조건 |
|---|---|---|
| D0 | 로컬 수치·상태·encoding 테스트 | 9개 unit test와 7,473건 token scan 통과 |
| E0 | Spark 전체 preflight + 5-step smoke | 자원 gate, 모델 load, finite loss/update, spool cleanup |
| A4-P | 동일 runner 20-step pilot | 전 probe stage 유한, AdamW/Audit 로그 완성 |
| A4-M | 117-step 1-epoch main | 사전등록 16개 probe와 completed manifest |
| A4-A | 집계 | `a4_summary.json`의 conflict/event/`ΔCE` 완성 |

**순서 정책:** A1–A3는 다른 팀원이 병렬 수행한다. A4는 별도 checkpoint에 의존하지 않고 고정
Qwen 초기 가중치에서 자체 SPF 궤적을 만든다. 단일 Spark에서 A4 내부의 smoke→pilot→main은
자원과 수치 실패를 조기에 잡기 위해 순차 실행한다.

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

- [x] model/tokenizer/dataset revision과 training config 고정
- [x] private projection sample 1개, 독립 audit 최소 16개, hash/중복 gate 구현
- [x] FP32 optimizer state를 custom AdamW에서 강제
- [x] microbatch 1 × accumulation 64에서 각 loss를 실제 예제 수로 평균
- [x] 모든 2D weight, rank clipping, 매 step FP32 randomized SVD 정의
- [x] 기준 `torch.optim.AdamW` 2-step parity와 branch nonmutation 테스트 통과
- [x] GSM8K 7,473건 token scan과 truncation 0 확인
- [ ] DGX에서 CUDA 최소 48 GiB 여유·55 GiB cap, CPU 최소 32 GiB, NVMe 140 GiB 여유 및 model-load preflight 통과
- [ ] 5-step smoke에서 실제 BF16 parameter 변화, peak memory/time, spool cleanup 확인

**중단 규칙:** preflight 실패, OOM, NaN/Inf, safety/audit 중복, assistant token 절단, state dtype
불일치, 실제 update 전부 0, spool 삭제 실패 시 main에 진입하지 않는다. 이번 117-step A4는 대형
optimizer checkpoint를 저장하지 않으며, 실패 run은 manifest를 보존하고 새 run ID로 처음부터 재시작한다.

## 10. 평가·시간·비용 산정

- **표본 수:** GSM8K `train=7473`. `global_batch=64`이면 이번 1 epoch는 `ceil(7473/64)=117` optimizer steps다.
- **A4 학습 시간:** `T_A4 = 117 × t_SPF_step + 16 × t_probe_overhead`; smoke에서 측정한 평균으로만 일정 산정한다.
- **A4 probe 비용:** 독립 gradient 1개와 panel 16개의 step 전후 forward, 세 gradient spool I/O가 포함된다.
- **평가 비용:** `N_generation × average generation seconds + judge_calls × unit_API_price` (판정기 종류가 정해져야 계산 가능).
- **전력:** `E_kWh = mean_power_kW × wall_time_h`; `cost = E_kWh × actual_KRW_per_kWh`. DGX Spark 240W는 전원 공급 장치 정격이지 측정된 평균 전력은 아님.
- **저장:** BF16 model checkpoint 약 15.2GB + optimizer states(수십 GB 이상) + RNG/scheduler metadata. 전체 optimizer checkpoint를 3개만 저장해도 수백 GB가 될 수 있어 보관 정책 필요.

**성능 성공 기준:** SPF의 Safety-ASR 및 GSM8K 정확도를 Standard SFT와 paired 설정에서 비교. A4는 residual conflict 및 원인별 부호 역전의 빈도·크기와 safety loss drift를 보고. 유의미한 잔여 충돌이 확인될 때에만 MomentGuard 구현으로 확장. 재현 실패도 실험 조건·원인과 함께 결과로 기록한다.

## 11. 권장안 확정 기록

| 결정 | 확정안 | 이유 |
|---|---|---|
| D1 Safety direction | 접근 권한이 있는 원 논문 one-shot 1개를 저장소 밖에서 주입; 없으면 사전 고정한 거부 쌍 1개 | 원문 유출 방지, audit와 hash 중복 검사 |
| D2 optimizer | custom `NVMeInspectableAdamW`, BF16 parameter + NVMe FP32 `m/v` | 60 GB CUDA 목표에서 동일 AdamW 상태와 단계별 update 관찰 |
| D3 범위 | 논문 전체 재현이 아닌 독립 A4 기전 검증 | A1–A3와 병렬이며 결과 해석 범위를 과장하지 않음 |
| D4 평가 | 유료 judge 없음; 16개 독립 audit gradient와 실제 panel CE | A4 질문에 직접 필요하고 외부 API 변동성 없음 |
| D5 SPF | global paper-zero gate, 모든 2D weight, k=10, FP32 randomized SVD, 1D skip | 규칙이 완전히 고정되고 streamed 구현 가능 |
| D6 sequence | 576, truncation 금지 | GSM8K 전수 scan 최대 567 |
| D7 probes | smoke 5개; main 사전등록 16개 | 초기·중기·후기를 포함하면서 probe spool 비용 제한 |
| D8 checkpoint | main 중간 checkpoint 없음; 실패 run 보존 후 처음부터 재시작 | FP32 optimizer checkpoint의 큰 저장·메모리 비용 방지 |
| D9 확장 | seed 42 한 epoch를 먼저 완료 | 현상 확인 전 다중 seed 비용을 쓰지 않음 |

## 12. 실행 전에 외부에서 필요한 입력

코드 선택은 모두 확정됐다. 사용자가 준비할 것은 다음 두 가지뿐이다.

1. 저장소 밖의 private 경로에 projection용 1개 JSON과 서로 중복되지 않는 audit 16개 JSONL을 둔다.
2. DGX Spark 작업에 CUDA 최소 48 GiB 여유(55 GiB cap), CPU 컨테이너 최소 32 GiB,
   NVMe 140 GiB 이상 여유 공간을 확보한다.

그 다음 `bash scripts/a4_preflight.sh` → `bash scripts/a4_smoke.sh` → 20-step pilot →
`bash scripts/a4_main.sh` 순서로 실행한다. preflight 또는 smoke 실패는 설정을 몰래 낮추는 신호가
아니라 `failed-capacity`/`failed-numerical` 근거로 보존한다.

## 13. SafetyProject 정합성 검토 및 통합 실행 계약

이 절은 2026-11-21 기준 `PROJECTPLAN.md`, `docs/DECISIONS.md`, `docs/EVALUATION_PROTOCOL.md`, `docs/RUNBOOK.md`, `configs/pilot.yaml`과 본 명세의 관계를 정리한다. A4 runner는 독립 실행 가능하며, 기존 파일럿 설계와 다른 조건은 이번 Qwen/GSM8K 트랙의 독립 설정으로 분리한다.

### 13.1 기존 계획과 일치하는 원칙

1. `PROJECTPLAN.md §11.4`의 `g_u → g_u_prime → m_hat → Dm_hat → delta_theta` 전이 분석을 A4의 상위 정의로 유지한다.
2. `PROJECTPLAN.md §13`의 동일 체크포인트 반사실적 진단을 사용해 과거 1차 모멘트와 누적 2차 모멘트의 효과를 분리한다. A4 결과는 후속 C 실험의 설계 근거가 될 수 있다.
3. Full-parameter BF16, AdamW, no-LoRA, safety CE/NLL은 proxy이고 행동 ASR은 별도라는 원칙을 유지한다.
4. Qwen2.5-7B는 기존 계획의 anchor이며, 이번에는 계산 예산상 Qwen을 먼저 수행하고 다모델 확장은 후속 승인으로 미룬다.

### 13.2 기존 파일럿과 다른 조건의 처리

| 우선순위 | 항목 | 기존 SafetyProject | 이번 명세 | 통합 규칙 |
|---|---|---|---|---|
| **P0** | 모델 범위 | Qwen/Llama/Gemma 병렬 | Qwen 1종 우선 | 이번 실행은 Qwen만; 다모델 확장은 별도 승인 |
| **P0** | 주 학습 데이터 | Dolly 2,000 | GSM8K 7,473 | Qwen SPF 재현용 별도 config 사용; Dolly 행렬은 이력으로 유지 |
| **P0** | GPU | 4×H100/A100·ZeRO/FSDP | DGX Spark 단일 | 실제 통합 메모리 기준 E0 gate 필수; 단일 장비 sharding 이득을 가정하지 않음 |
| **P0** | AdamW β₁ | 주 비교 0.9 | A4 고정 0.5 | 0.9는 후속 별도 민감도 궤적 |
| **P0** | batch·학습량 | global 16, 125 steps, 1 epoch | global 64, 117 steps, 1 epoch | A4 전용으로 독립 고정; E0 smoke는 결과가 아님 |
| **P0** | 평가 | JailbreakBench·WildJailbreak·StrongREJECT + XSTest | HEx-PHI/HarmBench + XSTest | XSTest 필수; HEx-PHI는 직접 재현, 저장소 평가군은 외부 검증으로 분리 |
| **P1** | SVD rank | k={1,4,8} 탐색 | k=10 고정 | 이번 결과는 A4 기전 검증이며 완전 재현으로 표기하지 않음 |
| **P1** | sequence length | 1024 | **576 확정** | GSM8K 7,473건 전수 스캔 최대 567; truncation 0 |
| **P1** | 실행 코드 | 공통 runner 미완성 | A4 전용 실행 package·config·scripts·tests | A4 전용 경로만 사용 |

기존 `configs/pilot.yaml`과 `experiments/pilot_matrix.csv`는 2026-11-10 Dolly/세 모델 파일럿의 이력 자료이며 이번 Qwen/GSM8K 실행에는 사용하지 않는다. 이번 트랙의 선언형 설정은 `configs/qwen25_spf_a4.yaml`, 실행 순서는 `experiments/qwen25_spf_a4_matrix.csv`가 담당한다.

### 13.3 A4 해석의 필수 구분

- `s_fresh`는 현재 파라미터와 참조 안전성 데이터의 gradient, `s_audit`는 분리된 독립 Audit panel의 gradient다. 전자의 충돌 0은 후자의 충돌 0을 의미하지 않는다.
- 잔여 충돌은 **SPF residual**, **optimizer-induced**, **realized-update residual**로 분리하고, 세부 전이는 §5의 momentum/preconditioner/decay 분류로 기록한다.
- `s·delta > 0`은 safety loss의 1차 증가 방향일 뿐 ASR 상승의 증명이 아니다. signed dot, cosine, update norm, 독립 Audit의 실제 `ΔCE`와 행동 평가를 분리해 보고한다.
- 실제 `torch.optim.AdamW`와 비변형 functional replica의 초기 2 step 이상 parity, state dtype, bias correction, weight decay, 원 state 비변형을 확인한다.
- SVD 대상 block, 제외 목록, rank clipping, singular-vector convention, dtype과 실행 시간을 manifest에 고정한다.

### 13.4 실행 준비도와 완료 기준

| Gate | 현재 상태 | 완료 기준 |
|---|---|---|
| D0 구현·로컬 검증 | **완료** | NVMe chunk parity·probe·spool tests, package import/compile |
| D0 데이터 encoding | **완료** | 고정 GSM8K 7,473건 scan, 최대 567, truncation 0 |
| E0 DGX Spark 접근·자원 | **대기** | private data, CUDA/BF16/model load, 55 GiB cap, NVMe≥140 GiB |
| E0 5-step smoke | **대기** | finite update·로그, spool cleanup, peak memory/time |
| A4-P 20-step | **대기** | 모든 probe stage와 Audit `ΔCE` 유한 |
| A4-M 117-step | **대기** | 사전등록 16개 probe, completed manifest, summary |

따라서 남은 blocker는 구현 결정이 아니라 private 입력과 DGX 자원 배정 및 실측이다.

### 13.5 통합 실행 계약

1. Qwen BF16 Full AdamW+SPF로 E0 5 step을 수행한 뒤 A4-P 20 step, A4-M 117 step으로 확장한다.
2. 고정 초기 가중치·GSM8K batch 순서·β₁=0.5·lr=2e-5·effective batch=64를 전 구간 유지한다.
3. 각 probe에서 raw utility/fresh/audit gradient를 NVMe에 순차 spool하고 optimizer step 직후 삭제한다.
4. Safety Audit CE와 행동 ASR을 분리하며, 행동 평가는 병렬 A1–A3 트랙 결과와 나중에 결합한다.
5. 용량이나 수치 문제로 실패해도 LoRA, 소형 모델, BF16 moment로 조용히 대체하지 않는다. 실패 상태를 manifest에 남긴다.
6. 유료 judge 예산은 이번 A4 기전 실험에서 사용하지 않는다.

## 참고자료 (2026-11-21 확인)

1. SPF 원문 (algorithm, experimental setup, appendix): https://arxiv.org/html/2601.10141
2. Qwen2.5-7B-Instruct 공식 모델 카드: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
3. GSM8K 원 데이터: https://huggingface.co/datasets/openai/gsm8k
4. NVIDIA DGX Spark 기술 사양: https://docs.nvidia.com/dgx/dgx-spark/hardware.html
5. PyTorch DGX Spark 8B Full Fine-tuning 사례: https://pytorch.org/blog/unlock-reasoning-in-llama-3-1-8b-via-full-fine-tuning-on-nvidia-dgx-spark/
6. NVIDIA DGX Spark PyTorch Fine-tune Playbook: https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-pytorch-fine-tune/README.md
7. SPF 소프트웨어 아카이브 후보 (실제 코드 내용 검증 전): https://zenodo.org/records/21289041

---
**문서 버전 상태:** v0.9 통합 실행판. 코드와 권장안은 확정됐으며, private 입력을 연결한 DGX preflight와 smoke를 통과하면 main을 실행한다.
