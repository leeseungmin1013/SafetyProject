# Qwen2.5-7B SPF/A4 계획의 SafetyProject 적합성 검토

**검토일:** 2026-11-21  
**대상 문서:** [첨부한 실행 명세서 v0.9](Qwen25_7B_SPF_A4_실험명세서_v0_9.md)  
**기준 저장소:** `SafetyProject`의 `PROJECTPLAN.md`, `docs/DECISIONS.md`, `docs/EVALUATION_PROTOCOL.md`, `docs/RUNBOOK.md`, `configs/pilot.yaml` (2026-11-21 현재 main)  
**판정:** **연구 목적과 A4 질문에는 적합. 현재 상태로 전체 실험 실행은 불가. 기존 파일럿 설계와 실행 조건이 충돌하므로 우선순위 및 config를 분리·동기화해야 함.**

## 1. 저장소와 일치하는 점

1. `PROJECTPLAN.md §11.4`에 이미 A4가 `g_u → g_u_prime → m_hat → Dm_hat → delta_theta` 전이 분석으로 정의되어 있어 명세서의 핵심 가설과 동일하다.
2. `§13`의 C 실험은 같은 체크포인트에서 과거 1차 모멘트·누적 2차 모멘트의 효과를 따로 계측하도록 설계되어 있다. A4 계측 결과를 후속 C의 설계 근거로 사용할 수 있다.
3. Full-parameter BF16, AdamW, no-LoRA, 안전성 CE/NLL은 proxy이며 ASR은 별도라는 원칙, XSTest over-refusal은 기존 저장소와 일치한다.
4. Qwen2.5-7B는 기존 계획에서도 anchor였고, 기존 세 모델 동시 진행을 Qwen 선행으로 축소하는 것은 계산 예산 관점에서 타당하다.

## 2. 충돌·보정이 필요한 지점

| 우선순위 | 항목 | 기존 SafetyProject | 새 v0.9 | 정리 방안 |
|---|---|---|---|---|
| **P0** | 현재 모델 범위 | Qwen/Llama/Gemma 병렬 | Qwen 1종 우선 | **이번 실행 범위**는 Qwen만. 후속 다모델 확장은 별도 승인 |
| **P0** | 주 학습 데이터 | Dolly 2,000 | GSM8K 7,473 | Qwen SPF 원 논문 재현용 **GSM8K 별도 config** 사용. 이전 Dolly 실험 행렬 비활성(아카이브 유지) |
| **P0** | GPU | 4×H100/A100·ZeRO/FSDP | DGX Spark 단일 | 단일 128GB *통합 메모리* 기준 E0 gate 필요. 1 GPU의 ZeRO-3/FSDP만으로 실질적 model sharding 이득 기대하지 말 것 |
| **P0** | AdamW β₁ | 주 비교 0.9 | SPF 논문 재현 0.5 | **이번 Track R/A4의 공통 baseline β₁=0.5**. 0.9는 명명된 민감도 군으로 분리 |
| **P0** | Batch·학습량 | global 16, 125 steps, 1 epoch | global 64, 117 steps/epoch, 최대 5 epochs | 원 논문 재현 설정 64/5로 독립 고정. E0의 smoke는 방법 비교용 아님 |
| **P0** | 평가 | JailbreakBench·WildJailbreak·StrongREJECT + XSTest 필수 | HEx-PHI/HarmBench + XSTest 선택적 | **XSTest는 반드시 유지**. 논문 직접 비교에는 HEx-PHI 프로토콜 확인, 별도 외부 평가에는 저장소 규약 유지 |
| **P1** | SVD rank | k={1,4,8} 구조 탐색 | k=10 잠정 | 논문 구현의 k/블록 규칙 우선 추적. 임의 k=10을 논문 재현으로 표기 금지 |
| **P1** | Seq 길이 | 1024 | 512 | GSM8K 원문·답변의 실제 토큰 길이 분포와 truncation 비율을 측정 후 결정 |
| **P1** | 실행 코드 | 실행 스크립트 미완성 | 명세만 존재 | 이 PR은 **문서와 계획의 업데이트**이며 Spark 실행 코드/결과는 포함하지 않음 |

## 3. A4 수학·측정 설계 보완 (중요)

### A. 잔여 충돌은 세 개로 분리한다

- `SPF residual`: `g_u'` 직후 아직 안전성 방향에 충돌이 남는지. 블록 제외·truncated rank·condition gating의 한계.
- `optimizer-induced`: `g_u'`는 비충돌인데 과거 1차 모멘트(`m_prev`) 또는 좌표별 사전조건화(`v_prev`) 뒤에 부호가 역전되는지.
- `realized-update residual`: decay 및 실제 BF16 파라미터 저장/반올림까지 반영한 `theta_after-theta_before`가 충돌하는지.

반드시 `s_fresh`(현재 파라미터·참조 안전성 데이터의 gradient)와 `s_audit`(분리된 독립 Audit panel의 gradient)를 구분한다. `s_fresh`에서 충돌이 0이어도 `s_audit`의 충돌이 0이라는 결론은 아니다.

### B. `β₁=0`의 한-step 비교는 주의한다

서로 다른 β₁로 훈련된 이전 모멘트 상태는 동일하지 않으므로 `β₁=0.9`로 *한 step만* 바꾸고 같은 `m_prev`를 재사용하면 합당한 full-run β₁=0.9 실험이 아니다. 주 인과 진단은 **β₁, 현재 gradient, v_prev, step, θ를 고정하고 `m_prev`를 유지 vs 0으로 리셋**하는 반사실적 branch(`PROJECTPLAN §13.2`)를 우선 사용한다. `β₁=0` 및 `β₁=0.9`는 별도의 전체 학습 궤적 실험으로 구분한다.

### C. update 방향·크기 및 근사 손실을 같이 보고한다

`s·delta > 0`은 안전성 손실 *1차 증가 방향*이지 실제 ASR 상승을 보장하지 않는다. cosine뿐 아니라 `s·delta`, `||delta||`, 독립 Safety Audit의 정확한 `ΔCE`를 함께 측정하고, norm-matched 대조군을 포함한다. 수치 임계값은 `PROJECTPLAN §8.4`의 norm-scaled `tau`와 일치시킨다. v0.9의 `cos>1e-3`은 별도 민감도 분석으로만 사용한다.

### D. 학습 충실도 조건

- accumulation 후 utility gradient에 SPF 적용. 매 micro-step 투영은 다름.
- 실제 `torch.optim.AdamW` update와 비변형 functional replica의 parity(초기 2 step 이상), weight decay·bias correction·state dtype 검사.
- `D=I`는 대조용 가상 업데이트로 두되 step norm을 실제 update와 맞춘 후 독립 Audit CE로 해석.
- `theta_after-theta_before` 실제 저장 파라미터 변화를 포함. 가상 `-ηD m_hat`만으로 실제 업데이트라고 명명하지 않기.
- SVD의 대상 block, exclusion, rank clipping, singular vector convention, 정밀도, 성능을 고정하여 기록.

## 4. 실행 준비도 (as of 2026-11-21)

| Gate | 현 상태 | 완료 기준 |
|---|---|---|
| P0 논문·repo 정렬 | **부분 완료** | 정확한 safety sample, SPF 구현, optimizer 설정, 평가 프로토콜 pin |
| E0 DGX Spark 접근·자원 | **미측정** | `nvidia-smi`, `free -h`, `df -h`, torch/driver, 3~5 optimizer steps, checkpoint restore |
| P1 Baseline 20→50 | **미구현/미실행** | Qwen full parameter에 실제 업데이트, 손실·메모리 로그 |
| P2 SPF 20→50 | **미구현/미실행** | 정확한 투영 및 원본과 공정 비교 |
| A4-P 계측 | **미구현/미실행** | m/v/precision 보존, branch nonmutation, parity, 10~20 step 로그 |
| R1 1 epoch | **시작 불가** | 앞선 게이트 통과 후 실행 |
| A4 본 분석 | **시작 불가** | 초기·중기·후기 ≥48 유효 step 및 독립 audit probes |

**결론:** 장비 점검과 구현 착수는 지금 가능. 학습 코드는 아직 없고 실기기 접근도 없어 *지금 바로 정식 실험 실행 가능*이라고 판단하면 안 된다.

## 5. 권장 실행 계약

1. 파일럿: Qwen BF16 Full AdamW 3~5 step `E0` ⇒ 20/50 step `P1`.
2. SPF 재현: 같은 Qwen 초기 가중치와 GSM8K 배치 순서, β₁=0.5, lr=2e-5, effective batch=64로 `P2` ⇒ `R1`. 다만 원 논문 나머지 하이퍼파라미터는 공식 코드/appendix 확인 후 고정한다.
3. A4는 `P2` 실행 과정에서 low-overhead scalar logging 설계, 단일 Spark에서 replay는 별도 순차 프로세스로 실행. ProjectPlan의 A4 정의를 존중하면서 C의 반사실적 분석 일부를 A4 본 계측에 사용.
4. Safety Audit과 행동 ASR은 분리하며 XSTest 과잉 거부를 유지한다.
5. 실패하면 LoRA나 더 작은 모델로 대체하지 않고 `failed-capacity` 또는 `failed-numerical`로 기록, Full-parameter 경로의 메모리/정밀도 옵션을 진단한다.

## 6. 정식 실행 전 미결정 사항 (결정 필요)

1. **SPF 참조 안전성 쌍의 원문/재현 가능 source** 및 A4 Audit panel의 분리 정책.
2. **DGX Spark 실측 환경**: 64/128GB SKU, PyTorch/드라이버·디스크, FP32 moment/full weight 상태의 실제 메모리 사용 및 save/load.
3. **학습 정밀도 구현**: FP32 moment를 어떻게 유지할지, BF16 weight의 update 손실을 어떻게 검증할지.
4. **원 논문 SPF 설정 확정**: k, 파라미터 허용목록, scheduler, clipping, decay, 1D 처리, data/chat format.
5. **평가 접근성/비용**: HEx-PHI 사용 가능성과 judge 구현. 기존 `D-006`의 OpenAI $70은 *나중 MomentGuard vs One-shot 확인용*이므로 이번 재현 파일럿 비용으로 자동 집행하지 않는다.

## 7. 저장소 문서 우선순위 / 참고

- [Qwen 현재 실행 설정(선언형)](../configs/qwen25_spf_a4.yaml): 실제 Trainer 입력 파일이 아님.
- [Qwen 실행 행렬](../experiments/qwen25_spf_a4_matrix.csv): 이번 단계의 작업 순서.
- 기존 `configs/pilot.yaml` 및 `experiments/pilot_matrix.csv`: **2026-11-10의 Dolly/세 모델 파일럿을 설명하는 이력 자료**, 이번 Qwen/GSM8K 작업에는 사용하지 않는다.
- `PROJECTPLAN.md §11.4`의 A4 정의 및 `§13.2`의 진단 비교군을 원칙으로 유지한다.
- 외부 기술 참고: SPF https://arxiv.org/abs/2601.10141 · Qwen https://huggingface.co/Qwen/Qwen2.5-7B-Instruct · Spark https://pytorch.org/blog/unlock-reasoning-in-llama-3-1-8b-via-full-fine-tuning-on-nvidia-dgx-spark/

**본 문서는 기존 v0.9를 훼손하지 않고 SafetyProject와 일치시키기 위한 검토·통합 정정이다.**
