# 실행 Runbook

이 문서는 팀원이 같은 방식으로 실행하고 실패를 남기기 위한 운영 절차다. 실제 명령은 LLaMA-Factory commit과 GPU backend를 고정한 뒤 추가한다.

## 1. 실행 전 확인

1. Issue에서 run ID를 예약한다.
2. `main` 최신 commit에서 새 branch를 만든다.
3. `configs/pilot.yaml`을 복사하지 말고 run manifest가 해당 파일의 Git SHA를 참조하게 한다.
4. 모델과 데이터에 접근 가능한지 확인한다.
5. GPU가 다른 작업 없이 비어 있는지 확인한다.
6. artifact storage 경로와 쓰기 권한을 확인한다.
7. credential이 shell history나 config에 들어가지 않게 한다.

Run ID 형식:

```text
pilot-<model>-<condition>-s<seed>-<YYYYMMDD>-r<revision>
```

예:

```text
pilot-qwen25-7b-adamw-s11-20261110-r01
```

## 2. 사전 점검

다음 정보가 없으면 장시간 실행을 시작하지 않는다.

- Git SHA
- 모델 immutable revision
- 데이터 split manifest hash
- GPU type/count
- software environment lock
- output URI
- seed
- 예상 GPU-hours

10 step preflight에서 확인할 것:

- loss가 유한함
- 모든 대상 파라미터에 gradient가 존재함
- trainable parameter 수가 전체 LM parameter 수와 일치함
- LoRA/adapter parameter가 없음
- optimizer state가 생성됨
- checkpoint save/load 성공
- 재개 후 다음 step의 loss가 연속 실행과 허용오차 내 일치
- VRAM headroom이 10% 이상

## 3. 학습 실행

실행 중 다음을 step 단위로 기록한다.

- optimizer step
- examples와 input/answer tokens
- learning rate
- task loss
- gradient norm
- step wall time
- allocated/reserved/NVML VRAM
- dataloader time
- checkpoint save time

checkpoint는 `0, 25, 50, 75, 100, 125` step에 저장한다. checkpoint에는 optimizer, scheduler, sampler position, RNG state를 포함한다.

## 4. 중단 규칙

즉시 중단:

- loss 또는 optimizer state에 NaN/Inf
- 잘못된 dataset split 또는 evaluation leakage 발견
- trainable parameter 범위가 full training과 다름
- artifact 경로가 다른 run을 덮어쓰는 상태
- credential 또는 유해 원문이 로그에 노출

진단 후 재시도:

- OOM
- GPU 통신 오류
- 일시적 storage/API 오류
- judge rate limit

재시도는 같은 run ID에 결과를 덮어쓰지 않는다. `r02`처럼 revision을 올리고 이전 run과 원인을 연결한다.

## 5. OOM 대응 순서

1. GPU당 microbatch 감소
2. gradient accumulation 증가로 global batch 16 유지
3. activation checkpointing 확인
4. ZeRO-3/FSDP CPU offload는 별도 조건으로 기록
5. 동일 종류 GPU 수 증가

금지:

- 몰래 max sequence length 축소
- BF16에서 4-bit로 변경
- full training을 LoRA/QLoRA로 변경
- 특정 모델만 다른 데이터 수 또는 optimizer step 사용

## 6. 프로파일링

- 첫 10 step은 warm-up/compile 구간으로 표시한다.
- step 11 이후 중앙값과 p95를 계산한다.
- 측정 전후 CUDA synchronize를 사용한다.
- wall time과 GPU-hours를 모두 계산한다.
- download와 preprocessing 시간은 end-to-end에 포함하되 train-only 시간도 별도 기록한다.

## 7. 평가 실행

1. checkpoint hash를 확인한다.
2. evaluation manifest를 read-only로 연다.
3. Initial과 AdamW에 같은 prompt IDs를 사용한다.
4. generation output은 private artifact storage에 쓴다.
5. judge input/output과 API response ID를 보관한다.
6. parser 실패와 API 실패를 별도 status로 남긴다.
7. aggregate 결과만 Git에 올린다.

## 8. 실행 종료

각 run은 다음 상태 중 하나로 종료한다.

- `completed`
- `failed-capacity`
- `failed-numerical`
- `failed-infrastructure`
- `invalid-data`
- `invalid-config`
- `cancelled-budget`

완료 후 Issue에 다음을 남긴다.

- run status
- artifact URI와 checksum
- 핵심 지표 요약
- 예상 대비 GPU-hours
- anomaly
- 다음 권장 행동

## 9. 결과 병합 조건

aggregate 결과 PR은 다음이 있어야 한다.

- 실행 manifest
- config hash
- 결과 schema validation
- reviewer가 재계산할 수 있는 metric input ID 목록
- raw harmful text가 없다는 확인
- 실패 및 누락 분모
- Initial과의 paired delta
