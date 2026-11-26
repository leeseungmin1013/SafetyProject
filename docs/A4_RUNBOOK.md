# Qwen2.5 SPF/A4 실행 Runbook

이 runbook은 A1–A3와 독립적으로 A4만 실행한다. 대상은 Qwen2.5-7B-Instruct, GSM8K,
매 optimizer step fresh one-shot safety gradient, SPF 후 AdamW 잔여 충돌이다.

## 1. 현재 실행 범위

- Qwen2.5-7B-Instruct full-parameter BF16
- GSM8K `main`, revision pin
- AdamW β=(0.5, 0.999), lr=2e-5, FP32 moments를 NVMe에 조각별 저장
- global batch 64, microbatch 1, accumulation 64, 1 GPU
- SPF rank 10, 모든 2D weight matrix
- 5-step smoke, 20-step pilot, 117-step 1-epoch run
- `s_fresh`와 독립 `s_audit`에 대한 `d_raw`, `d_m`, `d_pre`, `d_adamw`, `d_actual`
- 독립 Audit panel의 실제 CE 전후 변화

이 runner는 단일 CUDA device 전용이다. PyTorch CUDA 할당 상한을 **55 GiB(약 59.1 GB)**로
고정해 기존 약 60.8 GiB 프로세스 제한과 60 GB 목표 안에서 시도한다. FP32 `m/v`는 NVMe에 두고 한 번에 8,388,608개
원소(상태 tensor당 32 MiB)만 읽는다. **60 GB 이내 실기기 성공은 아직 측정되지 않았다.**
preflight는 최소 48 GiB의 CUDA 여유, 32 GiB의 CPU 컨테이너 한도와 140 GiB의 NVMe 여유를
요구한다. 실제 peak와 step 시간은 smoke에서 확인한다.

## 2. private 입력 준비

원시 안전성 문항은 Git에 넣지 않는다. 접근 제한 경로에 UTF-8 파일 두 개를 준비한다.

One-shot JSON은 정확히 한 레코드다.

```json
{"messages":[{"role":"user","content":"<private prompt>"},{"role":"assistant","content":"<private safe response>"}]}
```

Audit JSONL은 projection sample과 겹치지 않는 레코드를 최소 16개 둔다. 한 줄에 JSON 객체
하나이며 같은 `messages` 형식 또는 `prompt`/`response` 형식을 사용할 수 있다.

```bash
export A4_SAFETY_SAMPLE_PATH=/workspace/private/a4/one_shot.json
export A4_AUDIT_PANEL_PATH=/workspace/private/a4/audit.jsonl
export A4_OUTPUT_ROOT=/workspace/runs/qwen25-spf-a4
export A4_GRADIENT_SPOOL_ROOT=/workspace/runs/.a4-gradient-spool
export A4_OPTIMIZER_STATE_ROOT=/workspace/runs/.a4-optimizer-state
```

preflight는 private 파일이 저장소 밖에 있는지, 중복이 없는지, 레코드 수와 SHA256을 확인한다.
내용은 로그나 manifest에 기록하지 않는다.

## 3. 설치와 테스트

DGX 컨테이너의 잠긴 환경을 활성화한 뒤 editable install 또는 `PYTHONPATH` 방식 중 하나를 쓴다.

```bash
cd /workspace/projects/<name>/SafetyProject
python -m pip install -e '.[dev]'
python -m pytest
```

기존 컨테이너 패키지를 바꾸지 않으려면 설치 없이 실행할 수 있다.

```bash
export PYTHONPATH="$PWD/src"
python -m unittest discover -s tests -v
```

## 4. 실행

장비·데이터·token length·모델 로드를 포함한 전체 점검:

```bash
bash scripts/a4_preflight.sh
```

5-step smoke는 모든 step을 A4 probe로 기록한다. 로그 디렉터리를 먼저 만든다.

```bash
mkdir -p /workspace/runs/logs
tmux new -s a4-smoke
bash scripts/a4_smoke.sh 2>&1 | tee /workspace/runs/logs/a4-smoke.log
```

smoke의 다섯 step이 유한하고 spool이 정리되며 `a4.jsonl`에 다섯 행이 생긴 뒤 본 실행을 시작한다.

먼저 20-step pilot을 실행해 전체 probe 경로를 확인한다.

```bash
tmux new -s a4-pilot
bash scripts/a4_pilot.sh 2>&1 | tee /workspace/runs/logs/a4-pilot.log
```

20개 probe의 stage metric과 Audit `ΔCE`가 모두 유한한 경우에만 main으로 진행한다.

```bash
tmux new -s a4-main
bash scripts/a4_main.sh 2>&1 | tee /workspace/runs/logs/a4-main.log
```

본 실행은 사전 고정한 16개 step에서 probe를 기록한다. FP32 optimizer state는 NVMe에서
원소 조각별로 읽고 갱신한다. probe step의 raw utility/fresh/audit gradient도 NVMe에 임시
저장하고 optimizer step 직후 삭제한다. optimizer state는 완료·일반 오류 시 삭제되며
강제 종료 시에는 해당 run ID의 잔여 디렉터리를 확인 후 정리한다. 이 파일들은 재시작
checkpoint가 아니므로 실패한 run은 처음부터 다시 시작한다.
설정의 `checkpoint_policy: save_final_main_only`는 117-step 본 실행이 끝나면
`checkpoint/`에 모델 가중치(safetensors)와 tokenizer를 저장한다. 파일별 SHA-256을
`checkpoint/checkpoint_manifest.json`에 기록하고, 저장이 완료된 뒤에만
`run_manifest.json` 상태를 `completed`로 바꾼다. 5-step smoke와 20-step pilot은
체크포인트를 저장하지 않는다. 최종 체크포인트는 추론·평가용이며 optimizer 상태가 없어
학습 중간 재시작에는 사용할 수 없다.

## 5. 출력

각 run은 `$A4_OUTPUT_ROOT/<run-id>/`에 다음을 만든다.

- `run_manifest.json`: revision, config hash, 장비, 상태
- `train.jsonl`: utility/safety loss, SPF 발동, 메모리, 시간
- `a4.jsonl`: fresh/audit 방향별 단계 metric, 전이 event, Audit ΔCE
- `checkpoint/`: 117-step 본 실행의 최종 모델·tokenizer·파일 해시 목록

약 7.62B parameter 기준 optimizer state는 약 56.8 GiB, probe gradient 3벌은 최대 약
42.6 GiB다. 모델 cache와 임시 파일 여유를 포함해 preflight에서 140 GiB의 남은 공간을
요구한다. preflight는 남은 공간을 확인하지만 저장 장치가 실제 NVMe인지 판별하지 않는다.
DGX에서 `lsblk -o NAME,TYPE,SIZE,MOUNTPOINT`로 `/workspace`의 backing device를 확인한다.
모멘트 파일은 매 step 약 56.8 GiB를 읽고 같은 양을 쓰므로 I/O로 step 시간이 길어질 수 있다.

run이 끝난 뒤 집계 파일을 만든다.

```bash
bash scripts/a4_analyze.sh /workspace/runs/qwen25-spf-a4/<run-id>
```

`a4_summary.json`에는 reference별 단계 conflict rate, 전이 event rate, Audit ΔCE 요약이 저장된다.

`a4.jsonl`의 conflict는 `s·d > max(1e-12, 1e-6||s||||d||)`이다. SPF intervention 자체는
논문과 같이 `s·g_u < 0`에서 발동한다.

## 6. 중단 규칙

- preflight memory/disk/token check 실패
- CUDA OOM
- utility/reference loss 또는 gradient NaN/Inf
- private sample과 Audit panel 중복
- assistant token 절단
- trainable parameter 수가 전체 parameter 수와 다름
- probe 종료 뒤 임시 spool 삭제 실패

실패 run의 manifest와 로그는 보존한다. 설정을 바꾼 재시도는 새 run ID를 사용한다.
