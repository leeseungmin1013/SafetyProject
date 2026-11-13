# DGX Spark 팀 컨테이너 사용 가이드

팀이 함께 쓰는 GPU 환경에 접속해서 학습을 돌리기까지의 절차와 규칙이다.
팀원은 DGX Spark 본체가 아니라 그 위에 격리해 둔 **컨테이너 하나**에만 들어간다.

> 접속 주소, 기기 이름, 토큰, 키는 이 저장소에 적지 않는다. `[비공개]`로 표시된 값은
> 관리자에게 받는다. 비밀번호 로그인은 쓰지 않는다.

실험 절차(run ID, 사전 점검, 중단 규칙)는 [`RUNBOOK.md`](RUNBOOK.md)가 기준이다.

## 한눈에 보기

| 항목 | 값 |
| --- | --- |
| 접속 | Tailscale로 공유받은 기기 1대에 SSH (개인 공개키) |
| 계정 | `ubuntu` 하나를 팀 전체가 함께 쓴다. root와 sudo는 없다 |
| 작업 공간 | `/workspace` (400GB 상한). 그 밖의 경로는 읽기 전용이다 |
| 메모리 | 컨테이너 64 GiB + GPU는 프로세스당 약 60.8 GiB 상한 |
| CPU / 프로세스 수 | 16코어 / 4096개 |
| 네트워크 | 인터넷으로 나가는 것만 된다. 관리자의 집 네트워크와 개인 기기에는 닿지 않는다 |
| 기본 환경 | Ubuntu 24.04 (ARM64), Python 3.12, PyTorch 2.14.1+cu130, uv, tmux, git |

```bash
ssh safety-gpu                                   # 2절
tmux new -s <이름>                               # 4절
nvidia-smi && free -h                            # 5절: 다른 사람이 쓰고 있는지 확인
cd /workspace/projects/<이름>
python /opt/project/smoke_test.py                # 환경 점검
```

## 1. 알아 둘 특성

- **ARM64다.** x86용 설치 가이드의 wheel 주소를 그대로 쓰면 맞지 않는 빌드가 깔린다.
  PyTorch는 이미지에 들어 있는 `2.14.1+cu130`(aarch64, CUDA 13.0)을 쓴다. 새 패키지는
  PyPI에 `aarch64` wheel이 있는지 먼저 확인한다.
- **GPU 전용 메모리가 없다.** CPU와 GPU가 같은 메모리(전체 121.7 GiB)를 나눠 쓴다.
  `nvidia-smi`의 Memory-Usage는 `Not Supported`로 나오며 고장이 아니다.
- **컨테이너의 64 GiB 제한은 CPU 쪽 메모리에만 걸린다.** 실측으로 확인했다: 4 GiB로
  제한한 컨테이너가 GPU 메모리 12 GiB를 잡는 동안 컨테이너 사용량은 0.44 GiB로 집계됐다.
  그래서 GPU 쪽은 별도 장치로 막는다.
  - PyTorch 프로세스는 전체의 50%(약 60.8 GiB)까지만 GPU 메모리를 쓸 수 있다. 이미지의
    전역 `sitecustomize.py`가 `torch.cuda.set_per_process_memory_fraction(0.5)`를 건다.
    직접 만든 가상환경에도 자동으로 적용된다.
  - vLLM 서버는 `vllm-serve` 래퍼로 띄운다. `--gpu-memory-utilization 0.45`로 고정된다.
  - 이 상한은 **프로세스마다 따로** 적용된다. 프로세스 여러 개의 합계는 막아 주지 않으므로
    5절의 규칙으로 관리한다.
- **메모리가 바닥나면 내 작업만 죽지 않는다.** 같은 머신의 다른 작업까지 죽을 수 있다.
- `free -h`는 컨테이너가 아니라 머신 전체의 메모리를 보여 준다.

## 2. 접속

처음 한 번:

1. 자기 PC에 [Tailscale](https://tailscale.com/download)을 설치하고 본인 계정으로 로그인한다.
2. 관리자가 보낸 기기 공유 초대를 수락한다. 기기 목록에 공유받은 기기 1대가 나타난다.
   이 기기 말고 관리자의 다른 기기는 보이지 않고 접속할 수도 없다.
3. SSH 키를 만든다. 이미 쓰는 키가 있으면 그것을 써도 된다.

   ```bash
   ssh-keygen -t ed25519 -C "<이름>"
   ```

4. **공개키**(`~/.ssh/id_ed25519.pub`의 한 줄)를 관리자에게 보낸다. 비밀키는 누구에게도
   보내지 않는다. 관리자가 등록하면 접속할 수 있다.
5. 자기 PC의 `~/.ssh/config`에 별칭을 만든다.

   ```text
   Host safety-gpu
       HostName [비공개]
       User ubuntu
       IdentityFile ~/.ssh/id_ed25519
       ServerAliveInterval 30
   ```

접속과 확인:

```bash
ssh safety-gpu
uname -m && nvidia-smi | head -12        # aarch64, NVIDIA GB10
```

TensorBoard나 Jupyter처럼 컨테이너 안에서 연 포트는 SSH 로컬 포워딩으로 본다.
그 밖의 포워딩과 에이전트 포워딩은 꺼져 있다.

```bash
ssh -L 6006:localhost:6006 safety-gpu
```

## 3. 환경

### 기본 환경

접속하면 `/opt/venv`의 Python이 바로 잡힌다. 읽기 전용이며 다음이 들어 있다.

| 패키지 | 버전 |
| --- | --- |
| torch | 2.14.1+cu130 |
| transformers | 5.22.2 |
| datasets | 5.2.0 |
| accelerate | 1.15.0 |
| peft | 0.21.3 |
| lm-eval | 0.4.13 |

전체 목록은 `infra/dgx_spark/uv.lock`에 있다. 점검용 스크립트는 `/opt/project/`에 있다.

### 작업 폴더

| 경로 | 용도 |
| --- | --- |
| `/workspace/projects/<이름>/` | 각자의 작업 폴더. 레포는 여기에 clone 한다 |
| `/workspace/hf-cache` | Hugging Face 모델·데이터셋 캐시 (공용, `HF_HOME`) |
| `/workspace/runs` | 실행 결과와 로그 |
| `/workspace/home` | `ubuntu` 계정의 홈 (공용) |

홈이 공용이므로 개인 설정은 홈에 두지 않는다. git 사용자 정보는 `--global` 없이 자기
레포 안에서 설정한다.

```bash
cd /workspace/projects/<이름>/SafetyProject
git config user.name "<이름>" && git config user.email "<메일>"
```

### 패키지 추가

시스템 패키지는 설치할 수 없다. 필요하면 관리자에게 이미지에 넣어 달라고 요청한다.
Python 패키지는 자기 폴더의 가상환경에 자유롭게 설치한다.

```bash
cd /workspace/projects/<이름>/SafetyProject/infra/dgx_spark
uv sync                       # uv.lock 과 같은 환경을 .venv 에 만든다
source .venv/bin/activate
uv add <패키지>               # 내 환경에만 추가된다
```

패키지 파일은 공용 캐시(`/workspace/.cache/uv`)에서 하드링크되므로 가상환경 하나가
수십 MB만 차지한다. 캐시가 비어 있는 첫 실행만 수 분이 걸린다. 팀 전체가 써야 하는
패키지는 `pyproject.toml`과 `uv.lock`을 함께 고쳐 pull request로 올린다.

### 토큰과 인증

- Hugging Face: 팀 공용 읽기 전용 토큰이 자동으로 적용된다 (`HF_TOKEN_PATH`).
  gated 모델(Llama-3.1 등)도 그대로 받을 수 있다. 개인 토큰은 넣지 않는다.
- GitHub: 계정을 함께 쓰므로 **개인 토큰이나 키를 컨테이너에 저장하지 않는다.**
  공개 레포는 인증 없이 `git pull` 된다. push는 자기 PC에서 하거나, 수명이 짧은
  fine-grained 토큰을 입력창에 직접 넣는 방식으로만 한다.

## 4. 실행

접속이 끊기면 그 안에서 돌던 프로세스도 죽는다. 몇 분 넘게 걸리는 작업은 tmux에서 돌린다.

```bash
tmux new -s <이름>
cd /workspace/projects/<이름>/SafetyProject
mkdir -p /workspace/runs/logs
python <스크립트> 2>&1 | tee /workspace/runs/logs/<이름>-$(date +%Y%m%d-%H%M%S).log
```

| 하려는 일 | 방법 |
| --- | --- |
| 세션에서 빠져나오기 (작업은 계속) | `Ctrl-b` 누른 뒤 `d` |
| 다시 들어가기 | `tmux attach -t <이름>` |
| 세션 목록 | `tmux ls` |
| 중단 | 세션에 들어가 `Ctrl-c`. 안 멈추면 자기 프로세스의 PID에 `kill <PID>` |

모두 같은 계정이라 남의 세션에 들어가거나 남의 프로세스를 끌 수 있다. 자기 이름이 붙은
세션만 쓴다. 중단한 뒤에는 `nvidia-smi`에서 프로세스가 사라졌는지 확인한다.

## 5. 자원 사용 규칙

1. **시작 전에 확인한다.** `nvidia-smi` 프로세스 목록과 `tmux ls`로 누가 쓰고 있는지 본다.
2. **쓰기 전에 팀 채널에 알린다.**

   ```text
   [GPU 사용] 이름 / 작업(run ID) / 예상 GPU 메모리 / 시작 ~ 예상 종료
   ```

3. **GPU 메모리 합계를 64 GiB 안에서 쓴다.** 프로세스당 상한은 합계를 막지 못한다.
   학습은 한 번에 하나를 원칙으로 하고, 동시에 띄울 때는 채널에서 합계를 맞춘다.
4. **상한을 우회하지 않는다.** `set_per_process_memory_fraction`을 다시 호출하거나
   `vllm-serve` 대신 `vllm serve`를 직접 쓰지 않는다.
5. **디스크는 400GB를 함께 쓴다.** 체크포인트를 여러 개 남길 때는 채널에 알린다.
   `/tmp`는 4GB이고 메모리에서 차감되므로 큰 파일은 `/workspace`에 둔다.
6. **끝나면 알린다.** 프로세스를 띄운 채 자리를 뜨지 않는다.
7. 모델 가중치, 체크포인트, 원시 생성 결과, 토큰은 커밋하지 않는다
   ([`SECURITY.md`](../SECURITY.md)).

## 6. 사전 점검과 측정 결과

```bash
python /opt/project/smoke_test.py                 # GPU, bf16, 패키지 import 검사
python /opt/project/generate_test.py              # 모델 로드와 짧은 생성
python /opt/project/preflight_lora.py             # LoRA 메모리·시간 측정
python /opt/project/preflight_lora.py --help      # 모든 설정
```

`preflight_lora.py`는 Qwen2.5-7B-Instruct에 LoRA(rank 32, alpha 64)를 붙여 짧게 학습을
돌리고 메모리 피크와 step 시간을 잰다. 입력은 고정 길이의 무작위 토큰이라 loss 값에는
의미가 없다. 결과는 `/workspace/runs/preflight/` 아래 `result.json`으로 남는다.

2026-11-14에 컨테이너 안에서 잰 값이다.

| 항목 | 값 |
| --- | --- |
| 모델 | `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28` (bf16) |
| LoRA | rank 32, alpha 64, 대상 `all-linear`. 학습 파라미터 80.7M개 (fp32, 텐서 392개) |
| 입력 | 길이 1024의 무작위 토큰, gradient checkpointing 켬, 12 step |

| 배치 구성 | GPU 피크 (reserved) | 통합 메모리 피크 | step당 시간 | 처리량 |
| --- | --- | --- | --- | --- |
| microbatch 1 x 누적 4 | 19.4 GiB | 21.4 GiB | 4.65초 | 880 토큰/초 |
| microbatch 4 x 누적 1 | 24.9 GiB | 26.3 GiB | 5.39초 | 760 토큰/초 |

- 모델을 올린 직후가 14.5 GiB이고, 학습 중 피크는 64 GiB 예산의 33~41%다.
- 전역 배치 16, 길이 1024(step당 16,384 토큰)로 환산하면 step당 약 19~22초, 125 step에
  약 39~45분이다. 학습 step만 계산한 환산값이며 평가와 체크포인트 저장은 빠져 있다.
- microbatch 1 구성의 step 시간은 표본이 2개뿐이라 참고값이다 (A4 기록 step을 뺐기 때문).
  microbatch 4 구성은 표본 8개다.
- A4 로거가 기록하는 step은 평균 6.25초로, 안전 배치 역전파 한 번과 기록에 약 1.6초가
  더 든다.
- 컨테이너의 CPU 쪽 사용량은 피크 26 GiB였다. 모델 파일을 읽으며 쌓인 페이지 캐시가
  포함된 값이고, 이 캐시도 컨테이너의 64 GiB 제한에 잡힌다 (필요하면 자동으로 회수된다).
- LoRA 가중치는 fp32라서, bf16 가중치를 직접 학습할 때 생기는 반올림 손실 문제는 없다.

## 7. A4 로거

`infra/dgx_spark/a4_logger.py`는 한 optimizer step 안에서 태스크 그래디언트가 투영과
optimizer를 거치며 안전 그래디언트와 충돌하는지를 기록한다.

| 기호 | 뜻 | 충돌 판정 |
| --- | --- | --- |
| `g_s` | 안전 손실의 그래디언트 | 기준 벡터 |
| `g_u` | 투영 전 태스크 그래디언트 | `<g_s, g_u> < 0` |
| `g_u'` | 투영 후 태스크 그래디언트 | `<g_s, g_u'> < 0` |
| `Δθ` | optimizer step이 실제로 만든 변화 | `<g_s, Δθ> > 0` (부호가 반대) |

그래디언트는 `-g` 방향으로 움직이므로 내적이 음수일 때, 실제 업데이트는 내적이 양수일 때
안전 손실이 커진다. `reintroduced`는 `g_u'`는 충돌이 아닌데 `Δθ`가 충돌인 경우다.

```python
from a4_logger import A4Config, A4Logger, named_grads

a4 = A4Logger(run_dir / "a4.jsonl", A4Config(interval=8, always_first=4))
...
if a4.should_log(step):                # 처음 4 step 과 이후 8 step 마다만 기록
    before = a4.snapshot(params)       # optimizer.step() 직전의 파라미터
    optimizer.step()
    a4.log(step, g_s, g_u, g_u_proj, before, params)
```

기록은 JSONL이며 한 줄이 step 하나다. 전체(`global`)와 층별(`groups`)로 내적, 코사인,
노름, 충돌 여부가 들어간다. `preflight_lora.py --a4-demo`로 실제 모델에서 로거가 도는지
확인할 수 있다. 이때는 안전 배치 대신 무작위 배치를, SPF 대신 단순 투영을 쓰므로 기록된
값은 실험 결과가 아니다.

## 8. 평가

- vLLM은 기본 환경에 없다. PyTorch 버전을 따로 고정하므로 별도 가상환경에 설치하고,
  서버는 반드시 `vllm-serve <모델> ...`로 띄운다.
- `lm-eval`(GSM8K 등)은 기본 환경에 들어 있다.
- vLLM 기반 종합 평가 레포는 아직 정해지지 않았다. 정해지면 이 절에 실행법을 적는다.

## 9. 관리자용: 호스트에서 하는 일

팀원은 이 절의 작업을 할 수 없다. `infra/dgx_spark/`에서 실행한다.

| 순서 | 명령 | 하는 일 |
| --- | --- | --- |
| 1 | `sudo host/setup_workspace_image.sh` | 400GB 이미지 파일을 만들어 `/srv/safety-workspace`에 마운트 |
| 2 | `container/run_container.sh build` 후 `up` | 이미지 빌드, 컨테이너 생성 (재부팅 후 자동 시작) |
| 3 | `sudo host/setup_firewall.sh install` | 컨테이너에서 사설망·개인 Tailscale 대역·호스트로 가는 트래픽 차단 |
| 4 | `host/check_isolation.sh` | 차단이 실제로 동작하는지 점검. 인터넷만 "연결됨"이어야 한다 |

비밀 값은 `container/secrets/`에 둔다. 이 폴더는 git에서 제외된다.

| 파일 | 내용 |
| --- | --- |
| `secrets/team.env` | `HF_TOKEN=...`, `TS_AUTHKEY=...` (권한 600) |
| `secrets/public/authorized_keys` | 팀원 공개키, 한 줄에 하나. 고친 뒤 `container/run_container.sh reload-keys` |

- 팀원을 뺄 때는 `authorized_keys`에서 그 줄을 지우고 `reload-keys`를 실행한 뒤,
  Tailscale 관리 콘솔에서 기기 공유를 해제한다.
- Tailscale 관리 콘솔에서 컨테이너 기기만 팀원에게 공유한다. 컨테이너 기기에는 키 만료를
  끄고, 다른 개인 기기로 나가는 접근을 ACL로 막아 둔다.
- 호스트의 다른 서비스는 `127.0.0.1`에만 바인딩한다. Docker로 띄우는 서비스는
  `-p 127.0.0.1:8000:8000`처럼 주소를 붙인다. 주소 없이 `-p 8000:8000`으로 띄우면
  같은 네트워크의 모든 기기에 공개된다.
- 호스트에서 서빙을 함께 돌릴 때는 메모리를 40% 이하로 제한한다. 팀 컨테이너 64 GiB와
  합쳐 전체 121.7 GiB를 넘지 않게 하기 위해서다.
- 접속 기록은 `docker logs safety-team`에 남는다. 어떤 공개키로 들어왔는지 지문이 찍힌다.

한계도 적어 둔다. 컨테이너는 호스트와 커널, GPU 드라이버를 함께 쓰므로 가상 머신만큼
강한 경계는 아니다. GPU 메모리 상한은 프로세스 단위이고 사용자가 코드로 바꿀 수 있어서,
합계 64 GiB는 규칙과 모니터링으로 지킨다.

## 10. 자주 겪는 오류

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| `Permission denied (publickey)` | 공개키가 등록되지 않았거나 다른 키로 접속했다 | 관리자에게 등록을 확인하고 `IdentityFile`을 맞춘다 |
| 접속 주소를 찾지 못한다 | Tailscale이 꺼져 있거나 공유 초대를 수락하지 않았다 | Tailscale 로그인 상태와 기기 목록을 확인한다 |
| `Read-only file system` | `/workspace` 밖에 쓰려고 했다 | `/workspace` 아래에서 작업한다 |
| `sudo: command not found`, `apt` 실패 | 설계상 root 권한이 없다 | 시스템 패키지는 관리자에게 요청한다 |
| 60 GiB 근처에서 `CUDA out of memory` | 프로세스당 GPU 상한에 걸렸다 | microbatch를 줄이고 누적을 늘린다 (`RUNBOOK.md` 5절) |
| 여유가 있어 보이는데 `CUDA out of memory` | 다른 사람이 GPU를 쓰고 있거나 페이지 캐시가 크다 | `nvidia-smi`, `free -h`를 보고 채널에서 확인한다 |
| `torch.cuda.is_available()`이 `False` | x86용 torch를 설치했다 | `python -c "import torch; print(torch.__version__)"`이 `+cu130`으로 끝나는지 본다. 아니면 `uv sync`로 다시 맞춘다 |
| `Cannot access gated repo` (403) | 팀 토큰에 그 모델 권한이 없다 | 관리자에게 알린다 |
| `torch_dtype` 관련 경고 | transformers 5에서 인자 이름이 `dtype`으로 바뀌었다 | `from_pretrained(..., dtype=torch.bfloat16)` |
| 접속이 끊긴 뒤 작업이 사라졌다 | tmux 밖에서 실행했다 | 4절대로 tmux에서 실행한다 |
| 집·사내망 주소로 연결이 안 된다 | 사설망 대역은 의도적으로 막혀 있다 | 인터넷에 공개된 주소만 쓸 수 있다 |
