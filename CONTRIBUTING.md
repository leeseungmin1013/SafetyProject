# 기여 가이드

## 작업 흐름

1. Issue를 만들고 실험 ID 또는 변경 목적을 적는다.
2. `main`에서 짧은 feature branch를 만든다.
3. 문서·설정·코드 변경을 같은 pull request에 섞지 않는 것을 원칙으로 한다.
4. 실험 조건이 바뀌면 `docs/DECISIONS.md`와 해당 config를 함께 갱신한다.
5. 최소 1명의 review와 자동 검사를 통과한 뒤 squash merge한다.

권장 branch 이름:

- `experiment/<run-id>`
- `eval/<benchmark>`
- `infra/<topic>`
- `docs/<topic>`
- `fix/<topic>`

## 실험 재현성 규칙

모든 실행은 다음을 남겨야 한다.

- run ID
- Git commit SHA와 dirty 여부
- 모델 repository와 immutable revision
- 데이터셋 repository, revision, split manifest hash
- 완전한 설정 파일
- seed 및 RNG 상태
- GPU 모델·수, driver, CUDA, PyTorch, Transformers, LLaMA-Factory 버전
- 시작/종료 시각과 종료 상태
- peak allocated/reserved/NVML VRAM
- optimizer step 수와 실제 처리 토큰 수

결과를 재실행했다면 기존 파일을 덮어쓰지 말고 새 run ID를 발급한다.

## Pull request 검토 체크리스트

- 파일럿의 no-LoRA 범위를 위반하지 않는가?
- Initial과 AdamW가 동일한 prompt/template/evaluation 조건을 사용하는가?
- 평가 test를 보고 하이퍼파라미터를 선택하지 않았는가?
- 서로 다른 토크나이저의 절대 CE를 모델 간 직접 비교하지 않는가?
- 실패·누락·길이 제한을 ASR 성공/실패로 임의 처리하지 않는가?
- 유해 원문이나 credential이 Git에 포함되지 않았는가?
- 비용 증가가 예상되면 GPU-hours 추정이 있는가?

## 결과 수정

평가 버그가 발견되면 기존 결과를 삭제하지 않는다. 해당 run을 `invalid`로 표시하고 원인, 영향 범위, 대체 run을 연결한다.
