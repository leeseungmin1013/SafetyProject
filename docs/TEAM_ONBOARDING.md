# 팀 온보딩과 역할

## 1. 첫 30분

새 팀원은 다음 순서로 확인한다.

1. `README.md`에서 범위와 no-LoRA 원칙 확인
2. `EXPERIMENT_PLAN.md`에서 연구 질문과 단계 확인
3. `EVALUATION_PROTOCOL.md`에서 지표와 분모 확인
4. `RUNBOOK.md`에서 실행·중단·재시도 규칙 확인
5. `DECISIONS.md`에서 아직 열려 있는 결정 확인
6. GitHub Issue에서 본인 역할과 첫 작업 선택

## 2. 권장 역할

| 역할 | 책임 | 승인해야 하는 변경 |
| --- | --- | --- |
| Research lead | 질문·가설·중단 기준 | 범위, 비교군, 성공 기준 |
| Training owner | 학습 설정과 checkpoint | optimizer, LR, batch, sharding |
| Data owner | split, 중복, manifest | 데이터 revision과 제외 규칙 |
| Safety eval owner | 공격과 ASR 평가 | behavior/attack/judge 변경 |
| Utility eval owner | CE와 task metric | prompt, parser, metric 변경 |
| Infra owner | GPU, storage, 환경 | backend, container, 비용 산정 |
| Audit owner | blinded human review | rubric와 adjudication |
| Reproducibility reviewer | manifest와 재실행 | 결과 PR 최종 검토 |

한 사람이 여러 역할을 맡을 수 있지만, 자신의 장시간 run 결과를 혼자 승인하지 않는다.

## 3. GitHub 설정 권장값

저장소 생성 후 관리자가 설정한다.

- visibility: private
- default branch: `main`
- branch protection 또는 ruleset
  - pull request 필수
  - approving review 최소 1개
  - stale approval dismiss
  - force push 및 branch deletion 금지
- Issues, Discussions 활성화
- Actions 권한 최소화
- secret scanning과 push protection 사용 가능 시 활성화

팀원이 조직 team으로 관리되면 개인별 초대보다 team 단위 `write` 권한을 권장한다. 외부 검토자는 `triage` 또는 `read`, 저장소 관리자는 최소 인원만 `admin`을 준다.

## 4. 회의와 기록

- 매주 1회 30분 experiment review
- 장시간 run 시작 전 비동기 config review
- 모든 조건 변경은 Issue와 `DECISIONS.md`에 기록
- 구두 합의만으로 봉인된 test를 열거나 평가 조건을 변경하지 않음
- 실패 run도 주간 review에 포함

## 5. Issue 라벨 권장

- `experiment`
- `evaluation`
- `data`
- `infrastructure`
- `documentation`
- `blocked`
- `needs-decision`
- `invalid-run`
- `cost-review`
- `security-sensitive`

## 6. 팀원 초대에 필요한 정보

저장소 관리자는 다음 중 하나를 수집한다.

- 개인 초대: GitHub username과 권한 수준
- 조직 초대: organization 이름, team slug, 권한 수준

권장 기본 권한은 `write`다. Admin은 branch rule, secret, collaborator 관리를 맡는 최소 인원으로 제한한다.

## 7. 첫 번째 업무 분배 예시

- Data owner: Dolly split manifest와 중복 검사
- Training owner: Qwen 10-step full AdamW preflight
- Infra owner: 2×/4×80GB capacity profile
- Utility owner: 5개 task prompt/parser 고정
- Safety owner: Tier 0 behavior/attack/judge pipeline
- Audit owner: refusal/harmfulness rubric calibration
- Research lead: open decision 종료 및 P1 승인
