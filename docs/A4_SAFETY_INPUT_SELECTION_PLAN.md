# A4 safety 입력 선정·동결 계획

**상태:** Audit 64개 모델 보조 제안본·검토 workbook 생성 완료, 사람 검토·중복 검증·활성화 전 — projection/Audit 실사용 파일은 아직 교체하지 않았고 GPU 실험도 시작하지 않음  
**범위:** A4의 one-shot safety gradient 1개, A4가 관리하는 독립 Audit 64개, 그 안의 A4 gradient/CE 패널 16개  
**상세 절차:** [A4 중심 Safety Audit 64개 선정·검토 계획](SAFETY_AUDIT64_MODEL_ASSISTED_REVIEW_PLAN.md)이 Audit 선정·검토에 대한 최신 명세이며, 이 문서와 충돌하는 기존 16개 직접 선정 규칙을 대체한다.

## 1. 바로잡을 점

기존 `/workspace/private/a4/one_shot.json`과 `audit.jsonl`은 작성자가 만든 임시 입력이다. `preflight --core-only` 통과는 파일 개수, JSON 구조, 정확한 레코드 중복, 메모리·디스크만 확인한 결과이지 연구 입력의 타당성을 보증하지 않는다. 이 두 파일을 그대로 A4 본 실험에 쓰지 않는다. 기존 파일은 해시와 함께 private 보관 경로에 보존하고, 새 파일이 검증되기 전에는 덮어쓰지 않는다.

SPF 원 논문의 one-shot 쌍은 비공개가 아니라 논문 §5.1, PDF 9쪽에 공개되어 있다. 원문은 이 위치에서 직접 전사한다. 원시 프롬프트와 응답은 이 저장소에 적지 않는다. 논문은 safety sample 유형에 따라 범주 간 전이가 달라질 수 있음을 §5.6에서 보고하므로, 기존 임의의 계정 침입 쌍으로 원 논문 조건을 대체하지 않는다.

## 2. 출처와 고정 범위

| 용도 | 사전 고정 출처 | 출력 |
| --- | --- | --- |
| Projection | [SPF 논문 arXiv:2601.10141v1, §5.1](https://arxiv.org/pdf/2601.10141v1)의 Algorithm 1 safety sample | 정확히 1개 `messages` JSON |
| Audit | [PKU-Alignment/PKU-SafeRLHF 데이터 카드](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF), `default/train`, revision `9421ffafec3fa40a1f1a7d567b4d525079477ecb` | 정확히 16개 `messages` JSONL |

Audit 후보 풀은 원 논문의 projection 출처와 다른 공개 데이터셋이다. PKU-SafeRLHF에는 두 응답의 안전성 레이블과 harm-category 레이블이 있으며, 데이터 라이선스는 **CC BY-NC 4.0**이다. 사용자는 이 A4가 비상업 연구임을 확인했다. 용도가 바뀌면 이용 조건을 다시 검토한다. PKU 레이블은 *응답*의 안전성에 관한 것이므로, 안전 레이블만으로 유해 프롬프트나 적절한 거부를 자동 확정하지 않는다.

## 3. Audit 64개 사전 선정 규칙

1. `default/train`의 고정 revision만 사용하고 원본 row index를 보존한다. `default/test`는 선정에 쓰지 않는다.
2. 응답별 안전 레이블은 후보 추출 신호로만 쓰고, 안전한 쪽을 target response 후보로 사용한다. unsafe response 본문은 출력 파일이나 모델 판정 입력에 넣지 않는다.
3. 고정된 8개 위험 범주에서 각 8개, 총 64개를 구성하고 범주별 4개씩 총 32개를 예비 후보로 둔다. A4가 실제 사용하는 16개는 각 범주 2개씩 64개 안에서 사전 지정한다.
4. 기계 필터·Qwen 고정 tokenizer의 576토큰 제한·정규화 exact 중복 검사를 통과한 후보만 모델 보조 판정에 보낸다. 원문을 축약·수정해 통과시키지 않는다.
5. 모델 보조 판정은 고정 rubric과 structured output을 사용하고, 모델 점수나 A4 실험 지표로 후보 순위를 바꾸지 않는다. 후보·예비 후보의 순서는 고정 해시로 정한다.
6. 사람 검토자 1명이 최종 64개의 prompt와 target response 전체 원문을 각각 `승인/반려/보류`로 판정한다. 모델 근거는 사람의 1차 판정 뒤에만 대조한다. 반려·보류는 활성화하지 않으며, 예비 후보도 같은 전수 검토를 거친다.

## 4. 독립성 및 누출 점검

- Projection과 Audit 사이, Audit 내부에서 **prompt 기준** 정규화 exact match와 유사 문항 후보를 확인한다. 기존 코드의 전체 JSON hash 비교만으로는 같은 프롬프트/다른 응답을 걸러낼 수 없다.
- 기존 프로젝트 규약에 따라 character n-gram 및 embedding retrieval로 유사 후보를 찾고 사람이 의미 중복을 판정한다. 최소한 HEx-PHI와 A4에 결합할 안전성 평가셋(JailbreakBench, WildJailbreak, StrongREJECT, XSTest)의 확정된 prompt ID와 대조한다. 평가셋은 변경하지 않고 겹치는 Audit 후보를 제외한다.
- 아직 확보하지 못한 평가셋은 `미검증`으로 기록한다. 모든 의도된 평가셋과의 분리가 확인되기 전에는 Audit을 완전한 독립 평가라고 부르거나 본 실험 해석을 확정하지 않는다.
- Audit은 optimizer 업데이트에 사용하지 않는다. 다만 매 step 진단에 관찰되므로 최종 행동 평가용 완전한 blind test가 아니라 **독립 입력의 진단 패널**로 표기한다.

## 5. 생성·검증·동결 산출물

비공개 파일은 저장소 밖 `/workspace/private/a4/`에만 두고 Linux 실행 환경에서 디렉터리 `700`, 파일 `600`을 유지한다. 원문 전사와 후보 검토가 끝나면 다음을 만든다.

| 파일 | 내용 |
| --- | --- |
| `one_shot.json` | 논문 §5.1의 user/assistant 원문 1쌍. UTF-8, Qwen chat-template 입력 형식 |
| `audit64.jsonl` | 사람이 승인한 64쌍. 범주별 8개, 고정 순서 |
| `audit.jsonl` | 승인된 64개 안에서 사전 지정한 A4용 16쌍. 범주별 2개, 고정 순서 |
| `selection_manifest.json` | 논문 버전, 데이터셋 ID/revision/split, 선정 알고리즘·모델 판정 버전, 64개 및 A4 16개 ID·row index·범주·정규화 prompt hash·응답 hash, 검토자 판정, 제외·대체 사유, 생성시각, 파일 SHA256, 라이선스 확인 결과. 원시 문항은 넣지 않음 |

원문과 JSON 직렬화(Unicode, 공백, 줄바꿈)를 확정한 후 세 파일의 SHA256을 기록한다. 코드 설정의 `projection_source`를 기존 대체 입력 표시에서 논문 원문 출처로 수정하고, run manifest에는 입력 두 파일과 선정 manifest의 hash를 남긴다. 파일 또는 선정 규칙이 바뀌면 새 버전·새 hash·새 run ID를 사용하며 이전 결과에 덮어쓰지 않는다. 원본 데이터셋 덤프와 raw API response는 Git에 올리지 않는다. 64개 원문과 구조화된 모델 판정의 유일한 Git 예외는 팀 전용 링크 키로 보호되는 공개 Site 저장소 `sites/safety-audit-review/`이며, 주 실험 저장소의 추적 파일에는 넣지 않는다.

## 6. 실행 게이트와 완료 조건

1. 논문 원문 전사 대조, Audit 64개 원문 전수 검토, 라이선스 확인, 평가셋 중복 검사를 완료한다.
2. `preflight --core-only`로 정확한 개수·중복·경로·디스크를 확인하고, 추가된 prompt-level 중복 및 manifest 일치 검사를 통과한다.
3. 팀 GPU 사용 공지 후 전체 `preflight --load-model`에서 토큰 길이·CUDA·모델 로드를 확인한다.
4. 새 입력 hash를 고정한 동일 설정으로 5-step smoke → 20-step pilot을 순서대로 실행한다. 파일럿 지표를 보고 Audit 문항을 교체하지 않는다. 입력 오류가 발견되면 실패 run을 보존하고 새 버전으로 처음부터 재실행한다.

**현재 단계의 완료:** 출처·선정·중단 기준 문서화와 비공개 후보 생성까지다. 기존 private 파일과 DGX 결과는 바꾸지 않았고 GPU 작업도 시작하지 않았다.

## 7. 현재 생성·검토 상태 (2026-11-22)

- `scripts/a4_stage_safety_inputs.py`가 논문 원문 전사 파일의 SHA256을 검사한다. `scripts/a4_prepare_audit64.py`가 고정 데이터셋 revision에서 후보를 재현 가능하게 생성하고 모델 보조 판정을 기록한다. 원본 덤프와 raw API response는 저장소에 쓰지 않으며, 64개 원문·구조화 판정만 팀 전용 링크 키로 보호되는 공개 Site 저장소에 포함한다.
- 로컬 private `C:\workspace\private\a4\staging-v3\one_shot.proposed.json`: 논문 원문 전사, SHA256 `c74f0a2f75ffd16d010f96df3904b301ca90735407c6a960b8d242779f2dabf7`.
- 로컬 private `C:\workspace\private\a4\audit64-v1\`: 모델 보조 제안 64개, 예비 후보 32개, A4 부분집합 ID 16개, 판정·비용·선정 manifest가 있다. manifest 상태는 `awaiting_one_human_full_text_review`다.
- 1,024개 후보 판정의 누계는 입력 232,158토큰, 출력 144,381토큰, 추정 비용 USD 7.632504다. 제안 64개는 8범주×8, A4 부분집합은 8범주×2로 구성했다.
- 정규화 exact와 character 4-gram 내부 검사는 통과했다. 이는 사람의 원문 승인, embedding 의미 중복, Projection·행동 평가셋 분리를 완료했다는 뜻이 아니다.
- 검토 workbook은 `outputs/safety-audit64-20261122/Safety_Audit64_Human_Review.xlsx`다. 한 건씩 검토하는 로그인 없는 팀 링크 웹 앱은 `sites/safety-audit-review/`에 있다. 사용자 요청에 따라 64개 원문과 구조화된 모델 판정은 Site 소스·배포물에도 포함하고, 전체 링크의 고엔트로피 키를 제시한 요청만 API에 접근할 수 있다. 검토 상황은 팀 공용 D1 상태로 즉시 저장한다. 전체 링크를 가진 사람은 누구나 보고 수정할 수 있으므로 링크를 비밀로 관리한다. API 키와 raw API response는 올리지 않는다. 사람 검토·평가셋 중복 확인·입력 해시 동결을 마치기 전까지 `/workspace/private/a4/one_shot.json`, `/workspace/private/a4/audit.jsonl`을 교체하지 않는다. 5-step/20-step 실행도 하지 않는다.

### 이후 교체 절차 (아직 실행하지 않음)

1. 검토자 1명은 workbook의 64개 prompt와 target response 전체를 읽고 각 쌍을 `승인/반려/보류`로 판정한다. 모델 판정은 1차 판정 이후에만 대조한다. 반려·보류가 있으면 고정 순서의 예비 후보를 같은 방식으로 검토한다.
2. 승인 후보에 대해 prompt exact/character n-gram/embedding 유사 중복 및 모든 의도된 평가셋과의 중복을 검토한다. 충돌이 있으면 거절 사유를 남기고 **실험 지표를 보기 전**에만 예비 후보로 대체한다.
3. 승인 64개와 그 안의 A4 16개 ID, 최종 row index·순서, 논문 전사 대조 결과, 라이선스 확인, 파일별 SHA256을 private 최종 manifest에 고정한다. proposed 파일을 직접 수정하지 않고 새 버전 디렉터리에 최종 파일을 만든다.
4. DGX의 기존 `one_shot.json`/`audit.jsonl`은 해시와 함께 private 백업에 보존한다. 새 파일을 임시 경로로 전송·해시 대조한 뒤 같은 파일시스템 안에서 교체하고, 코드 설정의 projection 출처와 run manifest hash를 갱신한다. 교체 중 검증에 실패하면 기존 파일을 유지한다.
5. 교체 후 `preflight --core-only`를 재실행한다. 통과 전에는 GPU smoke/pilot/main을 실행하지 않는다. 입력을 다시 바꾸면 새 버전·새 run ID로 시작한다.

## 근거

- [SPF 원 논문 PDF](https://arxiv.org/pdf/2601.10141v1): §5.1 원문 safety sample, §5.6 sample 유형별 ablation, Appendix B 안전 데이터 구성.
- [PKU-SafeRLHF 데이터 카드](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF): 응답별 안전성·범주 레이블, 데이터 구성, CC BY-NC 4.0.
- [프로젝트 평가 규약](EVALUATION_PROTOCOL.md), [중복 방지 원칙](EXPERIMENT_PLAN.md): 평가셋 분리와 유사 문항 확인.
