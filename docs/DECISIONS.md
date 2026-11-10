# 결정 기록

실험 결과를 본 뒤 조건을 소급 변경하지 않도록 중요한 결정을 날짜순으로 기록한다.

## D-001 · 파일럿에서 LoRA 제외

- 날짜: 2026-11-10
- 상태: 확정
- 결정: LoRA, QLoRA, SafeLoRA, SaLoRA를 파일럿 및 이 저장소의 직접 실행 범위에서 제외한다.
- 이유: 초기 목적은 Initial 대비 일반 full-parameter AdamW의 ASR, task CE, 시간, VRAM을 파악하는 것이다.
- 영향: full parameter가 불가능해도 adapter 방식으로 대체하지 않고 capacity failure로 기록한다.

## D-002 · 모델 범위

- 날짜: 2026-11-10
- 상태: 확정
- 결정: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Gemma-3-12B-IT를 사용하고 Qwen3-30B MoE는 제외한다.
- 영향: exact repository revision은 실행 전 별도 고정한다.

## D-003 · 파일럿 핵심 비교

- 날짜: 2026-11-10
- 상태: 확정
- 결정: 각 모델의 Initial과 full SFT + AdamW를 paired 비교한다.
- 영향: 방어 방법 간 순위는 본 파일럿의 결론이 아니다.

## D-004 · 과잉 거부 포함

- 날짜: 2026-11-10
- 상태: 확정
- 결정: XSTest benign over-refusal을 필수 지표로 사용한다.
- 이유: 낮은 ASR이 무차별 거절로 인한 것인지 구분해야 한다.

## D-005 · 원격 저장소 visibility

- 날짜: 2026-11-10
- 상태: 잠정
- 결정: 팀 초대와 안전성 평가 자료를 고려해 private 저장소를 기본값으로 한다.
- 변경 조건: 사용자가 public 또는 조직 정책을 명시하면 변경한다.

## D-006 · 자동 judge와 OpenAI API 예산

- 날짜: 2026-11-10
- 상태: 확정
- 결정: 전체 sweep은 benchmark-native evaluator를 사용하고, 핵심 `MomentGuard vs One-shot` 확인군에는 OpenAI `gpt-6-astra`를 사용한다.
- 설정: `reasoning.effort=low`, structured JSON, 입력 최대 1,500 tokens, 출력 최대 256 tokens. 최종 대량 판정은 Batch API를 사용한다.
- 예산: 총 `$70` 중 calibration `$8`, 핵심 판정 `$45`, 불일치 재판정 `$10`, 예비비 `$7`. 예상 누적 비용 `$63`에서 신규 제출을 중단한다.
- 영향: 모든 방법·공격의 Cartesian product를 OpenAI judge로 판정하지 않는다.

## D-007 · 인원 및 일정 관리 제외

- 날짜: 2026-11-10
- 상태: 확정
- 결정: 인원 배정, 역할표, 회의 주기, 주차별 일정은 이 저장소에서 관리하지 않는다.
- 영향: 저장소에는 실험 순서, 의존성, 완료 조건과 독립 검수 규칙만 기록한다.

## 열린 결정

| ID | 결정 | 선택지 | 기한 | 상태 |
| --- | --- | --- | --- | --- |
| O-001 | GitHub owner | 개인 계정 / organization | 원격 협업 시작 전 | open |
| O-002 | GPU backend | 4×H100 80GB / 4×A100 80GB / 기타 | P0 | open |
| O-003 | sharding | DeepSpeed ZeRO-3 / FSDP | P0 | open |
| O-004 | LLaMA-Factory revision | immutable commit | P0 | open |
| O-005 | judge version | `gpt-6-astra` exact version ID와 prompt hash | P1 | open |
| O-006 | artifact storage | private bucket/volume과 접근 정책 | P0 | open |
| O-007 | audit | calibration set과 blinded audit rubric | P1 | open |
| O-008 | GuardSpace/SPA | full-parameter 재현 포함 여부 | 파일럿 종료 | open |
