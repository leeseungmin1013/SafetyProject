# A4 5-step smoke 실행 인계 — 2026-11-23

## 동결된 선택

- 사람 검토: 승인 55, 반려 4, 보류 5
- A4 panel: `configs/a4_safety_audit16_v1.json`의 고정 순서 16개
- 교체: `SA64-034→035`, `SA64-049→051`, `SA64-057→059`
- 반려·보류 9개는 panel에서 제외하며 실험 지표를 본 뒤 재선정하지 않는다.

## private 동결본

로컬과 DGX에 같은 세 파일을 준비했다. 기존 활성 `one_shot.json`과 `audit.jsonl`은 아직 교체하지 않았다.

- 로컬: `C:\workspace\private\a4\audit16-frozen-v1\`
- DGX: `/workspace/private/a4/audit16-frozen-v1/`
- `audit.jsonl`: SHA256 `7031df5fa93aaa6f27e1c76da7524c5156e16d9cb01f3d189f63ec22961b8efc`
- `one_shot.json`: SHA256 `c74f0a2f75ffd16d010f96df3904b301ca90735407c6a960b8d242779f2dabf7`
- `selection_manifest.json`: SHA256 `dc7dceb48a85b9fd2181c57db226a8e5b8d0213f0790d712463250e0b9ecc537`

동결본 생성기는 `scripts/a4_finalize_reviewed_selection.py`다. 16개 고유 ID, 8범주×2, 반려·보류와의 비중복, source/meta 존재, prompt/response hash 일치를 확인한 뒤 새 디렉터리에만 쓴다.

## 확인된 실행 환경

- SSH 별칭 `safety-gpu` 접속 성공
- 아키텍처 `aarch64`, GPU `NVIDIA GB10`
- 확인 시 GPU 프로세스 없음, GPU 사용률 0%
- 메모리: 121 GiB 중 117 GiB available
- `/workspace`: 393 GiB available
- DGX 저장소: `/workspace/projects/a4/SafetyProject`, Git `1a51201a5d83165783936a5a25db0aeb2e8950f3`, clean

## 실행 전 남은 게이트

1. 동결 16개와 Projection·확정 행동 평가셋의 exact/character n-gram/embedding 중복·의미 중복 확인을 완료하고 manifest 상태를 갱신한다.
2. 기존 DGX 활성 입력의 SHA256을 별도 기록·백업한다.
3. 환경 변수는 동결 디렉터리를 가리킨 상태로 먼저 full preflight를 실행한다. preflight 통과 전 활성 경로를 덮어쓰지 않는다.
4. full preflight가 통과하면 새 run ID와 로그 경로를 예약하고 5-step smoke만 실행한다.
5. smoke의 finite metric, 실제 parameter update, peak memory/time, 5개 `a4.jsonl` 행, spool cleanup을 확인하기 전 20-step pilot을 시작하지 않는다.

## 게이트 통과 후 준비 명령 (현재 실행 보류)

```bash
cd /workspace/projects/a4/SafetyProject
export A4_SAFETY_SAMPLE_PATH=/workspace/private/a4/audit16-frozen-v4-proposed/one_shot.json
export A4_AUDIT_PANEL_PATH=/workspace/private/a4/audit16-frozen-v4-proposed/audit.jsonl
export A4_OUTPUT_ROOT=/workspace/runs/qwen25-spf-a4
export A4_GRADIENT_SPOOL_ROOT=/workspace/runs/.a4-gradient-spool
export A4_OPTIMIZER_STATE_ROOT=/workspace/runs/.a4-optimizer-state
bash scripts/a4_preflight.sh
```

preflight 또는 smoke가 실패하면 설정을 임의로 낮추지 않고 실패 manifest와 로그를 보존한 뒤 새 run ID로 재시도한다.

## 2026-11-23 실행 재개 기록

- DGX 기존 활성 입력은 `/workspace/private/a4/active-backup-20261123-pre-smoke/`에 복사하고 `SHA256SUMS` 검증을 통과했다. 기존 활성 `one_shot.json` SHA256은 `4dcc3e3caa521a304b3a6d9470f1e8f915af4524cee17bd90e2a74b4dffa9c94`, `audit.jsonl`은 `6cca4ea9a5e4d245505b6fe19d58ea2f9b6c9fc5645a22d70fb1a1562864f869`이다. 활성 파일은 변경하지 않았다.
- private 동결 디렉터리의 `evaluation_overlap_manifest.json`에 평가 출처와 prompt ID/hash, 원본 파일 hash, 검색 결과를 기록했다. 로컬과 DGX 사본의 현재 SHA256은 `11f4a34e2eda320ec47f73424c492edeaaa21bea69929cad83a74f4f2bd9ce3e`로 같다. JailbreakBench harmful 100개(`886acc352a31533ffbcf4ef22c744658688086fc`), StrongREJECT 공식 원본 full 313개(`f7cad6c17e624e21d8df2278e918ae1dddb4cb56`), XSTest 전체 450개(`d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d`)를 고정했다. 동결 입력 17개와 이 세 평가셋의 정규화 exact 일치는 0건, character 4-gram Jaccard 최고값은 0.190476, MiniLM embedding cosine 최고값은 0.670439다. projection/audit 내부 136쌍의 exact 일치는 0건, 4-gram 최고값은 0.296296, embedding 최고값은 0.549392다.
- embedding cosine ≥0.60인 공개 평가셋 근접 후보 7쌍은 private manifest에 ID만 남겼고, 원문 쌍은 로컬 private `C:\workspace\private\a4\audit16-frozen-v1\semantic_review_candidates.json`에 별도로 준비했다. 에이전트의 원문 직접 검토에서는 SA64-035와 StrongREJECT 두 문항을 같은 핵심 요청으로 판정했다. 나머지 다섯 쌍은 관련 주제지만 요청 행동·대상이 달랐다. 이는 사람 검토 판정을 대체하지 않는다. SA64-035는 기존 panel에 그대로 두고 의미 중복 게이트를 막는다. 같은 범주의 이미 승인된 후순위 SA64-036~040은 공개 평가셋 exact 0건으로 예비 검사했으나, 동결본을 교체하거나 새로운 panel을 확정하지 않았다.
- DGX의 PJH720 인증으로 [HEx-PHI](https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI) revision `83128b46334b80cc567bd7a2caf7af11c5b0bab7`와 [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) revision `5ddc12a7894f842b0619b8e1c7ee496b198af009`의 공식 파일 요청이 모두 HTTP 200으로 성공했다. 따라서 접근 승인을 더 기다릴 필요는 없다. HEx-PHI 공개 300문항과 WildJailbreak eval 2,210문항을 DGX 메모리에서 비교했으며, 라이선스 대상 원문은 로컬이나 Git으로 복사하지 않았다. v1 동결 입력과의 정규화 exact 일치는 0건, character 4-gram Jaccard 최고값은 0.136111, MiniLM embedding cosine 최고값은 0.724533이다. 유사도가 높은 한 쌍 `SA64-051` 대 `HEx-PHI category_11.csv:row-006`은 의미 중복 판정이 남았다. ID와 점수만 `gated_overlap_manifest.json`에 기록하고 원문 검토 자료는 DGX private `gated_semantic_review.private.json`에만 보관했다.
- 공개 평가셋 의미 중복 후보 `SA64-035`를 같은 범주의 사전 승인 후순위 `SA64-036`으로 바꾼 **미확정 v2 후보**를 별도 생성했다. 로컬과 DGX의 `/workspace/private/a4/audit16-frozen-v2-proposed/`에 두었고 v1 및 DGX 활성 입력은 그대로다. v2 `audit.jsonl` SHA256은 `451a8b9f4e27ebe6a5b236beade5403cd464c0c05d8262a2ef5185329ef4dcc8`, `one_shot.json` SHA256은 v1과 같은 `c74f0a2f75ffd16d010f96df3904b301ca90735407c6a960b8d242779f2dabf7`이다. 공개 비교 exact 0건, 4-gram 최고 0.190476, embedding 최고 0.669136이고, 게이트 대상 비교 exact 0건, 4-gram 최고 0.136111, embedding 최고 0.724533이다. v2에서도 위 HEx-PHI 한 쌍의 의미 판정은 남는다.
- v2 후보를 명시한 DGX `--core-only` preflight는 별도 출력 경로 `/workspace/runs/qwen25-spf-a4-v2-preflight`에서 통과했다. `projection_and_evaluation_overlap_check_passed` 게이트는 **아직 미통과**다. 공개 평가셋 근접 후보의 사람 판정, HEx-PHI 한 쌍의 라이선스 보유자 판정, v2 교체 확정이 필요하다. full `--load-model` preflight, tmux smoke, run ID 예약은 시작하지 않았다. 새 동결 입력은 DGX 활성 경로를 덮어쓰지 않았다.

## 2026-11-23 추가 의미 중복 검토

- 사용자 요청에 따라 HEx-PHI 근접 쌍을 DGX 안의 Qwen2.5-3B-Instruct로 구조화 비교했다. 첫 출력은 요약과 boolean 판정이 서로 충돌했으므로 차이점을 명시하는 두 번째 검토를 수행했다. 두 번째 출력과 요약에 근거한 에이전트 판정은 `SA64-051` 대 `HEx-PHI category_11.csv:row-006`이 **같은 핵심 요청**이라는 것이다. 라이선스 대상 원문은 DGX 밖으로 내보내지 않았다. 이는 사람 판정으로 기록하지 않는다.
- 고정 순서의 같은 범주 승인 후보를 순서대로 검토했다. `SA64-052`는 JailbreakBench `35`와 같은 탈세·적발 회피 목적이어서 제외했다. `SA64-053`은 panel 내부 `SA64-041`과 같은 대출 사기 목적이어서 제외했다. `SA64-054`를 선택한 **미확정 v5 후보**는 로컬 `C:\workspace\private\a4\audit16-frozen-v5-proposed\`와 DGX `/workspace/private/a4/audit16-frozen-v5-proposed/`에 별도 저장했다. v1·v2와 DGX 활성 입력은 변경하지 않았다.
- v5 `audit.jsonl` SHA256은 `5785124cd18d09e6a938a6652c1a528b81c9838a5b10a002184e3aaf97f60783`, `one_shot.json`은 `c74f0a2f75ffd16d010f96df3904b301ca90735407c6a960b8d242779f2dabf7`이다. 공개 세 평가셋과의 exact 0건, 최고 4-gram 0.190476, embedding 0.669136이다. 공개 manifest SHA256은 `d04acf3f3cf6474ec9adf0597d3b67df1df7246a94d91c9e9208e25f6e143ae4`다. HEx-PHI·WildJailbreak와의 exact 0건, 최고 4-gram 0.136111, embedding 0.589103이며 ≥0.60 의미 검토 대상은 0건이다. 게이트 평가 manifest SHA256은 `e9a3774460933cfebc52d28384f452e0617f20e045c4493e0d643d19011f5b46`이다. 공개 남은 근접 쌍 5개는 에이전트가 행동·대상이 서로 다르다고 검토했으며 ID만 private `agent_semantic_review.json`에 기록했다.
- v5 후보 DGX `--core-only` preflight는 별도 출력 경로 `/workspace/runs/qwen25-spf-a4-v5-preflight`에서 통과했다. **사람의 의미 중복 판정과 v5 선정 확정 전**에는 `projection_and_evaluation_overlap_check_passed`를 통과로 표시하지 않는다. full `--load-model` preflight와 tmux smoke는 아직 시작하지 않았다. 사용자 요청대로 smoke가 시작되면 작업을 종료하고 결과 검토는 후속 요청 때 수행한다.

## 2026-11-24 사람 판정 반영

- 사용자는 표로 제시한 10쌍 중 `SA64-035` 대 StrongREJECT 137·138, `SA64-051` 대 HEx-PHI `category_11.csv:row-006`, `SA64-052` 대 JailbreakBench 35를 같은 핵심 요청으로 판정했다. 나머지 6쌍은 다른 요청으로 판정했다. HEx-PHI 문항은 라이선스 제한으로 원문 대신 요약을 제시한 판정이라는 한계를 `human_semantic_review.json`에 명시했다.
- `SA64-053` 대 내부 `SA64-041`이 다르다는 사람 판정에 따라 순서상 앞선 `SA64-053`을 선택하는 **v4 후보**로 되돌렸다. 추가로 이 후보와 XSTest 199도 사용자가 다른 요청으로 판정했다. 따라서 v5는 채택하지 않는다. v4 사람 판정 11쌍은 로컬과 DGX의 `/workspace/private/a4/audit16-frozen-v4-proposed/human_semantic_review.json`에 ID·결론만 저장했다.
- v4 `audit.jsonl` SHA256은 `bf604999f6bbaa4a30773477c99a02a2e25cc6df4e797590b8c0995272cac39b`, `one_shot.json`은 `c74f0a2f75ffd16d010f96df3904b301ca90735407c6a960b8d242779f2dabf7`, 선정 manifest는 `6fb6f1252124ae5d532d134ed50a0062b6d832ee34bd55d31e9fe84ab0b4f8e7`이다. 공개 3셋과 HEx-PHI·WildJailbreak 모두 exact 중복 0건이다. 공개 근접 6쌍과 panel 내부 근접 1쌍은 사람 판정 `다름`, gated 근접 0쌍이다. 검증 스크립트가 `overlap_gate_decision.json`에 `passed`를 기록했다.
- DGX v4 `--core-only` preflight 통과. `configs/qwen25_spf_a4.yaml`의 `projection_source`를 실제 출처 `arxiv:2601.10141v1 section 5.1`로 갱신했고 DGX에 동기화했다. full `--load-model` preflight는 별도 출력 경로 `/workspace/runs/qwen25-spf-a4-v4-full-preflight`에서 진행 중이다. 이것이 통과하기 전에는 tmux smoke를 시작하지 않는다.
- v4 full `--load-model` preflight가 `passed`로 종료했다. 7,615,616,512개 파라미터 전부 trainable이고 모델 로드 CUDA peak reserved는 14.354 GiB다. `a4-qwen25-7b-spf-s42-20261124-smoke-v4` 실행 ID, `a4-smoke-v4-20261124` tmux 세션, `/workspace/runs/qwen25-spf-a4-v4-smoke-20261124/tmux-smoke.log` 로그 경로를 `smoke_reservation.json`에 예약했다. run manifest에는 선정 manifest SHA256을 기록하도록 `train.py`를 갱신했다.
