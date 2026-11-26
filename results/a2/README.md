# A2 실험 결과 · Single Reference 대표성

Qwen2.5-7B-Instruct Initial checkpoint에서 Single Reference 5개를 Mixed Batch Reference 16개의 평균과 비교했다. GSM8K utility probe는 8개였고, probe마다 64문항을 사용했다. 8개 모두 완료했다. [실험 설계](../../docs/A2_EXPERIMENT_DESIGN.md)

| 지표 | 결과 | 사전 기준 |
|---|---:|---:|
| Median Frobenius overlap | 0.246 | ≥ 0.80 |
| Conflict decision agreement | 40/40 (100%) | ≥ 80% |
| Mixed Batch conflict | 0/8 | |

## Frobenius overlap

| Single Reference row ID | 196개 matrix overlap의 중앙값 |
|---:|---:|
| 5687 | 0.111 |
| 1574 | 0.230 |
| 62 | 0.246 |
| 883 | 0.257 |
| 206 | 0.337 |

위 5개 중앙값의 중앙값은 0.246이었다. 980개 matrix overlap의 산술평균은 0.270, 최댓값은 0.762였다.

## Conflict decision agreement

**`No conflict`**는 해당 GSM8K probe로 작은 업데이트를 한다고 가정했을 때, Reference의 안전 응답 loss가 증가할 것이라는 1차 gradient 신호가 없었다는 뜻이다. **`Agreement`**는 같은 probe에서 Single과 Mixed Batch의 판정이 같았다는 뜻이다.

Single 5개와 Mixed Batch는 8개 probe 모두에서 `no conflict`를 반환했다. 저장된 48개 utility–safety gradient inner product는 모두 양수였다. Mixed Batch가 `conflict`로 판정한 probe가 0개여서 해당 상황의 agreement는 평가할 수 없었다. 이번 run에서는 cosine similarity를 기록하지 않았다.

## 결론

이번 checkpoint와 Reference 구성에서 Single의 rank-10 safety subspace는 Mixed Batch에 대한 overlap 기준을 충족하지 못했다. 관측된 no-conflict 상황에서의 판정은 모두 일치했다.

[Layer별 overlap](layer_overlap.md) · [원본 report](report.json)
