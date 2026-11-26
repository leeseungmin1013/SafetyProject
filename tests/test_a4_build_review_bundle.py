from __future__ import annotations

import json
from pathlib import Path

from scripts.a4_build_review_bundle import build_bundle


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_bundle_keeps_model_judgment_separate(tmp_path: Path) -> None:
    audits = []
    metadata = []
    judgments = []
    ids = []
    for index in range(64):
        audit_id = f"audit-{index:02d}"
        ids.append(audit_id)
        audits.append(
            {
                "audit_id": audit_id,
                "messages": [
                    {"role": "user", "content": f"prompt {index}"},
                    {"role": "assistant", "content": f"target {index}"},
                ],
            }
        )
        metadata.append(
            {
                "audit_id": audit_id,
                "assigned_category": "Cybercrime",
                "rank": index + 1,
                "row_index": index,
                "prompt_sha256": "a" * 64,
                "response_sha256": "b" * 64,
            }
        )
        judgments.append(
            {
                "row_index": index,
                "verdict": "eligible",
                "rationale": "reason",
                "defect_codes": [],
                "harmful_request": "yes",
                "assigned_category_fit": "yes",
                "target_is_safe_refusal_or_alternative": "yes",
                "target_contains_actionable_harm": "no",
                "natural_single_turn_pair": "yes",
                "suggested_category": "Cybercrime",
            }
        )

    _write_jsonl(tmp_path / "audit64.proposed.private.jsonl", audits)
    _write_jsonl(tmp_path / "audit64_metadata.proposed.jsonl", metadata)
    _write_jsonl(
        tmp_path / "model_calls.private.jsonl", [{"judgments": judgments}]
    )
    (tmp_path / "selection_manifest.proposed.json").write_text(
        json.dumps(
            {
                "a4_subset_ids": ids[:16],
                "dataset": {"id": "example", "revision": "fixed"},
                "status": "awaiting_one_human_full_text_review",
            }
        ),
        encoding="utf-8",
    )

    bundle = build_bundle(tmp_path)

    assert bundle["item_count"] == 64
    assert bundle["items"][0]["prompt"] == "prompt 0"
    assert bundle["items"][0]["model"]["rationale"] == "reason"
    assert bundle["items"][0]["a4_subset"] is True
    assert bundle["items"][63]["a4_subset"] is False
