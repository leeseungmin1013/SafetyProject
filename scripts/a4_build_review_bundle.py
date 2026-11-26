"""Build a browser-local review bundle for the proposed Safety Audit 64.

The output contains private prompt and target text and must remain outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PRIVATE_DIR = Path(r"C:\workspace\private\a4\audit64-v1")
DEFAULT_OUTPUT = Path(
    r"C:\workspace\SafetyProject\outputs\safety-audit64-20261122"
    r"\Safety_Audit64_Review_Bundle.private.json"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _content(messages: list[dict[str, Any]], role: str) -> str:
    for message in messages:
        if message.get("role") == role:
            content = message.get("content", "")
            if isinstance(content, str):
                return content
    raise ValueError(f"missing {role!r} message")


def build_bundle(
    private_dir: Path,
    privacy: str = "private-local-review-data-do-not-commit",
) -> dict[str, Any]:
    audits = _read_jsonl(private_dir / "audit64.proposed.private.jsonl")
    metadata = _read_jsonl(private_dir / "audit64_metadata.proposed.jsonl")
    manifest_path = private_dir / "selection_manifest.proposed.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    metadata_by_id = {row["audit_id"]: row for row in metadata}
    judgments_by_row: dict[int, dict[str, Any]] = {}
    for call in _read_jsonl(private_dir / "model_calls.private.jsonl"):
        for judgment in call.get("judgments", []):
            judgments_by_row[int(judgment["row_index"])] = judgment

    a4_ids = set(manifest["a4_subset_ids"])
    items: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    for audit in audits:
        audit_id = audit["audit_id"]
        meta = metadata_by_id[audit_id]
        row_index = int(meta["row_index"])
        judgment = judgments_by_row.get(row_index)
        if judgment is None:
            raise ValueError(f"missing model judgment for row {row_index}")
        category = meta["assigned_category"]
        category_counts[category] = category_counts.get(category, 0) + 1
        items.append(
            {
                "audit_id": audit_id,
                "row_index": row_index,
                "category": category,
                "category_rank": category_counts[category],
                "a4_subset": audit_id in a4_ids,
                "prompt": _content(audit["messages"], "user"),
                "target_response": _content(audit["messages"], "assistant"),
                "prompt_sha256": meta["prompt_sha256"],
                "response_sha256": meta["response_sha256"],
                "model": {
                    "verdict": judgment["verdict"],
                    "rationale": judgment["rationale"],
                    "defect_codes": judgment["defect_codes"],
                    "harmful_request": judgment["harmful_request"],
                    "assigned_category_fit": judgment["assigned_category_fit"],
                    "target_is_safe_refusal_or_alternative": judgment[
                        "target_is_safe_refusal_or_alternative"
                    ],
                    "target_contains_actionable_harm": judgment[
                        "target_contains_actionable_harm"
                    ],
                    "natural_single_turn_pair": judgment["natural_single_turn_pair"],
                    "suggested_category": judgment["suggested_category"],
                },
            }
        )

    if len(items) != 64:
        raise ValueError(f"expected 64 audits, got {len(items)}")

    return {
        "schema_version": "safety-audit-review-bundle-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "privacy": privacy,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "dataset": manifest["dataset"],
        "status": manifest["status"],
        "item_count": len(items),
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--hosted",
        action="store_true",
        help="Mark the bundle as embedded in the owner-private review Site.",
    )
    args = parser.parse_args()

    privacy = (
        "embedded-in-owner-private-site-source-and-deployment"
        if args.hosted
        else "private-local-review-data-do-not-commit"
    )
    bundle = build_bundle(args.private_dir, privacy=privacy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"path": str(args.output), "items": 64, "sha256": digest}))


if __name__ == "__main__":
    main()
