"""Freeze the human-approved A4 audit panel without activating it.

The command reads private proposed records and a Git-safe ID-only selection decision,
validates source hashes and category balance, then writes a new versioned private
directory. It never overwrites an existing directory or the active A4 inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import Any


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalise_prompt(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}")
        records.append(value)
    return records


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _jsonl_bytes(values: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for value in values
    )


def _message_text(record: dict[str, Any], role: str) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Record {record.get('audit_id')} has no messages list")
    matches = [message.get("content") for message in messages if message.get("role") == role]
    if len(matches) != 1 or not isinstance(matches[0], str) or not matches[0]:
        raise ValueError(f"Record {record.get('audit_id')} must have one non-empty {role} message")
    return matches[0]


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = Path(args.selection).resolve()
    proposed_path = Path(args.proposed).resolve()
    metadata_path = Path(args.metadata).resolve()
    one_shot_path = Path(args.one_shot).resolve()
    output_dir = Path(args.output_dir).resolve()

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    selection = _read_json(selection_path)
    selected_ids = selection.get("selected_audit_ids")
    if not isinstance(selected_ids, list) or len(selected_ids) != 16 or len(set(selected_ids)) != 16:
        raise ValueError("selected_audit_ids must contain 16 unique IDs")

    excluded = selection.get("excluded", {})
    excluded_ids = set(excluded.get("rejected", [])) | set(excluded.get("held", []))
    if excluded_ids.intersection(selected_ids):
        raise ValueError("Selected IDs overlap rejected or held IDs")

    proposed = {record["audit_id"]: record for record in _read_jsonl(proposed_path)}
    metadata = {record["audit_id"]: record for record in _read_jsonl(metadata_path)}
    missing = [audit_id for audit_id in selected_ids if audit_id not in proposed or audit_id not in metadata]
    if missing:
        raise ValueError(f"Selected IDs missing from proposed records or metadata: {missing}")

    category_counts: dict[str, int] = {}
    output_records: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    for audit_id in selected_ids:
        record = proposed[audit_id]
        meta = metadata[audit_id]
        prompt = _message_text(record, "user")
        response = _message_text(record, "assistant")
        prompt_sha256 = _sha256_bytes(_normalise_prompt(prompt).encode("utf-8"))
        response_sha256 = _sha256_bytes(response.encode("utf-8"))
        if prompt_sha256 != meta.get("prompt_sha256") or response_sha256 != meta.get("response_sha256"):
            raise ValueError(f"Source hash mismatch for {audit_id}")

        category = meta.get("assigned_category")
        if not isinstance(category, str):
            raise ValueError(f"Missing category for {audit_id}")
        category_counts[category] = category_counts.get(category, 0) + 1
        output_records.append({"messages": record["messages"]})
        manifest_records.append(
            {
                "audit_id": audit_id,
                "row_index": meta.get("row_index"),
                "category": category,
                "fixed_rank": meta.get("rank"),
                "human_decision": "approved",
                "prompt_sha256": prompt_sha256,
                "response_sha256": response_sha256,
            }
        )

    if len(category_counts) != 8 or any(count != 2 for count in category_counts.values()):
        raise ValueError(f"Expected 8 categories with 2 selected records each: {category_counts}")

    one_shot_bytes = one_shot_path.read_bytes()
    one_shot_value = json.loads(one_shot_bytes.decode("utf-8"))
    if not isinstance(one_shot_value, dict):
        raise ValueError("One-shot input must be a JSON object")

    audit_bytes = _jsonl_bytes(output_records)
    manifest = {
        "schema_version": "a4-reviewed-input-freeze-v1",
        "status": "frozen_pending_independence_check_and_dgx_preflight",
        "selection_decision": selection,
        "selected_records": manifest_records,
        "category_counts": category_counts,
        "source_files": {
            "proposed_sha256": _sha256_bytes(proposed_path.read_bytes()),
            "metadata_sha256": _sha256_bytes(metadata_path.read_bytes()),
            "selection_sha256": _sha256_bytes(selection_path.read_bytes()),
            "one_shot_source_sha256": _sha256_bytes(one_shot_bytes),
        },
        "output_files": {
            "audit.jsonl": _sha256_bytes(audit_bytes),
            "one_shot.json": _sha256_bytes(one_shot_bytes),
        },
        "activation_gates_remaining": [
            "projection_and_evaluation_overlap_check_passed",
            "dgx_preflight_passed",
        ],
    }
    manifest_bytes = _json_bytes(manifest)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        (temporary / "audit.jsonl").write_bytes(audit_bytes)
        (temporary / "one_shot.json").write_bytes(one_shot_bytes)
        (temporary / "selection_manifest.json").write_bytes(manifest_bytes)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "output_dir": str(output_dir),
        "selected_count": len(selected_ids),
        "category_count": len(category_counts),
        "audit_sha256": manifest["output_files"]["audit.jsonl"],
        "one_shot_sha256": manifest["output_files"]["one_shot.json"],
        "status": manifest["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--proposed", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--one-shot", required=True)
    parser.add_argument("--output-dir", required=True)
    result = freeze(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
