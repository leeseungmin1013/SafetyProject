"""Stage, but never activate, provenance-pinned A4 safety inputs.

Raw prompts and responses are written only under a private directory outside Git.
This script deliberately does not replace the active inputs: the proposed Audit
panel still needs two independent human reviews and evaluation-set overlap checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DATASET_ID = "PKU-Alignment/PKU-SafeRLHF"
DATASET_REVISION = "9421ffafec3fa40a1f1a7d567b4d525079477ecb"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"
DATASET_LICENSE = "cc-by-nc-4.0"
ALGORITHM = "a4-audit-v3"
PROJECTION_FILE_SHA256 = "c74f0a2f75ffd16d010f96df3904b301ca90735407c6a960b8d242779f2dabf7"
CATEGORIES = (
    "Cybercrime",
    "Privacy Violation",
    "Violence",
    "Physical Harm",
    "Drugs",
    "Economic Crime",
    "White-Collar Crime",
    "Psychological Harm",
)


def normalise_prompt(prompt: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def rank_key(row_index: int, prompt: str) -> str:
    return digest(f"{ALGORITHM}|{row_index}|{normalise_prompt(prompt)}")


_REFUSAL = re.compile(
    r"\b(?:i\s+(?:cannot|can['’]t|will\s+not|won['’]t|am\s+unable\s+to)\b"
    r"|i['’]m\s+unable\s+to\b|i\s+(?:do\s+not|don['’]t)\s+(?:provide|assist|help)\b)",
    re.IGNORECASE,
)


def has_explicit_refusal(response: str) -> bool:
    """A high-precision candidate filter, not a substitute for human review."""
    return bool(_REFUSAL.search(response[:240]))


def eligible_candidate(row_index: int, row: dict[str, Any]) -> dict[str, Any] | None:
    safe_flags = (row.get("is_response_0_safe"), row.get("is_response_1_safe"))
    if safe_flags not in ((True, False), (False, True)):
        return None
    safe_id = 0 if safe_flags[0] else 1
    unsafe_id = 1 - safe_id
    harm = row.get(f"response_{unsafe_id}_harm_category")
    if not isinstance(harm, dict):
        return None
    target_categories = [name for name in CATEGORIES if harm.get(name) is True]
    if not target_categories:
        return None
    prompt = row.get("prompt")
    response = row.get(f"response_{safe_id}")
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    if not isinstance(response, str) or not response.strip():
        return None
    if not has_explicit_refusal(response):
        return None
    return {
        "row_index": row_index,
        "target_categories": target_categories,
        "all_harm_categories": sorted(name for name, value in harm.items() if value is True),
        "safe_response_id": safe_id,
        "rank": rank_key(row_index, prompt),
        "prompt": prompt,
        "response": response,
        "prompt_sha256": digest(normalise_prompt(prompt)),
        "response_sha256": digest(response),
    }


def _record(prompt: str, response: str) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]}


def _pair(record: dict[str, Any]) -> tuple[str, str]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("projection must have exactly one user and one assistant message")
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise ValueError("projection roles must be user, assistant")
    prompt, response = messages[0].get("content"), messages[1].get("content")
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(response, str) or not response.strip():
        raise ValueError("projection prompt and response must be nonempty strings")
    return prompt, response


def _outside_repository(path: Path) -> None:
    repository = Path(__file__).resolve().parent.parent
    if path.resolve().is_relative_to(repository):
        raise ValueError("private output must be outside the repository")


def _write_exclusive(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _jsonl_bytes(values: Iterable[Any]) -> bytes:
    return b"".join(_json_bytes(value) for value in values)


def stage(projection_path: Path, output_dir: Path, *, noncommercial_confirmed: bool) -> dict[str, Any]:
    if not noncommercial_confirmed:
        raise ValueError("PKU-SafeRLHF is CC BY-NC 4.0; confirm noncommercial research use")
    _outside_repository(projection_path)
    _outside_repository(output_dir)
    if file_digest(projection_path) != PROJECTION_FILE_SHA256:
        raise ValueError("projection bytes do not match the verified paper transcription")
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    projection_prompt, _ = _pair(projection)
    projection_hash = digest(normalise_prompt(projection_prompt))
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from safety_project_a4.data import encode_record

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", revision="a09a35458c702b33eeacc393d103063234e8bc28"
    )
    encode_record(tokenizer, projection, 576)
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT, revision=DATASET_REVISION)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(dataset):
        candidate = eligible_candidate(index, row)
        if candidate is not None and candidate["prompt_sha256"] != projection_hash:
            for category in candidate["target_categories"]:
                by_category[category].append(candidate)

    queue_by_category: dict[str, list[dict[str, Any]]] = {}
    selected_by_category: dict[str, list[dict[str, Any]]] = {}
    exclusions: list[dict[str, Any]] = []
    seen_prompt_hashes = {projection_hash}
    # Allocate scarce categories first. A multi-label row can be assigned only once.
    allocation_order = sorted(CATEGORIES, key=lambda name: (len(by_category[name]), name))
    for category in allocation_order:
        candidates = sorted(by_category[category], key=lambda item: item["rank"])
        if len(candidates) < 4:
            raise ValueError(f"not enough eligible candidates for {category}: {len(candidates)}")
        category_queue: list[dict[str, Any]] = []
        for candidate in candidates:
            if candidate["prompt_sha256"] in seen_prompt_hashes:
                exclusions.append({"row_index": candidate["row_index"], "reason": "duplicate_prompt"})
                continue
            try:
                encode_record(tokenizer, _record(candidate["prompt"], candidate["response"]), 576)
            except ValueError as error:
                exclusions.append({"row_index": candidate["row_index"], "reason": str(error)})
                continue
            seen_prompt_hashes.add(candidate["prompt_sha256"])
            category_queue.append({**candidate, "category": category})
            if len(category_queue) == 4:
                break
        if len(category_queue) != 4:
            raise ValueError(f"not enough unique tokenizer-valid candidates for {category}: {len(category_queue)}")
        queue_by_category[category] = category_queue
        selected_by_category[category] = category_queue[:2]

    queue = [item for category in CATEGORIES for item in queue_by_category[category]]
    selected = [item for category in CATEGORIES for item in selected_by_category[category]]

    audit_path = output_dir / "audit.proposed.jsonl"
    queue_path = output_dir / "audit_review_queue.jsonl"
    manifest_path = output_dir / "selection_manifest.proposed.json"
    for path in (audit_path, queue_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite staged artifact: {path}")

    audit_bytes = _jsonl_bytes(_record(item["prompt"], item["response"]) for item in selected)
    queue_bytes = _jsonl_bytes(
        {
            "row_index": item["row_index"],
            "category": item["category"],
            "target_categories": item["target_categories"],
            "all_harm_categories": item["all_harm_categories"],
            "rank": item["rank"],
            "messages": _record(item["prompt"], item["response"])["messages"],
        }
        for item in queue
    )
    manifest = {
        "status": "proposed_not_approved",
        "projection_source": "arxiv:2601.10141v1 section 5.1",
        "projection_file_sha256": file_digest(projection_path),
        "projection_prompt_sha256": projection_hash,
        "audit_source": DATASET_ID,
        "audit_revision": DATASET_REVISION,
        "audit_config": DATASET_CONFIG,
        "audit_split": DATASET_SPLIT,
        "audit_license": DATASET_LICENSE,
        "noncommercial_research_confirmed": True,
        "selection_algorithm": ALGORITHM,
        "categories": list(CATEGORIES),
        "category_allocation_order": allocation_order,
        "category_eligibility_counts": {name: len(by_category[name]) for name in CATEGORIES},
        "selected": [
            {
                "row_index": item["row_index"],
                "category": item["category"],
                "target_categories": item["target_categories"],
                "all_harm_categories": item["all_harm_categories"],
                "safe_response_id": item["safe_response_id"],
                "rank": item["rank"],
                "prompt_sha256": item["prompt_sha256"],
                "response_sha256": item["response_sha256"],
            }
            for item in selected
        ],
        "machine_exclusions": exclusions,
        "human_reviews": "pending_two_independent_reviewers",
        "evaluation_overlap_check": "pending",
        "audit_file_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "review_queue_file_sha256": hashlib.sha256(queue_bytes).hexdigest(),
    }
    _write_exclusive(audit_path, audit_bytes)
    _write_exclusive(queue_path, queue_bytes)
    _write_exclusive(manifest_path, _json_bytes(manifest))
    return {
        "projection_sha256": manifest["projection_file_sha256"],
        "audit_sha256": manifest["audit_file_sha256"],
        "manifest_sha256": file_digest(manifest_path),
        "selected_count": len(selected),
        "review_queue_count": len(queue),
        "status": manifest["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--noncommercial-confirmed", action="store_true")
    args = parser.parse_args()
    print(json.dumps(stage(args.projection, args.output_dir, noncommercial_confirmed=args.noncommercial_confirmed), indent=2))


if __name__ == "__main__":
    main()
