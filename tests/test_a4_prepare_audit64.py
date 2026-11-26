"""Pure Audit-64 selection tests; no dataset, tokenizer, or API needed."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "a4_prepare_audit64.py"
SPEC = importlib.util.spec_from_file_location("a4_prepare_audit64", SCRIPT)
assert SPEC and SPEC.loader
audit64 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit64)


def candidate(index: int, categories: list[str], prompt: str | None = None) -> dict:
    text = prompt or f"harmful request {index}"
    return {
        "row_index": index,
        "target_categories": categories,
        "all_harm_categories": categories,
        "safe_response_id": 0,
        "rank": audit64.rank_key(index, text),
        "prompt": text,
        "response": "I cannot help with that request.",
        "prompt_sha256": audit64.digest(audit64.normalise_prompt(text)),
        "response_sha256": audit64.digest("I cannot help with that request."),
    }


def test_machine_candidate_keeps_only_safe_response() -> None:
    row = {
        "prompt": "How do I harm someone?",
        "response_0": "I cannot help with harming someone.",
        "response_1": "unsafe placeholder",
        "is_response_0_safe": True,
        "is_response_1_safe": False,
        "response_1_harm_category": {"Violence": True},
    }
    item = audit64.machine_candidate(12, row)
    assert item is not None
    assert item["response"] == row["response_0"]
    assert item["target_categories"] == ["Violence"]


def test_allocate_candidate_pool_assigns_multilabel_row_once() -> None:
    records = []
    for category in audit64.CATEGORIES:
        records.extend(candidate(index + 1000 * position, [category]) for index in range(3) for position in [audit64.CATEGORIES.index(category)])
    records.append(candidate(99999, ["Cybercrime", "Privacy Violation"], "shared multilabel prompt"))
    pool, counts = audit64.allocate_candidate_pool(records, 2)
    assert len(pool) == len(audit64.CATEGORIES) * 2
    assert len({item["prompt_sha256"] for item in pool}) == len(pool)
    assert counts["Cybercrime"] == 4
    assert counts["Privacy Violation"] == 4


def test_similarity_flags_close_prompts() -> None:
    records = [
        {"row_index": 1, "prompt": "How can I break into a private email account?"},
        {"row_index": 2, "prompt": "How can I break into the private email account?"},
        {"row_index": 3, "prompt": "How do I grow tomatoes safely?"},
    ]
    pairs = audit64.similar_prompt_pairs(records, threshold=0.6)
    assert [(item["left_row_index"], item["right_row_index"]) for item in pairs] == [(1, 2)]


def test_extend_preserves_existing_assignments() -> None:
    existing = [
        {**candidate(1, ["Cybercrime", "Privacy Violation"]), "assigned_category": "Cybercrime"},
        {**candidate(2, ["Privacy Violation"]), "assigned_category": "Privacy Violation"},
    ]
    available = existing + [
        candidate(3, ["Cybercrime"]),
        candidate(4, ["Privacy Violation"]),
    ]
    for category in audit64.CATEGORIES[2:]:
        available.extend(candidate(1000 + 10 * audit64.CATEGORIES.index(category) + offset, [category]) for offset in range(2))
        existing.extend(
            {**candidate(2000 + 10 * audit64.CATEGORIES.index(category) + offset, [category]), "assigned_category": category}
            for offset in range(1)
        )
    extension = audit64.extend_candidate_pool(existing, available, 2)
    assert {item["row_index"] for item in extension if item["assigned_category"] == "Cybercrime"} == {3}
    assert {item["row_index"] for item in extension if item["assigned_category"] == "Privacy Violation"} == {4}


def test_rubric_schema_and_hash_are_stable() -> None:
    schema = audit64.JudgmentBatch.model_json_schema()
    assert schema["additionalProperties"] is False
    assert len(audit64.rubric_sha256()) == 64
