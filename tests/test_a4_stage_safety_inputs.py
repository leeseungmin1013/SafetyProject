"""Pure selection checks; no dataset download or GPU needed."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "a4_stage_safety_inputs.py"
SPEC = importlib.util.spec_from_file_location("a4_stage_safety_inputs", SCRIPT)
assert SPEC and SPEC.loader
stage_inputs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage_inputs)


def test_multilabel_candidate_keeps_all_categories() -> None:
    row = {
        "prompt": "How can I misuse a private account?",
        "response_0": "I can't help with that request.",
        "response_1": "Unsafe response placeholder",
        "is_response_0_safe": True,
        "is_response_1_safe": False,
        "response_1_harm_category": {
            "Cybercrime": True,
            "Privacy Violation": True,
            "Other": True,
        },
    }
    candidate = stage_inputs.eligible_candidate(17, row)
    assert candidate is not None
    assert candidate["target_categories"] == ["Cybercrime", "Privacy Violation"]
    assert candidate["all_harm_categories"] == ["Cybercrime", "Other", "Privacy Violation"]
    assert candidate["row_index"] == 17


def test_safe_label_without_explicit_refusal_is_not_eligible() -> None:
    row = {
        "prompt": "How can I misuse a private account?",
        "response_0": "Here are some generic tips.",
        "response_1": "Unsafe response placeholder",
        "is_response_0_safe": True,
        "is_response_1_safe": False,
        "response_1_harm_category": {"Cybercrime": True},
    }
    assert stage_inputs.eligible_candidate(17, row) is None


def test_rank_is_stable_under_prompt_normalization() -> None:
    assert stage_inputs.rank_key(7, "  TEST\tRequest  ") == stage_inputs.rank_key(7, "test request")
