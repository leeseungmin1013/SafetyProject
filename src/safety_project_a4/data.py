from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]


def _normalise_record(record: dict[str, Any]) -> tuple[str, str]:
    if "question" in record and "answer" in record:
        return str(record["question"]), str(record["answer"])
    if "prompt" in record and "response" in record:
        return str(record["prompt"]), str(record["response"])
    messages = record.get("messages")
    if isinstance(messages, list) and len(messages) == 2:
        if messages[0].get("role") == "user" and messages[1].get("role") == "assistant":
            return str(messages[0]["content"]), str(messages[1]["content"])
    raise ValueError("record must contain question/answer, prompt/response, or user+assistant messages")


def load_private_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
        raise ValueError(f"invalid JSON records in {source}")
    records = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} in {source} is not an object")
            records.append(value)
    if not records:
        raise ValueError(f"no records in {source}")
    return records


def _extract_token_ids(value: Any) -> list[int]:
    """Normalise token IDs across Transformers tokenizer return types."""
    if isinstance(value, dict) or hasattr(value, "keys"):
        try:
            return _extract_token_ids(value["input_ids"])
        except (KeyError, TypeError):
            pass
    if isinstance(value, Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(token_id, int) for token_id in value):
            return value
        if len(value) == 1 and isinstance(value[0], (list, tuple)):
            return _extract_token_ids(value[0])
    raise TypeError(f"tokenizer returned unsupported token IDs: {type(value).__name__}")


def encode_pair(tokenizer, question: str, answer: str, max_length: int) -> EncodedExample:
    user_message = [{"role": "user", "content": question}]
    full_messages = user_message + [{"role": "assistant", "content": answer}]
    prompt_text = tokenizer.apply_chat_template(
        user_message,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = _extract_token_ids(tokenizer(prompt_text, add_special_tokens=False))
    full_ids = _extract_token_ids(tokenizer(full_text, add_special_tokens=False))
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("assistant prefix mismatch; tokenizer chat template cannot be safely masked")
    if len(full_ids) > max_length:
        raise ValueError(
            f"encoded example has {len(full_ids)} tokens, exceeding max_sequence_length={max_length}; "
            "increase the configured limit instead of truncating"
        )
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if all(label == -100 for label in labels):
        raise ValueError("encoded example has no assistant tokens")
    return EncodedExample(full_ids, labels)


def encode_record(tokenizer, record: dict[str, Any], max_length: int) -> EncodedExample:
    question, answer = _normalise_record(record)
    return encode_pair(tokenizer, question, answer, max_length)


class CausalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, examples: Iterable[EncodedExample | dict[str, list[int]]]) -> dict[str, Tensor]:
        normalised = []
        for example in examples:
            if isinstance(example, EncodedExample):
                normalised.append(example)
            else:
                normalised.append(EncodedExample(example["input_ids"], example["labels"]))
        maximum = max(len(example.input_ids) for example in normalised)
        input_ids, labels, attention_mask = [], [], []
        for example in normalised:
            padding = maximum - len(example.input_ids)
            input_ids.append(example.input_ids + [self.pad_token_id] * padding)
            labels.append(example.labels + [-100] * padding)
            attention_mask.append([1] * len(example.input_ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: tensor.to(device=device, non_blocking=True) for name, tensor in batch.items()}
