from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(f"required environment variable is not set: {name}")

    return _ENV_PATTERN.sub(replace, value)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("top-level config must be a mapping")
    config = _expand_env(copy.deepcopy(raw))
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = ("runtime", "model", "data", "training", "spf", "a4", "artifact")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"missing config sections: {', '.join(missing)}")

    training = config["training"]
    world_size = int(config["hardware"]["devices"])
    expected = (
        int(training["per_device_microbatch"])
        * world_size
        * int(training["gradient_accumulation_steps"])
    )
    configured = int(training["effective_batch_examples"])
    if expected != configured:
        raise ValueError(
            "effective batch mismatch: "
            f"microbatch({training['per_device_microbatch']}) * devices({world_size}) * "
            f"accumulation({training['gradient_accumulation_steps']}) = {expected}, "
            f"configured={configured}"
        )
    if int(training["per_device_microbatch"]) != 1:
        raise ValueError("the A4 runner currently requires per_device_microbatch=1")
    if config["spf"]["conflict_gating"] not in {"paper_zero", "norm_scaled_tau"}:
        raise ValueError("spf.conflict_gating must be paper_zero or norm_scaled_tau")
    if config["training"]["optimizer_state_dtype"] not in {"fp32", "parameter"}:
        raise ValueError("training.optimizer_state_dtype must be fp32 or parameter")
    if config["hardware"]["optimizer_state_storage"] != "nvme_chunked_fp32":
        raise ValueError("A4 requires hardware.optimizer_state_storage=nvme_chunked_fp32")
    if config["training"]["optimizer_state_dtype"] != "fp32":
        raise ValueError("NVMe A4 requires fp32 optimizer state")
    if int(config["training"]["optimizer_chunk_elements"]) < 1:
        raise ValueError("optimizer_chunk_elements must be positive")
    if config["artifact"].get("checkpoint_policy") not in {
        "disabled_restart_run", "save_final_main_only"
    }:
        raise ValueError("artifact.checkpoint_policy must be disabled_restart_run or save_final_main_only")


def with_overrides(
    config: dict[str, Any], *, max_steps: int | None = None, probe_every: int | None = None
) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        updated["training"]["max_optimizer_steps"] = max_steps
    if probe_every is not None:
        if probe_every < 1:
            raise ValueError("probe_every must be positive")
        updated["a4"]["probe_every"] = probe_every
    return updated
