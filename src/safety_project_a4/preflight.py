from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from .config import load_config
from .data import encode_record, load_private_records
from .io_utils import atomic_json, sha256_file, utc_now


GIB = 1024**3


def _record_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _memory_limit_bytes() -> int | None:
    candidates = (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text != "max":
            value = int(text)
            if value < 1 << 60:
                return value
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError):
            pass
    return None


def _ensure_outside_repository(path: Path, repository: Path) -> None:
    try:
        path.resolve().relative_to(repository.resolve())
    except ValueError:
        return
    raise ValueError(f"private safety data must be outside the Git repository: {path}")


def run_preflight(config_path: str, *, core_only: bool, load_model: bool, allow_low_memory: bool) -> dict[str, Any]:
    config = load_config(config_path)
    repository = Path(config["_config_path"]).parent.parent.resolve()
    runtime = config["runtime"]
    output_root = Path(runtime["output_root"]).resolve()
    spool_root = Path(runtime["gradient_spool_root"]).resolve()
    state_root = Path(runtime["optimizer_state_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    spool_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)

    safety_path = Path(config["data"]["safety"]["projection_sample_path"]).resolve()
    audit_path = Path(config["data"]["safety"]["audit_panel_path"]).resolve()
    if runtime["require_private_data_outside_repository"]:
        _ensure_outside_repository(safety_path, repository)
        _ensure_outside_repository(audit_path, repository)
        _ensure_outside_repository(spool_root, repository)
        _ensure_outside_repository(state_root, repository)
    if not safety_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("A4_SAFETY_SAMPLE_PATH and A4_AUDIT_PANEL_PATH must point to files")

    safety_records = load_private_records(safety_path)
    audit_records = load_private_records(audit_path)
    if len(safety_records) != 1:
        raise ValueError(f"projection safety file must contain exactly one record; found {len(safety_records)}")
    required_audit = int(config["data"]["safety"]["independent_audit_probes"])
    if len(audit_records) < required_audit:
        raise ValueError(f"audit panel needs at least {required_audit} records; found {len(audit_records)}")
    safety_hash = _record_hash(safety_records[0])
    audit_hashes = [_record_hash(record) for record in audit_records]
    if safety_hash in audit_hashes:
        raise ValueError("projection sample overlaps the independent audit panel")
    if len(set(audit_hashes)) != len(audit_hashes):
        raise ValueError("audit panel contains duplicate records")

    memory_limit = _memory_limit_bytes()
    required_host_memory = int(runtime["minimum_host_memory_gib"]) * GIB
    if memory_limit is not None and memory_limit < required_host_memory and not allow_low_memory:
        raise MemoryError(
            f"host container memory limit is {memory_limit / GIB:.1f} GiB; "
            f"chunked NVMe A4 requires at least {required_host_memory / GIB:.0f} GiB by policy"
        )
    required_disk = int(runtime["minimum_free_disk_gib"]) * GIB
    output_disk = shutil.disk_usage(output_root)
    spool_disk = shutil.disk_usage(spool_root)
    state_disk = shutil.disk_usage(state_root)
    for label, path, disk in (
        ("output", output_root, output_disk),
        ("gradient spool", spool_root, spool_disk),
        ("optimizer state", state_root, state_disk),
    ):
        if disk.free < required_disk:
            raise OSError(
                f"only {disk.free / GIB:.1f} GiB free under {label} path {path}; "
                f"at least {required_disk / GIB:.0f} GiB is required"
            )

    report: dict[str, Any] = {
        "checked_at": utc_now(),
        "status": "passed-core" if core_only else "passed",
        "config": str(Path(config_path).resolve()),
        "memory_limit_gib": None if memory_limit is None else round(memory_limit / GIB, 3),
        "output_disk_free_gib": round(output_disk.free / GIB, 3),
        "spool_disk_free_gib": round(spool_disk.free / GIB, 3),
        "state_disk_free_gib": round(state_disk.free / GIB, 3),
        "safety_sample_sha256": sha256_file(safety_path),
        "audit_panel_sha256": sha256_file(audit_path),
        "audit_records": len(audit_records),
    }
    if core_only:
        atomic_json(output_root / "preflight.json", report)
        return report

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not support BF16")
    expected_devices = int(config["hardware"]["devices"])
    if torch.cuda.device_count() != expected_devices:
        raise RuntimeError(
            f"configured devices={expected_devices}, visible CUDA devices={torch.cuda.device_count()}"
        )
    device_properties = torch.cuda.get_device_properties(0)
    current_fraction = float(torch.cuda.get_per_process_memory_fraction(0))
    budget_fraction = min(current_fraction, runtime["gpu_budget_gib"] * GIB / device_properties.total_memory)
    torch.cuda.set_per_process_memory_fraction(budget_fraction, 0)
    process_fraction = float(torch.cuda.get_per_process_memory_fraction(0))
    process_limit = int(device_properties.total_memory * process_fraction)
    free_cuda, total_cuda = torch.cuda.mem_get_info(0)
    effective_cuda_capacity = min(process_limit, free_cuda)
    minimum_cuda = int(runtime["minimum_free_cuda_gib"]) * GIB
    if effective_cuda_capacity < minimum_cuda and not allow_low_memory:
        raise MemoryError(
            f"effective CUDA capacity is {effective_cuda_capacity / GIB:.1f} GiB "
            f"(free={free_cuda / GIB:.1f}, process-limit={process_limit / GIB:.1f}); "
            f"chunked NVMe A4 requires at least {minimum_cuda / GIB:.0f} GiB by policy"
        )

    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["repository"], revision=model_config["tokenizer_revision"]
    )
    maximum = int(config["training"]["max_sequence_length"])
    encode_record(tokenizer, safety_records[0], maximum)
    for record in audit_records[:required_audit]:
        encode_record(tokenizer, record, maximum)

    utility = config["data"]["utility"]
    dataset = load_dataset(
        utility["repository"], utility["config"], split=utility["train_split"], revision=utility["revision"]
    )
    encoded_lengths = []
    for record in dataset:
        encoded = encode_record(tokenizer, record, maximum)
        encoded_lengths.append(len(encoded.input_ids))
    sorted_lengths = sorted(encoded_lengths)
    p99_index = min(len(sorted_lengths) - 1, int(len(sorted_lengths) * 0.99))
    report.update(
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "cuda_total_gib": round(total_cuda / GIB, 3),
            "cuda_free_gib": round(free_cuda / GIB, 3),
            "cuda_process_fraction": process_fraction,
            "cuda_process_limit_gib": round(process_limit / GIB, 3),
            "optimizer_state_storage": "nvme_chunked_fp32",
            "optimizer_chunk_elements": config["training"]["optimizer_chunk_elements"],
            "utility_records": len(dataset),
            "shortest_encoded_tokens": sorted_lengths[0],
            "p99_encoded_tokens": sorted_lengths[p99_index],
            "longest_encoded_tokens": sorted_lengths[-1],
            "encoded_records_over_512": sum(length > 512 for length in encoded_lengths),
        }
    )
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_config["repository"],
            revision=model_config["revision"],
            torch_dtype=torch.bfloat16,
            attn_implementation=model_config["attention_implementation"],
            low_cpu_mem_usage=True,
        ).to("cuda:0")
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if total != trainable:
            raise RuntimeError(f"not all model parameters are trainable: {trainable}/{total}")
        report.update(
            {
                "parameter_count": total,
                "trainable_parameter_count": trainable,
                "estimated_fp32_state_gib": round(total * 8 / GIB, 3),
                "estimated_three_bf16_gradient_spools_gib": round(total * 6 / GIB, 3),
                "model_load_peak_cuda_reserved_gib": round(
                    torch.cuda.max_memory_reserved(0) / GIB, 3
                ),
            }
        )
        del model
        torch.cuda.empty_cache()
        for label, path in (
            ("output", output_root),
            ("gradient spool", spool_root),
            ("optimizer state", state_root),
        ):
            remaining = shutil.disk_usage(path).free
            if remaining < required_disk:
                raise OSError(
                    f"only {remaining / GIB:.1f} GiB free under {label} after model download; "
                    f"at least {required_disk / GIB:.0f} GiB is required"
                )
        report.update(
            {
                "output_disk_free_gib": round(shutil.disk_usage(output_root).free / GIB, 3),
                "spool_disk_free_gib": round(shutil.disk_usage(spool_root).free / GIB, 3),
                "state_disk_free_gib": round(shutil.disk_usage(state_root).free / GIB, 3),
            }
        )

    atomic_json(output_root / "preflight.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Qwen SPF/A4 execution environment")
    parser.add_argument("--config", default="configs/qwen25_spf_a4.yaml")
    parser.add_argument("--core-only", action="store_true", help="skip CUDA, tokenizer, and dataset checks")
    parser.add_argument("--load-model", action="store_true", help="also load the full model and count parameters")
    parser.add_argument(
        "--allow-low-memory",
        action="store_true",
        help="diagnostic override; does not make a low-memory run scientifically valid",
    )
    args = parser.parse_args()
    try:
        report = run_preflight(
            args.config,
            core_only=args.core_only,
            load_model=args.load_model,
            allow_low_memory=args.allow_low_memory,
        )
    except Exception as error:
        print(f"preflight failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
