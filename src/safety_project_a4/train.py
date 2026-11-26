from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .checkpoint import save_final_checkpoint
from .config import load_config, with_overrides
from .data import CausalCollator, EncodedExample, encode_record, load_private_records, move_batch
from .gradient_capture import GradientSpool, apply_spf_projection, capture_reference_gradient
from .io_utils import append_jsonl, atomic_json, git_state, sha256_file, sha256_json, utc_now
from .metrics import MetricAccumulator, norm_scaled_tau, transition_events
from .nvme_optimizer import NVMeInspectableAdamW


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_probe_step(step: int, config: dict[str, Any]) -> bool:
    a4 = config["a4"]
    if step in {int(value) for value in a4["probe_steps"]}:
        return True
    every = int(a4.get("probe_every", 0))
    return every > 0 and step % every == 0


def _finite_gradients(model: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def _batch(encoded: EncodedExample, collator: CausalCollator, device: torch.device) -> dict[str, torch.Tensor]:
    return move_batch(collator([encoded]), device)


@torch.no_grad()
def _panel_loss(model, encoded_panel: list[EncodedExample], collator, device) -> float:
    was_training = model.training
    model.eval()
    token_weighted_loss = 0.0
    token_count = 0
    for encoded in encoded_panel:
        batch = _batch(encoded, collator, device)
        count = int((batch["labels"] != -100).sum().item())
        output = model(**batch)
        token_weighted_loss += float(output.loss.item()) * count
        token_count += count
    if was_training:
        model.train()
    return token_weighted_loss / max(token_count, 1)


def _manifest(config: dict[str, Any], run_id: str, output_dir: Path, status: str) -> dict[str, Any]:
    repository = Path(config["_config_path"]).parent.parent
    commit, dirty = git_state(repository)
    safety_path = config["data"]["safety"]["projection_sample_path"]
    audit_path = config["data"]["safety"]["audit_panel_path"]
    selection_path = os.environ.get("A4_SELECTION_MANIFEST_PATH")
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "created_at": utc_now(),
        "git": {"commit": commit, "dirty": dirty},
        "model": {
            "repository": config["model"]["repository"],
            "revision": config["model"]["revision"],
            "condition": "spf-a4",
        },
        "data": {
            "repository": config["data"]["utility"]["repository"],
            "revision": config["data"]["utility"]["revision"],
            "projection_sample_sha256": sha256_file(safety_path),
            "audit_panel_sha256": sha256_file(audit_path),
            "selection_manifest_sha256": sha256_file(selection_path) if selection_path else None,
        },
        "training": {
            "seed": config["training"]["pilot_seed"],
            "finetuning_type": "full",
            "precision": "bf16",
            "optimizer": "adamw",
            "optimizer_state_dtype": config["training"]["optimizer_state_dtype"],
            "optimizer_state_storage": "nvme_chunked_fp32",
            "optimizer_chunk_elements": config["training"]["optimizer_chunk_elements"],
            "global_batch_examples": config["training"]["effective_batch_examples"],
            "config_sha256": sha256_json({key: value for key, value in config.items() if key != "_config_path"}),
        },
        "spf": {
            "rank": config["spf"]["rank"],
            "gate": config["spf"]["conflict_gating"],
            "allowlist": config["spf"]["parameter_allowlist"],
        },
        "hardware": {
            "gpu_type": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "cuda_process_limit_gib": round(
                torch.cuda.get_device_properties(0).total_memory
                * torch.cuda.get_per_process_memory_fraction(0)
                / 1024**3,
                3,
            ),
            "configured_cuda_budget_gib": config["runtime"]["gpu_budget_gib"],
        },
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": str(torch.version.cuda),
            "transformers": importlib.metadata.version("transformers"),
            "datasets": importlib.metadata.version("datasets"),
            "safetensors": importlib.metadata.version("safetensors"),
        },
        "artifacts": {
            "uri": str(output_dir),
            "contains_raw_harmful_text": False,
            "checkpoint_policy": config["artifact"]["checkpoint_policy"],
        },
    }


def run(config: dict[str, Any]) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("A4 training requires CUDA")
    if torch.cuda.device_count() != 1 or int(config["hardware"]["devices"]) != 1:
        raise RuntimeError("this streamed A4 runner currently supports one visible CUDA device")
    device = torch.device("cuda:0")
    total_cuda = torch.cuda.get_device_properties(device).total_memory
    current_fraction = float(torch.cuda.get_per_process_memory_fraction(device))
    budget_fraction = min(
        current_fraction,
        float(config["runtime"]["gpu_budget_gib"]) * 1024**3 / total_cuda,
    )
    torch.cuda.set_per_process_memory_fraction(budget_fraction, device)
    seed = int(config["training"]["pilot_seed"])
    _set_seed(seed)

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    run_id = os.environ.get("A4_RUN_ID") or f"{config['runtime']['run_id_prefix']}-s{seed}-{timestamp}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("A4_RUN_ID must be a safe directory name")
    output_dir = Path(config["runtime"]["output_root"]).resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = _manifest(config, run_id, output_dir, "running")
    atomic_json(output_dir / "run_manifest.json", manifest)
    training_log = output_dir / "train.jsonl"
    a4_log = output_dir / "a4.jsonl"

    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["repository"], revision=model_config["tokenizer_revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = CausalCollator(tokenizer.pad_token_id)
    maximum = int(config["training"]["max_sequence_length"])

    utility_config = config["data"]["utility"]
    raw_dataset = load_dataset(
        utility_config["repository"],
        utility_config["config"],
        split=utility_config["train_split"],
        revision=utility_config["revision"],
    )
    utility_examples = [encode_record(tokenizer, record, maximum) for record in raw_dataset]
    safety_records = load_private_records(config["data"]["safety"]["projection_sample_path"])
    audit_records = load_private_records(config["data"]["safety"]["audit_panel_path"])
    safety_example = encode_record(tokenizer, safety_records[0], maximum)
    panel_size = int(config["a4"]["audit_loss_panel_size"])
    audit_examples = [encode_record(tokenizer, record, maximum) for record in audit_records[:panel_size]]

    model = AutoModelForCausalLM.from_pretrained(
        model_config["repository"],
        revision=model_config["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation=model_config["attention_implementation"],
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    if model_config["activation_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if total_parameters != trainable_parameters:
        raise RuntimeError(f"full fine-tuning invariant failed: {trainable_parameters}/{total_parameters}")
    manifest["model"].update(
        {"parameter_count": total_parameters, "trainable_parameter_count": trainable_parameters}
    )
    atomic_json(output_dir / "run_manifest.json", manifest)

    training = config["training"]
    chunk_elements = int(training["optimizer_chunk_elements"])
    optimizer = NVMeInspectableAdamW(
        model.named_parameters(),
        state_root=config["runtime"]["optimizer_state_root"],
        run_id=run_id,
        chunk_elements=chunk_elements,
        lr=float(training["learning_rate"]),
        betas=(float(training["beta1"]), float(training["beta2"])),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    maximum_steps = int(training["max_optimizer_steps"])
    effective_batch = int(training["effective_batch_examples"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(utility_examples), generator=generator).tolist()
    step_count = min(maximum_steps, math.ceil(len(order) / effective_batch))
    probe_count = 0
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()

    try:
        for zero_based_step in range(step_count):
            step = zero_based_step + 1
            indices = order[zero_based_step * effective_batch : (zero_based_step + 1) * effective_batch]
            optimizer.zero_grad(set_to_none=True)
            utility_loss_sum = 0.0
            for index in indices:
                batch = _batch(utility_examples[index], collator, device)
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite utility loss at step {step}")
                utility_loss_sum += float(loss.detach().item())
                (loss / len(indices)).backward()
            if not _finite_gradients(model):
                raise FloatingPointError(f"non-finite utility gradient at step {step}")

            probe = _is_probe_step(step, config)
            raw_utility_spool = None
            if probe:
                raw_utility_spool = GradientSpool(
                    config["runtime"]["gradient_spool_root"],
                    f"{run_id}-step{step}-utility",
                    chunk_elements=chunk_elements,
                )
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    if parameter.grad is None:
                        raise RuntimeError(f"utility gradient is missing for {name}")
                    raw_utility_spool.write(name, parameter.grad)
            torch.manual_seed(seed + step * 1009)

            def safety_loss():
                return model(**_batch(safety_example, collator, device)).loss

            capture = capture_reference_gradient(
                model,
                safety_loss,
                compute_dot=True,
                compute_bases=True,
                spool_root=config["runtime"]["gradient_spool_root"] if probe else None,
                spool_label=f"{run_id}-step{step}-fresh",
                rank=int(config["spf"]["rank"]),
                oversampling=int(config["spf"]["svd_oversampling"]),
                niter=int(config["spf"]["svd_power_iterations"]),
                allowlist_policy=config["spf"]["parameter_allowlist"],
                chunk_elements=chunk_elements,
            )
            reference_norm = math.sqrt(max(capture.reference_norm_sq, 0.0))
            utility_norm = math.sqrt(max(capture.utility_norm_sq, 0.0))
            gate = config["spf"]["conflict_gating"]
            if gate == "paper_zero":
                conflict = capture.dot_with_utility < 0.0
            else:
                conflict = capture.dot_with_utility < -norm_scaled_tau(reference_norm, utility_norm)
            if conflict:
                apply_spf_projection(model, capture.bases, chunk_elements=chunk_elements)
            del capture.bases

            audit_before = None
            audit_after = None
            audit_capture = None
            fresh_accumulator = None
            audit_accumulator = None
            nvme_read_before = optimizer.bytes_read
            nvme_written_before = optimizer.bytes_written
            if probe:
                audit_before = _panel_loss(model, audit_examples, collator, device)
                audit_record_index = probe_count % len(audit_examples)
                probe_count += 1
                audit_example = audit_examples[audit_record_index]

                def audit_gradient_loss():
                    return model(**_batch(audit_example, collator, device)).loss

                audit_capture = capture_reference_gradient(
                    model,
                    audit_gradient_loss,
                    compute_dot=False,
                    compute_bases=False,
                    spool_root=config["runtime"]["gradient_spool_root"],
                    spool_label=f"{run_id}-step{step}-audit",
                    rank=1,
                    oversampling=0,
                    niter=0,
                    allowlist_policy=config["spf"]["parameter_allowlist"],
                    chunk_elements=chunk_elements,
                )
                fresh_accumulator = MetricAccumulator()
                audit_accumulator = MetricAccumulator()
                fresh_accumulator.add_reference_norm_sq(capture.reference_norm_sq)
                audit_accumulator.add_reference_norm_sq(audit_capture.reference_norm_sq)

                def probe_callback(name, start, end, stages):
                    fresh = capture.spool.read_slice(name, start, end, device=device).float()
                    audit = audit_capture.spool.read_slice(name, start, end, device=device).float()
                    for stage, vector in stages:
                        vector_fp32 = vector.float()
                        norm_sq = float(vector_fp32.square().sum(dtype=torch.float64).item())
                        fresh_accumulator.add(
                            stage,
                            float((fresh * vector_fp32).sum(dtype=torch.float64).item()),
                            norm_sq,
                        )
                        audit_accumulator.add(
                            stage,
                            float((audit * vector_fp32).sum(dtype=torch.float64).item()),
                            norm_sq,
                        )

                optimizer_started = time.monotonic()
                optimizer.step(
                    probe_callback=probe_callback,
                    no_spf_gradient_provider=lambda name, start, end, parameter_device: raw_utility_spool.read_slice(
                        name, start, end, device=parameter_device
                    ),
                )
                audit_after = _panel_loss(model, audit_examples, collator, device)
                fresh_metrics = fresh_accumulator.finalize()
                audit_metrics = audit_accumulator.finalize()
                append_jsonl(
                    a4_log,
                    {
                        "run_id": run_id,
                        "step": step,
                        "spf_gate": gate,
                        "spf_triggered": conflict,
                        "pre_projection_dot_s_g": capture.dot_with_utility,
                        "fresh": {name: metric.to_dict() for name, metric in fresh_metrics.items()},
                        "fresh_events": transition_events(fresh_metrics),
                        "audit": {name: metric.to_dict() for name, metric in audit_metrics.items()},
                        "audit_events": transition_events(audit_metrics),
                        "audit_record_index": audit_record_index,
                        "audit_panel_ce_before": audit_before,
                        "audit_panel_ce_after": audit_after,
                        "audit_panel_delta_ce": audit_after - audit_before,
                    },
                )
            else:
                optimizer_started = time.monotonic()
                optimizer.step()
            optimizer_step_seconds = time.monotonic() - optimizer_started

            if capture.spool is not None:
                capture.spool.cleanup()
            if audit_capture is not None and audit_capture.spool is not None:
                audit_capture.spool.cleanup()
            if raw_utility_spool is not None:
                raw_utility_spool.cleanup()
            optimizer.zero_grad(set_to_none=True)
            append_jsonl(
                training_log,
                {
                    "run_id": run_id,
                    "step": step,
                    "examples": len(indices),
                    "utility_loss_mean": utility_loss_sum / len(indices),
                    "safety_loss": capture.loss,
                    "spf_triggered": conflict,
                    "probe": probe,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "optimizer_step_seconds": optimizer_step_seconds,
                    "optimizer_nvme_read_bytes": optimizer.bytes_read - nvme_read_before,
                    "optimizer_nvme_written_bytes": optimizer.bytes_written - nvme_written_before,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )

        peak_reserved = torch.cuda.max_memory_reserved(device)
        budget_bytes = int(float(config["runtime"]["gpu_budget_gib"]) * 1024**3)
        if peak_reserved > budget_bytes:
            raise MemoryError(
                f"CUDA peak reserved {peak_reserved / 1024**3:.2f} GiB exceeds "
                f"configured budget {budget_bytes / 1024**3:.2f} GiB"
            )
        optimizer.cleanup()
        checkpoint_policy = config["artifact"]["checkpoint_policy"]
        full_run_steps = int(training["one_epoch_optimizer_steps"])
        if checkpoint_policy == "save_final_main_only" and step_count == full_run_steps:
            model.eval()
            model.gradient_checkpointing_disable()
            model.config.use_cache = True
            manifest["artifacts"]["checkpoint"] = save_final_checkpoint(
                model, tokenizer, output_dir
            )
        elif checkpoint_policy == "save_final_main_only":
            manifest["artifacts"]["checkpoint"] = {
                "status": "skipped_partial_run",
                "optimizer_steps": step_count,
                "full_run_steps": full_run_steps,
            }
        manifest["status"] = "completed"
        manifest["completed_at"] = utc_now()
        manifest["training"]["optimizer_steps"] = step_count
        manifest["hardware"].update(
            {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "wall_time_seconds": time.monotonic() - started,
                "optimizer_nvme_read_bytes": optimizer.bytes_read,
                "optimizer_nvme_written_bytes": optimizer.bytes_written,
            }
        )
        atomic_json(output_dir / "run_manifest.json", manifest)
        return output_dir
    except (torch.cuda.OutOfMemoryError, MemoryError):
        manifest["status"] = "failed-capacity"
        manifest["completed_at"] = utc_now()
        atomic_json(output_dir / "run_manifest.json", manifest)
        raise
    except FloatingPointError:
        manifest["status"] = "failed-numerical"
        manifest["completed_at"] = utc_now()
        atomic_json(output_dir / "run_manifest.json", manifest)
        raise
    except Exception:
        manifest["status"] = "failed-infrastructure"
        manifest["completed_at"] = utc_now()
        atomic_json(output_dir / "run_manifest.json", manifest)
        raise
    finally:
        if optimizer.state_dir.exists():
            optimizer.cleanup()
        for spool in Path(config["runtime"]["gradient_spool_root"]).glob(f"{run_id}-*"):
            if spool.is_dir():
                import shutil

                if spool.parent.resolve() != Path(config["runtime"]["gradient_spool_root"]).resolve():
                    raise RuntimeError("gradient spool escaped its root")
                shutil.rmtree(spool)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run streamed Qwen SPF training with A4 probes")
    parser.add_argument("--config", default="configs/qwen25_spf_a4.yaml")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--probe-every", type=int)
    args = parser.parse_args()
    try:
        config = with_overrides(
            load_config(args.config), max_steps=args.max_steps, probe_every=args.probe_every
        )
        output = run(config)
    except Exception as error:
        print(f"A4 run failed: {error}", file=sys.stderr)
        raise
    print(json.dumps({"status": "completed", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
