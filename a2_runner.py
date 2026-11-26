"""A2 Initial-checkpoint gradient diagnostic."""

import argparse
import hashlib
import json
import math
import random
import statistics
import tempfile
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


def probe_rows(population, count, size, seed):
    if count < 1 or size < 1:
        raise ValueError("count and size must be positive")
    ids = random.Random(seed).sample(range(population), count * size)
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def agreement(batch, single):
    if not batch or len(batch) != len(single):
        raise ValueError("gate sequences must be nonempty and have equal length")
    matches = [a == b for a, b in zip(batch, single)]
    result = dict(correct=sum(matches), total=len(batch),
                  rate=statistics.mean(matches), batch_conflict_rate=statistics.mean(batch))
    for label, value in (("conflict", True), ("no_conflict", False)):
        group = [match for gate, match in zip(batch, matches) if gate == value]
        result[label] = statistics.mean(group) if group else None
    return result


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def viewer_row(manifest, index):
    query = urlencode(dict(dataset=manifest["dataset"], config=manifest["config"],
                           split=manifest["split"], offset=index, length=1))
    with urlopen(f"https://datasets-server.huggingface.co/rows?{query}", timeout=30) as response:
        rows = json.load(response)["rows"]
    if len(rows) != 1 or rows[0]["row_idx"] != index:
        raise ValueError(f"Dataset Viewer returned the wrong row for {index}")
    return rows[0]["row"]


def load_references(dataset, entries, manifest):
    items, provenance = [], []
    for entry in entries:
        row, index = entry["row_idx"], entry["response_index"]
        item = dataset[row]
        prompt, response = item["prompt"], item[f"response_{index}"]
        expected = item[f"response_{index}_sha256"]
        if not item[f"is_response_{index}_safe"] or not prompt.strip() or not response.strip():
            raise ValueError(f"Invalid safe reference at row {row}")
        reviewed_source = viewer_row(manifest, row)
        if (prompt, response, expected) != (
            reviewed_source["prompt"], reviewed_source[f"response_{index}"],
            reviewed_source[f"response_{index}_sha256"],
        ):
            raise ValueError(f"Pinned revision differs from Dataset Viewer at row {row}")
        items.append((prompt, response))
        provenance.append(dict(entry, prompt_sha256=sha(prompt), response_sha256=sha(response),
                               dataset_response_sha256=expected))
    return items, provenance


def encode(tokenizer, prompt, response, limit):
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=False)
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": response}], tokenize=True, return_dict=False
    )
    if full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("Chat template did not produce a separable assistant continuation")
    suffix = full[len(prefix):]
    if len(suffix) >= limit:
        raise ValueError("Assistant continuation exceeds the configured sequence limit")
    prefix = prefix[-(limit - len(suffix)):]
    return prefix + suffix, [-100] * len(prefix) + suffix


def mean_gradient(torch, model, tokenizer, params, examples, limit, progress_label,
                  accumulation_device="cpu"):
    # Accumulate per-example means in FP32; only one backward is live.
    sums = {}
    try:
        for index, (prompt, response) in enumerate(examples, start=1):
            ids, labels = encode(tokenizer, prompt, response, limit)
            loss = model(input_ids=torch.tensor([ids], device="cuda"),
                         labels=torch.tensor([labels], device="cuda"), use_cache=False).loss
            if not torch.isfinite(loss).item():
                raise ValueError("Nonfinite sample loss")
            loss.backward()
            for name, param in params:
                if param.grad is None:
                    continue
                if name in sums:
                    sums[name].add_(param.grad.detach().to(device=accumulation_device))
                else:
                    sums[name] = param.grad.detach().to(device=accumulation_device,
                                                        dtype=torch.float32, copy=True)
                param.grad = None
            del loss
            print(f"{progress_label}: sample {index}/{len(examples)}", flush=True)
    finally:
        model.zero_grad(set_to_none=True)
    for name, grad in sums.items():
        grad.div_(len(examples))
        if not torch.isfinite(grad).all().item():
            raise ValueError(f"Nonfinite gradient for {name}")
    return sums


def basis(torch, matrix, rank):
    rank = min(rank, *matrix.shape)
    u, singular, _ = torch.svd_lowrank(matrix.to("cuda"), q=min(rank + 8, *matrix.shape), niter=4)
    if singular[rank - 1].item() <= 0:
        raise ValueError("Zero singular value at requested rank")
    return u[:, :rank].cpu().contiguous()


def overlap(torch, a, b):
    return float((a.T @ b).square().sum().div(min(a.shape[1], b.shape[1])).clamp(0, 1))


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def run(args):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    singles, batches = manifest["single_reference"], manifest["batch_reference"]
    if len(singles) != 5 or len(batches) != 16 or len({x["row_idx"] for x in singles + batches}) != 21:
        raise ValueError("Expected 5 Single and 16 Batch References with distinct row IDs")
    if args.smoke:
        singles, batches = singles[:1], batches[:2]
    pk = load_dataset(manifest["dataset"], manifest["config"], split=manifest["split"],
                      revision=manifest["revision_to_verify"])
    batch_examples, batch_meta = load_references(pk, batches, manifest)
    single_examples, single_meta = load_references(pk, singles, manifest)
    gsm = load_dataset("openai/gsm8k", "main", split="train", revision=args.gsm_revision)
    probes = probe_rows(len(gsm), 32, 64, args.seed)[:args.probe_count]
    if args.smoke:
        probes = [probes[0][:args.smoke_samples]]
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    # Attention and MLP matrix scope for overlap; the gate uses all trainable parameters.
    blocks = {name for name, p in params if p.ndim == 2 and
              (".self_attn." in name or ".mlp." in name)}
    if not blocks:
        raise ValueError("No attention/MLP parameter matrices found")

    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    report = dict(status="running", smoke=args.smoke, model=args.model,
                  model_revision=args.model_revision, gsm_revision=args.gsm_revision,
                  dataset_revision=manifest["revision_to_verify"], seed=args.seed,
                  sequence_limit=args.sequence_limit, rank=10, device=torch.cuda.get_device_name(),
                  accumulation_device=args.accumulation_device,
                  single_reference=single_meta, batch_reference=batch_meta, probe_rows=probes,
                  overlap_blocks=sorted(blocks), gate_parameters=[name for name, _ in params],
                  audit_overlap_verified=False, overlaps=[], gates=[])
    target = args.output / "report.json"
    write_json(target, report)
    try:
        with tempfile.TemporaryDirectory(prefix="a2-", dir=args.scratch) as scratch:
            root = Path(scratch)
            references = [batch_examples] + [[example] for example in single_examples]
            batch_bases = {}
            safety_norms = []
            for i, examples in enumerate(references):
                grad = mean_gradient(torch, model, tokenizer, params, examples,
                                     args.sequence_limit, f"Reference {i+1}/{len(references)}",
                                     args.accumulation_device)
                if missing := blocks.difference(grad):
                    raise ValueError(f"Missing overlap gradient blocks: {sorted(missing)}")
                scores = {}
                norm_squared = 0.
                for j, (name, _) in enumerate(params):
                    if name not in grad:
                        continue
                    saved = grad[name].cpu()
                    flat = saved.reshape(-1)
                    norm_squared += float(torch.dot(flat, flat))
                    torch.save(saved, root / f"{i}-{j}.pt")
                    if name in blocks:
                        u = basis(torch, grad[name], 10)
                        if i == 0:
                            batch_bases[name] = u
                        else:
                            scores[name] = overlap(torch, u, batch_bases[name])
                del grad, saved, flat
                safety_norms.append(math.sqrt(norm_squared))
                report["safety_gradient_norms"] = safety_norms
                if i:
                    report["overlaps"].append(dict(row_idx=singles[i-1]["row_idx"],
                                                   median=statistics.median(scores.values()),
                                                   blocks=scores))
                print(f"Reference gradient {i+1}/{len(references)} saved", flush=True)
                write_json(target, report)

            for probe_index, ids in enumerate(probes, start=1):
                examples = [(gsm[i]["question"], gsm[i]["answer"]) for i in ids]
                grad = mean_gradient(torch, model, tokenizer, params, examples,
                                     args.sequence_limit, f"Utility probe {probe_index}/{len(probes)}",
                                     args.accumulation_device)
                dots = [0.] * len(references)
                utility_norm_squared = 0.
                for j, (name, _) in enumerate(params):
                    if name not in grad:
                        continue
                    utility = grad.pop(name).cpu().reshape(-1)
                    utility_norm_squared += float(torch.dot(utility, utility))
                    for i in range(len(references)):
                        path = root / f"{i}-{j}.pt"
                        if path.exists():
                            safety = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                            dots[i] += float(torch.dot(utility, safety.reshape(-1)))
                if not all(math.isfinite(value) for value in dots):
                    raise ValueError("Nonfinite conflict inner product")
                gates = [value < 0 for value in dots]
                utility_norm = math.sqrt(utility_norm_squared)
                if not all(math.isfinite(norm) for norm in [utility_norm, *safety_norms]):
                    raise ValueError("Nonfinite gradient norm")
                cosines = [max(-1., min(1., dot / (utility_norm * norm)))
                           if utility_norm and norm else None
                           for dot, norm in zip(dots, safety_norms)]
                report["gates"].append(dict(inner_products=dots, conflict=gates,
                                            utility_gradient_norm=utility_norm,
                                            cosine_similarities=cosines))
                print(f"Utility probe {len(report['gates'])}/{len(probes)}", flush=True)
                write_json(target, report)

        batch = [probe["conflict"][0] for probe in report["gates"]]
        outcomes = [agreement(batch, [probe["conflict"][i] for probe in report["gates"]])
                    for i in range(1, len(references))]
        med = statistics.median(x["median"] for x in report["overlaps"])
        overall = sum(x["correct"] for x in outcomes) / (len(batch) * len(outcomes))
        report.update(status="complete", median_overlap=med, overall_agreement=overall,
                      agreement_by_single=[dict(row_idx=s["row_idx"], **x)
                                           for s, x in zip(singles, outcomes)],
                      meets_thresholds=None if args.smoke else med >= .8 and overall >= .8)
    except BaseException as error:
        report.update(status="failed", error=type(error).__name__)
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(target, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/a2_reference_manifest.json"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gsm-revision", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-limit", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--probe-count", type=int, choices=range(1, 33), default=32,
                        metavar="1..32")
    parser.add_argument("--smoke-samples", type=int, choices=range(1, 65), default=64,
                        metavar="1..64")
    parser.add_argument("--accumulation-device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not args.smoke and args.smoke_samples != 64:
        parser.error("--smoke-samples requires --smoke")
    for rev in (args.model_revision, args.gsm_revision):
        if len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev):
            parser.error("Model and GSM8K revisions must be 40-character Git commit SHAs")
    if not args.scratch.is_dir() or args.sequence_limit < 2:
        parser.error("An existing scratch directory and sequence limit >= 2 are required")
    run(args)


if __name__ == "__main__":
    main()
