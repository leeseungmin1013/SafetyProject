from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .io_utils import atomic_json, utc_now


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return None if not finite else statistics.fmean(finite)


def analyze_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("no A4 records")
    summary: dict[str, Any] = {
        "generated_at": utc_now(),
        "run_id": records[0]["run_id"],
        "probe_steps": [record["step"] for record in records],
        "probe_count": len(records),
        "spf_trigger_rate": statistics.fmean(bool(record["spf_triggered"]) for record in records),
        "references": {},
    }
    for reference in ("fresh", "audit"):
        stage_names = records[0][reference].keys()
        stages: dict[str, Any] = {}
        for stage in stage_names:
            metrics = [record[reference][stage] for record in records]
            valid = [metric for metric in metrics if metric["classification"] not in {"invalid", "zero-norm"}]
            stages[stage] = {
                "n": len(metrics),
                "valid_n": len(valid),
                "conflict_n": sum(metric["classification"] == "conflict" for metric in valid),
                "conflict_rate": None
                if not valid
                else statistics.fmean(metric["classification"] == "conflict" for metric in valid),
                "mean_dot": _mean([float(metric["dot"]) for metric in valid]),
                "mean_cosine": _mean(
                    [float(metric["cosine"]) for metric in valid if metric["cosine"] is not None]
                ),
                "mean_vector_norm": _mean([float(metric["vector_norm"]) for metric in valid]),
            }
        event_names = records[0][f"{reference}_events"].keys()
        events = {
            name: {
                "count": sum(bool(record[f"{reference}_events"][name]) for record in records),
                "rate": statistics.fmean(
                    bool(record[f"{reference}_events"][name]) for record in records
                ),
            }
            for name in event_names
        }
        summary["references"][reference] = {"stages": stages, "events": events}

    deltas = [float(record["audit_panel_delta_ce"]) for record in records]
    summary["audit_panel_delta_ce"] = {
        "n": len(deltas),
        "mean": statistics.fmean(deltas),
        "median": statistics.median(deltas),
        "min": min(deltas),
        "max": max(deltas),
        "positive_n": sum(value > 0.0 for value in deltas),
        "positive_rate": statistics.fmean(value > 0.0 for value in deltas),
    }
    return summary


def analyze_run(run_dir: str | Path) -> Path:
    directory = Path(run_dir).resolve()
    source = directory / "a4.jsonl"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    summary = analyze_records(records)
    destination = directory / "a4_summary.json"
    atomic_json(destination, summary)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate one SPF/A4 run")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    destination = analyze_run(args.run_dir)
    print(json.dumps({"summary": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

