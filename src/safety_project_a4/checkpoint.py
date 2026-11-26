from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import atomic_json, sha256_file, utc_now


def save_final_checkpoint(model: Any, tokenizer: Any, output_dir: str | Path) -> dict[str, Any]:
    """Save an inference-ready model atomically within a unique run directory."""
    run_dir = Path(output_dir).resolve()
    checkpoint_dir = run_dir / "checkpoint"
    incomplete_dir = run_dir / ".checkpoint-incomplete"
    if checkpoint_dir.exists() or incomplete_dir.exists():
        raise FileExistsError(f"checkpoint output already exists under {run_dir}")

    incomplete_dir.mkdir()
    model.save_pretrained(incomplete_dir, max_shard_size="4GB", safe_serialization=True)
    tokenizer.save_pretrained(incomplete_dir)

    shards = sorted(incomplete_dir.glob("*.safetensors"))
    required = (
        incomplete_dir / "config.json",
        incomplete_dir / "tokenizer_config.json",
        incomplete_dir / "tokenizer.json",
    )
    if not shards or any(not path.is_file() for path in required):
        raise RuntimeError("final checkpoint is missing model shards, config, or tokenizer")
    if any(path.stat().st_size == 0 for path in shards):
        raise RuntimeError("final checkpoint contains an empty model shard")

    files = sorted(path for path in incomplete_dir.iterdir() if path.is_file())
    checkpoint_manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "format": "safetensors",
        "files": [
            {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in files
        ],
    }
    total_bytes = sum(path.stat().st_size for path in files)
    atomic_json(incomplete_dir / "checkpoint_manifest.json", checkpoint_manifest)
    incomplete_dir.rename(checkpoint_dir)
    return {
        "path": str(checkpoint_dir),
        "format": "safetensors",
        "shard_count": len(shards),
        "total_bytes": total_bytes,
        "manifest_sha256": sha256_file(checkpoint_dir / "checkpoint_manifest.json"),
    }
