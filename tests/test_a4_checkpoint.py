import json
import tempfile
import unittest
from pathlib import Path

from safety_project_a4.checkpoint import save_final_checkpoint
from safety_project_a4.io_utils import sha256_file


class _Model:
    def __init__(self, include_weights=True):
        self.include_weights = include_weights

    def save_pretrained(self, output_dir, **kwargs):
        assert kwargs["safe_serialization"] is True
        (output_dir / "config.json").write_text("{}", encoding="utf-8")
        if self.include_weights:
            (output_dir / "model.safetensors").write_bytes(b"test weights")


class _Tokenizer:
    def save_pretrained(self, output_dir):
        (output_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (output_dir / "tokenizer.json").write_text("{}", encoding="utf-8")


class CheckpointTests(unittest.TestCase):
    def test_complete_checkpoint_has_verified_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            saved = save_final_checkpoint(_Model(), _Tokenizer(), run_dir)
            checkpoint_dir = run_dir / "checkpoint"
            self.assertEqual(saved["path"], str(checkpoint_dir))
            self.assertEqual(saved["shard_count"], 1)
            self.assertFalse((run_dir / ".checkpoint-incomplete").exists())
            manifest_path = checkpoint_dir / "checkpoint_manifest.json"
            self.assertEqual(saved["manifest_sha256"], sha256_file(manifest_path))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in manifest["files"]:
                path = checkpoint_dir / item["name"]
                self.assertEqual(item["sha256"], sha256_file(path))
                self.assertEqual(item["size_bytes"], path.stat().st_size)
            self.assertEqual(saved["total_bytes"], sum(item["size_bytes"] for item in manifest["files"]))

    def test_missing_weights_never_publishes_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with self.assertRaises(RuntimeError):
                save_final_checkpoint(_Model(include_weights=False), _Tokenizer(), run_dir)
            self.assertFalse((run_dir / "checkpoint").exists())
            self.assertTrue((run_dir / ".checkpoint-incomplete").exists())


if __name__ == "__main__":
    unittest.main()
