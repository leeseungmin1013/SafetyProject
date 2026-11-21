#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

: "${A4_SAFETY_SAMPLE_PATH:?Set A4_SAFETY_SAMPLE_PATH to the private one-shot JSON file}"
: "${A4_AUDIT_PANEL_PATH:?Set A4_AUDIT_PANEL_PATH to the private audit JSONL file}"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
python -m safety_project_a4.preflight --config configs/qwen25_spf_a4.yaml --load-model
python -m safety_project_a4.train --config configs/qwen25_spf_a4.yaml
