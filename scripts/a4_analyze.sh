#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/a4-run-directory" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
python -m safety_project_a4.analyze --run-dir "$1"

