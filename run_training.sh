#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
python_bin="${LUMBAR_PYTHON:-python3}"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" "$project_dir/scripts/run_full_training.py" "$@"
