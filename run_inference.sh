#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ $# != 2 ]]; then
  echo 'Usage: ./run_inference.sh <input_dir> <output_dir>' >&2
  exit 2
fi
if [[ ! -d "$1" ]]; then
  echo 'Input must be a directory containing CT .nii or .nii.gz files' >&2
  exit 2
fi

# An already prepared Python environment saves the first-run download.
python_bin="${LUMBAR_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  python_bin="$project_dir/.inference_venv/bin/python"
fi

requirements_ready() {
  "$python_bin" - "$project_dir/requirements-cpu.lock" <<'PY'
import importlib.metadata
from pathlib import Path
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(1)
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line or line.startswith("#"):
        continue
    name, expected = line.split("==", 1)
    try:
        installed = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit(1)
    if installed != expected:
        raise SystemExit(1)
PY
}

if ! requirements_ready 2>/dev/null; then
  if [[ -n "${LUMBAR_PYTHON:-}" ]]; then
    echo 'LUMBAR_PYTHON does not have the pinned Python 3.11 inference requirements; using a local environment.' >&2
  fi
  command -v python3.11 >/dev/null || {
    echo 'Python 3.11 is required for automatic setup.' >&2
    exit 1
  }
  python_bin="$project_dir/.inference_venv/bin/python"
  if [[ ! -x "$python_bin" ]]; then
    python3.11 -m venv "$project_dir/.inference_venv"
  fi
  if ! requirements_ready 2>/dev/null; then
    echo 'Installing pinned inference requirements in .inference_venv (first run only)...' >&2
    "$python_bin" -m pip install -r "$project_dir/requirements-cpu.lock"
  fi
  requirements_ready || {
    echo 'Inference dependencies could not be verified after installation.' >&2
    exit 1
  }
fi

# The package is loaded from this folder, with no separate install command.
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" - "$project_dir" "$1" "$2" <<'PY'
import os
import subprocess
import sys
from pathlib import Path

project = Path(sys.argv[1]).resolve()
source = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3]).resolve()
files = []
for path in source.iterdir():
    if path.is_file() and path.name.endswith((".nii", ".nii.gz")):
        files.append(path)
files.sort()
if not files:
    raise SystemExit("No NIfTI CT files found")
ids = []
for path in files:
    case_id = path.name.removesuffix(".gz").removesuffix(".nii")
    ids.append(case_id)
if len(ids) != len(set(ids)):
    raise SystemExit("Duplicate case IDs in input directory")
checkpoint = project / "outputs/nnunet/model.pt"
if not checkpoint.is_file():
    raise SystemExit("Packaged model weights are missing")
if output == source or source in output.parents or output in source.parents:
    raise SystemExit("Input and output directories must not overlap")
if output.exists() and any(output.iterdir()):
    raise SystemExit("Output directory must be new or empty")
for path, case_id in zip(files, ids):
    command = [
        sys.executable,
        "-m",
        "lumbar_ct.inference",
        str(path),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output / case_id),
        "--case-id",
        case_id,
        "--device",
        os.environ.get("LUMBAR_DEVICE", "auto"),
        "--units-policy",
        os.environ.get("LUMBAR_UNITS_POLICY", "strict"),
    ]
    subprocess.run(command, check=True)
PY
