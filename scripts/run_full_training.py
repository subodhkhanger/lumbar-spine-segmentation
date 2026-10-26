#!/usr/bin/env python3
"""Prepare, train and validate a fresh ResEnc M on the full source training split."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from lumbar_ct.artifacts import file_hash, reject_path_overlap, write_json
from lumbar_ct.training import choose_device

PROJECT = Path(__file__).resolve().parents[1]


def run(script, *arguments):
    command = [sys.executable, str(PROJECT / "scripts" / script)]
    command.extend(str(argument) for argument in arguments)
    subprocess.run(command, cwd=PROJECT, check=True)


# Keep new training results separate from the packaged weights.
def check_work_directory(root, data):
    reject_path_overlap(root, data)
    reject_path_overlap(root, PROJECT / "outputs")
    for name in ("src", "scripts", "tests", "configs", "manifests"):
        reject_path_overlap(root, PROJECT / name)
    if root.exists():
        raise ValueError(
            "Use a new --work-dir; an existing run will not be overwritten"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "data/raw/totalsegmentator_v201"
    )
    parser.add_argument(
        "--work-dir", type=Path, default=PROJECT / "training/full_source"
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check setup without preparing data or starting training",
    )
    args = parser.parse_args()
    choice = args.device
    if choice is None and sys.stdin.isatty():
        choice = (
            input("Training device [auto/cuda/mps/cpu; default auto]: ").strip()
            or "auto"
        )
    device = choose_device(choice or "auto")
    root = args.work_dir.resolve()
    data = args.data_dir.resolve()
    check_work_directory(root, data)
    if not (data / "meta.csv").is_file():
        raise ValueError(
            "--data-dir must contain TotalSegmentator v2.0.1 meta.csv and case folders"
        )
    from importlib.metadata import version

    if version("nnunetv2") != "2.8.1":
        raise ValueError("Install the pinned requirements-nnunet.txt before training")
    print(
        f"Device: {device}; full official training split; 57 source validation scans",
        flush=True,
    )
    print(f"New results: {root}", flush=True)
    if args.check:
        return
    selected = PROJECT / "outputs/nnunet/model.pt"
    selected_hash = file_hash(selected)
    root.mkdir(parents=True)
    os.environ["nnUNet_n_proc_DA"] = "4" if device.type == "cuda" else "0"
    run("prepare_full_source.py", "--data-dir", data, "--root", root)
    manifest = root / "manifest.json"
    normalizer = root / "normalization.json"
    nnunet = root / "nnunet"
    run(
        "prepare_nnunet.py",
        "--root",
        nnunet,
        "--manifest",
        manifest,
        "--normalizer",
        normalizer,
        "--adapter-output",
        root / "adapter.json",
    )
    run("train_nnunet.py", "--root", nnunet, "--device", device)
    result = (
        nnunet
        / "results/Dataset501_Lumbar/nnUNetTrainerLumbar__nnUNetResEncUNetMPlans__3d_fullres/fold_0"
    )
    candidates = {}
    for name in ("midway", "final"):
        checkpoint = root / f"{name}.pt"
        run(
            "export_nnunet.py",
            result / f"checkpoint_{name}.pth",
            "--root",
            nnunet,
            "--normalizer",
            normalizer,
            "--output",
            checkpoint,
        )
        validation = root / f"validation_{name}"
        run(
            "validate_model.py",
            "--checkpoint",
            checkpoint,
            "--manifest",
            manifest,
            "--normalizer",
            normalizer,
            "--device",
            device,
            "--output",
            validation,
        )
        summary = json.loads((validation / "summary.json").read_text())
        if summary["summary"]["attempted_cases"] != 57:
            raise ValueError("Selection requires all 57 source validation cases")
        if summary["checkpoint_sha256"] != file_hash(checkpoint):
            raise ValueError("Validation checkpoint changed")
        candidates[name] = summary
    winner = max(candidates, key=lambda name: candidates[name]["summary"]["macro"])
    summary = candidates[winner]["summary"]
    useful = summary["macro"] >= 0.5
    for level in summary["levels"].values():
        useful = useful and level["mean"] is not None and level["mean"] >= 0.2
    write_json(
        root / "selection.json",
        {
            "selected": winner,
            "validation": summary,
            "checkpoint_sha256": file_hash(root / f"{winner}.pt"),
            "candidates": {
                name: result["summary"] for name, result in candidates.items()
            },
            "useful_baseline_check_passed": useful,
            "criterion": "Internal check: macro Dice >= 0.50 and every level mean >= 0.20",
            "test_data_used_for_selection": False,
        },
    )
    if file_hash(selected) != selected_hash:
        raise ValueError("Packaged inference weights changed")
    if not useful:
        raise RuntimeError("Full-volume learning check failed; inspect selection.json")
    shutil.copy2(root / f"{winner}.pt", root / "model.pt")
    print(
        f"New model: {root / 'model.pt'}; source validation macro Dice: {summary['macro']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
