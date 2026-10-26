#!/usr/bin/env python3
"""Convert our own framework checkpoint to the reference-free inference package."""

import argparse
import json
from pathlib import Path
import torch
from lumbar_ct.artifacts import file_hash
from lumbar_ct.nnunet_inference import tensor_order


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--root", default="cache/nnunet")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--parent-checkpoint",
        type=Path,
        help="Our own starting weights when exporting a fine-tuned model",
    )
    p.add_argument("--normalizer", type=Path, required=True)
    args = p.parse_args()
    dataset = Path(args.root) / "preprocessed" / "Dataset501_Lumbar"
    plans = json.loads((dataset / "nnUNetResEncUNetMPlans.json").read_text())
    dataset_json = json.loads((dataset / "dataset.json").read_text())
    split = json.loads((dataset / "splits_final.json").read_text())[0]
    # Only load checkpoints produced by this project's own trainer.
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    patch = [0, 0, 0]
    for i, axis in enumerate(tensor_order(plans)):
        patch[axis] = plans["configurations"]["3d_fullres"]["patch_size"][i]
    result = {
        "kind": "nnunet_resenc_m",
        "state_dict": source["network_weights"],
        "plans": plans,
        "dataset_json": dataset_json,
        "config": {"patch_xyz": patch, "spacing_mm": [2, 2, 2]},
        "normalizer": json.loads(args.normalizer.read_text()),
        "steps_completed": source["current_epoch"] * 250,
        "train_ids": split["train"],
        "validation_ids": split["val"],
        "source_checkpoint_sha256": file_hash(args.checkpoint),
        "nnunet_version": "2.8.1",
        "axis_order": "prepared XYZ converted to reader ZYX then plans transpose; inverse applied to scores",
        "inference": "official Gaussian sliding-window logits; 50% overlap; no mirroring",
    }
    if args.parent_checkpoint:
        parent = torch.load(
            args.parent_checkpoint, map_location="cpu", weights_only=True
        )
        assert parent["plans"] == plans
        assert sorted(parent["validation_ids"]) == sorted(split["val"])
        assert set(parent["train_ids"]).issubset(split["train"])
        result["fine_tune_steps"] = result["steps_completed"]
        result["parent_steps"] = parent["steps_completed"]
        result["steps_completed"] += parent["steps_completed"]
        result["parent_checkpoint_sha256"] = file_hash(args.parent_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
