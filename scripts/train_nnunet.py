#!/usr/bin/env python3
""" train a fresh nnU-Net ResEnc M."""

import argparse
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from lumbar_ct.artifacts import write_json
from lumbar_ct.training import choose_device
from contextlib import nullcontext


def seed_all():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def autocast(device):
    if device.type == "cuda":
        return torch.autocast("cuda")
    return nullcontext()


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--root", default="cache/nnunet")
    p.add_argument("--resume", action="store_true")

    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = p.parse_args()
    root = Path(args.root).resolve()

    for env, folder in [
        ("nnUNet_raw", "raw"),
        ("nnUNet_preprocessed", "preprocessed"),
        ("nnUNet_results", "results"),
    ]:
        os.environ[env] = str(root / folder)
    os.environ["nnUNet_compile"] = "false"
    os.environ.setdefault("nnUNet_n_proc_DA", "4")
    torch.set_num_threads(4)
    device = choose_device(args.device)

    from lumbar_ct.nnunet_trainer import nnUNetTrainerLumbar
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

    dataset = root / "preprocessed" / "Dataset501_Lumbar"
    plans = json.loads((dataset / "nnUNetResEncUNetMPlans.json").read_text())
    dataset_json = json.loads((dataset / "dataset.json").read_text())
    cfg = plans["configurations"]["3d_fullres"]

    evidence = root / "repair_evidence"
    evidence.mkdir(parents=True, exist_ok=True)

    if not args.resume:
        seed_all()
        trainer = nnUNetTrainerLumbar(
            {**plans, "continue_training": False},
            "3d_fullres",
            0,
            dataset_json,
            device=device,
        )
        # A diagnostic is kept outside the final training result folder.
        trainer.output_folder = str(evidence / "overfit")
        Path(trainer.output_folder).mkdir(exist_ok=True)
        trainer.initialize()
        stored = nnUNetDatasetBlosc2(
            str(dataset / cfg["data_identifier"]), identifiers=["s0739"]
        )
        data, seg, _, _ = stored.load_case("s0739")
        data = np.asarray(data)
        seg = np.asarray(seg)
        points = np.argwhere(seg[0] > 0)
        center = (points.min(0) + points.max(0)) // 2
        patch = np.array(cfg["patch_size"])
        start = center - patch // 2

        x = np.zeros((1, *patch), dtype=np.float32)
        y = np.zeros((1, *patch), dtype=np.int64)

        lo = np.maximum(start, 0)
        hi = np.minimum(start + patch, data.shape[1:])
        src = (slice(None), *[slice(a, b) for a, b in zip(lo, hi)])
        dst = (slice(None), *[slice(a - s, b - s) for a, b, s in zip(lo, hi, start)])
        x[dst] = data[src]
        y[dst] = seg[src]
        for level in range(1, 6):
            assert (y == level).sum() == (seg == level).sum() > 0
        x = torch.from_numpy(np.stack([x, x])).to(device)
        y = torch.from_numpy(np.stack([y, y])).to(device)
        trainer.network.train()
        with torch.no_grad(), autocast(device):
            outputs = trainer.network(x)
        targets = [
            F.interpolate(y.float(), size=o.shape[2:], mode="nearest").long()
            for o in outputs
        ]
        batch = {"data": x, "target": targets}
        history = []
        start_time = time.monotonic()
        passed = False

        for step in range(1, 501):
            trainer.network.train()
            loss = trainer.train_step(batch)
            if step % 25 == 0:
                trainer.network.eval()
                with torch.no_grad(), autocast(device):
                    pred = trainer.network(x)[0].argmax(1)
                target = y[:, 0]
                dice = []
                for level in range(1, 6):
                    prediction_mask = pred == level
                    reference_mask = target == level
                    intersection = (prediction_mask & reference_mask).sum()
                    total = prediction_mask.sum() + reference_mask.sum()
                    level_dice = 2 * intersection / total
                    dice.append(float(level_dice.item()))
                row = {
                    "step": step,
                    "loss": float(loss["loss"]),
                    "dice": dice,
                    "seconds": time.monotonic() - start_time,
                }
                history.append(row)
                print("OVERFIT " + json.dumps(row), flush=True)
                if min(dice) >= 0.8 and np.mean(dice) >= 0.9:
                    passed = True
                    break
        write_json(
            evidence / "overfit.json",
            {
                "case_id": "s0739",
                "passed": passed,
                "history": history,
                "criterion": "every level Dice >= 0.8 and mean >= 0.9 on a fixed source-training patch",
                "weights_reused_for_final_training": False,
            },
        )
        if not passed:
            raise RuntimeError("All-level recovery check failed; full run not launched")
        del trainer, stored, x, y, targets, outputs, batch, pred, target
        if device.type == "cuda":
            torch.cuda.empty_cache()
    seed_all()
    trainer = nnUNetTrainerLumbar(
        {**plans, "continue_training": args.resume},
        "3d_fullres",
        0,
        dataset_json,
        device=device,
    )

    if args.resume:
        trainer.load_checkpoint(
            str(Path(trainer.output_folder) / "checkpoint_latest.pth")
        )
    write_json(
        evidence / "run_config.json",
        {
            "framework": "nnunetv2 2.8.1",
            "architecture": cfg["architecture"],
            "patch": cfg["patch_size"],
            "batch_size": cfg["batch_size"],
            "epochs": trainer.num_epochs,
            "iterations_per_epoch": trainer.num_iterations_per_epoch,
            "initial_lr": trainer.initial_lr,
            "optimizer": "official Nesterov SGD",
            "loss": "official deep-supervision Dice + cross entropy",
            "foreground_sampling_probability": 0.5,
            "mirroring": False,
            "initialization": "random; diagnostic and old custom model weights not reused",
            "seed": 42,
            "device": str(device),
            "torch_version": str(torch.__version__),
            "checkpoint_selection": "full-volume source-validation macro Dice between midway and final; per-level failures retained",
            "split": json.loads((dataset / "splits_final.json").read_text())[0],
        },
    )
    trainer.run_training()
    print("FINAL TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
