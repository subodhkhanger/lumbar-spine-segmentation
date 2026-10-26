#!/usr/bin/env python3
"""Adapt frozen prepared development data to official nnU-Net without renormalizing."""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import nibabel as nib
from lumbar_ct.artifacts import write_json, file_hash

DATASET = "Dataset501_Lumbar"
PLANS = "nnUNetResEncUNetMPlans"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("cache/nnunet"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--adapter-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    for name, folder in [
        ("nnUNet_raw", "raw"),
        ("nnUNet_preprocessed", "preprocessed"),
        ("nnUNet_results", "results"),
    ]:
        os.environ[name] = str(root / folder)
        (root / folder).mkdir(parents=True, exist_ok=True)
    rows = [
        r
        for r in json.loads(args.manifest.read_text())["cases"]
        if r["role"] in ("train", "validation")
    ]
    splits = [
        {
            "train": [r["case_id"] for r in rows if r["role"] == "train"],
            "val": [r["case_id"] for r in rows if r["role"] == "validation"],
        }
    ]
    normalizer = json.loads(args.normalizer.read_text())
    assert sorted(splits[0]["train"]) == sorted(normalizer["fit_ids"])
    raw = root / "raw" / DATASET
    for sub in ["imagesTr", "labelsTr"]:
        (raw / sub).mkdir(parents=True, exist_ok=True)
    for row in rows:
        for source, destination in [
            (
                Path(row["normalized_ct"]),
                raw / "imagesTr" / (row["case_id"] + "_0000.nii.gz"),
            ),
            (
                Path(row["path"]) / "labels.nii.gz",
                raw / "labelsTr" / (row["case_id"] + ".nii.gz"),
            ),
        ]:
            if not destination.exists():
                os.link(source, destination)
    dataset = {
        "channel_names": {"0": "nonorm"},
        "labels": {"background": 0, **{f"L{i}": i for i in range(1, 6)}},
        "numTraining": len(rows),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIO",
    }
    write_json(raw / "dataset.json", dataset)
    from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import (
        DatasetFingerprintExtractor,
    )
    from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import (
        nnUNetPlannerResEncM,
    )
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
        DefaultPreprocessor,
    )
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

    extractor = DatasetFingerprintExtractor(DATASET, num_processes=2)
    extractor.dataset = {
        k: v for k, v in extractor.dataset.items() if k in splits[0]["train"]
    }
    extractor.num_foreground_voxels_for_intensitystats = 320000
    extractor.run()
    planner = nnUNetPlannerResEncM(DATASET)
    plans = planner.plan_experiment()
    prepared = root / "preprocessed" / DATASET
    write_json(prepared / "splits_final.json", splits)
    cfg = plans["configurations"]["3d_fullres"]
    assert cfg["normalization_schemes"] == ["NoNormalization"]
    assert cfg["spacing"] == [2.0, 2.0, 2.0]
    DefaultPreprocessor(verbose=False).run(
        DATASET, "3d_fullres", PLANS, num_processes=2
    )
    manager = PlansManager(plans)
    reader = manager.image_reader_writer_class()
    stored = nnUNetDatasetBlosc2(
        str(prepared / cfg["data_identifier"]), identifiers=[r["case_id"] for r in rows]
    )
    for row in rows:
        images, properties = reader.read_images(
            [str(raw / "imagesTr" / (row["case_id"] + "_0000.nii.gz"))]
        )
        expected = images.transpose([0, *[i + 1 for i in manager.transpose_forward]])
        data, seg, previous, record = stored.load_case(row["case_id"])
        np.testing.assert_array_equal(np.asarray(data), expected)
        ref = np.asarray(
            nib.load(str(raw / "labelsTr" / (row["case_id"] + ".nii.gz"))).dataobj
        )
        expected_seg = ref.transpose(2, 1, 0).transpose(manager.transpose_forward)
        np.testing.assert_array_equal(np.asarray(seg)[0], expected_seg)
    write_json(
        args.adapter_output,
        {
            "dataset": DATASET,
            "plans": PLANS,
            "configuration": "3d_fullres",
            "split": splits[0],
            "normalizer_sha256": file_hash(args.normalizer),
            "fingerprint_ids": sorted(splits[0]["train"]),
            "normalization": "existing preprocessing only; framework NoNormalization",
            "reader": "NibabelIO: XYZ to ZYX",
            "transpose_forward": manager.transpose_forward,
            "patch_tensor_axes": cfg["patch_size"],
            "batch_size": cfg["batch_size"],
            "architecture": cfg["architecture"],
            "checked_cases": len(rows),
            "checks": "All cached image and label arrays exactly match the prepared inputs after documented axis permutation; no second interpolation or normalization",
        },
    )
    print(
        "nnU-Net adapter verified:",
        cfg["patch_size"],
        "batch",
        cfg["batch_size"],
        flush=True,
    )


if __name__ == "__main__":
    main()
