#!/usr/bin/env python3
"""Run the fixed source-validation cohort; references never enter inference."""

import argparse
import json
from pathlib import Path
import time
import nibabel as nib
import numpy as np
import torch
from lumbar_ct.artifacts import file_hash, write_json
from lumbar_ct.body import body_centroids
from lumbar_ct.evaluate import score_case, summarize
from lumbar_ct.geometry import Grid
from lumbar_ct.inference import export_prepared
from lumbar_ct.nnunet_inference import load_checkpoint


def review_panels(ct, ref, pred, body, refbody, path):
    # Centres come from reference only for displaying alignment, never for inference.
    levels = [i for i in range(1, 6) if np.any(ref == i)]
    if not levels:
        return
    level = levels[len(levels) // 2]
    points = np.argwhere(ref == level)
    center = np.rint(points.mean(0)).astype(int)
    panels = {}
    for axis in range(3):
        varying = [i for i in range(3) if i != axis]
        selection = [
            slice(max(0, int(c) - 38), min(n, int(c) + 39))
            for c, n in zip(center, ct.shape)
        ]
        selection[axis] = int(center[axis])
        for name, data in [
            ("ct", ct),
            ("reference", ref),
            ("prediction", pred),
            ("body", body),
            ("reference_body", refbody),
        ]:
            panels[f"{name}_{axis}"] = np.flip(data[tuple(selection)].T, axis=0)
    np.savez_compressed(path, **panels, level=level, center_xyz=center)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/nnunet/model.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--output", type=Path, default=Path("outputs/nnunet/validation_selected")
    )
    p.add_argument(
        "--manifest", type=Path, required=True,
        help="Verified source-development manifest",
    )
    p.add_argument("--normalizer", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--reuse",
        type=Path,
        help="Reuse saved predictions from a hash-verified baseline validation run",
    )
    args = p.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    source_manifest = args.manifest
    manifest = json.loads(source_manifest.read_text())
    sha = file_hash(args.checkpoint)
    manifest_sha = file_hash(source_manifest)
    if args.manifest:
        from lumbar_ct.source_development import validate_development

        official = [
            {
                "case_id": case_id,
                "split": split,
                "dataset": "totalsegmentator_v201",
            }
            for case_id, split in manifest["official_splits"].items()
        ]
        if manifest.get("selection_mode") == "full_source":
            from lumbar_ct.training import validate_full_source

            validate_full_source(manifest, official)
            if set(checkpoint["train_ids"]) != set(manifest["train_ids"]):
                raise ValueError(
                    "Checkpoint training cohort differs from the full-source manifest"
                )
            if set(checkpoint["validation_ids"]) != set(manifest["validation_ids"]):
                raise ValueError(
                    "Checkpoint validation cohort differs from the full-source manifest"
                )
        else:
            validate_development(manifest, official)
        training_ids = {
            r["case_id"]
            for r in official
            if r["dataset"] == "totalsegmentator_v201" and r["split"] == "train"
        }
        if not set(checkpoint["train_ids"]) <= training_ids:
            raise ValueError("Checkpoint contains non-training source IDs")
        if file_hash(args.normalizer) != manifest["normalizer_sha256"]:
            raise ValueError("Frozen normalization changed")
    if checkpoint["normalizer"] != json.loads(args.normalizer.read_text()):
        raise ValueError("Checkpoint normalization differs")
    rows = [r for r in manifest["cases"] if r["role"] == "validation"]
    if set(r["case_id"] for r in rows) & set(checkpoint["train_ids"]):
        raise ValueError("Validation overlaps checkpoint training IDs")
    if args.manifest:
        args.output.mkdir(parents=True, exist_ok=args.resume)
    reuse = {}
    if args.reuse:
        saved = json.loads((args.reuse / "summary.json").read_text())
        if saved["checkpoint_sha256"] != sha:
            raise ValueError("Reused predictions belong to a different checkpoint")
        reuse = {r["case_id"]: r for r in saved["cases"]}
    results = []
    start = time.monotonic()
    for row in rows:
        path = Path(row["path"])
        out = args.output / row["case_id"]
        if args.manifest:
            record = json.loads((path / "record.json").read_text())
            if record["split"] != "validation" or record["case_id"] != row["case_id"]:
                raise ValueError("Prepared validation split or identity differs")
            for name, expected in row["prepared_sha256"].items():
                if file_hash(path / name) != expected:
                    raise ValueError("Prepared validation input changed")
        if args.resume and (out / "validation.json").exists():
            saved = json.loads((out / "validation.json").read_text())
            if (
                saved.get("checkpoint_sha256") != sha
                or saved.get("development_manifest_sha256") != manifest_sha
            ):
                raise ValueError("Resume provenance differs")
            if saved["case_id"] != row["case_id"]:
                raise ValueError("Resume case differs")
            if all(
                (out / name).exists()
                for name in [
                    "vertebrae_2mm.nii.gz",
                    "bodies_2mm.nii.gz",
                    "centroids.json",
                ]
            ):
                results.append(saved)
                continue
        img = nib.load(row["normalized_ct"])
        ct = np.asarray(img.dataobj, dtype=np.float32)
        grid = Grid(ct.shape, img.affine)
        support = np.asarray(nib.load(str(path / "support.nii.gz")).dataobj, dtype=bool)
        if row["case_id"] in reuse:
            import shutil

            source = args.reuse / row["case_id"]
            image = nib.load(source / "vertebrae_2mm.nii.gz")
            np.testing.assert_allclose(image.affine, grid.affine, atol=1e-5)
            if image.shape != grid.shape:
                raise ValueError("Reused prediction grid differs")
            pred = np.asarray(image.dataobj, dtype=np.uint8)
            body = np.asarray(
                nib.load(source / "bodies_2mm.nii.gz").dataobj, dtype=np.uint8
            )
            centroids = json.loads((source / "centroids.json").read_text())
            out.mkdir(parents=True, exist_ok=True)
            for name in ["vertebrae_2mm.nii.gz", "bodies_2mm.nii.gz", "centroids.json"]:
                shutil.copy2(source / name, out / name)
        else:
            _, pred, body, centroids = export_prepared(
                model,
                checkpoint,
                ct,
                grid,
                support,
                out,
                case_id=row["case_id"],
                device=args.device,
            )
        ref = np.asarray(nib.load(str(path / "labels.nii.gz")).dataobj, dtype=np.uint8)
        refbody, _ = body_centroids(ref, grid, case_id=row["case_id"])
        hu = np.asarray(nib.load(str(path / "ct_hu.nii.gz")).dataobj, dtype=np.float32)
        review_panels(hu, ref, pred, body, refbody, out / "review.npz")
        result = {
            "case_id": row["case_id"],
            "scores": score_case(pred, ref),
            "centroids": centroids["levels"],
            "checkpoint_sha256": sha,
            "development_manifest_sha256": manifest_sha,
            "status": "completed",
            "reused_prediction": row["case_id"] in reuse,
        }
        if (
            row["case_id"] in reuse
            and result["scores"] != reuse[row["case_id"]]["scores"]
        ):
            raise ValueError(
                "Reused prediction scores differ from the verified original run"
            )
        results.append(result)
        write_json(out / "validation.json", result)
        print(
            json.dumps(
                {
                    "case_id": row["case_id"],
                    "dice": {k: v["dice"] for k, v in result["scores"].items()},
                    "seconds": time.monotonic() - start,
                }
            ),
            flush=True,
        )
    assert file_hash(args.checkpoint) == sha
    write_json(
        args.output / "summary.json",
        {
            "checkpoint_sha256": sha,
            "cases": results,
            "development_manifest_sha256": manifest_sha,
            "validation_ids": [r["case_id"] for r in rows],
            "summary": summarize([r["scores"] for r in results]),
            "seconds": time.monotonic() - start,
            "grid": "prepared 2 mm; validation diagnostics, not native-grid final test metrics",
            "selection": "candidate checkpoint; compare complete validation summaries before selecting final weights",
        },
    )


if __name__ == "__main__":
    main()
