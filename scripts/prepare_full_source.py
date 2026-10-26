#!/usr/bin/env python3
"""Prepare every usable official training case and all source validation cases."""

import argparse
from collections import Counter
import json
from pathlib import Path

from lumbar_ct.artifacts import file_hash, write_json
from lumbar_ct.datasets import SOURCE_FILES, validate_splits
from lumbar_ct.geometry import load_volume, save_volume
from lumbar_ct.normalization import fit_normalizer
from lumbar_ct.prepare import prepare_source, source_metadata, write_prepared
from lumbar_ct.training import validate_full_source


def training_cases(rows):
    for row in rows:
        if row["role"] != "train":
            continue
        folder = Path(row["path"])
        yield {
            "case_id": row["case_id"],
            "split": "train",
            "ct": load_volume(folder / "ct_hu.nii.gz").data,
            "labels": load_volume(folder / "labels.nii.gz", categorical=True).data,
        }


# Reuse the existing geometry and label checks.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    metadata = source_metadata(args.data_dir / "meta.csv")
    official = [
        {**entry, "dataset": "totalsegmentator_v201"}
        for entry in metadata.values()
    ]
    validate_splits(official)
    counts = Counter(entry["split"] for entry in metadata.values())
    if (counts["train"], counts["validation"], counts["test"]) != (1082, 57, 89):
        raise ValueError("Expected the official 1,082/57/89 source split")
    for case_id, entry in metadata.items():
        folder = args.data_dir / case_id
        required = [folder / "ct.nii.gz"]
        if entry["split"] != "test":
            required.extend(
                folder / "segmentations" / name for name in SOURCE_FILES.values()
            )
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Complete source data is required; missing {path}"
                )
    rows = []
    exclusions = []
    validation_failures = []
    records = []
    for case_id, entry in sorted(metadata.items()):
        if entry["split"] == "test":
            continue
        role = entry["split"]
        folder = root / "cases" / case_id
        try:
            case = prepare_source(
                args.data_dir / case_id,
                case_id=case_id,
                split=role,
                allow_target_loss=role == "validation",
            )
            write_prepared(case, folder)
            records.append(case.record)
        except ValueError as exc:
            failure = {"case_id": case_id, "error": str(exc)}
            if role == "train":
                exclusions.append(failure)
            else:
                validation_failures.append(failure)
            print(f"REJECTED {role} {case_id}: {exc}", flush=True)
            continue
        rows.append(
            {
                "case_id": case_id,
                "dataset": "totalsegmentator_v201",
                "role": role,
                "status": "prepared",
                "path": str(folder),
            }
        )
        print(f"PREPARED {role} {case_id}", flush=True)
    write_json(root / "preparation_exclusions.json", exclusions)
    write_json(root / "validation_failures.json", validation_failures)
    if validation_failures:
        raise RuntimeError(
            "Validation preparation failed; resolve recorded failures before training"
        )
    test_cases = [
        {
            "case_id": case_id,
            "split": "test",
            "ct_sha256": file_hash(args.data_dir / case_id / "ct.nii.gz"),
        }
        for case_id, entry in metadata.items()
        if entry["split"] == "test"
    ]
    validate_splits(records + test_cases)
    train_ids = sorted(row["case_id"] for row in rows if row["role"] == "train")
    manifest = {
        "selection_mode": "full_source",
        "official_splits": {case_id: row["split"] for case_id, row in metadata.items()},
        "train_ids": train_ids,
        "validation_ids": sorted(
            row["case_id"] for row in rows if row["role"] == "validation"
        ),
        "preparation_exclusions": exclusions,
        "cases": rows,
        "test_data_used_for_selection": False,
    }
    validate_full_source(manifest, official)
    normalizer = fit_normalizer(training_cases(rows), allowed_ids=train_ids, seed=42)
    normalizer_path = root / "normalization.json"
    write_json(normalizer_path, normalizer.to_dict())
    for row in rows:
        folder = Path(row["path"])
        volume = load_volume(folder / "ct_hu.nii.gz")
        output = folder / "ct_normalized.nii.gz"
        save_volume(output, normalizer.apply(volume.data), volume.grid)
        record_path = folder / "record.json"
        record = json.loads(record_path.read_text())
        record["normalizer_sha256"] = normalizer.sha256()
        record["outputs_sha256"][output.name] = file_hash(output)
        record["normalization"] = "training_foreground_percentile_clip_then_zscore"
        record["eligible"] = True
        write_json(record_path, record)
        row["normalized_ct"] = str(output)
        row["prepared_sha256"] = {}
        for name in ("ct_normalized.nii.gz", "support.nii.gz", "labels.nii.gz"):
            row["prepared_sha256"][name] = file_hash(folder / name)
    manifest["normalizer_sha256"] = file_hash(normalizer_path)
    write_json(root / "manifest.json", manifest)
    print(
        f"READY: {len(train_ids)} training, 57 validation, {len(exclusions)} rejected training scans",
        flush=True,
    )


if __name__ == "__main__":
    main()
