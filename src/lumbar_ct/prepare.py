"""Prepare both datasets on a common grid defined by the CT, not its labels."""

from dataclasses import dataclass
import csv
import io
from pathlib import Path

import numpy as np
import nibabel as nib

from .artifacts import file_hash, reject_path_overlap, write_json
from .datasets import LEVELS, SOURCE_FILES, map_verse, validate_labels
from .geometry import (
    Grid,
    acquisition_support,
    load_volume,
    paired_mm_units_policy,
    resample,
    save_volume,
    shared_grid,
)

SOURCE_UNITS_POLICY = {
    "unit": "mm",
    "expected_spacing_mm": 1.5,
    "provenance": "TotalSegmentator source paper: images resampled to 1.5-mm isotropic resolution; "
    "publisher v2 notes retain the v1 public training/validation subjects. "
    "Explicit project interpretation of missing unit codes, not inferred solely from affine numbers.",
    "sources": [
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC10546353/",
        "https://github.com/wasserth/TotalSegmentator/blob/master/resources/improvements_in_v2.md",
    ],
    "scope": "TotalSegmentator CT v2.0.1 adapter only; reject inconsistent spacing",
}


@dataclass
class PreparedCase:
    ct: np.ndarray
    labels: np.ndarray | None
    support: np.ndarray
    grid: Grid
    record: dict
    binary_labels: dict | None = None


def source_metadata(path):
    return parse_source_metadata(Path(path).read_text(encoding="utf-8-sig"))


def parse_source_metadata(text):
    rows = list(csv.DictReader(io.StringIO(text.lstrip("\ufeff")), delimiter=";"))
    if not rows:
        raise ValueError("expected release meta.csv with image_id and split columns")

    if not {"image_id", "split"} <= set(rows[0]):
        raise ValueError("expected release meta.csv with image_id and split columns")
    result = {}
    for row in rows:
        key = row["image_id"]
        if key in result:
            raise ValueError(f"duplicate metadata case: {key}")
        split = {"train": "train", "val": "validation", "test": "test"}.get(
            row["split"]
        )
        if split is None:
            raise ValueError(f"unknown official split: {row['split']}")
        result[key] = {
            "case_id": key,
            "split": split,
            "subject_id": None,
            "series_id": None,
        }
    return result


def _base_record(ct_path, ct, target, case_id, split, release):
    if split not in {"train", "validation", "test"}:
        raise ValueError("case identity and declared split required")

    if not case_id:
        raise ValueError("case identity and declared split required")
    return {
        "case_id": case_id,
        "split": split,
        "release": release,
        "subject_id": None,
        "series_id": None,
        "ct_path": str(Path(ct_path).resolve()),
        "ct_sha256": file_hash(ct_path),
        "native_grid": ct.grid.to_dict(),
        "shared_grid": target.to_dict(),
        "ct_audit": ct.audit,
        "array_order": "XYZ",
        "coordinate_system": "RAS",
        "unit": "mm",
        "ct_interpolation": "linear, nearest centre extended to voxel boundary; no antialiasing",
        "label_interpolation": "nearest neighbour",
        "outside_hu": -1024,
        "raw_hu_preserved": True,
        "normalization": "not_fitted",
        "audit_status": "passed",
        "audit_eligible": True,
        "eligible": False,
        "release_integrity": "not_verified",
        "eligibility_status": "awaiting_release_verification_and_split_audit",
        "exclusion_reason": None,
    }


def _check_targets(before, after, *, allow_loss=False):
    lost = [
        name
        for i, name in enumerate(LEVELS, 1)
        if np.any(before == i) and not np.any(after == i)
    ]
    if lost and not allow_loss:
        raise ValueError(f"targets lost in resampling: {lost}")
    return lost


# Prepare the source CT and its five lumbar masks.
def prepare_source(
    case_dir,
    *,
    case_id,
    split,
    spacing=2.0,
    allow_overlaps=False,
    allow_target_loss=False,
):
    case_dir = Path(case_dir)
    ct_path = case_dir / "ct.nii.gz"
    ct = load_volume(ct_path, units_policy=SOURCE_UNITS_POLICY)
    target = shared_grid(ct.grid, spacing)
    native_labels = np.zeros(ct.grid.shape, np.uint8)
    labels = np.zeros(target.shape, np.uint8)
    audits = {}
    paths = {}
    hashes = {}
    coverage = {}
    binary_labels = {}
    overlaps = []
    for index, (name, filename) in enumerate(SOURCE_FILES.items(), 1):
        path = case_dir / "segmentations" / filename
        if not path.is_file():
            raise ValueError(f"missing source reference {name}: {path}")
        volume = load_volume(path, categorical=True, units_policy=SOURCE_UNITS_POLICY)
        validate_labels(volume.data, [0, 1])
        # Native alignment is an overlap/coverage audit only. Training labels
        # are interpolated directly from their original grids to the shared grid.
        native_mask = resample(volume.data, volume.grid, ct.grid, kind="label").astype(
            bool
        )
        target_mask = resample(volume.data, volume.grid, target, kind="label").astype(
            bool
        )
        native_overlap = int(np.count_nonzero(native_mask & (native_labels > 0)))
        target_overlap = int(np.count_nonzero(target_mask & (labels > 0)))
        if native_overlap or target_overlap:
            if not allow_overlaps:
                raise ValueError(f"source overlap while merging {name}: {path}")
            overlaps.append(
                {
                    "level": name,
                    "native_voxels": native_overlap,
                    "target_voxels": target_overlap,
                }
            )
        if allow_overlaps:
            binary_labels[name] = target_mask
        native_labels[native_mask] = index
        labels[target_mask] = index
        audits[name] = volume.audit
        paths[name] = str(path.resolve())
        hashes[name] = file_hash(path)
        present = bool(volume.data.any())
        if present:
            coverage[name] = "annotated_present"
        else:
            coverage[name] = "empty_annotation"
        if present and not native_mask.any():
            raise ValueError(f"source reference {name} has no CT-grid support")
    support = acquisition_support(ct.grid, target)
    labels[~support] = 0
    lost = []
    if allow_overlaps:
        for name, mask in binary_labels.items():
            mask[~support] = False
            if coverage[name] == "annotated_present" and not mask.any():
                lost.append(name)
        if lost and not allow_target_loss:
            raise ValueError(f"targets lost in resampling: {lost}")
    else:
        lost = _check_targets(native_labels, labels, allow_loss=allow_target_loss)
    record = _base_record(
        ct_path, ct, target, case_id, split, "TotalSegmentator CT v2.0.1"
    )
    observed_labels = []
    for level, name in enumerate(LEVELS, 1):
        if allow_overlaps:
            present = binary_labels[name].any()
        else:
            present = np.any(labels == level)
        if present:
            observed_labels.append(level)

    if overlaps:
        label_format = "separate_binary"
        merged_labels = None
        separate_labels = binary_labels
    else:
        label_format = "multiclass"
        merged_labels = labels
        separate_labels = None

    record.update(
        label_paths=paths,
        label_sha256=hashes,
        label_audits=audits,
        coverage=coverage,
        observed_labels=observed_labels,
        variant_flags=[],
        overlaps=overlaps,
        lost_target_levels=lost,
        label_format=label_format,
    )
    return PreparedCase(
        resample(ct.data, ct.grid, target, kind="ct"),
        merged_labels,
        support,
        target,
        record,
        separate_labels,
    )


# Assume millimetres only for the audited VerSe release.
def verse_release_units_policy(path):
    image = nib.load(str(path))
    if image.header.get_xyzt_units()[0] != "unknown":
        return None
    spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
    if not np.isfinite(spacing).all():
        raise ValueError("spacing inconsistent with declared VerSe release policy")

    if np.any(spacing < 0.1):
        raise ValueError("spacing inconsistent with declared VerSe release policy")

    if np.any(spacing > 5.0):
        raise ValueError("spacing inconsistent with declared VerSe release policy")
    return {
        "unit": "mm",
        "expected_spacing_mm": spacing.tolist(),
        "provenance": (
            "Explicit project interpretation for the verified VerSe 2020 release: publisher "
            "data utilities interpret header zooms/affines in mm when resampling and "
            "rescaling centroids. Missing units are assumed mm under that convention; raw "
            "files are unchanged. Sidecar acquisition spacing is not substituted."
        ),
        "source": "https://github.com/anjany/verse/blob/main/utils/data_utilities.py",
        "scope": "verified VerSe release only; explicit opt-in, not a generic NIfTI repair",
    }


# Prepare the VerSe CT and convert its lumbar labels.
def prepare_verse(
    ct_path,
    label_path,
    *,
    case_id,
    spacing=2.0,
    allow_release_units=False,
    allow_target_loss=False,
):
    if allow_release_units:
        ct_policy = verse_release_units_policy(ct_path)
    else:
        ct_policy = None
    ct = load_volume(ct_path, units_policy=ct_policy)
    units_policy = paired_mm_units_policy(
        label_path, ct.grid, ct.audit["units"], ct_path
    )
    if units_policy is None and allow_release_units:
        units_policy = verse_release_units_policy(label_path)
    reference = load_volume(label_path, categorical=True, units_policy=units_policy)
    native_mapped, variants = map_verse(reference.data)
    target = shared_grid(ct.grid, spacing)
    labels = resample(native_mapped, reference.grid, target, kind="label").astype(
        np.uint8
    )
    support = acquisition_support(ct.grid, target)
    labels[~support] = 0
    lost = _check_targets(native_mapped, labels, allow_loss=allow_target_loss)
    record = _base_record(
        ct_path, ct, target, case_id, "test", "VerSe 2020 restructured 03_test"
    )
    coverage = {}
    for level, name in enumerate(LEVELS, 1):
        if np.any(native_mapped == level):
            coverage[name] = "annotated_present"
        else:
            coverage[name] = "empty_annotation"

    record.update(
        label_paths={"verse": str(Path(label_path).resolve())},
        label_sha256={"verse": file_hash(label_path)},
        label_audits={"verse": reference.audit},
        observed_labels=sorted(int(v) for v in np.unique(labels) if v),
        variant_flags=variants,
        label_format="multiclass",
        lost_target_levels=lost,
        coverage=coverage,
    )
    return PreparedCase(
        resample(ct.data, ct.grid, target, kind="ct"), labels, support, target, record
    )


def write_prepared(case, output_dir):
    output_dir = Path(output_dir)
    for path in [case.record["ct_path"], *case.record["label_paths"].values()]:
        reject_path_overlap(Path(path).parent, output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("prepared output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_volume(output_dir / "ct_hu.nii.gz", case.ct, case.grid)
    outputs = ["ct_hu.nii.gz", "support.nii.gz"]
    if case.binary_labels is not None:
        for name, mask in case.binary_labels.items():
            filename = f"{name}.nii.gz"
            save_volume(output_dir / filename, mask.astype(np.uint8), case.grid)
            outputs.append(filename)
    else:
        save_volume(
            output_dir / "labels.nii.gz", case.labels.astype(np.uint8), case.grid
        )
        outputs.append("labels.nii.gz")
    save_volume(output_dir / "support.nii.gz", case.support.astype(np.uint8), case.grid)
    record = dict(case.record)
    record["outputs_sha256"] = {name: file_hash(output_dir / name) for name in outputs}
    # Written last: record.json is the completion marker for this cache entry.
    write_json(output_dir / "record.json", record)
