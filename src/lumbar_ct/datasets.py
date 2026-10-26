"""Lumbar label mapping and checks for data leakage."""

import random

import numpy as np

LEVELS = ("L1", "L2", "L3", "L4", "L5")
SOURCE_FILES = {name: f"vertebrae_{name}.nii.gz" for name in LEVELS}

# VerSe label number -> our model's label number


VERSE_MAP = {
    20: 1,
    21: 2,
    22: 3,
    23: 4,
    24: 5,
}


def validate_labels(data, allowed):
    data = np.asarray(data)

    if data.ndim != 3:
        raise ValueError("labels must be a finite integer XYZ array")

    if not np.isfinite(data).all():
        raise ValueError("labels must be a finite integer XYZ array")

    rounded = np.rint(data)
    if not np.array_equal(data, rounded):
        raise ValueError("labels must be a finite integer XYZ array")

    observed_labels = set(np.unique(data).tolist())
    allowed_labels = set(allowed)
    unexpected_labels = observed_labels - allowed_labels

    if unexpected_labels:
        unexpected = sorted(unexpected_labels)
        raise ValueError(f"unexpected labels: {unexpected}")

    return data


# Convert VerSe numbers to L1–L5 and flag extra vertebrae.
def map_verse(data):

    raw = validate_labels(data, range(29))
    result = np.zeros(raw.shape, np.uint8)

    for old, new in VERSE_MAP.items():
        result[raw == old] = new

    flags = []
    if np.any(raw == 25):
        flags.append("L6")
    if np.any(raw == 28):
        flags.append("T13")

    return result, flags


def combine_source_masks(masks):

    missing = [name for name in LEVELS if masks.get(name) is None]

    if set(masks) != set(LEVELS):
        raise ValueError(
            f"missing source references: {missing}; supplied keys={sorted(masks)}"
        )

    if missing:
        raise ValueError(
            f"missing source references: {missing}; supplied keys={sorted(masks)}"
        )

    shape = np.asarray(masks["L1"]).shape
    result = np.zeros(shape, np.uint8)
    states = {}

    for index, name in enumerate(LEVELS, start=1):
        mask = validate_labels(masks[name], [0, 1]).astype(bool)

        if mask.shape != shape:
            raise ValueError(f"grid shape mismatch: {name}")

        if np.any(mask & (result > 0)):
            raise ValueError(f"source overlap while merging {name}")

        result[mask] = index
        if mask.any():
            states[name] = "annotated_present"
        else:
            states[name] = "empty_annotation"

    return result, states


# Check that cases do not cross dataset splits.


def validate_splits(cases):
    seen = {}
    subjects = {}
    series = {}
    hashes = {}

    for case in cases:
        key, split = case["case_id"], case["split"]

        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split {split}: {key}")

        if key in seen:
            raise ValueError(f"duplicate case ID: {key}")

        seen[key] = split
        for field, index in [
            ("subject_id", subjects),
            ("series_id", series),
            ("ct_sha256", hashes),
        ]:
            value = case.get(field)
            if value:
                if value in index and index[value] != split:
                    raise ValueError(f"split leakage for {field}={value}")
                index[value] = split


def select_training_ids(
    cases, count, *, seed=42, max_negative=None, require_coverage=False
):
    validate_splits(cases)

    if count < 0:
        raise ValueError("selection counts must be nonnegative")

    if max_negative is not None and max_negative < 0:
        raise ValueError("selection counts must be nonnegative")
    eligible = []
    for case in cases:
        if case["split"] != "train":
            continue
        if case.get("eligible") is not True:
            continue
        eligible.append(case)
    eligible.sort(key=lambda case: case["case_id"])

    random.Random(seed).shuffle(eligible)
    if max_negative is None and not require_coverage:
        return [c["case_id"] for c in eligible[:count]]
    if any("observed_labels" not in c for c in eligible):
        raise ValueError("stratified selection requires audited observed_labels")

    selected = []
    covered = set()

    if require_coverage:
        for c in eligible:
            labels = set(c["observed_labels"]) & set(range(1, 6))
            if labels - covered:
                selected.append(c)
                covered |= labels
            if covered == set(range(1, 6)):
                break
        if covered != set(range(1, 6)):
            raise ValueError("requested subset cannot cover all five levels")

        if len(selected) > count:
            raise ValueError("requested subset cannot cover all five levels")

    negatives = 0
    selected_ids = {c["case_id"] for c in selected}

    for c in eligible:
        if len(selected) >= count:
            break
        if c["case_id"] in selected_ids:
            continue
        negative = not (set(c["observed_labels"]) & set(range(1, 6)))
        if negative and max_negative is not None and negatives >= max_negative:
            continue
        selected.append(c)
        selected_ids.add(c["case_id"])
        negatives += int(negative)

    return [c["case_id"] for c in selected]
