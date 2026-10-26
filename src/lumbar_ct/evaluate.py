"""Per-level Dice scoring, including missed predictions and failed cases."""

import numpy as np

from .datasets import LEVELS, validate_labels


# Measure Dice for each lumbar level in one scan.
def score_case(prediction, reference):
    if reference is None:
        result = {}
        for name in LEVELS:
            result[name] = {
                "dice": None,
                "status": "missing_reference",
                "reference_voxels": None,
                "prediction_voxels": None,
            }
        return result
    ref = validate_labels(reference, range(6))
    if prediction is None:
        pred = None
    else:
        pred = validate_labels(prediction, range(6))
    if pred is not None and pred.shape != ref.shape:
        raise ValueError(
            "prediction/reference grid shapes differ; align physically first"
        )
    result = {}
    for i, name in enumerate(LEVELS, 1):
        reference_voxels = int(np.count_nonzero(ref == i))
        prediction_voxels = None
        if pred is not None:
            prediction_voxels = int(np.count_nonzero(pred == i))

        if reference_voxels == 0:
            dice, status = None, "empty_reference"
        elif pred is None:
            dice, status = 0.0, "inference_failed"
        else:
            intersection = int(np.count_nonzero((ref == i) & (pred == i)))
            dice = 2.0 * intersection / (reference_voxels + prediction_voxels)
            if prediction_voxels == 0:
                status = "missed_positive"
            else:
                status = "scored"
        result[name] = {
            "dice": dice,
            "status": status,
            "reference_voxels": reference_voxels,
            "prediction_voxels": prediction_voxels,
        }
    return result


def summarize(case_scores):
    levels = {}
    for name in LEVELS:
        values = []
        for row in case_scores:
            dice = row[name]["dice"]
            if dice is not None:
                values.append(dice)

        mean = None
        standard_deviation = None
        if values:
            mean = float(np.mean(values))
        if len(values) > 1:
            standard_deviation = float(np.std(values, ddof=1))
        levels[name] = {
            "mean": mean,
            "sd": standard_deviation,
            "n": len(values),
        }
    means = [levels[name]["mean"] for name in LEVELS]
    macro = None
    if all(mean is not None for mean in means):
        macro = float(np.mean(means))
    return {
        "levels": levels,
        "macro": macro,
        "macro_denominator": 5,
        "attempted_cases": len(case_scores),
    }
