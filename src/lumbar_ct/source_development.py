"""Source-only cohort selection and validation checks."""

import random

SELECTED_HASH = "3ecff599ffe127fe09c686ba440733f1a021b58b742b6dd2db516f2845706334"


def coverage_group(row):
    present = sum(v > 0 for v in row["lumbar_voxels"].values())
    if present == 0:
        return "negative"
    if present == 5:
        return "full"
    return "partial"


def expansion_candidates(rows, original_train):
    groups = {name: [] for name in ("full", "partial", "negative")}
    for row in sorted(rows, key=lambda r: r["case_id"]):
        if row["dataset"] != "totalsegmentator_v201":
            continue
        if row["split"] != "train":
            continue
        if not row["eligible"]:
            continue
        if row["case_id"] in original_train:
            continue
        group = coverage_group(row)
        groups[group].append(row)
    rng = random.Random(42)
    for group in groups.values():
        rng.shuffle(group)
    return groups


# Check training and validation IDs against official splits.
def validate_development(manifest, official_rows):
    official = {
        r["case_id"]: r
        for r in official_rows
        if r["dataset"] == "totalsegmentator_v201"
    }
    train, val = manifest["train_ids"], manifest["validation_ids"]
    if len(train) != len(set(train)):
        raise ValueError("Development IDs are duplicated or overlap")

    if len(val) != len(set(val)):
        raise ValueError("Development IDs are duplicated or overlap")

    if set(train) & set(val):
        raise ValueError("Development IDs are duplicated or overlap")
    for case_id in train:
        if case_id not in official:
            raise ValueError("Training must use the official source training split")
        if official[case_id]["split"] != "train":
            raise ValueError("Training must use the official source training split")
    expected = {c for c, r in official.items() if r["split"] == "validation"}
    if len(expected) != 57:
        raise ValueError(
            "Validation must include all 57 official source validation cases"
        )

    if set(val) != expected:
        raise ValueError(
            "Validation must include all 57 official source validation cases"
        )
    original = manifest["original_train_ids"]
    patch_val = manifest["patch_validation_ids"]
    if len(original) != 64:
        raise ValueError("Original 64/8 cohort changed")

    if len(set(original)) != 64:
        raise ValueError("Original 64/8 cohort changed")

    if not set(original) <= set(train):
        raise ValueError("Original 64/8 cohort changed")

    if len(patch_val) != 8:
        raise ValueError("Original 64/8 cohort changed")

    if len(set(patch_val)) != 8:
        raise ValueError("Original 64/8 cohort changed")

    if not set(patch_val) <= set(val):
        raise ValueError("Original 64/8 cohort changed")
    cases = manifest["cases"]
    if len(cases) != len(train) + len(val):
        raise ValueError("Prepared case membership differs")

    if len({r["case_id"] for r in cases}) != len(cases):
        raise ValueError("Prepared case membership differs")
    for row in cases:
        if row["case_id"] in train:
            expected_role = "train"
        else:
            expected_role = "validation"
        if row["case_id"] not in set(train) | set(val):
            raise ValueError("Only prepared source development cases are allowed")

        if row["role"] != expected_role:
            raise ValueError("Only prepared source development cases are allowed")

        if row["dataset"] != "totalsegmentator_v201":
            raise ValueError("Only prepared source development cases are allowed")

        if row["status"] != "prepared":
            raise ValueError("Only prepared source development cases are allowed")


# Compare models using the same source validation cases.
def compare_source_validation(baseline, candidate, manifest):
    from .evaluate import summarize

    expected = set(manifest["validation_ids"])
    for summary in (baseline, candidate):
        ids = [r["case_id"] for r in summary["cases"]]
        if len(ids) != 57:
            raise ValueError("Comparison requires the same 57 validation cases")

        if set(ids) != expected:
            raise ValueError("Comparison requires the same 57 validation cases")

        if len(set(ids)) != 57:
            raise ValueError("Comparison requires the same 57 validation cases")
        if summary.get("development_manifest_sha256") != baseline.get(
            "development_manifest_sha256"
        ):
            raise ValueError("Validation manifest provenance differs")

        if not summary.get("development_manifest_sha256"):
            raise ValueError("Validation manifest provenance differs")
        if summary["summary"] != summarize([r["scores"] for r in summary["cases"]]):
            raise ValueError("Validation aggregate differs from per-case scores")

    def reference_counts(summary):
        counts = {}
        for case in summary["cases"]:
            counts[case["case_id"]] = {
                level: score["reference_voxels"]
                for level, score in case["scores"].items()
            }
        return counts

    if reference_counts(baseline) != reference_counts(candidate):
        raise ValueError("Per-case reference membership differs")
    old, new = baseline["summary"], candidate["summary"]
    deltas = {
        k: new["levels"][k]["mean"] - old["levels"][k]["mean"] for k in old["levels"]
    }

    def negative_counts(summary):
        counts = {}
        for case in summary["cases"]:
            scores = case["scores"].values()
            if any(score["reference_voxels"] for score in scores):
                continue
            counts[case["case_id"]] = sum(
                score["prediction_voxels"] or 0 for score in scores
            )
        return counts

    before, after = negative_counts(baseline), negative_counts(candidate)
    no_negative_regression = before.keys() == after.keys() and all(
        after[k] <= before[k] for k in before
    )
    same_counts = all(
        old["levels"][k]["n"] == new["levels"][k]["n"] for k in old["levels"]
    )
    complete = all(
        r.get("status", "completed") == "completed" for r in candidate["cases"]
    )
    candidate_preferred = False
    if new["macro"] > old["macro"]:
        if min(deltas.values()) >= -0.02:
            candidate_preferred = bool(
                no_negative_regression and same_counts and complete
            )

    return {
        "candidate_preferred": candidate_preferred,
        "baseline_macro": old["macro"],
        "candidate_macro": new["macro"],
        "per_level_delta": deltas,
        "negative_predicted_voxels": sum(after.values()),
        "negative_cases_before": before,
        "negative_cases_after": after,
        "test_data_used_for_selection": False,
        "criterion": (
            "Higher 57-case macro, no level drop >0.02, no increase in predicted voxels on "
            "any negative case, no failed cases"
        ),
    }
