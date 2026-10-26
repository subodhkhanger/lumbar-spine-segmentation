import numpy as np
import pytest

from lumbar_ct.datasets import combine_source_masks, map_verse, validate_splits, select_training_ids


def test_DATA_01_exact_mapping_and_variants():
    raw = np.array([0, 19, 20, 21, 22, 23, 24, 25, 28]).reshape(3, 3, 1)
    mapped, flags = map_verse(raw)
    assert mapped.ravel().tolist() == [0, 0, 1, 2, 3, 4, 5, 0, 0]
    assert flags == ["L6", "T13"]


@pytest.mark.parametrize("bad", [1.5, np.nan, 99, -1])
def test_DATA_02_invalid_verse_label(bad):
    with pytest.raises(ValueError):
        map_verse(np.array([bad]).reshape(1, 1, 1))


def test_DATA_03_missing_is_not_empty():
    masks = {f"L{i}": np.zeros((2, 2, 2)) for i in range(1, 6)}
    merged, states = combine_source_masks(masks)
    assert not merged.any()
    assert states["L1"] == "empty_annotation"
    del masks["L1"]
    with pytest.raises(ValueError, match="missing.*L1"):
        combine_source_masks(masks)


def test_DATA_04_overlapping_source_masks():
    masks = {f"L{i}": np.zeros((2, 2, 2)) for i in range(1, 6)}
    masks["L1"][0, 0, 0] = masks["L2"][0, 0, 0] = 1
    with pytest.raises(ValueError, match="overlap.*L2"):
        combine_source_masks(masks)


@pytest.mark.parametrize("subject", [False, True])
def test_SPLIT_01_leakage_rejected(subject):
    cases = [
        {"case_id": "a", "split": "train", "subject_id": "p"},
        {"case_id": "b" if subject else "a", "split": "test", "subject_id": "p"},
    ]
    with pytest.raises(ValueError, match="leakage|duplicate"):
        validate_splits(cases)


def test_SPLIT_02_selection_order_independent_nested_train_only():
    cases = [{"case_id": str(i), "split": "train", "eligible": True} for i in range(12)]
    cases += [{"case_id": "heldout", "split": "test", "eligible": True}]
    small = select_training_ids(cases, 4, seed=42)
    large = select_training_ids(list(reversed(cases)), 8, seed=42)
    assert small == large[:4]
    assert "heldout" not in large


def test_SPLIT_03_positive_negative_quota_and_level_coverage():
    cases = [{"case_id": "positive", "split": "train", "eligible": True,
              "observed_labels": [1, 2, 3, 4, 5]}]
    cases += [{"case_id": f"negative{i}", "split": "train", "eligible": True,
               "observed_labels": []} for i in range(8)]
    selected = select_training_ids(cases, 4, seed=42, max_negative=1, require_coverage=True)
    assert "positive" in selected
    assert len(selected) == 2
