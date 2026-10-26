import numpy as np
import pytest

from lumbar_ct.evaluate import score_case, summarize


@pytest.mark.parametrize("prediction, expected", [([1, 1, 0, 0], 1),
                                                ([0, 0, 1, 1], 0),
                                                ([1, 0, 1, 0], .5)])
def test_MET_01_03_known_dice(prediction, expected):
    ref = np.array([1, 1, 0, 0]).reshape(4, 1, 1)
    scores = score_case(np.array(prediction).reshape(ref.shape), ref)
    assert scores["L1"]["dice"] == expected


def test_MET_04_failed_positive_kept_as_zero():
    scores = score_case(None, np.ones((2, 2, 2), np.uint8))
    assert scores["L1"]["dice"] == 0
    assert scores["L1"]["status"] == "inference_failed"
    missed = score_case(np.zeros((2, 2, 2)), np.ones((2, 2, 2)))
    assert missed["L1"]["dice"] == 0


def test_MET_05_empty_reference_excluded_false_positive_counted():
    scores = score_case(np.ones((2, 2, 2)), np.zeros((2, 2, 2)))
    assert scores["L1"]["dice"] is None
    assert scores["L1"]["prediction_voxels"] == 8
    assert scores["L1"]["status"] == "empty_reference"


def test_MET_06_undefined_macro_and_sample_sd():
    one = score_case(np.ones((2, 2, 2)), np.ones((2, 2, 2)))
    result = summarize([one])
    assert result["levels"]["L1"] == {"mean": 1, "sd": None, "n": 1}
    assert result["macro"] is None
    failed = score_case(None, np.ones((2, 2, 2)))
    result = summarize([one, failed])
    assert result["levels"]["L1"]["mean"] == .5
    assert result["levels"]["L1"]["sd"] == pytest.approx(np.sqrt(.5))


def test_missing_reference_does_not_become_zero():
    result = score_case(None, None)
    assert all(row["dice"] is None and row["status"] == "missing_reference" for row in result.values())


def test_macro_uses_five_level_means():
    reference = np.arange(1, 6).reshape(5, 1, 1)
    prediction = reference.copy()
    prediction[0] = 0
    assert summarize([score_case(prediction, reference)])["macro"] == .8
