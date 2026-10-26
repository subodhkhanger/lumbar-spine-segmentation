import numpy as np
import pytest

from lumbar_ct.normalization import Normalizer, fit_normalizer


def test_NORM_02_hand_calculation_and_frozen_roundtrip():
    normalizer = Normalizer(0, 10, 5, 2, ["a"], 42)
    np.testing.assert_allclose(normalizer.apply(np.array([-10, 3, 20])), [-2.5, -1, 2.5], atol=1e-6)

    assert Normalizer.from_dict(normalizer.to_dict()) == normalizer


def test_NORM_03_heldout_fit_rejected():
    cases = [{"case_id": "test", "split": "test", "ct": np.ones((2, 2, 2)),
              "labels": np.ones((2, 2, 2))}]

    with pytest.raises(ValueError, match="unauthorised"):
        fit_normalizer(cases, allowed_ids=["train"])


def test_training_statistics_known_and_order_independent():
    cases = [{"case_id": "a", "split": "train", "ct": np.array([0., 2.]).reshape(2, 1, 1),
              "labels": np.ones((2, 1, 1))},
             {"case_id": "b", "split": "train", "ct": np.array([4., 6.]).reshape(2, 1, 1),
              "labels": np.ones((2, 1, 1))}]
    result = fit_normalizer(cases, allowed_ids=["a", "b"])
    assert result.mean == 3
    assert result.sd == pytest.approx(np.sqrt(5))

    assert result.lower == pytest.approx(.03)
    assert result.upper == pytest.approx(5.97)

    assert result == fit_normalizer(list(reversed(cases)), allowed_ids=["a", "b"])


def test_empty_training_foreground_is_error():

    with pytest.raises(ValueError, match="foreground"):
        fit_normalizer([{"case_id": "a", "split": "train", "ct": np.zeros((2, 2, 2)),
                         "labels": np.zeros((2, 2, 2))}], allowed_ids=["a"])
