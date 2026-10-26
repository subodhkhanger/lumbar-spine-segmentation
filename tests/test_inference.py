import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from lumbar_ct import inference
from lumbar_ct.geometry import Grid, save_volume, world_point
from lumbar_ct.nnunet_inference import load_checkpoint


def test_loader_rejects_other_model_types(tmp_path):
    checkpoint = tmp_path / "old_model.pt"
    torch.save({"kind": "legacy"}, checkpoint)

    with pytest.raises(ValueError, match="ResEnc M checkpoint"):
        load_checkpoint(checkpoint)


def test_ct_pipeline_uses_selected_nnunet_checkpoint(tmp_path, monkeypatch):
    checkpoint = Path(__file__).resolve().parents[1] / "outputs/nnunet/model.pt"
    ct_path = tmp_path / "ct.nii.gz"
    output = tmp_path / "result"
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-30, 12, 50]
    grid = Grid((16, 16, 16), affine)
    save_volume(ct_path, np.zeros(grid.shape, dtype=np.float32), grid)

    # Use fixed scores to check CT loading and output geometry separately from prediction.
    def fixed_scores(model, saved, ct, device):
        assert type(model).__name__ == "ResidualEncoderUNet"
        assert saved["kind"] == "nnunet_resenc_m"
        assert not model.training
        assert device == "cpu"
        scores = np.zeros((6, *ct.shape), dtype=np.float32)
        scores[2] = 1
        return scores

    monkeypatch.setattr(inference, "predict_nnunet_scores", fixed_scores)
    inference.main(
        [
            str(ct_path),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--device",
            "cpu",
        ]
    )

    result = nib.load(str(output / "vertebrae_native.nii.gz"))
    np.testing.assert_allclose(result.affine, affine)
    assert np.all(np.asarray(result.dataobj) == 2)
    record = json.loads((output / "inference.json").read_text())
    assert record["reference_masks_used"] is False
    centroids = json.loads((output / "centroids.json").read_text())
    expected = world_point([7.5, 7.5, 7.5], affine)
    np.testing.assert_allclose(centroids["centroids"]["L2"], expected)


def test_auto_device_uses_cpu_when_accelerators_are_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert inference.choose_inference_device("auto") == "cpu"
    assert inference.choose_inference_device("mps") == "cpu"


def test_mps_inference_failure_retries_on_cpu(tmp_path, monkeypatch):
    ct_path = tmp_path / "ct.nii.gz"
    output = tmp_path / "result"
    grid = Grid((8, 8, 8), np.diag([2.0, 2.0, 2.0, 1.0]))
    save_volume(ct_path, np.zeros(grid.shape, dtype=np.float32), grid)

    class Model:
        def to(self, device):
            assert device == "cpu"
            return self

    model = Model()
    normalizer = {
        "lower": 0.0,
        "upper": 1.0,
        "mean": 0.0,
        "sd": 1.0,
        "fit_ids": ["train"],
        "seed": 42,
    }
    checkpoint = {"config": {"spacing_mm": [2.0, 2.0, 2.0]}, "normalizer": normalizer}
    devices = []

    def scores_for_device(model, saved, ct, target, support, folder, *, case_id, device):
        devices.append(device)
        if device == "mps":
            raise NotImplementedError("3D pooling is unavailable")
        scores = np.zeros((6, *ct.shape), dtype=np.float32)
        scores[0] = 1
        return scores, None, None, None

    monkeypatch.setattr(inference, "choose_inference_device", lambda requested: "mps")
    monkeypatch.setattr(inference, "load_checkpoint", lambda path, device: (model, checkpoint))
    monkeypatch.setattr(inference, "export_prepared", scores_for_device)
    inference.main([str(ct_path), "--checkpoint", str(ct_path), "--output", str(output)])
    assert devices == ["mps", "cpu"]
    assert json.loads((output / "inference.json").read_text())["device"] == "cpu"


def test_auto_device_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert inference.choose_inference_device("auto") == "cuda"
