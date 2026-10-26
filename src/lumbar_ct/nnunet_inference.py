"""Rebuild the saved nnU-Net and convert between its tensor axes and our XYZ arrays."""

import numpy as np
import torch


def tensor_order(plans):
    xyz_to_zyx = [2, 1, 0]
    return [xyz_to_zyx[axis] for axis in plans["transpose_forward"]]


def build_network(checkpoint):
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    architecture = checkpoint["plans"]["configurations"]["3d_fullres"]["architecture"]
    network = get_network_from_plans(
        architecture["network_class_name"],
        architecture["arch_kwargs"],
        architecture["_kw_requires_import"],
        1,
        6,
        allow_init=False,
        deep_supervision=False,
    )
    network.load_state_dict(checkpoint["state_dict"], strict=True)
    return network


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != "nnunet_resenc_m":
        raise ValueError("Expected an nnU-Net ResEnc M checkpoint")

    model = build_network(checkpoint)
    model = model.to(device)
    model.eval()
    return model, checkpoint


# Run nnU-Net over the scan in overlapping patches.
@torch.inference_mode()
def predict_nnunet_scores(model, checkpoint, ct, device):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    manager = PlansManager(checkpoint["plans"])
    configuration = manager.get_configuration("3d_fullres")
    assert configuration.normalization_schemes == ["NoNormalization"]
    assert list(configuration.spacing) == [2.0, 2.0, 2.0]
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=str(device).startswith("cuda"),
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.manual_initialization(
        model,
        manager,
        configuration,
        None,
        checkpoint["dataset_json"],
        "nnUNetTrainerLumbar",
        (),
    )
    # Change spatial axes for nnU-Net, then undo the permutation on its output.
    order = tensor_order(checkpoint["plans"])
    reordered_ct = ct.transpose(order)[None]
    data = torch.from_numpy(np.ascontiguousarray(reordered_ct, dtype=np.float32))
    logits = predictor.predict_sliding_window_return_logits(data).float().cpu()
    scores = logits.softmax(0).numpy()

    spatial_axes = [int(axis) + 1 for axis in np.argsort(order)]
    xyz_scores = scores.transpose([0, *spatial_axes])
    return np.ascontiguousarray(xyz_scores)
