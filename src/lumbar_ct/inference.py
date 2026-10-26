"""CT-only segmentation and vertebral-body centroids in RAS millimetres."""

import argparse
from pathlib import Path
import time
import numpy as np
import torch
from .artifacts import file_hash, reject_path_overlap, write_json
from .body import body_centroids
from .geometry import (
    load_volume,
    shared_grid,
    resample,
    acquisition_support,
    save_volume,
)
from .nnunet_inference import load_checkpoint, predict_nnunet_scores
from .normalization import Normalizer
from .prepare import SOURCE_UNITS_POLICY, verse_release_units_policy


# Use MPS only when this PyTorch build supports the model's 3D operations.
def choose_inference_device(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        return "cuda"
    if requested == "cpu":
        return "cpu"
    if requested not in {"auto", "mps"}:
        raise ValueError(f"Unknown inference device: {requested}")
    if requested == "auto" and torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        try:
            sample = torch.ones((1, 1, 4, 4, 4), device="mps")
            torch.nn.functional.avg_pool3d(sample, 2)
            kernel = torch.ones((1, 1, 3, 3, 3), device="mps")
            torch.nn.functional.conv3d(sample, kernel)
            return "mps"
        except (RuntimeError, NotImplementedError) as exc:
            print(f"MPS 3D operations unavailable; using CPU: {exc}", flush=True)
    elif requested == "mps":
        print("MPS unavailable; using CPU", flush=True)
    return "cpu"


# Move scores to the original grid and choose each voxel label.
def native_argmax(scores, source, native, support):
    native_support = resample(support.astype(np.uint8), source, native, kind="label")
    valid = native_support.astype(bool)
    best_score = resample(scores[0], source, native, kind="score")
    prediction = np.zeros(native.shape, dtype=np.uint8)

    for level in range(1, 6):
        level_score = resample(scores[level], source, native, kind="score")
        better = level_score > best_score
        prediction[better] = level
        np.maximum(best_score, level_score, out=best_score)

    prediction[~valid] = 0
    return prediction


# Predict the labels and save body masks and centres.
def export_prepared(model, checkpoint, ct, grid, support, output, *, case_id, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    scores = predict_nnunet_scores(model, checkpoint, ct, device)

    # Force padded regions outside the acquisition to background.
    scores[:, ~support] = 0
    scores[0, ~support] = 1
    labels = scores.argmax(0).astype(np.uint8)
    bodies, centroids = body_centroids(labels, grid, case_id=case_id)
    centroids["checkpoint_steps"] = checkpoint["steps_completed"]

    save_volume(output / "vertebrae_2mm.nii.gz", labels, grid)
    save_volume(output / "bodies_2mm.nii.gz", bodies, grid)
    write_json(output / "centroids.json", centroids)
    return scores, labels, bodies, centroids


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="L1-L5 CT segmentation and body centroids; no reference masks used"
    )
    parser.add_argument("ct", type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/nnunet/model.pt")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument(
        "--units-policy",
        choices=["strict", "totalsegmentator-v201", "verse2020"],
        default="strict",
        help="Explicit release-specific assumption only when input units are absent",
    )
    args = parser.parse_args(argv)

    reject_path_overlap(args.ct, args.output)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be new or empty")
    torch.set_num_threads(4)
    device = choose_inference_device(args.device)
    print(f"Inference device: {device}", flush=True)

    policy = None
    if args.units_policy == "totalsegmentator-v201":
        policy = SOURCE_UNITS_POLICY
    if args.units_policy == "verse2020":
        policy = verse_release_units_policy(args.ct)

    start = time.monotonic()
    ct = load_volume(args.ct, units_policy=policy)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    spacing = checkpoint["config"]["spacing_mm"]
    if len(set(spacing)) != 1:
        raise ValueError("checkpoint spacing must be isotropic")

    grid = shared_grid(ct.grid, spacing=float(spacing[0]))
    support = acquisition_support(ct.grid, grid)
    hu = resample(ct.data, ct.grid, grid, kind="ct")
    normalizer = Normalizer.from_dict(checkpoint["normalizer"])
    normalized = normalizer.apply(hu)
    case_id = args.case_id or args.ct.name.removesuffix(".nii.gz").removesuffix(".nii")

    try:
        scores, _, _, _ = export_prepared(
            model,
            checkpoint,
            normalized,
            grid,
            support,
            args.output,
            case_id=case_id,
            device=device,
        )
    except (RuntimeError, NotImplementedError) as exc:
        if device != "mps":
            raise
        print(f"MPS inference failed; retrying on CPU: {exc}", flush=True)
        device = "cpu"
        model = model.to(device)
        scores, _, _, _ = export_prepared(
            model,
            checkpoint,
            normalized,
            grid,
            support,
            args.output,
            case_id=case_id,
            device=device,
        )
    native_labels = native_argmax(scores, grid, ct.grid, support)
    save_volume(args.output / "vertebrae_native.nii.gz", native_labels, ct.grid)

    record = {
        "input": str(args.ct.resolve()),
        "input_sha256": file_hash(args.ct),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "device": device,
        "seconds": time.monotonic() - start,
        "input_audit": ct.audit,
        "prepared_grid": grid.to_dict(),
        "native_grid": ct.grid.to_dict(),
        "centroid_grid": "2 mm RAS",
        "reference_masks_used": False,
    }
    write_json(args.output / "inference.json", record)
    print(f"Wrote segmentation and centroids to {args.output}", flush=True)


if __name__ == "__main__":
    main()
