# Lumbar CT Segmentation

**Status: ongoing.** A 3D medical-imaging project for segmenting lumbar vertebrae L1–L5 in CT scans and estimating vertebral body centres in world coordinates. The pipeline works end to end; the next phase focuses on improving segmentation accuracy and performance across datasets.

## Visual result

![Predicted L1–L5 vertebrae on CT, shown in sagittal, coronal and axial views](images/s0235_prediction_overlay.png)

*CT-only prediction on TotalSegmentator case s0235. Red: L1, orange: L2, green: L3, blue: L4, purple: L5. This illustrates the output; quantitative results are below.*

## How it works

CT scans are resampled to a 2 mm RAS grid and normalized with frozen training-only statistics. **nnU-Net v2 ResEnc M**, trained from scratch without external pretrained spine weights, predicts L1–L5. A morphological postprocessing step estimates each vertebral body and exports its centre in RAS millimetres. Segmentation masks are also returned on the original CT grid.

The current model started with 32 TotalSegmentator training scans and continued on 64 scans. Continuation epoch 15 was selected using all 57 official source-validation scans. Inference uses CT only. Reference labels are used for training and evaluation, never as inference inputs.

## Current results

The frozen model was evaluated on **89 TotalSegmentator test scans** and **73 VerSe 2020 test scans**. Dice is averaged over scans where the reference level is present, including missed levels as zero; `n` is the number of those scans. Macro Dice averages the five level means.

| Level | TotalSegmentator Dice | VerSe Dice |
| --- | ---: | ---: |
| L1 | 0.846 (n=64) | 0.775 (n=58) |
| L2 | 0.884 (n=59) | 0.754 (n=58) |
| L3 | 0.865 (n=59) | 0.699 (n=58) |
| L4 | 0.894 (n=60) | 0.602 (n=57) |
| L5 | 0.923 (n=57) | 0.597 (n=51) |
| **Macro Dice** | **0.882** | **0.685** |

All 162 scans completed inference. Performance drops on VerSe, particularly at L4/L5. Observed failures include partial vertebrae and incorrect level numbering: nine annotated source levels had zero overlap, and 39/282 annotated VerSe levels had Dice below 0.10. False lumbar predictions occurred on 1/19 entirely negative source scans and 3/15 negative VerSe scans. Body-centre accuracy has not been measured against independent reference points.

VerSe labels 20–24 map to L1–L5; L6 (25) is background, so predicting it as a target level counts as an error. Earlier baseline test scores were inspected during development; the final checkpoint was selected using source validation.

## Next steps

- Expand training to 541 official source-training scans, keeping all 57 validation scans separate. This run has not been performed.
- Improve partial-scan detection and vertebral numbering using source-validation failures.
- Investigate the VerSe performance gap, especially at L4/L5, while keeping both test sets out of model selection.
- Validate vertebral body centres against independent reference points.

## Run inference

Download and extract the [runnable package with trained weights](https://github.com/subodhkhanger/lumbar-spine-segmentation/releases/download/v0.1.0/lumbar-ct-segmentation.zip). Use Python 3.11 on Linux or macOS. Place CT `.nii` or `.nii.gz` files directly in an input directory, then run from the extracted project folder:

```bash
./run_inference.sh /path/to/input_cts /path/to/empty_output
```

The first run installs the pinned dependencies in a local environment. It selects CUDA when available, tries supported MPS operations, then falls back to CPU. Docker is not required. The release ZIP includes the trained weights at `outputs/nnunet/model.pt`. A Git clone or GitHub source archive contains source code only; copy that weights file from the release ZIP to the same path before inference. Training images and reference masks are excluded.

Each scan produces native-grid vertebra labels, 2 mm vertebra/body masks, `centroids.json`, and inference provenance. Labels 1–5 mean L1–L5; missing predictions and failed body estimates are recorded explicitly.

For audited dataset files with missing spatial units, set `LUMBAR_UNITS_POLICY=totalsegmentator-v201` or `LUMBAR_UNITS_POLICY=verse2020` before the command. Other inputs must declare their units. Input and output directories must not overlap.

## Development

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m pytest -q
```

The training launcher starts a new model on the full official training split and writes to `training/full_source/`, preserving the supplied weights. It is separate from the planned 541-scan experiment and does not reproduce the current model's staged training or its scores:

```bash
LUMBAR_PYTHON="$PWD/.venv/bin/python" ./run_training.sh --data-dir /path/to/totalsegmentator_v201 --device auto
```

The data directory must contain `meta.csv` and the case folders. Add `--check` to verify setup without training. New runs use only official training cases and select checkpoints on source validation.
