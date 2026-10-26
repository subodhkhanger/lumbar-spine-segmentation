"""Device and split checks for a new full-source training run."""

import torch


# Check the backend before preparing the dataset.
def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Use --device cpu or a CUDA machine.")
    elif name == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable in this Python environment.")
        try:
            sample = torch.ones((1, 1, 4, 4, 4), device="mps", requires_grad=True)
            torch.nn.functional.avg_pool3d(sample, 2).sum().backward()
            convolution = torch.nn.Conv3d(1, 2, 3).to("mps")
            convolution(sample).sum().backward()
        except (RuntimeError, NotImplementedError) as exc:
            raise ValueError(
                "This MPS build cannot train ResEnc M's 3D operations. "
                "Use --device cuda or --device cpu."
            ) from exc
    elif name != "cpu":
        raise ValueError(f"Unknown device: {name}")
    return torch.device(name)


# Account for every official training case, including rejected annotations.
def validate_full_source(manifest, official_rows):
    official = {}
    for row in official_rows:
        if row["dataset"] == "totalsegmentator_v201":
            official[row["case_id"]] = row["split"]
    expected_train = {key for key, split in official.items() if split == "train"}
    expected_val = {key for key, split in official.items() if split == "validation"}
    expected_test = {key for key, split in official.items() if split == "test"}
    if (len(expected_train), len(expected_val), len(expected_test)) != (1082, 57, 89):
        raise ValueError("Expected the official 1,082/57/89 source split")
    train = manifest["train_ids"]
    validation = manifest["validation_ids"]
    excluded = [row["case_id"] for row in manifest["preparation_exclusions"]]
    if len(train) != len(set(train)) or len(excluded) != len(set(excluded)):
        raise ValueError("Duplicate training or exclusion IDs")
    if set(train) & set(excluded):
        raise ValueError("A training case is also excluded")
    if set(train) | set(excluded) != expected_train:
        raise ValueError("Every official training case must be accounted for")
    if len(validation) != 57 or set(validation) != expected_val:
        raise ValueError("Keep all 57 official validation cases")
    expected_cases = set(train) | set(validation)
    cases = manifest["cases"]
    if len(cases) != len(expected_cases):
        raise ValueError("Prepared membership differs from the split")
    if {row["case_id"] for row in cases} != expected_cases:
        raise ValueError("Prepared membership differs from the split")
    for row in cases:
        role = "train" if row["case_id"] in train else "validation"
        if row["role"] != role or row["status"] != "prepared":
            raise ValueError("Prepared role or status differs")
        if row["dataset"] != "totalsegmentator_v201":
            raise ValueError("Only source data may enter training and validation")
