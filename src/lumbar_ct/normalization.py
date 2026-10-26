"""CT normalization fitted on source training cases and reused unchanged."""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from .datasets import validate_labels


@dataclass(frozen=True)
class Normalizer:
    lower: float
    upper: float
    mean: float
    sd: float
    fit_ids: list
    seed: int
    epsilon: float = 1e-8
    max_voxels_per_case: int = 10000
    statistics: str = (
        "numpy percentile linear; population std ddof=0; raw foreground samples"
    )

    def __post_init__(self):
        object.__setattr__(self, "fit_ids", tuple(self.fit_ids))
        if not np.isfinite(
            [self.lower, self.upper, self.mean, self.sd, self.epsilon]
        ).all():
            raise ValueError("normalizer statistics must be finite")
        if self.lower > self.upper:
            raise ValueError("invalid normalizer statistics")

        if self.sd < 0:
            raise ValueError("invalid normalizer statistics")

        if self.epsilon <= 0:
            raise ValueError("invalid normalizer statistics")

        if self.max_voxels_per_case < 1:
            raise ValueError("invalid normalizer statistics")

    def apply(self, hu):
        hu = np.asarray(hu, dtype=np.float32)
        if not np.isfinite(hu).all():
            raise ValueError("nonfinite CT intensities")
        clipped = np.clip(hu, self.lower, self.upper)
        scaled = (clipped - self.mean) / max(self.sd, self.epsilon)
        return scaled.astype(np.float32)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    def sha256(self):
        settings = json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)
        return hashlib.sha256(settings.encode()).hexdigest()


# Calculate intensity statistics from training cases only.
def fit_normalizer(cases, *, allowed_ids, seed=42, max_voxels_per_case=10000):
    if max_voxels_per_case < 1:
        raise ValueError("sample cap must be positive")
    allowed = set(allowed_ids)
    sampled = {}
    seen = set()
    # Consume cases one at a time. Keep only capped foreground samples in RAM.
    for case in cases:
        key = case["case_id"]
        if key not in allowed:
            raise ValueError(f"unauthorised normalizer fit case: {key}")

        if case["split"] != "train":
            raise ValueError(f"unauthorised normalizer fit case: {key}")
        if key in seen:
            raise ValueError(f"duplicate fit case: {key}")
        seen.add(key)
        labels = validate_labels(case["labels"], range(6))
        ct = np.asarray(case["ct"])
        if ct.shape != labels.shape:
            raise ValueError(f"invalid aligned CT: {key}")

        if not np.isfinite(ct).all():
            raise ValueError(f"invalid aligned CT: {key}")
        foreground = ct[labels > 0].astype(np.float64)
        if len(foreground) > max_voxels_per_case:
            case_seed = int.from_bytes(
                hashlib.sha256(key.encode()).digest()[:8], "little"
            )
            rng = np.random.default_rng(np.random.SeedSequence([seed, case_seed]))
            foreground = foreground[
                rng.choice(len(foreground), max_voxels_per_case, replace=False)
            ]
        if len(foreground):
            sampled[key] = foreground
    if seen != allowed:
        raise ValueError(f"missing authorised fit cases: {sorted(allowed-seen)}")
    if not sampled:
        raise ValueError("no source-training foreground for normalizer")
    voxels = np.concatenate([sampled[key] for key in sorted(sampled)])
    lower, upper = np.percentile(voxels, [0.5, 99.5], method="linear")
    return Normalizer(
        float(lower),
        float(upper),
        float(voxels.mean()),
        float(voxels.std(ddof=0)),
        sorted(seen),
        seed,
        max_voxels_per_case=max_voxels_per_case,
    )
