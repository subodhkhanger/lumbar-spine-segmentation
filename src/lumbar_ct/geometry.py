"""Explicit XYZ voxel grids and RAS millimetre transforms.

Interpolation extends the nearest voxel centre to the acquisition boundary.
Outside the half-open voxel-boundary extent,
CT is -1024 HU and masks are 0.
No antialiasing is applied. Scores are continuous until native-grid argmax.
"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import os
import tempfile

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform


def validate_affine(affine):
    a = np.asarray(affine, dtype=np.float64)
    if a.shape != (4, 4):
        raise ValueError("invalid affine")

    if not np.isfinite(a).all():
        raise ValueError("invalid affine")

    if not np.allclose(a[3], [0, 0, 0, 1]):
        raise ValueError("invalid affine")
    if np.linalg.matrix_rank(a[:3, :3]) != 3:
        raise ValueError("singular affine")
    return a


@dataclass(frozen=True, eq=False)
class Grid:
    shape: tuple
    affine: np.ndarray

    def __post_init__(self):
        if len(self.shape) != 3:
            raise ValueError("grid must have three positive integer dimensions")

        for size in self.shape:
            if not isinstance(size, (int, np.integer)):
                raise ValueError("grid must have three positive integer dimensions")
            if size <= 0:
                raise ValueError("grid must have three positive integer dimensions")
        a = validate_affine(self.affine).copy()
        a.flags.writeable = False
        object.__setattr__(self, "shape", tuple(int(n) for n in self.shape))
        object.__setattr__(self, "affine", a)

    def to_dict(self):
        return {"shape_xyz": list(self.shape), "affine_ras_mm": self.affine.tolist()}


@dataclass
class Volume:
    data: np.ndarray
    grid: Grid
    audit: dict


# Convert a voxel position to millimetres.
def world_point(index, affine, coordinate_system="RAS"):
    index = np.asarray(index, dtype=float)
    if index.shape != (3,):
        raise ValueError("point must contain three finite coordinates")

    if not np.isfinite(index).all():
        raise ValueError("point must contain three finite coordinates")
    a = validate_affine(affine)
    point = a[:3, :3] @ index + a[:3, 3]
    if coordinate_system == "LPS":
        point *= [-1, -1, 1]
    elif coordinate_system != "RAS":
        raise ValueError("coordinate system must be RAS or LPS")
    return point


def boundary_corners(grid):
    indices = np.array(list(product(*[(-0.5, n - 0.5) for n in grid.shape])))
    return indices @ grid.affine[:3, :3].T + grid.affine[:3, 3]


def crop_grid(grid, offset, shape):
    if len(offset) != 3:
        raise ValueError("crop needs three dimensions")

    if len(shape) != 3:
        raise ValueError("crop needs three dimensions")
    if any(not isinstance(v, (int, np.integer)) for v in (*offset, *shape)):
        raise ValueError("crop indices must be integers")
    for start, length, full_length in zip(offset, shape, grid.shape):
        if start < 0:
            raise ValueError("crop exceeds grid")
        if length <= 0:
            raise ValueError("crop exceeds grid")
        if start + length > full_length:
            raise ValueError("crop exceeds grid")
    a = grid.affine.copy()
    a[:3, 3] = world_point(offset, grid.affine)
    return Grid(tuple(shape), a)


def paired_mm_units_policy(reference_path, ct_grid, ct_spatial_units, ct_path):
    if ct_spatial_units != "mm":
        return None
    image = nib.load(str(reference_path))
    if image.header.get_xyzt_units()[0] != "unknown":
        return None
    if not (int(image.header["sform_code"]) or int(image.header["qform_code"])):
        return None
    affine = validate_affine(image.affine)
    if tuple(image.shape) != ct_grid.shape:
        return None
    reference_grid = Grid(tuple(image.shape), affine)
    corner_delta = float(
        np.linalg.norm(
            boundary_corners(ct_grid) - boundary_corners(reference_grid), axis=1
        ).max()
    )
    indices = np.array(list(product(*[(0, n - 1) for n in ct_grid.shape])), dtype=float)
    transform = np.linalg.solve(affine, ct_grid.affine)
    index_delta = float(
        np.max(np.abs(indices @ transform[:3, :3].T + transform[:3, 3] - indices))
    )
    if corner_delta > 0.1:
        return None
    if index_delta >= 0.49:
        return None
    return {
        "unit": "mm",
        "expected_spacing_mm": np.linalg.norm(affine[:3, :3], axis=0).tolist(),
        "provenance": (
            "Paired CT explicitly declares mm; reference shape matches and full coded affines"
            " agree within 0.1 mm at every boundary corner and 0.49 voxel at every voxel "
            "centre. Interpret missing reference units as mm; preserve original files."
        ),
        "paired_ct": str(Path(ct_path).resolve()),
        "shape": list(ct_grid.shape),
        "matching_affine_ras_mm": ct_grid.affine.tolist(),
        "reference_affine_native_units": affine.tolist(),
        "maximum_corner_difference_mm": corner_delta,
        "maximum_index_displacement": index_delta,
    }


# Load a scan and check its units and coordinates.
def load_volume(path, *, categorical=False, units_policy=None):
    img = nib.load(str(path))
    if len(img.shape) != 3:
        raise ValueError(f"expected 3D NIfTI: {path}")
    units = img.header.get_xyzt_units()[0]
    resolved_units = units
    issues = []
    if units == "unknown" and units_policy is not None:
        if not units_policy.get("provenance"):
            raise ValueError(
                "unknown units require provenance and an expected-spacing policy"
            )

        if "expected_spacing_mm" not in units_policy:
            raise ValueError(
                "unknown units require provenance and an expected-spacing policy"
            )
        resolved_units = units_policy["unit"]
        issues.append(
            {
                "type": "unknown_spatial_units",
                "severity": "warning",
                "observed": units,
                "action": "resolve from declared units policy",
                "resolved_unit": resolved_units,
                "policy": units_policy,
            }
        )
    factor = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}.get(resolved_units)
    if factor is None:
        raise ValueError(f"unresolved spatial units ({units}): {path}")
    q, qc = img.get_qform(coded=True)
    s, sc = img.get_sform(coded=True)
    forms = {}
    for name, form, code in [("qform", q, qc), ("sform", s, sc)]:
        if code:
            a = validate_affine(form).copy()
            a[:3] *= factor
            forms[name] = a
    if not forms:
        raise ValueError(f"no coded coordinate frame: {path}")
    conflict = 0.0
    if len(forms) == 2:
        # Compare all eight voxel-boundary corners, in millimetres.
        qcorners = boundary_corners(Grid(img.shape, forms["qform"]))
        scorners = boundary_corners(Grid(img.shape, forms["sform"]))
        conflict = float(np.linalg.norm(qcorners - scorners, axis=1).max())
        if conflict > 0.1:
            raise ValueError(f"coded qform/sform conflict {conflict:.6g} mm: {path}")
    if "sform" in forms:
        selected = "sform"
    else:
        selected = "qform"
    grid = Grid(img.shape, forms[selected])
    data = np.asarray(img.dataobj, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"nonfinite voxel values: {path}")
    if categorical:
        if not np.equal(data, np.rint(data)).all():
            raise ValueError(f"noninteger labels: {path}")
        if data.min() < 0:
            raise ValueError(f"labels outside supported integer range: {path}")

        if data.max() > np.iinfo(np.int32).max:
            raise ValueError(f"labels outside supported integer range: {path}")
        maximum = float(data.max())
        if maximum <= 255:
            label_dtype = np.uint8
        elif maximum <= 65535:
            label_dtype = np.uint16
        else:
            label_dtype = np.int32
        data = data.astype(label_dtype)
    basis = grid.affine[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    if units == "unknown" and not np.allclose(
        spacing, units_policy["expected_spacing_mm"], atol=1e-5, rtol=0
    ):
        raise ValueError("unknown-unit affine contradicts declared spacing policy")
    direction = basis / spacing
    audit = {
        "path": str(Path(path).resolve()),
        "shape_xyz": list(img.shape),
        "units": units,
        "resolved_units": resolved_units,
        "issues": issues,
        "unit_factor_to_mm": factor,
        "selected_form": selected,
        "qform_code": int(qc),
        "sform_code": int(sc),
        "coded_forms_ras_mm": {k: v.tolist() for k, v in forms.items()},
        "form_corner_difference_mm": conflict,
        "affine_ras_mm": grid.affine.tolist(),
        "orientation": list(nib.aff2axcodes(grid.affine)),
        "spacing_mm": spacing.tolist(),
        "has_shear": bool(
            not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5)
        ),
        "proxy_slope": float(getattr(img.dataobj, "slope", 1)),
        "proxy_intercept": float(getattr(img.dataobj, "inter", 0)),
        "scaling": "NiBabel proxy applied once",
    }
    return Volume(data, grid, audit)


def shared_grid(grid, spacing=2.0):
    if not np.isfinite(spacing):
        raise ValueError("spacing must be positive")

    if spacing <= 0:
        raise ValueError("spacing must be positive")
    corners = boundary_corners(grid)
    lower, upper = corners.min(0), corners.max(0)
    shape = tuple(np.maximum(1, np.ceil((upper - lower) / spacing - 1e-9).astype(int)))
    a = np.diag([spacing, spacing, spacing, 1.0])
    a[:3, 3] = lower + spacing / 2
    return Grid(shape, a)


def acquisition_support(source, target):
    transform = np.linalg.solve(source.affine, target.affine)
    axes = np.ogrid[tuple(slice(0, n) for n in target.shape)]
    inside = np.ones(target.shape, dtype=bool)
    for i, n in enumerate(source.shape):
        # Zero terms needlessly expand an axis-aligned coordinate to a full volume.
        coord = transform[i, 3] + sum(
            transform[i, j] * axes[j] for j in range(3) if transform[i, j] != 0
        )
        inside &= (coord >= -0.5) & (coord < n - 0.5)
    return inside


def same_grid(a, b):
    if a.shape != b.shape:
        return False
    return np.array_equal(a.affine, b.affine)


# Move the image or labels onto another grid.
def resample(data, source, target, *, kind):
    if kind not in {"ct", "label", "score"}:
        raise ValueError("kind must be ct, label or score")
    data = np.asarray(data)
    if data.shape != source.shape:
        raise ValueError("input values/shape do not match source grid")

    if not np.isfinite(data).all():
        raise ValueError("input values/shape do not match source grid")
    if kind == "label" and not np.equal(data, np.rint(data)).all():
        raise ValueError("categorical interpolation requires integer labels")
    if same_grid(source, target):
        return data.copy()

    transform = np.linalg.solve(source.affine, target.affine)
    if kind == "label":
        output_dtype = data.dtype
        interpolation_order = 0
    else:
        output_dtype = np.float32
        interpolation_order = 1

    result = affine_transform(
        data,
        transform[:3, :3],
        transform[:3, 3],
        output_shape=target.shape,
        output=output_dtype,
        order=interpolation_order,
        mode="nearest",
        prefilter=False,
    )
    outside = ~acquisition_support(source, target)
    if kind == "ct":
        result[outside] = -1024
    else:
        result[outside] = 0
    return result


def restore_scores(scores, source, native, *, support):
    scores = np.asarray(scores)
    support = np.asarray(support, dtype=bool)
    if scores.shape != (6, *source.shape):
        raise ValueError("expected six XYZ score channels and source support")

    if support.shape != source.shape:
        raise ValueError("expected six XYZ score channels and source support")
    restored = np.stack(
        [resample(channel, source, native, kind="score") for channel in scores]
    )
    valid = resample(support.astype(np.uint8), source, native, kind="label").astype(
        bool
    )
    restored[:, ~valid] = 0
    # All-zero exterior scores select background at argmax.
    return restored


def save_volume(path, data, grid):
    data = np.asarray(data)
    if data.shape != grid.shape:
        raise ValueError("output values/shape do not match grid")

    if not np.isfinite(data).all():
        raise ValueError("output values/shape do not match grid")
    path = Path(path)
    if not str(path).endswith((".nii", ".nii.gz")):
        raise ValueError("output must be .nii or .nii.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data, grid.affine)
    img.header.set_xyzt_units("mm")
    img.set_sform(grid.affine, code=1)
    # A sheared affine cannot be represented exactly by a qform.
    img.set_qform(None, code=0)
    if str(path).endswith(".gz"):
        suffix = ".nii.gz"
    else:
        suffix = ".nii"
    fd, temporary = tempfile.mkstemp(prefix=".volume-", suffix=suffix, dir=path.parent)
    os.close(fd)
    try:
        nib.save(img, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
