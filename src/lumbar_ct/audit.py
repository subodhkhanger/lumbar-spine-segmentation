"""Read original headers without NiBabel repairs; report every detected issue."""

import gzip
import warnings

import nibabel as nib
import numpy as np

from .geometry import Grid, boundary_corners, validate_affine
from .prepare import SOURCE_UNITS_POLICY


# Check the original header and record any problems.
def inspect_header(path, dataset, units_policy=None):
    issues = []

    def issue(kind, observed, detection, handling, blocking=False):
        issues.append(
            dict(
                type=kind,
                observed=observed,
                detection=detection,
                handling=handling,
                blocking=blocking,
            )
        )

    with gzip.open(path, "rb") as f:
        header = nib.Nifti1Header.from_fileobj(f, check=False)
    # Run every native header check on a copy, retaining warnings before any repair.
    from nibabel.batteryrunners import BatteryRunner

    _, reports = BatteryRunner(header._get_checks()).check_fix(header.copy())
    for report in reports:
        if report.problem_level:
            issue(
                "nifti_header_check",
                report.problem_msg,
                "NiBabel header check on an unmodified-header copy",
                "Quarantine; raw header was not repaired",
                True,
            )
    shape = tuple(int(n) for n in header.get_data_shape())
    if len(shape) != 3 or any(n <= 0 for n in shape):
        issue("invalid_dimensions", list(shape), "Raw dim field", "Quarantine", True)
    units = header.get_xyzt_units()[0]
    factor = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}.get(units)
    forms = {}
    codes = {}
    for name in ("qform", "sform"):
        codes[name] = int(header[name + "_code"])
        if codes[name]:
            try:
                forms[name] = validate_affine(getattr(header, "get_" + name)())
            except Exception as exc:
                issue(
                    "invalid_" + name,
                    str(exc),
                    "Validate coded affine",
                    "Quarantine",
                    True,
                )
    selected = forms.get("sform", forms.get("qform"))
    if selected is None:
        issue(
            "missing_coordinate_frame",
            codes,
            "No usable coded qform or sform",
            "Quarantine",
            True,
        )
    if units == "unknown":
        resolved = (
            dataset == "totalsegmentator_v201"
            and selected is not None
            and np.allclose(
                np.linalg.norm(selected[:3, :3], axis=0),
                SOURCE_UNITS_POLICY["expected_spacing_mm"],
                atol=1e-5,
                rtol=0,
            )
        )
        paired = bool(
            units_policy
            and units_policy.get("paired_ct")
            and selected is not None
            and shape == tuple(units_policy["shape"])
            and np.allclose(
                selected,
                units_policy["reference_affine_native_units"],
                atol=1e-5,
                rtol=0,
            )
        )
        resolved = resolved or paired
        if resolved:
            factor = 1.0
        else:
            factor = None

        observed = units
        if paired:
            observed = {
                "unit": units,
                "maximum_corner_difference_mm": units_policy[
                    "maximum_corner_difference_mm"
                ],
                "maximum_index_displacement": units_policy[
                    "maximum_index_displacement"
                ],
            }
            handling = (
                "Inherit mm from explicitly-mm paired CT after matching shape and full coded "
                "affines within 0.1 mm at boundary corners and 0.49 voxel at voxel centres; "
                "preserve raw file: "
            ) + units_policy["paired_ct"]
        elif resolved:
            handling = "Interpret as mm under SOURCE_UNITS_POLICY; affine spacing matches 1.5 mm; preserve raw file"
        else:
            handling = "Quarantine; no applicable documented unit policy"

        issue(
            "unknown_spatial_units",
            observed,
            "Raw xyzt_units spatial code is zero",
            handling,
            not resolved,
        )
    elif factor is None:
        issue("unsupported_spatial_units", units, "Raw xyzt_units", "Quarantine", True)
    elif factor != 1:
        issue(
            "non_mm_spatial_units",
            units,
            "Raw xyzt_units",
            "Convert all affine spatial rows, including origin, to mm",
        )
    grid = None
    if selected is not None and factor is not None and len(shape) == 3:
        affine = selected.copy()
        affine[:3] *= factor
        grid = Grid(shape, affine)
        if len(forms) == 2:
            corners = []
            for name in ("qform", "sform"):
                a = forms[name].copy()
                a[:3] *= factor
                corners.append(boundary_corners(Grid(shape, a)))
            delta = float(np.linalg.norm(corners[0] - corners[1], axis=1).max())
            if delta > 0.1:
                issue(
                    "qform_sform_conflict",
                    delta,
                    "Maximum disagreement across eight voxel-boundary corners in mm exceeds 0.1",
                    "Quarantine; do not choose a conflicting frame",
                    True,
                )
    if selected is not None and len(shape) == 3:
        spacing = np.linalg.norm(selected[:3, :3], axis=0)
        if not np.allclose(spacing, header.get_zooms()[:3], atol=1e-5, rtol=1e-5):
            extent_bound = float(
                np.sum(
                    np.abs(spacing - np.asarray(header.get_zooms()[:3]))
                    * np.asarray(shape)
                )
            )
            if factor is not None:
                bound_mm = extent_bound * factor
            else:
                bound_mm = None
            safe = bound_mm is not None and bound_mm <= 0.1
            if safe:
                handling = (
                    "Retain coded affine; accumulated discrepancy <=0.1 mm; do not rewrite raw "
                    "pixdim; resampling uses full affine"
                )
            else:
                handling = "Quarantine: accumulated discrepancy exceeds 0.1 mm or units remain unresolved"
            issue(
                "pixdim_affine_mismatch",
                {
                    "pixdim": [float(v) for v in header.get_zooms()[:3]],
                    "affine_spacing_native_units": spacing.tolist(),
                    "whole_extent_displacement_upper_bound_mm": bound_mm,
                },
                (
                    "Compare raw pixdim with selected affine column norms (atol/rtol 1e-5); bound "
                    "accumulated spacing discrepancy over the full image extent"
                ),
                handling,
                not safe,
            )
    slope, intercept = float(header["scl_slope"]), float(header["scl_inter"])
    if np.isfinite(slope) and slope != 0 and not np.isfinite(intercept):
        issue(
            "invalid_scaling",
            [slope, str(intercept)],
            "Finite nonzero raw scl_slope with nonfinite scl_inter",
            "Quarantine",
            True,
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        img = nib.load(str(path))
    for warning in caught:
        issue(
            "nibabel_load_warning",
            str(warning.message),
            "Captured warning during image load",
            "Quarantine pending review",
            True,
        )
    grid_affine = None
    orientation = None
    if grid is not None:
        grid_affine = grid.affine.tolist()
        orientation = list(nib.aff2axcodes(grid.affine))

    native_affine = None
    if selected is not None:
        native_affine = selected.tolist()

    record = dict(
        shape=list(shape),
        spatial_units=units,
        temporal_units=header.get_xyzt_units()[1],
        qform_code=codes["qform"],
        sform_code=codes["sform"],
        raw_pixdim=header["pixdim"].tolist(),
        raw_slope=str(slope),
        raw_intercept=str(intercept),
        dtype=str(header.get_data_dtype()),
        vox_offset=float(header["vox_offset"]),
        affine_ras_mm=grid_affine,
        affine_native_units=native_affine,
        coded_forms_native_units={name: a.tolist() for name, a in forms.items()},
        orientation=orientation,
        issues=issues,
    )
    if selected is not None:
        basis = selected[:3, :3]
        spacing = np.linalg.norm(basis, axis=0)
        record["spacing_native_units"] = spacing.tolist()
        if factor is not None:
            record["spacing_mm"] = (spacing * factor).tolist()
        record["has_shear"] = not np.allclose(
            (basis / spacing).T @ (basis / spacing), np.eye(3), atol=1e-5
        )
        record["has_shear"] = bool(record["has_shear"])
    return img, grid, record


def compare_reference_grid(ct, reference):
    if ct.shape != reference.shape:
        return False, {
            "ct_shape": list(ct.shape),
            "reference_shape": list(reference.shape),
        }
    delta = float(
        np.linalg.norm(boundary_corners(ct) - boundary_corners(reference), axis=1).max()
    )
    from itertools import product

    indices = np.array(list(product(*[(0, n - 1) for n in ct.shape])), dtype=float)
    transform = np.linalg.solve(reference.affine, ct.affine)
    mapped = indices @ transform[:3, :3].T + transform[:3, 3]
    displacement = float(np.max(np.abs(mapped - indices)))
    # Affine differences reach extrema at corners. This proves NN gives identical indices.
    safe = delta <= 0.1 and displacement < 0.49
    return safe, {
        "maximum_boundary_corner_difference_mm": delta,
        "maximum_voxel_centre_index_displacement": displacement,
    }
