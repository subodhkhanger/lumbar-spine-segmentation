"""Estimate vertebral bodies from masks on the 2 mm RAS grid."""

import numpy as np
from scipy import ndimage as ndi
from .geometry import world_point


def largest_component(mask):
    components, count = ndi.label(mask)
    if not count:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == sizes.argmax()


# Keep the thick body and trim the back of the vertebra.
def separate_body(mask, spacing, radius_mm=6.0):
    if radius_mm <= 0:
        raise ValueError("body erosion radius must be positive")
    mask = np.asarray(mask, dtype=bool)
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,):
        raise ValueError("invalid spacing")

    if not np.isfinite(spacing).all():
        raise ValueError("invalid spacing")

    if (spacing <= 0).any():
        raise ValueError("invalid spacing")
    output = np.zeros_like(mask)
    points = np.argwhere(mask)
    if not len(points):
        return output
    lower = points.min(0)
    upper = points.max(0) + 1
    region = tuple(slice(start, stop) for start, stop in zip(lower, upper))
    cropped_mask = np.pad(mask[region], 1)
    cropped_mask = largest_component(cropped_mask)

    distance = ndi.distance_transform_edt(cropped_mask, sampling=spacing)
    core = largest_component(distance > radius_mm)
    if not core.any():
        return output

    distance_to_core = ndi.distance_transform_edt(~core, sampling=spacing)
    body = (distance_to_core <= radius_mm) & cropped_mask

    # RAS +Y points forward; remove tissue behind the central body wall.
    core_depth = np.where(core, distance, 0)
    seed = np.unravel_index(core_depth.argmax(), cropped_mask.shape)
    centre_line = cropped_mask[seed[0], :, seed[2]]
    segments, _ = ndi.label(centre_line)
    body_segment = segments[seed[1]]
    posterior_wall = np.flatnonzero(segments == body_segment)[0]
    body[:, :posterior_wall, :] = False

    output[region] = largest_component(body)[1:-1, 1:-1, 1:-1]
    return output


# Find each body centre in millimetres.
def body_centroids(labels, grid, *, case_id, radius_mm=6.0):
    labels = np.asarray(labels)
    if labels.shape != grid.shape:
        raise ValueError("expected aligned labels 0..5")

    if not np.isin(labels, range(6)).all():
        raise ValueError("expected aligned labels 0..5")
    spacing = np.diag(grid.affine[:3, :3])
    axis_aligned = np.allclose(grid.affine[:3, :3], np.diag(spacing), atol=1e-5)
    if (spacing <= 0).any():
        raise ValueError("body extraction requires an orthogonal RAS grid")

    if not axis_aligned:
        raise ValueError("body extraction requires an orthogonal RAS grid")
    bodies = np.zeros(labels.shape, dtype=np.uint8)
    result = {
        "case_id": case_id,
        "coordinate_system": "RAS",
        "unit": "mm",
        "centroids": {},
        "method": (
            "physical erosion; largest core; constrained dilation; posterior cut at central "
            "body line back edge (RAS +Y anterior)"
        ),
        "radius_mm": radius_mm,
        "levels": {},
    }
    voxel_volume = float(np.prod(spacing))

    for level in range(1, 6):
        mask = labels == level
        body = separate_body(mask, spacing, radius_mm)
        vertebra_voxels = int(mask.sum())
        body_voxels = int(body.sum())
        warnings = []

        if vertebra_voxels:
            for axis in range(3):
                first_slice = np.take(mask, 0, axis=axis)
                last_slice = np.take(mask, -1, axis=axis)
                if first_slice.any() or last_slice.any():
                    warnings.append("prediction_touches_prepared_grid_boundary")
                    break
            if ndi.label(mask)[1] > 1:
                warnings.append("multiple_components_largest_used")

        if body_voxels:
            voxel_centre = np.argwhere(body).mean(0)
            centroid = world_point(voxel_centre, grid.affine).tolist()
            result["centroids"][f"L{level}"] = centroid
            status = "estimated"
            bodies[body] = level
        else:
            centroid = None
            if vertebra_voxels:
                status = "body_separation_failed"
            else:
                status = "missing_vertebra"

        result["levels"][f"L{level}"] = {
            "status": status,
            "centroid_ras_mm": centroid,
            "vertebra_volume_mm3": vertebra_voxels * voxel_volume,
            "body_volume_mm3": body_voxels * voxel_volume,
            "warnings": warnings,
        }
    return bodies, result
