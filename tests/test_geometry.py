from itertools import product

import nibabel as nib
import numpy as np
import pytest

from lumbar_ct.geometry import (Grid, world_point, crop_grid, load_volume,
                                shared_grid, resample, restore_scores, save_volume)


def image_file(tmp_path, affine, units="mm", data=None):
    img = nib.Nifti1Image(np.zeros((3, 4, 5), np.float32) if data is None else data, affine)
    img.header.set_xyzt_units(units)
    path = tmp_path / "image.nii.gz"
    nib.save(img, path)
    return path


def test_GEO_01_known_world_point():
    a = np.diag([2., 3., 4., 1.])
    a[:3, 3] = [10, -20, 30]
    assert np.allclose(world_point([2, 3, 4], a), [14, -11, 46], atol=1e-5)
    assert np.allclose(world_point([2, 3, 4], a, "LPS"), [-14, 11, 46], atol=1e-5)


def test_GEO_02_permutation_flip_has_exact_oracle():
    # Source x maps to -RAS z; source y to RAS x; source z to RAS y.
    a = np.array([[0, 2, 0, 0], [0, 0, 2, 0], [-2, 0, 0, 4], [0, 0, 0, 1.]])
    data = np.arange(27).reshape(3, 3, 3)
    target = Grid((3, 3, 3), np.diag([2., 2., 2., 1.]))
    result = resample(data, Grid(data.shape, a), target, kind="label")
    expected = np.empty_like(data)
    for x, y, z in product(range(3), repeat=3):
        expected[x, y, z] = data[2-z, x, y]
    np.testing.assert_array_equal(result, expected)


def test_GEO_03_crop_offset():
    grid = Grid((10, 11, 12), np.diag([2., 3., 4., 1.]))
    crop = crop_grid(grid, (2, 3, 4), (5, 5, 5))
    assert np.allclose(world_point([1, 1, 1], crop.affine), [6, 12, 20])


def test_GEO_04_metres_and_unknown_units(tmp_path):
    path = image_file(tmp_path, np.diag([.002, .003, .004, 1]), units="meter")
    volume = load_volume(path)
    np.testing.assert_allclose(np.diag(volume.grid.affine)[:3], [2, 3, 4], atol=1e-5)
    path = image_file(tmp_path, np.eye(4), units="unknown")
    with pytest.raises(ValueError, match="units"):
        load_volume(path)


def test_GEO_05_coded_form_conflict(tmp_path):
    img = nib.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4))
    img.header.set_xyzt_units("mm")
    img.set_qform(np.eye(4), code=1)
    a = np.eye(4)
    a[0, 3] = 10
    img.set_sform(a, code=1)
    path = tmp_path / "conflict.nii.gz"
    nib.save(img, path)
    with pytest.raises(ValueError, match="conflict"):
        load_volume(path)


def test_unknown_units_require_explicit_logged_release_policy(tmp_path):
    path = image_file(tmp_path, np.diag([1.5, 1.5, 1.5, 1]), units="unknown")
    policy = {"unit": "mm", "provenance": "fixture documented in mm", "expected_spacing_mm": 1.5}
    volume = load_volume(path, units_policy=policy)
    assert volume.audit["units"] == "unknown"
    assert volume.audit["resolved_units"] == "mm"
    assert volume.audit["issues"][0]["policy"] == policy
    bad = image_file(tmp_path, np.diag([3., 3., 3., 1.]), units="unknown")
    with pytest.raises(ValueError, match="contradicts"):
        load_volume(bad, units_policy=policy)


def test_GEO_06_shear_grid_covers_voxel_boundaries():
    a = np.array([[2, .5, 0, 10], [0, 2, .2, -4], [0, 0, 2, 3], [0, 0, 0, 1.]])
    source = Grid((3, 4, 5), a)
    target = shared_grid(source, spacing=2)
    corners = np.array(list(product(*[(-.5, n-.5) for n in source.shape])))
    world = corners @ a[:3, :3].T + a[:3, 3]
    lower = target.affine[:3, 3] - 1
    upper = lower + np.array(target.shape)*2
    assert np.all(world >= lower-1e-5)
    assert np.all(world <= upper+1e-5)
    assert nib.aff2axcodes(target.affine) == ("R", "A", "S")


def test_GEO_07_different_label_grid_alignment():
    label = np.zeros((3, 3, 3), np.uint8)
    label[1, 1, 1] = 3
    a = np.eye(4)
    a[:3, 3] = [1, 1, 1]
    result = resample(label, Grid(label.shape, a), Grid((5, 5, 5), np.eye(4)), kind="label")
    assert result[2, 2, 2] == 3
    assert np.count_nonzero(result) == 1


def test_GEO_08_native_export(tmp_path):
    a = np.diag([-2., 3., 4., 1.])
    a[:3, 3] = [10, 20, 30]
    grid = Grid((3, 4, 5), a)
    path = tmp_path / "mask.nii.gz"
    save_volume(path, np.ones(grid.shape, np.uint8), grid)
    result = load_volume(path, categorical=True)
    assert result.grid.shape == grid.shape
    np.testing.assert_allclose(result.grid.affine, a)
    assert result.audit["units"] == "mm"


def test_scores_restored_before_argmax_and_exterior_background():
    source = Grid((2, 1, 1), np.diag([2., 2., 2., 1.]))
    native = Grid((5, 1, 1), np.eye(4))
    scores = np.zeros((6, 2, 1, 1), np.float32)
    scores[1, :, 0, 0] = [.9, .4]
    scores[2, :, 0, 0] = [0, .8]
    restored = restore_scores(scores, source, native, support=np.ones(source.shape, bool))
    assert restored.argmax(0)[:, 0, 0].tolist() == [1, 1, 2, 0, 0]


def test_NORM_01_scaled_intensities_applied_once(tmp_path):
    img = nib.Nifti1Image(np.array([0, 1, 2], np.int16).reshape(3, 1, 1), np.eye(4))
    img.header.set_xyzt_units("mm")
    img.header.set_slope_inter(2, -100)
    path = tmp_path / "scaled.nii.gz"
    nib.save(img, path)
    volume = load_volume(path)
    assert volume.data.ravel().tolist() == [-100, -98, -96]
    assert volume.audit["proxy_slope"] == 2


def test_NORM_04_padding_is_hu_before_normalizing():
    source = Grid((2, 2, 2), np.eye(4))
    a = np.eye(4)
    a[0, 3] = -1
    target = Grid((4, 2, 2), a)
    result = resample(np.zeros(source.shape), source, target, kind="ct")
    assert np.all(result[0] == -1024)
    assert np.all(result[-1] == -1024)
