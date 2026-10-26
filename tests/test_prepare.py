import nibabel as nib
import numpy as np
import pytest

from lumbar_ct.artifacts import file_hash, reject_path_overlap, write_json
from lumbar_ct.datasets import SOURCE_FILES
from lumbar_ct.geometry import load_volume
from lumbar_ct.prepare import prepare_source, prepare_verse, source_metadata, write_prepared


def write_fixture(path, data, affine=None):
    img = nib.Nifti1Image(data, np.diag([2., 2., 2., 1.]) if affine is None else affine)
    img.header.set_xyzt_units("mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, path)


def source_case(tmp_path):
    folder = tmp_path / "input with spaces"
    write_fixture(folder / "ct.nii.gz", np.full((4, 4, 4), 100, np.float32))
    for i, (name, filename) in enumerate(SOURCE_FILES.items()):
        data = np.zeros((4, 4, 4), np.uint8)
        if i < 4:
            data[i, 1, 1] = 1
        write_fixture(folder / "segmentations" / filename, data)
    return folder


def test_source_case_roundtrip_and_inputs_unchanged(tmp_path):
    folder = source_case(tmp_path)
    before = file_hash(folder / "ct.nii.gz")
    prepared = prepare_source(folder, case_id="a", split="train")
    assert prepared.labels[:, 1, 1].tolist() == [1, 2, 3, 4]
    assert prepared.record["coverage"]["L5"] == "empty_annotation"
    write_prepared(prepared, tmp_path / "output with spaces")
    saved = load_volume(tmp_path / "output with spaces" / "ct_hu.nii.gz")
    np.testing.assert_array_equal(saved.data, prepared.ct)
    assert file_hash(folder / "ct.nii.gz") == before


def test_missing_source_file_is_failure(tmp_path):
    folder = source_case(tmp_path)
    (folder / "segmentations" / SOURCE_FILES["L2"]).unlink()
    with pytest.raises(ValueError, match="missing.*L2"):
        prepare_source(folder, case_id="a", split="train")


def test_verse_native_mapping_to_ct_derived_grid(tmp_path):
    ct_path, label_path = tmp_path / "ct.nii.gz", tmp_path / "mask.nii.gz"
    write_fixture(ct_path, np.ones((3, 3, 3), np.float32))
    raw = np.zeros((3, 3, 3), np.uint8)
    raw[1, 1, 1], raw[0, 0, 0] = 20, 25
    write_fixture(label_path, raw)
    prepared = prepare_verse(ct_path, label_path, case_id="verse-a")
    assert prepared.record["variant_flags"] == ["L6"]
    assert prepared.labels[1, 1, 1] == 1
    assert prepared.record["split"] == "test"


def test_input_output_overlap_rejected(tmp_path):
    for source, destination in [(tmp_path, tmp_path), (tmp_path, tmp_path / "child"),
                                (tmp_path / "child", tmp_path)]:
        with pytest.raises(ValueError, match="overlap"):
            reject_path_overlap(source, destination)


def test_json_rejects_nonfinite_before_replacing_existing_file(tmp_path):
    path = tmp_path / "artifact.json"
    write_json(path, {"valid": 1})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        write_json(path, {"bad": float("nan")})
    assert path.read_bytes() == before


def test_official_metadata_with_bom_and_semicolons(tmp_path):
    path = tmp_path / "meta.csv"
    path.write_text("\ufeffimage_id;split\ns0001;train\ns0002;val\ns0003;test\n")
    rows = source_metadata(path)
    assert [row["split"] for row in rows.values()] == ["train", "validation", "test"]


def test_unknown_split_in_metadata_is_error(tmp_path):
    path = tmp_path / "meta.csv"
    path.write_text("image_id;split\ns0001;unexpected\n")
    with pytest.raises(ValueError, match="unknown official split"):
        source_metadata(path)


def test_verse_missing_reference_units_inherit_only_matching_mm_ct(tmp_path):
    import nibabel as nib
    ct_path, label_path=tmp_path/'ct.nii.gz',tmp_path/'mask.nii.gz'
    write_fixture(ct_path,np.ones((3,3,3),np.float32))
    raw=np.zeros((3,3,3),np.uint8); raw[1,1,1]=20
    write_fixture(label_path,raw)
    image=nib.load(label_path); image.header.set_xyzt_units('unknown'); nib.save(image,label_path)
    prepared=prepare_verse(ct_path,label_path,case_id='paired')
    issue=prepared.record['label_audits']['verse']['issues'][0]
    assert issue['resolved_unit']=='mm' and issue['policy']['paired_ct']==str(ct_path)
    # Matching spacing alone cannot resolve missing units: full origin must match.
    image=nib.load(label_path); affine=image.affine.copy(); affine[0,3]+=1
    image.set_sform(affine,code=1); nib.save(image,label_path)
    with pytest.raises(ValueError,match='unresolved spatial units'):
        prepare_verse(ct_path,label_path,case_id='mismatched')


def test_overlapping_evaluation_masks_are_preserved_separately(tmp_path):
    folder=source_case(tmp_path)
    second=load_volume(folder/'segmentations'/SOURCE_FILES['L2'],categorical=True)
    second.data[0,1,1]=1  # preserve both L1 and L2 at this voxel
    write_fixture(folder/'segmentations'/SOURCE_FILES['L2'],second.data)
    with pytest.raises(ValueError,match='overlap'):
        prepare_source(folder,case_id='overlap',split='train')
    case=prepare_source(folder,case_id='overlap',split='test',allow_overlaps=True)
    assert case.labels is None and case.record['label_format']=='separate_binary'
    assert case.binary_labels['L1'][0,1,1] and case.binary_labels['L2'][0,1,1]
    dest=tmp_path/'prepared'; write_prepared(case,dest)
    assert not (dest/'labels.nii.gz').exists()
    assert load_volume(dest/'L1.nii.gz').data[0,1,1]==1
    assert load_volume(dest/'L2.nii.gz').data[0,1,1]==1


def test_verse_release_units_policy_is_explicit_opt_in(tmp_path):
    ct,mask=tmp_path/'ct.nii.gz',tmp_path/'mask.nii.gz'
    write_fixture(ct,np.ones((3,3,3),np.float32))
    raw=np.zeros((3,3,3),np.uint8);raw[1,1,1]=20
    write_fixture(mask,raw)
    for p in [ct,mask]:
        img=nib.load(p);img.header.set_xyzt_units('unknown');nib.save(img,p)
    with pytest.raises(ValueError,match='unresolved spatial units'):
        prepare_verse(ct,mask,case_id='release')
    case=prepare_verse(ct,mask,case_id='release',allow_release_units=True)
    assert case.labels[1,1,1]==1 and case.record['ct_audit']['resolved_units']=='mm'
    assert nib.load(ct).header.get_xyzt_units()[0]=='unknown'


def test_coarse_grid_target_loss_is_recorded_for_evaluation_only(tmp_path):
    folder=source_case(tmp_path)
    with pytest.raises(ValueError,match='targets lost'):
        prepare_source(folder,case_id='tiny',split='train',spacing=8.)
    case=prepare_source(folder,case_id='tiny',split='test',spacing=8.,allow_overlaps=True,allow_target_loss=True)
    assert case.record['lost_target_levels']==['L1','L2','L3','L4']
    assert case.record['coverage']['L1']=='annotated_present'
    assert case.record['observed_labels']==[]
    assert not case.labels.any()
