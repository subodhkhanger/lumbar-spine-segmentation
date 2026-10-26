import nibabel as nib
import numpy as np
from lumbar_ct.audit import inspect_header


def save(tmp_path, units='mm', affine=None):
    image=nib.Nifti1Image(np.zeros((4,5,6),dtype=np.int16), np.diag([1.5,1.5,1.5,1]) if affine is None else affine)
    image.header.set_xyzt_units(units)
    p=tmp_path/'image.nii.gz'; nib.save(image,p)
    return p


def test_source_unknown_units_explicit_policy_only(tmp_path):
    p=save(tmp_path,'unknown')
    _,g,r=inspect_header(p,'totalsegmentator_v201')
    assert g is not None and not r['issues'][0]['blocking']
    _,g,r=inspect_header(p,'verse2020_test')
    assert g is None and r['issues'][0]['blocking']


def test_source_policy_does_not_resolve_other_spacing(tmp_path):
    p=save(tmp_path,'unknown',np.eye(4))
    _,g,r=inspect_header(p,'totalsegmentator_v201')
    assert g is None and any(i['blocking'] for i in r['issues'])


def test_conflicting_coded_forms_reported(tmp_path):
    p=save(tmp_path)
    img=nib.load(p); q=img.affine.copy(); q[0,3]+=1
    img.set_qform(q,code=1); nib.save(img,p)
    _,_,r=inspect_header(p,'verse2020_test')
    assert any(i['type']=='qform_sform_conflict' and i['blocking'] for i in r['issues'])


def test_consumed_scaling_not_false_anomaly(tmp_path):
    p=save(tmp_path)
    img=nib.load(p); img.header.set_slope_inter(2, -1024); nib.save(img,p)
    image,_,r=inspect_header(p,'verse2020_test')
    assert r['raw_slope']=='2.0' and r['raw_intercept']=='-1024.0'
    assert not r['issues']
    assert np.asarray(image.dataobj).min()==-1024


def test_reference_grid_tolerance_requires_index_equivalence():
    from lumbar_ct.audit import compare_reference_grid
    from lumbar_ct.geometry import Grid
    a=np.eye(4); b=a.copy(); b[0,3]=.05
    assert compare_reference_grid(Grid((10,10,10),a),Grid((10,10,10),b))[0]
    b[0,3]=.2
    assert not compare_reference_grid(Grid((10,10,10),a),Grid((10,10,10),b))[0]
    # A small physical difference can exceed half a very small voxel.
    a=np.diag([.01,.01,.01,1]); b=a.copy(); b[0,3]=.008
    assert not compare_reference_grid(Grid((10,10,10),a),Grid((10,10,10),b))[0]


def test_raw_header_repairs_are_reported_without_changing_input(tmp_path):
    import gzip, hashlib, struct
    p=save(tmp_path)
    with gzip.open(p,'rb') as stream: raw=bytearray(stream.read())
    struct.pack_into('<f',raw,76,0.)  # invalid qfac, which NiBabel normally repairs
    with gzip.open(p,'wb') as stream: stream.write(raw)
    before=hashlib.sha256(p.read_bytes()).hexdigest()
    _,_,record=inspect_header(p,'verse2020_test')
    assert any(i['type']=='nifti_header_check' and i['blocking'] for i in record['issues'])
    assert hashlib.sha256(p.read_bytes()).hexdigest()==before


def test_pixdim_discrepancy_bounded_across_full_image(tmp_path):
    p=save(tmp_path)
    img=nib.load(p); img.header['pixdim'][1]=1.5001; nib.save(img,p)
    _,_,record=inspect_header(p,'verse2020_test')
    issue=next(i for i in record['issues'] if i['type']=='pixdim_affine_mismatch')
    assert not issue['blocking']
    img=nib.load(p); img.header['pixdim'][1]=1.7; nib.save(img,p)
    _,_,record=inspect_header(p,'verse2020_test')
    assert any(i['type']=='pixdim_affine_mismatch' and i['blocking'] for i in record['issues'])
