import copy
import pickle
import numpy as np
import pytest
from lumbar_ct.boundary_sampling import crop_pair,BoundaryTrainingDataset


@pytest.mark.parametrize('axis',[0,1,2])
@pytest.mark.parametrize('upper',[True,False])
def test_boundary_crop_preserves_alignment_labels_and_sampling_coordinates(axis,upper):
    seg=np.zeros((1,12,12,12),np.int16)
    seg[:,2:10,2:10,2:10]=2;seg[:,3:5,3:5,3:5]=1
    data=seg.astype(np.float32)*100
    props={'class_locations':{i:np.argwhere(seg==i) for i in [1,2]}}
    original=copy.deepcopy(props)
    x,y,p=crop_pair(data,seg,props,axis=axis,cut=6,keep_upper=upper)
    np.testing.assert_array_equal(x,y*100)
    assert set(np.unique(y)).issubset({0,1,2})
    for i,points in p['class_locations'].items():
        np.testing.assert_array_equal(points,np.argwhere(y==i))
        np.testing.assert_array_equal(props['class_locations'][i],original['class_locations'][i])
    assert seg.shape==(1,12,12,12)


def test_sampler_never_uses_unlisted_cases_or_changes_target_labels(tmp_path):
    class Dataset:
        identifiers=['training','negative']
        def load_case(self,case):
            seg=np.zeros((1,30,8,8),np.int16)
            if case=='training':
                for i in range(1,6):seg[:,i*4:i*4+3,2:6,2:6]=i
            return seg.astype(np.float32),seg,None,{'class_locations':{i:np.argwhere(seg==i) for i in range(1,6)}}
    dataset=Dataset()
    for name in ['training','negative','unlisted_validation']:
        with (tmp_path/(name+'.pkl')).open('wb') as f:pickle.dump(dataset.load_case(name)[3],f)
    wrapper=BoundaryTrainingDataset(dataset,tmp_path,crop_probability=0)
    assert all('unlisted_validation' not in cases for cases in wrapper.groups.values())
    for level in range(1,6):
        _,seg,_,properties=wrapper.load_case(f'L{level}')
        assert set(np.unique(seg))==set(range(6))
        assert list(properties['class_locations'])==[level]
