import numpy as np
from lumbar_ct.body import body_centroids
from lumbar_ct.geometry import Grid, world_point


def test_body_removes_posterior_process_and_exports_world_mm():
    x,y,z = np.indices((50,60,40))
    body = ((x-25)/11)**2+((y-38)/12)**2+((z-20)/9)**2<=1
    posterior = (abs(x-25)<=2)&(y>=8)&(y<=38)&(abs(z-20)<=2)
    labels = np.zeros(body.shape,dtype=np.uint8); labels[body|posterior] = 2
    affine = np.diag([2.,2.,2.,1.]); affine[:3,3] = [-80,-120,17]
    bodies,result = body_centroids(labels,Grid(labels.shape,affine),case_id='synthetic')
    assert not bodies[:, :24].any()
    np.testing.assert_allclose(result['levels']['L2']['centroid_ras_mm'],world_point([25,38,20],affine),atol=1.)
    assert result['levels']['L1']['centroid_ras_mm'] is None
    assert result['levels']['L1']['status']=='missing_vertebra'
    assert result['unit']=='mm' and result['coordinate_system']=='RAS'
    assert result['centroids']=={'L2':result['levels']['L2']['centroid_ras_mm']}
    assert np.all((bodies>0)<= (labels>0))


def test_thin_fragment_is_failure_not_whole_vertebra_fallback():
    labels = np.zeros((20,20,20),dtype=np.uint8); labels[10,4:16,4:16]=1
    _,result = body_centroids(labels,Grid(labels.shape,np.diag([2.,2.,2.,1.])),case_id='thin')
    assert result['levels']['L1']['status']=='body_separation_failed'
    assert result['levels']['L1']['centroid_ras_mm'] is None
    assert result['centroids']=={}


def test_native_score_restore_matches_continuous_channels():
    from lumbar_ct.geometry import restore_scores
    from lumbar_ct.inference import native_argmax
    source=Grid((7,8,9),np.diag([2.,2.,2.,1.]))
    affine=np.diag([1.5,1.5,1.5,1.]);affine[:3,3]=[-1.,.5,-.5]
    native=Grid((10,10,13),affine)
    rng=np.random.default_rng(13)
    scores=rng.random((6,*source.shape)).astype(np.float32)
    support=np.ones(source.shape,dtype=bool);support[:2]=False
    expected=restore_scores(scores,source,native,support=support).argmax(0)
    np.testing.assert_array_equal(native_argmax(scores,source,native,support),expected)


def test_posterior_cut_removes_broad_connected_prediction_fragment():
    from lumbar_ct.body import separate_body
    x,y,z=np.indices((60,80,40))
    body=((x-30)/14)**2+((y-50)/12)**2+((z-20)/10)**2<=1
    fragment=((x-30)/14)**2+((y-20)/10)**2+((z-20)/8)**2<=1
    bridge=(x>=36)&(x<=46)&(y>=20)&(y<=50)&(z>=12)&(z<=28)
    result=separate_body(body|fragment|bridge,[2,2,2])
    assert result[30,50,20]
    assert not result[:,:38].any()
