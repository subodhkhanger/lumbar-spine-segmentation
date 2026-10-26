import numpy as np
import torch
from lumbar_ct.nnunet_inference import build_network,predict_nnunet_scores
from lumbar_ct.nnunet_inference import load_checkpoint


def test_nnunet_axis_conversion_including_plan_transpose(monkeypatch):
    class Directional(torch.nn.Module):
        def forward(self,x):
            # First tensor spatial axis is S/I after this particular plan.
            x=torch.roll(x,1,2)
            return torch.cat([x,-x,x*0,x*0,x*0,x*0],1)

    class Predictor:
        def __init__(self,**kwargs):pass
        def manual_initialization(self,network,*args):self.model=network
        def predict_sliding_window_return_logits(self,data):return self.model(data[None])[0]

    monkeypatch.setattr('nnunetv2.inference.predict_from_raw_data.nnUNetPredictor',Predictor)

    plans={'transpose_forward':[0,2,1], 'configurations':{'3d_fullres':{
           'architecture':{},'normalization_schemes':['NoNormalization'],'spacing':[2,2,2]}}}

    ct=np.random.default_rng(12).normal(size=(11,13,17)).astype(np.float32)

    actual=predict_nnunet_scores(Directional(),{'plans':plans,'dataset_json':{}},ct,'cpu')

    rolled=torch.from_numpy(np.roll(ct,1,axis=2))[None]

    expected=torch.cat([rolled,-rolled,torch.zeros(4,*ct.shape)]).softmax(0).numpy()

    np.testing.assert_allclose(actual,expected,atol=1e-7)


def test_official_residual_network_checkpoint_loads_without_deep_supervision(tmp_path):
    architecture={'network_class_name':'dynamic_network_architectures.architectures.unet.ResidualEncoderUNet',
        'arch_kwargs':{'n_stages':3,'features_per_stage':[4,8,16],'conv_op':'torch.nn.Conv3d',
        'kernel_sizes':[[3]*3]*3,'strides':[[1]*3,[2]*3,[2]*3],'n_blocks_per_stage':[1,1,1],
        'n_conv_per_stage_decoder':[1,1],'conv_bias':True,'norm_op':'torch.nn.InstanceNorm3d',
        'norm_op_kwargs':{'eps':1e-5,'affine':True},'dropout_op':None,'dropout_op_kwargs':None,
        'nonlin':'torch.nn.LeakyReLU','nonlin_kwargs':{'inplace':True}},
        '_kw_requires_import':['conv_op','norm_op','dropout_op','nonlin']}

    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    torch.set_num_threads(2)

    network=get_network_from_plans(architecture['network_class_name'],architecture['arch_kwargs'],
          architecture['_kw_requires_import'],1,6,deep_supervision=True)

    checkpoint={'kind':'nnunet_resenc_m','plans':{'configurations':{'3d_fullres':{'architecture':architecture}}},
                'state_dict':network.state_dict()}

    path=tmp_path/'model.pt';torch.save(checkpoint,path)

    loaded,_=load_checkpoint(path)
    x=torch.randn(1,1,16,20,24)

    with torch.no_grad():
        expected=network(x)[0]
        np.testing.assert_allclose(loaded(x).numpy(),expected.numpy(),atol=1e-7)
