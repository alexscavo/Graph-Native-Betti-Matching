import numpy as np
import torch
from data.augmentations import AugmentationParameters
from data.source_augmentation import augment_source
from data.spline_targets import fit_spline


def test_warp_refits_splines_preserves_endpoints_and_rotates_metric():
    nodes = torch.tensor([[.2,.2,.5],[.8,.2,.5],[.5,.8,.5]])
    paths = np.stack([np.linspace(nodes[0], nodes[1], 64), np.linspace(nodes[1], nodes[2],64)])
    fitted = [fit_spline(p*64,8,64) for p in paths]
    geometry = dict(controls=torch.tensor(np.stack([x[0]/64 for x in fitted]),dtype=torch.float32),
        curves=torch.tensor(np.stack([x[1]/64 for x in fitted]),dtype=torch.float32),
        guides=torch.tensor(paths,dtype=torch.float32),metric=torch.eye(3)*64)
    image = torch.full((1,64,64,64),-.5)
    image[:,:,:,30:35] = .5
    policy = dict(nonplanar=True,textured_soft=True)
    identity = AugmentationParameters()
    params = AugmentationParameters(quarter_turns=(1,2,3),flip_axes=(True,False,True))
    raw, mask, xyz, geo = augment_source(image,image,nodes,geometry,identity,np.random.default_rng(9),policy)
    rotated = augment_source(image,image,nodes,geometry,params,np.random.default_rng(9),policy)
    torch.testing.assert_close(geo['controls'][:,[0,-1]],xyz[torch.tensor([[0,1],[1,2]])])
    torch.testing.assert_close(geo['curves'][:,[0,-1]],geo['controls'][:,[0,-1]])
    assert float(xyz[:,2].std())>.005
    assert torch.isfinite(raw).all() and -.5 <= raw.min() and raw.max() <= .5
    assert float(raw[0,:,:,0].std()) > .01  # Padding/background filled too.
    before=(geo['controls'][:,1:]-geo['controls'][:,:-1]) @ geo['metric'].T
    after=(rotated[3]['controls'][:,1:]-rotated[3]['controls'][:,:-1]) @ rotated[3]['metric'].T
    torch.testing.assert_close(before,after,atol=2e-5,rtol=2e-5)
    # No mutation of cached nodes/targets.
    torch.testing.assert_close(nodes[:,2],torch.full((3,),.5))
    assert torch.all(geometry['guides'][...,2]==.5)
