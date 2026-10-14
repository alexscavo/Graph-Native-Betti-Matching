import numpy as np
import torch
from data.augmentations import rotate_coordinates, rotate_volume
from data.spline_augmentation import rotate_spline_geometry
from models.spline import cubic_basis


def test_rotation_curve_equivariance_and_physical_distance():
    basis, _ = cubic_basis(8, 64)
    controls = torch.tensor([[[.1,.2,.3],[.2,.1,.4],[-.2,.3,.6],[.4,.8,1.1],
                              [.6,.7,.2],[.7,.5,.8],[.8,.6,.9],[.9,.8,.7]]])
    metric = torch.diag(torch.tensor([-64.,-32.,128.]))
    curves = basis @ controls
    geometry = dict(controls=controls, curves=curves, guides=curves.clone(), metric=metric)
    turns=(1,2,3)
    transformed=rotate_spline_geometry(geometry,turns,(64,64,64))
    assert torch.allclose(basis@transformed['controls'],transformed['curves'],atol=2e-7)
    expected=rotate_coordinates(controls[:,[0,-1]].reshape(-1,3),turns,(64,64,64))
    assert torch.allclose(transformed['controls'][:,[0,-1]].reshape(-1,3),expected,atol=2e-7)
    old=torch.diff(curves,dim=1)@metric.T
    new=torch.diff(transformed['curves'],dim=1)@transformed['metric'].T
    assert torch.allclose(old,new,atol=2e-5)
    assert not torch.equal(transformed['guides'],geometry['guides'])


def test_voxel_alignment_and_identity():
    image=torch.zeros(1,8,8,8);image[0,1,3,5]=1
    point=torch.tensor([[[1.,3.,5.]]])/8
    g=dict(controls=point,curves=point,guides=point,metric=torch.eye(3)*8)
    for turns in [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,2,3)]:
        result=rotate_spline_geometry(g,turns,(8,8,8))
        voxel=(result['guides'][0,0]*8).round().long()
        assert rotate_volume(image,turns)[0,voxel[0],voxel[1],voxel[2]]==1
