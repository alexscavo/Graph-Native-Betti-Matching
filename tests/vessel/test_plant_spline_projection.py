import torch
from data.augmentations import AugmentationParameters,apply_augmentation
from data.spline_augmentation import project_plant_geometry


def test_projected_curve_endpoints_follow_nodes_and_preserve_virtual_distances():
    # Nonzero residual and out-of-box control deliberately test no clipping.
    controls=torch.tensor([[[.2,.3,0.],[1.1,.5,0.],[.7,.6,0.],[.8,.7,0.]]])
    geometry={k:controls.clone() for k in ('controls','curves','guides')}
    params=AugmentationParameters(quarter_turns=(1,2,3),flip_axes=(True,False,True))
    result=project_plant_geometry(geometry,params,54,64,5,.55)
    image=torch.zeros(1,54,54,54)
    endpoints=controls[0,[0,-1]][:,[1,0,2]]+torch.tensor([0.,0.,.5])
    transformed=apply_augmentation(image,image,endpoints,params).nodes*54/64+5/64
    torch.testing.assert_close(result['controls'][0,[0,-1]],transformed)
    actual=(result['controls'][:,1:]-result['controls'][:,:-1])@result['metric'].T
    expected=(controls[:,1:]-controls[:,:-1])*54*.55
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
    # Affine invertibility protects controls outside the cube too.
    assert torch.allclose(torch.linalg.vector_norm(actual,dim=-1),torch.linalg.vector_norm(expected,dim=-1))
