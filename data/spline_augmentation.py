"""Exact affine targets for quarter-turn image augmentation (index/size coordinates)."""
import torch
from data.augmentations import _rotate_voxel_coordinates_once


def rotation_affine(turns, shape, *, dtype, device):
    """Return x' = x @ A.T + b, including the voxel-centre offset."""
    shape = tuple(int(s) for s in shape)
    probes = torch.cat((torch.zeros(1, 3, dtype=dtype, device=device),
                        torch.eye(3, dtype=dtype, device=device)))
    points = probes * torch.tensor(shape, dtype=dtype, device=device)
    current_shape = shape
    for k, (a, b) in zip(turns, ((0, 1), (0, 2), (1, 2))):
        points, current_shape = _rotate_voxel_coordinates_once(points, current_shape, k, a, b)
    points = points / torch.tensor(current_shape, dtype=dtype, device=device)
    offset = points[0]
    return (points[1:] - offset).T, offset


def rotate_spline_geometry(geometry, turns, shape):
    """Transform all target points without clipping controls outside the image.

    M' = M A^-1 retains physical distances in the original physical basis.
    Absolute physical positions may differ by a common translation; all losses
    use differences. Endpoints and residual controls share this same mapping.
    """
    controls = geometry['controls']
    matrix, offset = rotation_affine(turns, shape, dtype=controls.dtype, device=controls.device)
    result = {key: value @ matrix.T + offset for key, value in geometry.items()
              if key in ('controls', 'curves', 'guides', 'full_paths')}
    metric = geometry['metric']
    result['metric'] = metric @ torch.linalg.inv(matrix.to(metric))
    return result
