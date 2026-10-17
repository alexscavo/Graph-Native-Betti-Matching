"""Optional shared plant/leaf depth deformation and textured volume rendering."""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.distance import cdist

from data.augmentations import apply_augmentation
from data.spline_augmentation import rotation_affine
from data.spline_targets import fit_spline, resample


def augment_source(image, segmentation, nodes, geometry, parameters, rng, policy):
    """Input/output: padded cube, index/size coordinates, intensity [-.5,.5].

    Virtual physical distances measure the *deformed* geometry. The metric only
    undoes the final rigid rotation/flips, never the intended depth deformation.
    """
    raw = image[0].numpy().copy()
    mask = segmentation[0].numpy().copy()
    xyz = nodes.numpy().copy()
    size = raw.shape[0]
    assert raw.shape == (size,)*3
    height = np.zeros((size, size))
    if policy.get('nonplanar', True) and len(xyz):
        xy = xyz[:, :2]*size
        shifts = rng.normal(0, 5, len(xy)); shifts -= shifts.mean()
        kernel = np.exp(-cdist(xy, xy, 'sqeuclidean') / 200.)
        weights = np.linalg.solve(kernel + .05*np.eye(len(xy)), shifts)
        grid = np.stack(np.meshgrid(np.arange(size), np.arange(size), indexing='ij'), -1)
        height = (np.exp(-cdist(grid.reshape(-1, 2), xy, 'sqeuclidean')/200.) @ weights).reshape(size, size)
        slope = np.sqrt(sum(g*g for g in np.gradient(height))).max()
        occupied = np.where((raw > -.5) | (mask > 0))[2]
        clearance = min(10., float(occupied.min()), float(size-1-occupied.max())) if len(occupied) else 10.
        height *= min(1., clearance/max(abs(height).max(), 1e-8), 1./max(slope, 1e-8))
        grid = np.indices(raw.shape, dtype=float); grid[2] -= height[:, :, None]
        raw = map_coordinates(raw, grid, order=1, mode='constant', cval=-.5)
        mask = map_coordinates(mask, grid, order=0, mode='constant', cval=-.5)

    def warp(points):
        result = points.copy()
        result[..., 2] += map_coordinates(height, (points[..., :2]*size).reshape(-1, 2).T,
            order=1, mode='nearest').reshape(points.shape[:-1])/size
        return result

    xyz = warp(xyz)
    result = geometry
    if geometry is not None and policy.get('nonplanar', True):
        metric = geometry['metric'].numpy()
        inverse = np.linalg.inv(metric)
        controls, curves, guides = [], [], []
        # Densify original annotation polylines BEFORE the nonlinear warp.
        for i in range(len(geometry['controls'])):
            if 'full_paths' in geometry:
                start, stop = geometry['path_offsets'][i:i+2]
                path = geometry['full_paths'][int(start):int(stop)].numpy()
            else:
                path = geometry['guides'][i].numpy()
            path = resample(path @ metric.T, 257) @ inverse.T
            path = warp(path) @ metric.T
            fitted, curve = fit_spline(path, geometry['controls'].shape[1], geometry['curves'].shape[1])
            controls.append(fitted @ inverse.T); curves.append(curve @ inverse.T)
            guides.append(resample(path, geometry['guides'].shape[1]) @ inverse.T)
        result = dict(metric=geometry['metric'])
        for key, values in [('controls', controls), ('curves', curves), ('guides', guides)]:
            result[key] = torch.as_tensor(np.asarray(values).reshape(geometry[key].shape), dtype=torch.float32)

    if policy.get('textured_soft', True):
        def noise(sigma):
            x = gaussian_filter(rng.normal(size=raw.shape), sigma)
            return (x-x.mean()) / max(x.std(), 1e-8)
        slow, texture, fine = noise(8), noise(1.2), rng.normal(size=raw.shape)
        strength = rng.uniform(.75, 1.25)
        background = np.clip(rng.uniform(.18, .26) + strength*(.045*slow+.035*texture+.015*fine), 0, 1)
        foreground = np.clip(rng.uniform(.62, .82) + strength*(.04*slow+.025*texture+.015*fine), 0, 1)
        alpha = gaussian_filter(np.clip(raw+.5, 0, 1), rng.uniform(.5, .8))
        raw = ((1-alpha)*background+alpha*foreground-.5).astype('float32')

    transformed = apply_augmentation(torch.from_numpy(raw[None].astype('float32')),
        torch.from_numpy(mask[None].astype('float32')), torch.from_numpy(xyz.astype('float32')), parameters)
    if result is not None:
        matrix, offset = rotation_affine(parameters.quarter_turns, (size,)*3,
            dtype=nodes.dtype, device=nodes.device)
        for axis, flip in enumerate(parameters.flip_axes):
            if flip:
                matrix[axis] *= -1
                offset[axis] = (size-1)/size - offset[axis]
        result = {**{k: result[k] @ matrix.T + offset for k in ('controls','curves','guides')},
                  'metric': result['metric'] @ torch.linalg.inv(matrix)}
    return transformed.image, transformed.segmentation, transformed.nodes, result
