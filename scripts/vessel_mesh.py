"""Existing segmentation surface in normalized patch coordinates; display only."""
import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes


def mask_mesh(path):
    mask=nib.load(path).get_fdata(dtype=np.float32)>.5
    if not mask.any():return {'vertices':[],'faces':[]}
    vertices,faces,_,_=marching_cubes(mask.astype(np.float32),.5)
    # The viewer applies the same source-derived metric as for graph coordinates.
    vertices=vertices/np.array(mask.shape)
    return {'vertices':np.round(vertices,6).tolist(),'faces':faces.tolist()}
