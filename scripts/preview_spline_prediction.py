#!/usr/bin/env python3
"""Static projections of the exact exported scratch prediction and existing mask."""
import argparse,json
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--targets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
d=json.loads(a.data.read_text());sample=d['sample'];prediction=d['prediction'];shape=np.array([64,64,64])
image=nib.load(a.targets/'test/raw'/f'{sample}_data.nii.gz').get_fdata(dtype=np.float32)
mask=nib.load(a.targets/'test/seg'/f'{sample}_seg.nii.gz').get_fdata()>0
fig,axes=plt.subplots(2,3,figsize=(13,8));lo,hi=np.percentile(image,[1,99.5])
for col,omitted in enumerate((0,1,2)):
 vertical,horizontal=[axis for axis in range(3) if axis!=omitted]
 for row in range(2):
  ax=axes[row,col];ax.imshow(image.max(axis=omitted),cmap='gray',origin='lower',vmin=lo,vmax=hi)
  foreground=mask.any(axis=omitted)
  ax.contour(foreground.astype(float),levels=[.5],colors=['#71899b'],linewidths=.7)
  curves=d['guides'] if row==0 else prediction['spline_curves']
  nodes=np.asarray(d['target_nodes'] if row==0 else prediction['nodes_dhw']).reshape(-1,3)*shape
  color='#efad40' if row==0 else '#00c7b3'
  for curve in curves:
   coords=np.asarray(curve)*shape;ax.plot(coords[:,horizontal],coords[:,vertical],color=color,lw=1.5)
  ax.scatter(nodes[:,horizontal],nodes[:,vertical],s=18,color=color,edgecolors='black',linewidths=.3)
  ax.set_xlim(-.5,63.5);ax.set_ylim(-.5,63.5);ax.set_title(('Original guide' if row==0 else 'Saved model prediction')+f' · projection along axis {omitted}')
  ax.set_xlabel(f'Voxel axis {horizontal}');ax.set_ylabel(f'Voxel axis {vertical}')
fig.suptitle(f"B-spline scratch model · epoch 300 · {sample}\n{len(prediction['nodes_dhw'])} predicted nodes / {len(prediction['edges'])} edges; {len(d['target_nodes'])} reference nodes / {len(d['target_edges'])} edges · guides/mask are uncertain")
fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=150);plt.close(fig)
