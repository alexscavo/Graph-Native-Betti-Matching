#!/usr/bin/env python3
"""Freeze monitoring membership and length bins using IXI training data only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();r=a.root
manifest=json.loads((r/'manifest.json').read_text());train=manifest['train_ids'];val=manifest['validation_ids']
def choose(ids,n):return sorted(ids,key=lambda x:hashlib.sha256(('364505/monitor/'+x).encode()).digest())[:n]
arc=[];chord=[]
for name in train:
    with np.load(r/'train/spline'/f'{name}.npz') as c:
        for path in c['guides']@c['metric'].T:
            arc.append(float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()));chord.append(float(np.linalg.norm(path[-1]-path[0])))
subjects={x.split('_')[1] for x in train}
scales=[abs(np.linalg.det(np.asarray(manifest['patients'][s]['affine'])[:3,:3]))**(1/3) for s in sorted(subjects)]
result=dict(sample_ids={'train':choose(train,512),'val':choose(val,128)},plant_virtual_voxel_mm=float(np.median(scales)),
    virtual_scale_policy='Median geometric-mean voxel spacing over selected IXI training subjects; artificial plant scale, not anatomical plant millimetres',
    branch_bins={'gt_arc_quartile_cuts_mm':np.quantile(arc,[.25,.5,.75]).tolist(),'gt_chord_quartile_cuts_mm':np.quantile(chord,[.25,.5,.75]).tolist()})
(r/'monitoring.json').write_text(json.dumps(result,indent=2)+'\n');print('monitoring',len(train),len(val),result['plant_virtual_voxel_mm'])
