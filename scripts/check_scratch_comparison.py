#!/usr/bin/env python3
"""Small preflight: paired memberships, real batch, controls, common initialization."""
import hashlib, json, os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from configs import load_config, validate_config
from models import build_model
from data.loaders import build_evaluation_loader, discover_synthetic_mri
from training.evaluation.spline_visualization import draw_projections
import numpy as np

root=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/scratch_comparison_10k_300ep')
os.environ['GNBM_OUTPUT_DIR']=str(root.parent/'runs')
reference=json.loads((root/'comparison-manifest.json').read_text())
shared=None
for arm in ('old','spline'):
    os.environ['SYNTHETIC_MRI_DATASET']=str(root/arm)
    config=load_config(f'configs/scratch_comparison_{arm}_300ep.yaml'); validate_config(config)
    config['runtime']['workers']=0
    for split, key in (('train','train_ids'),('val','validation_ids'),('test','test_ids')):
        records=discover_synthetic_mri(root/arm,split)
        ids=sorted(r.sample_id for r in records)
        assert ids==reference[key], (arm,split,len(ids))
        for r in records:
            assert r.graph.resolve()==(root/'spline'/split/'vtp'/r.graph.name).resolve()
    loader=build_evaluation_loader(config,dataset_name='synthetic_mri',split='val',sample_ids=reference['validation_ids'][:1])
    batch=next(iter(loader)); assert tuple(batch[0].shape)==(1,1,64,64,64)
    if arm=='spline': assert batch[4][0]['controls'].shape[1:]==(8,3)
    else: assert batch[4][0] is None
    torch.manual_seed(config['experiment']['seed']); model=build_model(config)
    hashes={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items() if not k.startswith('spline_head.')}
    if shared is None: shared=hashes
    else: assert shared==hashes, 'Common initial weights differ'
    assert (model.spline_head is not None)==(arm=='spline')
    # Verify straight-edge visualization fallback on a real validation graph.
    nodes=batch[2][0].numpy(); edges=batch[3][0].numpy()
    draw_projections(batch[0][0].squeeze().numpy(),batch[1][0].squeeze().numpy()+.5,nodes,nodes[edges],{'nodes':nodes,'edges':edges},Path('/tmp')/f'comparison-{arm}-preflight.png',arm)
    del model
    print(f'{arm}: config, paired IDs, real sample, model and projection passed',flush=True)
print('Common model initialization is identical; neither model loaded a checkpoint.',flush=True)
