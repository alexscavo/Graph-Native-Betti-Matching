#!/usr/bin/env python3
"""Small real-data preflight for the degree-two comparison arm."""
import os,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from configs import load_config,validate_config
from data.loaders import build_evaluation_loader
from models import build_model
from metrics.branch_connectivity import contract_degree_two

root=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/scratch_comparison_10k_300ep')
os.environ['GNBM_OUTPUT_DIR']=str(root.parent/'runs');os.environ['SYNTHETIC_MRI_DATASET']=str(root/'adaptive')
config=load_config(root/'adaptive/training-config.yaml');validate_config(config);config['runtime']['workers']=0
report=json.loads((root/'adaptive/manifest.json').read_text());shared=json.loads((root/'comparison-manifest.json').read_text())
assert report['counts']==shared['counts'] and not report['failures']
assert config['model']['decoder']['object_queries']>=report['supervised_max_nodes']
for split,key in (('train','train_ids'),('val','validation_ids'),('test','test_ids')):
    ids=sorted(p.name.removesuffix('_graph.vtp') for p in (root/'adaptive'/split/'vtp').glob('*_graph.vtp'))
    assert ids==shared[key],split
loader=build_evaluation_loader(config,dataset_name='synthetic_mri',split='val',sample_ids=shared['validation_ids'][:2])
batch=next(iter(loader));assert tuple(batch[0].shape[1:])==(1,64,64,64)
for i,sample in enumerate(shared['validation_ids'][:len(batch[2])]):
    with np.load(root/'spline/val/spline'/f'{sample}.npz') as reference:
        contracted=contract_degree_two(batch[2][i].numpy(),batch[3][i].numpy())
        assert set(contracted.anchors)==set(range(len(reference['nodes'])))
        assert {tuple(sorted((b.start,b.end))) for b in contracted.branches}=={tuple(sorted(edge)) for edge in reference['edges']}
torch.manual_seed(config['experiment']['seed']);model=build_model(config)
assert model.spline_head is None
from training import build_criterion
criterion=build_criterion(config,model)
print('Adaptive exact split IDs, real batch, degree-two contraction, capacity, model and criterion passed.',flush=True)
(root/'adaptive/preflight.json').write_text(json.dumps({'passed':True,'object_queries':report['object_queries'],'supervised_max_nodes':report['supervised_max_nodes']})+'\n')
