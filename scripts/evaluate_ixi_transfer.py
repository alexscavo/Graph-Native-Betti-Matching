#!/usr/bin/env python3
"""Evaluate the validation-selected checkpoint once on the fixed IXI test set."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from data.loaders import build_evaluation_loader
from models import build_model
from models.checkpoint import load_checkpoint_file
from scripts.train_spline_weight_pilot import geometry,write_json,seed
from scripts.train_relation_distance_pilot import prediction_diagnostics
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/ixi_transfer.yaml'));p.add_argument('--arm',required=True);a=p.parse_args()
plan=yaml.safe_load(a.config.read_text());run=Path(plan['output_root'])/'runs'/a.arm
assert (run/'COMPLETE').is_file()
out=run/'held-out-test';out.mkdir(exist_ok=False)
checkpoint=load_checkpoint_file(run/'best.pt');cfg=checkpoint['config'];cfg['data']['train_augmentation']=False;cfg['tracking']['enabled']=False
seed(plan['seed']);model=build_model(cfg).cuda();model.load_state_dict(checkpoint['net']);model.eval()
loader=build_evaluation_loader(cfg,dataset_name='synthetic_mri',split='test')
root=Path(plan['ixi_root']);population=json.loads((root/'manifest.json').read_text())
assert [r.sample_id for r in loader.dataset.records]==population['test_ids']
metrics=geometry(model,loader,cfg,out,checkpoint['steps'],None,checkpoint['epoch'],True,split='test')
baseline=json.loads((root/'monitoring.json').read_text())['branch_bins']
prediction_diagnostics(out/f"test-{checkpoint['steps']:06d}"/'predictions.json',root,'test',baseline,core=True)
write_json(out/'summary.json',dict(checkpoint=str(run/'best.pt'),epoch=checkpoint['epoch'],samples=len(loader.dataset),metrics=metrics))
print(json.dumps(metrics))
