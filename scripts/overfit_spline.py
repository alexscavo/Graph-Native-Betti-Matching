#!/usr/bin/env python3
"""Bounded four-patch diagnostic: calibrate, smoke, or train; no held-out claims."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import subprocess
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from configs import load_config
from data.loaders import build_data_loaders
from models import build_model
from models.checkpoint import load_legacy_model_checkpoint
from models.spline import canonical_reverse
from training.engine import _move_graph_batch
from training.losses.criterion import build_criterion
from training.optim import build_optimizer, build_scheduler
from training.evaluation.inference import infer_graphs


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)


def construct(config, initial, device):
    seed(config['experiment']['seed'])
    train,monitor=build_data_loaders(config)
    model=build_model(config).to(device)
    if initial:
        report=load_legacy_model_checkpoint(model,initial,allow_missing_spline_head=True)
        print('Initialized existing SyntheticMRI weights; new tensors:',report.missing,flush=True)
    criterion=build_criterion(config,model).to(device)
    return model,criterion,train,monitor


def prepared(batch,config,device):
    return _move_graph_batch(batch,device,config['training']['input'],True)


def scalars(losses):
    return {k:float(v.detach().cpu()) for k,v in losses.items() if torch.is_tensor(v) and v.numel()==1}


def calibrate(config,initial,output,device):
    records=[]
    for label,weights in [('pretrained',initial),('scratch',None)]:
        model,criterion,loader,_=construct(config,weights,device)
        model.eval()
        volumes,targets=prepared(next(iter(loader)),config,device)
        tokens,predictions,_=model(volumes)
        losses=criterion(tokens,predictions,targets)
        base=losses['total']-losses['spline']*criterion.weights['spline']
        base_grad=torch.autograd.grad(base,tokens,retain_graph=True)[0].norm()
        spline_grad=torch.autograd.grad(losses['spline'],tokens)[0].norm()
        if not torch.isfinite(base_grad+spline_grad) or spline_grad <= 1e-12:
            raise RuntimeError('Cannot calibrate a non-finite or zero geometry gradient')
        records.append(dict(initialization=label,base_token_gradient_norm=float(base_grad),
                            spline_token_gradient_norm=float(spline_grad),ratio=float(base_grad/spline_grad),losses=scalars(losses)))
        del model,criterion,loader,tokens,predictions,losses,volumes,targets,base,base_grad,spline_grad
        gc.collect();torch.cuda.empty_cache()
    # One common weight for both initializations; neither run gets its own tuning.
    raw=float(np.exp(np.mean(np.log([r['ratio'] for r in records]))))
    weight=float(np.clip(raw,1e-3,100.))
    manifest=Path(config['data']['datasets']['synthetic_mri']['root'])/'manifest.json'
    result=dict(weight=weight,unclipped_weight=raw,mode=config['loss']['spline']['mode'],
                target_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                method='Geometric mean of graph/spline gradient-norm ratios at shared decoder tokens, measured before updates on the same four training patches in both initializations; clamp [0.001,100]; fixed thereafter.',
                records=records)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


@torch.no_grad()
def export_predictions(model,criterion,loader,config,device,output):
    from vessel_mesh import mask_mesh
    model.eval();records=[]
    source_records=loader.dataset.records if hasattr(loader.dataset,'records') else loader.dataset.datasets[0].records
    count=0
    for batch in loader:
        volumes,targets=prepared(batch,config,device)
        tokens,predictions,_=model(volumes)
        assignments=criterion.matcher(predictions,targets)
        decoded=infer_graphs(tokens,predictions,model.relation_embed,
                             object_queries=criterion.object_queries,relation_tokens=criterion.relation_tokens,
                             spline_head=model.spline_head,spline_metrics=[g['metric'] for g in targets['spline']])
        for j,(source,target) in enumerate(assignments):
            mapping=dict(zip(target.tolist(),source.tolist()))
            geometry=targets['spline'][j];metric=geometry['metric']
            edges=targets['edges'][j]
            pairs=torch.tensor([[mapping[a],mapping[b]] for a,b in edges.tolist()],device=device)
            endpoints=predictions['pred_nodes'][j,pairs,:3]
            reverse=canonical_reverse(endpoints @ metric.T)
            pairs[reverse]=pairs[reverse][:,[1,0]]
            features=criterion._relation_features(tokens[j,:criterion.object_queries],tokens[j,criterion.object_queries:criterion.object_queries+criterion.relation_tokens],pairs)
            controls,curves=model.spline_head(features,predictions['pred_nodes'][j,pairs,:3],metric)
            # Put each diagnostic curve back into its GT edge order for comparison.
            controls[reverse]=controls[reverse].flip(1);curves[reverse]=curves[reverse].flip(1)
            record=dict(sample=source_records[count].sample_id,metric=metric.cpu().tolist(),
                        mesh=mask_mesh(source_records[count].segmentation),
                        target_nodes=targets['nodes'][j].cpu().tolist(),target_edges=edges.cpu().tolist(),
                        target_controls=geometry['controls'].cpu().tolist(),target_curves=geometry['curves'].cpu().tolist(),
                        guide_samples=geometry['guides'].cpu().tolist(),
                        matched_controls=controls.cpu().tolist(),matched_curves=curves.cpu().tolist(),
                        matched_nodes=predictions['pred_nodes'][j,torch.tensor([mapping[i] for i in range(len(mapping))],device=device),:3].cpu().tolist(),
                        decoded={k:v.tolist() for k,v in decoded[j].items()})
            records.append(record);count+=1
    (output/'predictions.json').write_text(json.dumps(records,separators=(',',':')))
    from plotly.offline import get_plotlyjs
    template=(Path(__file__).parent/'spline_predictions.html').read_text()
    (output/'predictions.html').write_text(template.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',json.dumps(records,separators=(',',':'))))


def train(config,initial,output,device,steps,smoke):
    output.mkdir(parents=True,exist_ok=False)
    model,criterion,loader,monitor=construct(config,initial,device)
    optimizer=build_optimizer(config,model)
    scheduler=build_scheduler(config,optimizer,len(loader))
    (output/'resolved-config.yaml').write_text(yaml.safe_dump(config))
    metadata=dict(initial_weights=str(initial) if initial else None,seed=config['experiment']['seed'],steps=steps,
                  monitoring='Same four training patches; NOT held-out validation.',
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (output/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    logs=[]

    def monitor_record(step):
        model.eval();parts=[]
        with torch.no_grad():
            for batch in monitor:
                volumes,targets=prepared(batch,config,device)
                tokens,predictions,_=model(volumes);parts.append(scalars(criterion(tokens,predictions,targets)))
        row={'step':step,'seconds':time.monotonic()-started,'phase':'monitor',
             **{k:float(np.mean([r[k] for r in parts])) for k in parts[0]}}
        logs.append(row)
        with (output/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        model.train()

    started=time.monotonic();monitor_record(0)
    iterator=iter(loader)
    for step in range(1,steps+1):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        volumes,targets=prepared(batch,config,device)
        criterion.set_training_progress(step,100*step/steps)
        optimizer.zero_grad(set_to_none=True)
        tokens,predictions,_=model(volumes);losses=criterion(tokens,predictions,targets)
        if not torch.isfinite(losses['total']):raise FloatingPointError('Non-finite total loss')
        losses['total'].backward()
        gradient=sum(float(p.grad.abs().sum()) for p in model.spline_head.parameters() if p.grad is not None)
        if not np.isfinite(gradient) or gradient<=0:raise RuntimeError('Spline head has no finite nonzero gradient')
        optimizer.step();scheduler.step()
        row={'step':step,'seconds':time.monotonic()-started,'phase':'train','spline_gradient_l1':gradient,**scalars(losses)}
        with (output/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if step==1 or step%50==0:monitor_record(step)
    if steps%50!=0:monitor_record(steps)
    summary=dict(initialization='pretrained' if initial else 'scratch',initial=logs[0],final=logs[-1],steps=steps,
                 spline_weight=criterion.weights['spline'],loss_mode=config['loss']['spline']['mode'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    export_predictions(model,criterion,monitor,config,device,output)
    if not smoke:
        # Model-only weights; this file is initialization/export, not a resume checkpoint.
        torch.save({'net':model.state_dict(),'config':config,'steps':steps},output/'model.pt')
    print('Completed',output,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['calibrate','smoke','train'],required=True)
    p.add_argument('--config',default='configs/overfit_spline_v0.yaml')
    p.add_argument('--initial-weights',type=Path)
    p.add_argument('--calibration',type=Path,default=Path('experiments/spline_learning/calibration_v0.json'))
    p.add_argument('--output',type=Path,default=Path('experiments/spline_learning/runs/overfit'))
    p.add_argument('--steps',type=int,default=500)
    args=p.parse_args()
    config=load_config(args.config)
    device=torch.device('cuda')
    if args.action=='calibrate':
        if args.initial_weights is None:raise ValueError('Calibration needs the pretrained checkpoint for the common weight')
        calibrate(config,args.initial_weights,args.calibration,device);return
    calibration=json.loads(args.calibration.read_text())
    manifest=Path(config['data']['datasets']['synthetic_mri']['root'])/'manifest.json'
    if calibration['mode']!=config['loss']['spline']['mode'] or calibration['target_manifest_sha256']!=hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError('Calibration must match target manifest and loss mode')
    config['loss']['weights']['spline']=calibration['weight']
    train(config,args.initial_weights,args.output,device,3 if args.action=='smoke' else args.steps,args.action=='smoke')


if __name__=='__main__':main()
