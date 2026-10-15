#!/usr/bin/env python3
"""IXI-only, plants/IXI, SyntheticMRI/IXI; representation trained in both stages."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler
from configs import validate_config
from data.loaders import build_data_loaders, build_evaluation_loader, dataset_sample_manifest
from data.loaders.common import image_graph_collate, seed_data_worker
from models import build_model
from models.checkpoint import load_checkpoint_file
from models.matcher import build_matcher
from training.engine import train_step, _move_graph_batch
from training.optim import build_optimizer, build_scheduler
from training.tracking import build_tracker
from scripts.controlled_spline_unfreeze import RoutedCriterion
from scripts.extend_scratch_geometry import atomic_checkpoint, capture_rng, restore_rng
from scripts.train_spline_weight_pilot import seed, monitoring_rng, geometry, write_json
from scripts.frozen_spline_diagnostic import training_evaluation_loader, matched_geometry
from scripts.train_relation_distance_pilot import prediction_diagnostics


class TransferCriterion(RoutedCriterion):
    def loss_spline(self,tokens,predictions,targets,assignments):
        if float(self.config['loss']['weights']['spline'])==0:
            # Straight-edge controls have no geometry objective in either stage.
            return {'spline':predictions['pred_nodes'].sum()*0.}
        return super().loss_spline(tokens,predictions,targets,assignments)


def configuration(plan, arm):
    from scripts.train_plant_generalization import configuration as original_configuration
    pretrain=arm.startswith('pretrain_')
    strategy,representation=arm.removeprefix('pretrain_').rsplit('_',1)
    assert strategy in ('ixi','plants','synthetic')
    assert representation in ('straight','spline')
    cfg=original_configuration(plan,'pretrain_vessels' if pretrain else 'vessels_'+representation)
    cfg['experiment']['name']=arm
    cfg['model']['decoder']['object_queries']=256
    cfg['evaluation']['protocol']['max_detections']=256
    target=cfg['data']['datasets']['synthetic_mri']
    target.update(root=plan['ixi_root'],train_samples=None,validation_samples=None,
                  patch_selection='all',spline_controls=8,spline_family='bspline')
    if pretrain and strategy=='plants':
        cfg['data']['datasets']['plants']=dict(role='source',root=plan['plant_root'],train_samples=None,
            validation_samples=None,coordinate_order_on_disk=['y','x'],spline_controls=8,
            virtual_voxel_mm=plan['plant_virtual_voxel_mm'])
    elif pretrain and strategy=='synthetic':
        cfg['data']['datasets']['source_mri']=dict(target,role='source',root=plan['synthetic_root'])
    mixed=pretrain and strategy!='ixi'
    cfg['domain_adaptation'].update(enabled=mixed)
    if mixed:
        domain=plan['domain_adaptation']
        cfg['domain_adaptation'].update(image_adversarial=True,graph_adversarial=True,
            consistency_regularization=True,class_weights=domain['class_weights'],learning_rate=domain['learning_rate'])
        cfg['loss']['enabled'].append('domain');cfg['loss']['weights']['domain']=domain['weight']
    cfg['loss']['weights']['spline']=plan['spline_weight'] if representation=='spline' else 0.
    validate_config(cfg)
    return cfg


def check_domain_geometry(model, criterion, batch, config):
    """One positive graph per domain proves geometry gradients reach the head."""
    result={}
    for domain in sorted(set(batch[-1].tolist())):
        i=next(i for i,d in enumerate(batch[-1].tolist()) if d==domain and len(batch[3][i])>0)
        probe=(batch[0][i:i+1],batch[1][i:i+1],[batch[2][i]],[batch[3][i]],[batch[4][i]],batch[5][i:i+1])
        volumes,targets=_move_graph_batch(probe,'cuda',config['training']['input'],True)
        tokens,predictions,_=model(volumes)
        losses=criterion(tokens,predictions,targets)
        gradients=torch.autograd.grad(losses['spline'],[p for p in model.spline_head.parameters() if p.requires_grad],allow_unused=True)
        norm=sum(float(g.detach().abs().sum()) for g in gradients if g is not None)
        assert torch.isfinite(losses['spline']) and float(losses['spline_positive_edges'])>0 and norm>0
        result[str(domain)]=dict(spline_loss=float(losses['spline'].detach()),positive_edges=float(losses['spline_positive_edges']),head_gradient_l1=norm)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/ixi_transfer.yaml'))
    p.add_argument('--arm',required=True,choices=[prefix+strategy+'_'+rep for prefix in ('pretrain_','') for strategy in ('ixi','plants','synthetic') for rep in ('straight','spline')])
    p.add_argument('--until-epoch',type=int,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();plan=yaml.safe_load(a.config.read_text())
    population=json.loads((Path(plan['ixi_root'])/'manifest.json').read_text())
    plan['plant_virtual_voxel_mm']=json.loads((Path(plan['ixi_root'])/'monitoring.json').read_text())['plant_virtual_voxel_mm']
    cfg=configuration(plan,a.arm)
    pretrain=a.arm.startswith('pretrain_');weighted=cfg['loss']['weights']['spline']>0
    for root in ('plant_root','ixi_root','synthetic_root'):
        if not (Path(plan[root])/'READY').exists():raise ValueError(f'Preparation not complete: {root}')
    out=Path(plan['output_root'])/'runs'/(( 'smoke-' if a.smoke else '')+a.arm)
    out.mkdir(parents=True,exist_ok=a.smoke or (out/'latest.pt').exists())
    if (out/'COMPLETE').exists():print('Already complete',out);return
    seed(plan['seed']);train,_=build_data_loaders(cfg)
    mri_cfg=copy.deepcopy(cfg);mri_cfg['data']['datasets']={'synthetic_mri':mri_cfg['data']['datasets']['synthetic_mri']}
    mri_cfg['data']['train_augmentation']=False
    mri_cfg['domain_adaptation']['enabled']=False
    mri_cfg['loss']['enabled']=[x for x in mri_cfg['loss']['enabled'] if x!='domain']
    mri_train,_=build_data_loaders(mri_cfg)
    manifest=population
    assert [r.sample_id for r in mri_train.dataset.datasets[0].records]==manifest['train_ids']
    if pretrain and a.arm.startswith('pretrain_ixi_'):
        generator=torch.Generator().manual_seed(plan['seed'])
        sampler=RandomSampler(train.dataset,replacement=True,num_samples=3*plan['batch_size'] if a.smoke else plan['pretrain_draws_per_epoch'],generator=generator)
        train=DataLoader(train.dataset,batch_size=plan['batch_size'],sampler=sampler,num_workers=plan['workers'],
            collate_fn=image_graph_collate,worker_init_fn=seed_data_worker,generator=generator,pin_memory=True)
    if a.smoke and pretrain and not a.arm.startswith('pretrain_ixi_'):train.sampler.num_samples=3*plan['batch_size']
    if pretrain and not a.smoke:assert train.sampler.num_samples==plan['pretrain_draws_per_epoch']
    ids=json.loads((Path(plan['ixi_root'])/'monitoring.json').read_text())['sample_ids']
    val=build_evaluation_loader(mri_cfg,dataset_name='synthetic_mri',sample_ids=ids['val'][:4] if a.smoke else ids['val'])
    train_eval=training_evaluation_loader(mri_train,mri_cfg,ids['train'][:4] if a.smoke else ids['train'])
    write_json(out/'dataset-manifest.json',dict(train=dataset_sample_manifest(train.dataset),validation=dataset_sample_manifest(val.dataset)))
    write_json(out/'evaluation-sample-ids.json',ids)
    (out/'resolved-config.yaml').write_text(yaml.safe_dump(cfg))
    seed(plan['seed'])  # Dataset construction must not alter paired initialization.
    model=build_model(cfg).cuda()
    # Identical architecture/initialization in controls; unused curve parameters stay frozen.
    for parameter in model.spline_head.parameters():parameter.requires_grad_(weighted)
    criterion=TransferCriterion(cfg,build_matcher(cfg),model.relation_embed,spline_head=model.spline_head,route='graph_only').cuda()
    optimizer=build_optimizer(cfg,model);scheduler=build_scheduler(cfg,optimizer,len(train))
    start=0;step=0;best=(-1.,-1.);best_epoch=0;initial_weights=None
    if (out/'latest.pt').exists() and not a.smoke:
        checkpoint=load_checkpoint_file(out/'latest.pt');model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
        start=checkpoint['epoch'];step=checkpoint['steps'];best=tuple(checkpoint['best_score']);best_epoch=checkpoint['best_epoch']
        restore_rng(checkpoint['rng'],train)
    elif not pretrain:
        initial_weights=Path(plan['output_root'])/'runs'/(('smoke-' if a.smoke else '')+'pretrain_'+a.arm)/'best.pt'
        checkpoint=load_checkpoint_file(initial_weights)
        # Continue the same learned representation, including spline geometry, into IXI-only training.
        state={k:v for k,v in checkpoint['net'].items() if 'domain_discriminator' not in k}
        report=model.load_state_dict(state,strict=False)
        assert not report.unexpected_keys and not report.missing_keys
    digest=hashlib.sha256()
    if pretrain and start==0:
        for name,value in sorted(model.state_dict().items()):
            if 'domain_discriminator' not in name:
                digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    if pretrain and not a.smoke and start==0:
        references=[json.loads((Path(plan['output_root'])/'runs'/('smoke-pretrain_'+s+'_'+rep)/'provenance.json').read_text())['initial_shared_state_sha256'] for s in ('ixi','plants','synthetic') for rep in ('straight','spline')]
        assert len(set(references+[digest.hexdigest()]))==1,'Shared initialization mismatch'
    write_json(out/'provenance.json',dict(initial_shared_state_sha256=digest.hexdigest() if pretrain and start==0 else None,arm=a.arm,pretrain=pretrain,initial_weights=str(initial_weights),
        seed=plan['seed'],draws_per_epoch=train.sampler.num_samples if pretrain else len(train.dataset),source=plan,
        gradient_route='graph_only',geometry_supervised_in_both_stages=weighted,plant_geometry_supervised=weighted and a.arm=='pretrain_plants_spline',
        best_selection='validation common_branch_f1 then native_edge_f1, unchanged from previous paired study',
        bn='training statistics during updates; saved running statistics during evaluation'))
    cfg['tracking']['enabled']=not a.smoke
    tracker=build_tracker(cfg,out,resume=start>0,launch_metadata={'arm':a.arm})
    started=time.monotonic();epoch=start;history=[]
    baseline=json.loads((Path(plan['ixi_root'])/'monitoring.json').read_text())['branch_bins']
    target_root=Path(mri_cfg['data']['datasets']['synthetic_mri']['root'])
    def payload():
        return dict(net=model.state_dict(),config=cfg,epoch=epoch,steps=step,optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),rng=capture_rng(train),best_score=best,best_epoch=best_epoch)
    def monitor():
        nonlocal best,best_epoch
        with monitoring_rng():
            metrics=geometry(model,val,mri_cfg,out,step,tracker,epoch,True)
            relation=prediction_diagnostics(out/f'validation-{step:06d}'/'predictions.json',target_root,'val',baseline,core=False)
            train_metrics=None
            if epoch%10==0 or a.smoke:
                train_metrics=geometry(model,train_eval,mri_cfg,out,step,None,epoch,True,split='train')
                prediction_diagnostics(out/f'train-evaluation-{step:06d}'/'predictions.json',target_root,'train',baseline,core=False)
        score=(metrics['common_branch_f1'],metrics['native_edge_f1'])
        if score>best:
            best=score;best_epoch=epoch;atomic_checkpoint(out/'best.pt',payload())
        row=dict(epoch=epoch,step=step,elapsed_seconds=time.monotonic()-started,validation=metrics,train=train_metrics,relation_diagnostic=relation,best_epoch=best_epoch)
        history.append(row)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        if start==0 and a.smoke:monitor()
        limit=min(a.until_epoch,cfg['training']['epochs'])
        stop=False
        for epoch in range(start+1,limit+1):
            for batch_index,batch in enumerate(train):
                step+=1
                if a.smoke:torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();batch_started=time.monotonic()
                if a.smoke and batch_index==0 and weighted:
                    model.train()
                    write_json(out/'domain-geometry-check.json',check_domain_geometry(model,criterion,batch,cfg))
                loss=train_step(model,criterion,optimizer,scheduler,batch,cfg,'cuda',epoch=epoch,iteration=step,total_iterations=len(train)*cfg['training']['epochs'])
                if a.smoke:
                    torch.cuda.synchronize()
                    diagnostic=dict(spline_loss=float(loss['spline'].detach()) if 'spline' in loss else None, spline_edges=float(loss['spline_positive_edges'].detach()) if 'spline_positive_edges' in loss else None, step=step,batch_size=len(batch[2]),seconds=time.monotonic()-batch_started,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,domain_loss=float(loss['domain'].detach()) if 'domain' in loss else None,domain_labels=batch[-1].tolist())
                    if cfg['domain_adaptation']['enabled']:
                        assert set(batch[-1].tolist())=={0,1}
                        for module in [model.backbone_domain_discriminator,model.instance_domain_discriminator]:
                            assert any(x.grad is not None and x.grad.abs().sum()>0 for x in module.parameters())
                    if weighted:
                        assert all(isinstance(g,dict) and 'controls' in g for g in batch[4])
                        assert float(loss['spline_positive_edges'])>0
                    with (out/'smoke-batches.jsonl').open('a') as f:f.write(json.dumps(diagnostic)+'\n')
                if batch_index==0:
                    assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
                    assert any(x.grad is not None and x.grad.abs().sum()>0 for x in model.encoder.parameters())
                    assert any(x.grad is not None and x.grad.abs().sum()>0 for x in model.spline_head.parameters())==weighted
                if tracker and step%100==0:
                    tracker.log_training({k:float(v.detach()) for k,v in loss.items() if torch.is_tensor(v) and v.numel()==1},iteration=step,epoch=epoch,learning_rate=optimizer.param_groups[0]['lr'])
                if a.smoke and batch_index==2:break
            if a.smoke or epoch%plan['monitor_epochs']==0 or epoch==limit:
                monitor()
                stop=not pretrain and epoch>=plan['min_finetune_epochs'] and epoch-best_epoch>=plan['patience_epochs']
            atomic_checkpoint(out/'latest.pt',payload())
            if stop or a.smoke:break
        complete=stop or epoch==cfg['training']['epochs']
        if complete or a.smoke:
            with monitoring_rng():
                train_destination=out/f'train-evaluation-{step:06d}'
                if not train_destination.exists():
                    train_metrics=geometry(model,train_eval,mri_cfg,out,step,tracker,epoch,True,split='train')
                else:
                    train_metrics=json.loads((train_destination/'common-v2-summary.json').read_text())
                if not a.smoke:
                    full=build_evaluation_loader(mri_cfg,dataset_name='synthetic_mri')
                    assert len(full.dataset)==len(population['validation_ids'])
                    full_output=out/'full-validation';full_output.mkdir(exist_ok=True)
                    full_metrics=geometry(model,full,mri_cfg,full_output,step,tracker,epoch,True)
                    write_json(out/'final-full-validation.json',full_metrics)
                    prediction_diagnostics(full_output/f'validation-{step:06d}'/'predictions.json',target_root,'val',baseline,core=True)
                    best_checkpoint=load_checkpoint_file(out/'best.pt');model.load_state_dict(best_checkpoint['net'])
                    best_output=out/'best-full-validation';best_output.mkdir(exist_ok=True)
                    best_metrics=geometry(model,full,mri_cfg,best_output,best_checkpoint['steps'],None,best_checkpoint['epoch'],True)
                    write_json(out/'best-full-validation.json',best_metrics)
                    prediction_diagnostics(best_output/f"validation-{best_checkpoint['steps']:06d}"/'predictions.json',target_root,'val',baseline,core=True)
            write_json(out/'summary.json',dict(epoch=epoch,steps=step,early_stopped=stop,train=train_metrics,best_epoch=best_epoch))
            if not a.smoke:(out/'COMPLETE').write_text('complete\n')
        if tracker:tracker.finish()
    except BaseException:
        if tracker:tracker.finish(exit_code=1)
        raise

if __name__=='__main__':main()
