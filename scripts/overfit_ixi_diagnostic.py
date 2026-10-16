#!/usr/bin/env python3
"""Paired four-patch IXI fitting diagnostic; not a generalization benchmark."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,yaml
from data.loaders import build_data_loaders,build_evaluation_loader
from models import build_model
from models.checkpoint import load_checkpoint_file
from models.matcher import build_matcher
from training.engine import train_step,evaluate_loss
from training.optim import build_optimizer,build_scheduler
from training.tracking import build_tracker
from scripts.train_ixi_transfer import configuration,transfer_checkpoint,TransferCriterion
from scripts.frozen_spline_diagnostic import training_evaluation_loader
from scripts.train_spline_weight_pilot import seed,geometry,monitoring_rng,write_json
from training.evaluation.spline_visualization import draw_projections

def setup(plan,arm):
    source=yaml.safe_load(Path('configs/ixi_transfer.yaml').read_text())
    cfg=configuration(source,'ixi_'+arm)
    cfg['data'].update(train_augmentation=False,batch_size=4,validation_batch_size=4)
    cfg['runtime']['workers']=0
    cfg['training'].update(epochs=plan['steps'],warmup_epochs=10)
    cfg['experiment']['name']='ixi_overfit_'+arm
    cfg['tracking'].update(group='ixi-overfit-20261016',mode='offline')
    full,_=build_data_loaders(cfg)
    train=training_evaluation_loader(full,cfg,plan['train_ids'])
    val=build_evaluation_loader(cfg,dataset_name='synthetic_mri',sample_ids=plan['validation_ids'])
    return cfg,train,val

def previews(plan,train):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from data.loaders.plants import PlantsDataset
    from data.loaders.common import SamplePaths
    out=Path(plan['preview_root']);out.mkdir(parents=True,exist_ok=True)
    for i,record in enumerate(train.dataset.records):
        image,mask,nodes,edges,geo,_=train.dataset[i]
        draw_projections(image[0].squeeze().numpy(),mask[0].squeeze().numpy(),nodes[0].numpy(),geo[0]['guides'].numpy(),{'nodes':[],'edges':[]},out/(record.sample_id+'.png'),record.sample_id+' — reference audit',prediction_label='Image / mask only')
    roots={'plants':'/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/plant_generalization_20261014/plants-k8','leaves':'/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/HALVS_DATASET/patches_3d_transfer_20261015/leaves-k8'}
    fig,axes=plt.subplots(2,3,figsize=(11,7));stats={}
    for row,(name,root) in enumerate(roots.items()):
        root=Path(root);raw=sorted((root/'train/raw').glob('*_data.png'))[0]
        name_id=raw.name.removesuffix('_data.png')
        record=SamplePaths(raw,root/'train/seg'/(name_id+'_seg.png'),root/'train/vtp'/(name_id+'_graph.vtp'),name_id)
        ds=PlantsDataset([record],augment=False)
        im,mask,*_=ds[0];im=im[0].squeeze().numpy();mask=mask[0].squeeze().numpy()
        for col,axis in enumerate((0,1,2)):
            axes[row,col].imshow(im.max(axis=axis),origin='lower',cmap='gray',vmin=float(im.min()),vmax=float(im.max()))
            axes[row,col].set_title(f'{name}: MIP along axis {axis}')
        stats[name]=dict(sample=ds.records[0].sample_id,shape=list(im.shape),input_min=float(im.min()),input_max=float(im.max()),mask_occupied_z=np.flatnonzero((mask>0).any(axis=(0,1))).tolist())
    fig.suptitle('Actual loader output: unaugmented source slab, no rendering changes')
    fig.tight_layout();fig.savefig(out/'source-embedding.png',dpi=130);plt.close(fig)
    write_json(out/'source-embedding.json',stats)

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['straight','spline'],default='spline');p.add_argument('--smoke',action='store_true');p.add_argument('--preview',action='store_true');a=p.parse_args()
    plan=json.loads(Path('experiments/ixi_overfit/plan.json').read_text());seed(plan['seed'])
    cfg,train,val=setup(plan,a.arm)
    if a.preview:previews(plan,train);return
    out=Path(plan['output_root'])/(('smoke-' if a.smoke else '')+a.arm);out.mkdir(parents=True,exist_ok=False)
    checkpoint=load_checkpoint_file(plan['initial_checkpoint'])
    seed(plan['seed']);model=build_model(cfg).cuda()
    transfer_checkpoint(model,checkpoint,'ixi_delayed_spline')
    for p in model.spline_head.parameters():p.requires_grad_(a.arm=='spline')
    criterion=TransferCriterion(cfg,build_matcher(cfg),model.relation_embed,spline_head=model.spline_head,route='graph_only').cuda()
    optimizer=build_optimizer(cfg,model);scheduler=build_scheduler(cfg,optimizer,1)
    write_json(out/'provenance.json',dict(plan=plan,arm=a.arm,early_stopping=False,augmentation=False,initialization='same IXI graph-only pretraining; fresh geometry',loss_comparison='train and validation evaluated without augmentation, eval BN statistics, same criterion'))
    (out/'resolved-config.yaml').write_text(yaml.safe_dump(cfg))
    tracker=build_tracker(cfg,out,resume=False)
    batch=next(iter(train));steps=2 if a.smoke else plan['steps']
    def monitor(step):
        with monitoring_rng():
            row=dict(step=step,train_loss=evaluate_loss(model,criterion,train,cfg,'cuda'),validation_loss=evaluate_loss(model,criterion,val,cfg,'cuda'))
            row['train_metrics']=geometry(model,train,cfg,out,step,None,step,True,split='train')
            row['validation_metrics']=geometry(model,val,cfg,out,step,tracker,step,True)
            if tracker:
                tracker.log_validation(row['validation_loss'],iteration=step,epoch=step)
                tracker.run.log({'diagnostic/train_eval_'+k:v for k,v in row['train_loss'].items()})
            # All four fitting examples, not only first/last, for initial/final review.
            if step in (0,steps):
                predictions=json.loads((out/f'train-evaluation-{step:06d}'/'predictions.json').read_text())
                for i,record in enumerate(train.dataset.records):
                    im,mask,nodes,_,geo,_=train.dataset[i];pred=predictions[i]
                    draw_projections(im[0].squeeze().numpy(),mask[0].squeeze().numpy(),nodes[0].numpy(),geo[0]['guides'].numpy(),{'nodes':pred['nodes_dhw'],'spline_curves':pred['spline_curves']},out/'previews'/f'{step:04d}-{record.sample_id}.png',f'{a.arm} update {step} — {record.sample_id}')
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        return row
    try:
        initial=monitor(0)
        for step in range(1,steps+1):
            loss=train_step(model,criterion,optimizer,scheduler,batch,cfg,'cuda',epoch=step,iteration=step,total_iterations=plan['steps'])
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if step==1 and a.arm=='spline':assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.spline_head.parameters())
            if tracker and step%10==0:tracker.log_training({k:float(v.detach()) for k,v in loss.items() if torch.is_tensor(v) and v.numel()==1},iteration=step,epoch=step,learning_rate=optimizer.param_groups[0]['lr'])
            if step%plan['monitor_interval']==0 or step==steps:final=monitor(step)
        write_json(out/'summary.json',dict(initial=initial,final=final,steps=steps))
        if not a.smoke:torch.save(dict(net=model.state_dict(),config=cfg,steps=steps),out/'final.pt')
        if tracker:tracker.finish()
    except BaseException:
        if tracker:tracker.finish(exit_code=1)
        raise

if __name__=='__main__':main()
