#!/usr/bin/env python3
"""Bounded real-GPU gradient-routing, repeatability and BN-sensitivity checks."""
import copy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from data.loaders import build_data_loaders, build_evaluation_loader
from models import build_model
from models.checkpoint import load_checkpoint_file
from models.matcher import build_matcher
from scripts.controlled_spline_unfreeze import RoutedCriterion
from scripts.train_spline_weight_pilot import seed
from training.engine import _move_graph_batch, train_step
from training.optim import build_optimizer, build_scheduler

ROOT = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs')
OUTPUT = ROOT/'scratch_isolation_checks_20261012'
COUPLED = ROOT/'scratch_local_pilot_20261010_attempt2/local_spline'
ARMS = {'graph_control':ROOT/'scratch_local_pilot_20261010_attempt2/graph_control',
        'coupled':COUPLED, 'detached':ROOT/'scratch_detached_pilot_20261011/detached_spline'}


def compare(a, b):
    squared = reference = maximum = 0.
    unequal = 0
    for name, x in a.items():
        y = b[name]
        delta = (x.double()-y.double())
        maximum = max(maximum,float(delta.abs().max()))
        squared += float(delta.square().sum())
        reference += float(x.double().square().sum())
        unequal += int(not torch.equal(x,y))
    return dict(max_abs=maximum,l2=squared**.5,relative_l2=(squared/max(reference,1e-30))**.5,
                unequal_tensors=unequal)


@torch.no_grad()
def probe(model, volumes, batch_statistics):
    model.eval()
    if batch_statistics:
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm): module.train()
    memory=[]
    hook=model.decoder.encoder.register_forward_hook(lambda m,a,o:memory.append(o.detach()))
    try: tokens,predictions,features=model(volumes)
    finally:hook.remove()
    return dict(node_pair_max_delta=float((predictions['pred_nodes'][0]-predictions['pred_nodes'][1]).abs().max()),
                token_pair_mae=float((tokens[0]-tokens[1]).abs().mean()),
                projected_cnn_pair_mae=float((features[0]-features[1]).abs().mean()),
                transformer_memory_pair_mae=float((memory[0][0]-memory[0][1]).abs().mean()))


def main():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    checkpoint=load_checkpoint_file(COUPLED/'model-005000.pt')
    config=copy.deepcopy(checkpoint['config']);config['runtime']['workers']=0
    seed(364505);train,_=build_data_loaders(config)
    iterator=iter(train);batches=[next(iterator) for _ in range(3)]
    model=build_model(config).cuda()
    detector=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('spline_head.')]
    result=dict(source=str(COUPLED/'model-005000.pt'),steps_per_repeat=3,
                note='Diagnostics only: no checkpoint overwritten or full training. Same learned 5k state for all short repeats.')
    gradients={}
    for route in ('tokens_only','graph_only'):
        seed(364505);model.load_state_dict(checkpoint['net'],strict=True);model.train()
        criterion=RoutedCriterion(config,build_matcher(config),model.relation_embed,spline_head=model.spline_head,route=route).cuda()
        volumes,targets=_move_graph_batch(batches[0],'cuda','image',True)
        tokens,predictions,_=model(volumes);losses=criterion(tokens,predictions,targets)
        geometry=criterion.weights['spline']*losses['spline']
        gg=torch.autograd.grad(losses['total']-geometry,detector,retain_graph=True,allow_unused=True)
        sg=torch.autograd.grad(geometry,detector,allow_unused=True)
        graph_norm=sum(float(g.square().sum()) for g in gg if g is not None)**.5
        shape_norm=sum(float(g.square().sum()) for g in sg if g is not None)**.5
        gradients[route]=dict(graph_norm=graph_norm,geometry_norm=shape_norm,
                              nonzero_geometry_tensors=sum(int(torch.count_nonzero(g)>0) for g in sg if g is not None))
        if route=='graph_only':assert shape_norm==0.,'Geometry leaked into detector'
        del losses,geometry,gg,sg,tokens,predictions,criterion
    result['gradient_routes']=gradients
    states={};head_updates={}
    for name,route,weight in [('control_a','tokens_only',0.),('control_b','tokens_only',0.),
                              ('detached','graph_only',1.4739989612791453)]:
        seed(364505);model.load_state_dict(checkpoint['net'],strict=True)
        cfg=copy.deepcopy(config);cfg['loss']['weights']['spline']=weight
        criterion=RoutedCriterion(cfg,build_matcher(cfg),model.relation_embed,spline_head=model.spline_head,route=route).cuda()
        optimizer=build_optimizer(cfg,model);scheduler=build_scheduler(cfg,optimizer,2500)
        optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
        scheduler.load_state_dict(checkpoint['scheduler'])
        for step,batch in enumerate(batches,5001):
            train_step(model,criterion,optimizer,scheduler,batch,cfg,'cuda',epoch=3,iteration=step,total_iterations=750000)
        states[name]={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if not n.startswith('spline_head.')}
        head_updates[name]=max(float((v.detach().cpu()-checkpoint['net'][n]).abs().max())
                               for n,v in model.state_dict().items() if n.startswith('spline_head.') and v.is_floating_point())
        del optimizer,scheduler,criterion
    result['repeatability']=dict(control_repeat=compare(states['control_a'],states['control_b']),
                                 detached_vs_control=compare(states['control_a'],states['detached']),
                                 head_max_abs_updates=head_updates)
    del states,checkpoint
    bn={}
    for arm,path in ARMS.items():
        ckpt=load_checkpoint_file(path/'model.pt');model.load_state_dict(ckpt['net'],strict=True)
        provenance=json.loads((path/'provenance.json').read_text())
        loader=build_evaluation_loader(ckpt['config'],dataset_name='synthetic_mri',split='val',sample_ids=provenance['probe_ids'])
        volumes=next(iter(loader))[0].cuda()
        buffers={n:v.detach().clone() for n,v in model.named_buffers()}
        bn[arm]={'saved_statistics':probe(model,volumes,False),'batch_statistics':probe(model,volumes,True)}
        for n,v in model.named_buffers():v.copy_(buffers[n])
        assert all(torch.equal(v,buffers[n]) for n,v in model.named_buffers())
        bn[arm]['buffers_restored']=True
        del ckpt,volumes,buffers
    result['bn_sensitivity']=bn
    (OUTPUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
