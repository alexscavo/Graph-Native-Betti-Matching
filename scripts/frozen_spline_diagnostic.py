"""Head-only updates and paired geometry diagnostics; no new training objective."""
import hashlib
import numpy as np
import torch
from models.spline import canonical_reverse
from training.engine import _move_graph_batch
from metrics.representation_geometry import point_to_polyline_distance, sample_polyline_by_arclength


def freeze_detector(model):
    assert model.spline_head.family == 'bspline'
    assert model.spline_head.curve_evidence is None
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('spline_head.network.'))
    model.eval()  # Also freeze BN buffers and disable detector dropout.
    model.spline_head.train()


def detector_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith('spline_head.'):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def frozen_step(model, criterion, optimizer, scheduler, batch, config, device, **unused):
    model.eval()
    model.spline_head.train()
    volumes, targets = _move_graph_batch(batch, device, config['training']['input'], True)
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        tokens, predictions, _ = model(volumes)
        assignments = criterion.matcher(predictions, targets)
    losses = criterion.loss_spline(tokens, predictions, targets, assignments)
    losses['total'] = criterion.weights['spline'] * losses['spline']
    if not torch.isfinite(losses['total']):
        raise FloatingPointError('Non-finite frozen-head loss')
    losses['total'].backward()
    gradients = [p.grad for p in model.spline_head.network.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError('Missing/non-finite spline-head gradients')
    assert all(p.grad is None for name,p in model.named_parameters() if not name.startswith('spline_head.'))
    optimizer.step()
    scheduler.step()
    return losses


def _distance(curve, guide):
    return float((point_to_polyline_distance(sample_polyline_by_arclength(curve,.25), guide).mean()
                  + point_to_polyline_distance(sample_polyline_by_arclength(guide,.25), curve).mean())/2)


def summarize_pairs(rows):
    if not rows:
        return {'branches': 0}
    fields = [key for key in rows[0] if key not in ('sample_id','branch','max_endpoint_mm')]
    summary = {'branches':len(rows), **{key:float(np.mean([r[key] for r in rows])) for key in fields}}
    for name in ('fit','guide','endpoint_neutral_guide'):
        chord, curve = summary[name+'_chord_mm'], summary[name+'_curve_mm']
        summary[name+'_relative_improvement'] = (chord-curve)/chord if chord > 1e-8 else None
        summary[name+'_improved_fraction'] = float(np.mean([r[name+'_curve_mm'] < r[name+'_chord_mm'] for r in rows]))
    return summary


@torch.no_grad()
def matched_geometry(model, criterion, loader, config, device):
    """All Hungarian GT-positive pairs, independent of decoded edge selection.

    Endpoint-neutral curves are evaluation-only: transfer predicted residuals to
    the GT chord. Never substitute GT endpoints into the head or training loss.
    """
    model.eval()
    rows = []
    assignment_hash = hashlib.sha256()
    cursor = 0
    for batch in loader:
        volumes, targets = _move_graph_batch(batch, device, config['training']['input'], True)
        tokens, prediction, _ = model(volumes)
        assignments = criterion.matcher(prediction, targets)
        for b,(source,target) in enumerate(assignments):
            record = loader.dataset.records[cursor]; cursor += 1
            assignment_hash.update(record.sample_id.encode())
            assignment_hash.update(source.numpy().tobytes()); assignment_hash.update(target.numpy().tobytes())
            mapping = dict(zip(target.tolist(),source.tolist()))
            selected = [i for i,(a,c) in enumerate(targets['edges'][b].tolist()) if a in mapping and c in mapping]
            if not selected:
                continue
            pairs = torch.tensor([[mapping[a],mapping[c]] for a,c in targets['edges'][b][selected].tolist()], device=device)
            geometry = targets['spline'][b]
            metric = geometry['metric'].to(tokens)
            reverse = canonical_reverse(prediction['pred_nodes'][b,pairs,:3] @ metric.T)
            pairs[reverse] = pairs[reverse][:,[1,0]]
            endpoints = prediction['pred_nodes'][b,pairs,:3]
            relation = tokens[b,criterion.object_queries:criterion.object_queries+criterion.relation_tokens]
            features = criterion._relation_features(tokens[b,:criterion.object_queries],relation,pairs)
            controls, curves = model.spline_head(features,endpoints,metric)
            gt_controls = geometry['controls'][selected].clone()
            gt_curves = geometry['curves'][selected].clone()
            gt_controls[reverse] = gt_controls[reverse].flip(1)
            gt_curves[reverse] = gt_curves[reverse].flip(1)
            endpoints_mm = endpoints @ metric.T
            gt_endpoints = gt_controls[:,[0,-1]] @ metric.T
            t = torch.linspace(0,1,curves.shape[1],device=device)[None,:,None]
            chord = (1-t)*endpoints_mm[:,:1]+t*endpoints_mm[:,1:]
            gt_chord = (1-t)*gt_endpoints[:,:1]+t*gt_endpoints[:,1:]
            g = model.spline_head.greville[None,1:-1,None]
            pred_residual = controls[:,1:-1] @ metric.T - ((1-g)*endpoints_mm[:,:1]+g*endpoints_mm[:,1:])
            gt_residual = gt_controls[:,1:-1] @ metric.T - ((1-g)*gt_endpoints[:,:1]+g*gt_endpoints[:,1:])
            curve = curves @ metric.T
            neutral = curve-chord+gt_chord
            endpoint_error = (endpoints_mm-gt_endpoints).norm(dim=-1).amax(-1)
            fit_curve = (curve-gt_curves @ metric.T).norm(dim=-1).mean(-1)
            fit_chord = (chord-gt_curves @ metric.T).norm(dim=-1).mean(-1)
            residual_error = (pred_residual-gt_residual).norm(dim=-1).mean(-1)
            with np.load(record.graph.parent.parent/'spline'/f'{record.sample_id}.npz') as cache:
                paths, offsets = cache['full_paths'] @ metric.cpu().numpy().T, cache['path_offsets']
            for j,i in enumerate(selected):
                guide = paths[offsets[i]:offsets[i+1]]
                rows.append(dict(sample_id=record.sample_id,branch=i,max_endpoint_mm=float(endpoint_error[j]),
                    fit_curve_mm=float(fit_curve[j]),fit_chord_mm=float(fit_chord[j]),residual_control_mm=float(residual_error[j]),
                    guide_curve_mm=_distance(curve[j].cpu().numpy(),guide),guide_chord_mm=_distance(chord[j].cpu().numpy(),guide),
                    endpoint_neutral_guide_curve_mm=_distance(neutral[j].cpu().numpy(),guide),
                    endpoint_neutral_guide_chord_mm=_distance(gt_chord[j].cpu().numpy(),guide)))
    return {'assignment_sha256':assignment_hash.hexdigest(), 'all_pairs':summarize_pairs(rows),
            'endpoints_within_1mm':summarize_pairs([r for r in rows if r['max_endpoint_mm']<=1.]),
            'rows':rows}
