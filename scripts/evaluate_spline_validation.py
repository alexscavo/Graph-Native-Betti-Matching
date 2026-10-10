#!/usr/bin/env python3
"""Fixed full-validation comparison; no training or test-set evaluation."""
import argparse
import base64
import copy
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from configs import validate_config
from data.loaders import build_evaluation_loader
from models import build_model
from models.checkpoint import load_checkpoint_file
from training.losses.criterion import build_criterion
from training.tracking import build_tracker
from training.evaluation.spline_visualization import draw_projections
from scripts.train_spline_weight_pilot import geometry, seed, write_json
from scripts.frozen_spline_diagnostic import matched_geometry


def fixed_visualization_ids(ids, count, seed_value):
    by_subject = {}
    for sample in sorted(ids): by_subject.setdefault(sample.split('_')[1], []).append(sample)
    rng = random.Random(seed_value)
    subjects = rng.sample(sorted(by_subject), count)
    return [rng.choice(by_subject[s]) for s in sorted(subjects)]


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def collect(pilot):
    root = Path(pilot['output_root']); arms = list(pilot['checkpoints'])
    if not all((root/a/'summary.json').is_file() for a in arms):
        print('Other evaluations pending; last completed task will build the comparison.', flush=True)
        return
    summaries = {a: json.loads((root/a/'summary.json').read_text()) for a in arms}
    first = summaries[arms[0]]
    rows = []
    for arm, s in summaries.items():
        for key in ('sample_ids', 'visualization_ids', 'evaluation_protocol', 'input', 'dataset_root'):
            assert s['provenance'][key] == first['provenance'][key], (arm, key)
        assert s['provenance']['weights_unchanged'] and s['common']['samples']==1952
        native, common, match = s['native'], s['common'], s['matched']['all_pairs']
        rows.append(dict(model=arm, node_f1=native['node_f1'], edge_f1=native['edge_f1'],
            branch_f1=common['common_branch_f1'], curve_f1=common['common_curve_f1'],
            chord_curve_f1=s['chords']['common_curve_f1'],
            matched_guide_mm=match['guide_curve_mm'], matched_chord_mm=match['guide_chord_mm'],
            endpoint_neutral_mm=match['endpoint_neutral_guide_curve_mm'],
            endpoint_neutral_improvement=match['endpoint_neutral_guide_relative_improvement'],
            decoded_tortuosity_error=common['common_tortuosity_error'],
            betti0_error=common['common_betti0_abs_error_mean'], betti1_error=common['common_betti1_abs_error_mean'],
            anchor_count_error=common['common_anchor_count_abs_error_mean'],
            predicted_nodes=common['native_predicted_nodes_mean'],
            predicted_branches=common['common_predicted_branches_mean'],
            target_branches=common['common_target_branches_mean']))
    buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    def atomic(name, contents):
        target = root/name; temp = root/(name+f'.{os.getpid()}.tmp')
        temp.write_text(contents); temp.replace(target)
    atomic('comparison.csv', buffer.getvalue())
    columns = ['model','node_f1','edge_f1','branch_f1','curve_f1','chord_curve_f1','matched_guide_mm','endpoint_neutral_mm']
    table = '| ' + ' | '.join(columns) + ' |\n|'+'|'.join(['---']*len(columns))+'|\n'
    for row in rows:
        table += '| '+' | '.join(str(row[k]) if k=='model' else f'{row[k]:.5f}' for k in columns)+' |\n'
    note = ('All 1,952 validation patches, identical protocols; no test evaluation or BN recalibration. '
            'Matched-positive geometry is separate from full decoded graph metrics. Centerline guides are uncertain. '
            'Single seed; validation supports model selection, not an unbiased final test claim.')
    atomic('comparison.md', '# Full validation comparison\n\n'+note+'\n\n'+table)
    ids = first['provenance']['visualization_ids']; images = {}
    for sample in ids:
        images[sample] = {a: {kind: 'data:image/png;base64,'+base64.b64encode(
            (root/a/'visualizations'/f'{sample}-{kind}.png').read_bytes()).decode()
            for kind in ('curves','chords')} for a in arms}
    header = '<tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr>'
    body = ''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]) if k=='model' else f'{r[k]:.5f}')+'</td>' for k in columns)+'</tr>' for r in rows)
    gallery = '''<!doctype html><meta charset="utf-8"><title>Full validation comparison</title>
<style>body{font:16px system-ui;margin:24px;background:#101820;color:#eee}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #657080}select,label{margin:16px}main{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}img{width:100%}a{color:#83d8ff}h2{font-size:17px}@media(max-width:1000px){main{grid-template-columns:1fr}}</style>
<h1>Vessel graph: full validation comparison</h1><p>NOTE</p><div style="overflow:auto"><table>TABLE</table></div>
<p>Six cases were selected by a fixed seed, one per subject, before examining predictions. Each panel shows reference guides above predictions. Projected crossings need not be junctions. Open an image in a new tab to zoom.</p>
<select id="sample"></select><label><input id="chords" type="checkbox"> Show same-endpoint chords</label><main id="panels"></main>
<script>const pictures=DATA;const names=NAMES;const pick=document.getElementById('sample');
Object.keys(pictures).forEach(s=>pick.add(new Option(s,s)));
function render(){const mode=document.getElementById('chords').checked?'chords':'curves';const panels=document.getElementById('panels');panels.replaceChildren();for(const n of names){const box=document.createElement('section');const h=document.createElement('h2');h.textContent=n;const a=document.createElement('a');a.href=pictures[pick.value][n][mode];a.target='_blank';const img=document.createElement('img');img.src=a.href;img.alt=n+' '+pick.value+' '+mode;a.append(img);box.append(h,a);panels.append(box);}}
pick.onchange=render;document.getElementById('chords').onchange=render;render();</script>'''
    gallery = gallery.replace('NOTE',html.escape(note)).replace('TABLE',header+body).replace('NAMES',json.dumps(arms)).replace('DATA',json.dumps(images))
    atomic('comparison.html',gallery)
    atomic('comparison.json',json.dumps(dict(rows=rows,sample_ids=first['provenance']['sample_ids'],visualization_ids=ids,paired_protocol_verified=True),indent=2)+'\n')
    print('Saved comparison.csv, comparison.md and portable comparison.html:', root, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/spline_full_validation.yaml'))
    parser.add_argument('--arm',required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); pilot=yaml.safe_load(args.config.read_text())
    assert args.arm in pilot['checkpoints']
    manifest=json.loads(Path(pilot['population_manifest']).read_text())
    ids=manifest['validation_ids']; assert len(ids)==len(set(ids))==pilot['validation_samples']==1952
    assert set(ids).isdisjoint(manifest['train_ids']) and set(ids).isdisjoint(manifest['test_ids'])
    visual_ids=fixed_visualization_ids(ids,pilot['visualization_samples'],pilot['seed'])
    if args.smoke: ids=visual_ids[:2]; visual_ids=visual_ids[:2]
    checkpoint_path=Path(pilot['checkpoints'][args.arm]); checkpoint=load_checkpoint_file(checkpoint_path)
    config=copy.deepcopy(checkpoint['config'])
    config['experiment'].update(seed=pilot['seed'],name='full_validation_'+args.arm)
    config['runtime']['workers']=4
    config['data']['datasets']['synthetic_mri']['validation_samples']=1952
    config['tracking'].update(enabled=not args.smoke,mode='offline',group=pilot.get('tracking_group', 'spline-full-validation-20261009'))
    validate_config(config)
    output=Path(pilot['output_root'])/(('smoke-' if args.smoke else '')+args.arm)
    output.mkdir(parents=True,exist_ok=False)
    (output/'resolved-config.yaml').write_text(yaml.safe_dump(config))
    loader=build_evaluation_loader(config,dataset_name='synthetic_mri',split='val',sample_ids=ids)
    assert [r.sample_id for r in loader.dataset.records]==ids
    seed(pilot['seed']); model=build_model(config); model.load_state_dict(checkpoint['net'],strict=True);del checkpoint
    before=state_digest(model);model=model.cuda().eval();model.requires_grad_(False)
    assert all(not m.training for m in model.modules())
    provenance=dict(checkpoint=str(checkpoint_path),initial_state_sha256=before,sample_ids=ids,visualization_ids=visual_ids,
        evaluation_protocol=config['evaluation'],input=config['training']['input'],
        dataset_root=config['data']['datasets']['synthetic_mri']['root'],bn_recalibration_batches=0,split='val')
    write_json(output/'provenance.json',provenance)
    tracker=build_tracker(config,output,resume=False,launch_metadata=provenance)
    try:
        geometry(model,loader,config,output,0,tracker,0,True)
        destination=output/'validation-000000'
        criterion=build_criterion(config,model,validation=True).cuda()
        report=matched_geometry(model,criterion,loader,config,'cuda',
                                curvature_breakdown=pilot.get('curvature_breakdown', False))
        write_json(output/'matched-geometry.json',report);report.pop('rows')
        if tracker:
            tracker.log_metrics({'matched/'+k:v for k,v in report['all_pairs'].items() if v is not None},iteration=0,epoch=0)
        records={r['source_sample_id']:r for r in json.loads((destination/'predictions.json').read_text())}
        indices={r.sample_id:i for i,r in enumerate(loader.dataset.records)}
        paths=[]
        for sample in visual_ids:
            image,mask,nodes,_,guides,_=loader.dataset[indices[sample]];record=records[sample]
            for kind in ('curves','chords'):
                pred={'nodes':record['nodes_dhw']}
                if kind=='curves':pred['spline_curves']=record['spline_curves']
                else:pred['edges']=record['edges']
                path=output/'visualizations'/f'{sample}-{kind}.png'
                draw_projections(image[0].squeeze().numpy(),mask[0].squeeze().numpy(),nodes[0].numpy(),guides[0]['guides'].numpy(),pred,path,
                    f'{args.arm} · {sample} · {kind} · {len(record["nodes_dhw"])} nodes / {len(record["edges"])} edges')
                paths.append(path)
        if tracker:tracker.log_validation_images(paths,epoch=0)
        assert state_digest(model)==before,'Evaluation changed weights or buffers'
        provenance['weights_unchanged']=True
        write_json(output/'provenance.json',provenance)
        summary=dict(provenance=provenance,matched=report,
            native=json.loads((destination/'summary.json').read_text()),
            common=json.loads((destination/'common-v2-summary.json').read_text()),
            chords=json.loads((destination/'same-endpoint-chords-summary.json').read_text()))
        # Publish completion atomically; other tasks may be collecting results.
        write_json(output/'summary.pending.json',summary)
        (output/'summary.pending.json').replace(output/'summary.json')
        if tracker:tracker.finish()
        if not args.smoke:collect(pilot)
        print(args.arm,'evaluation complete:',len(ids),'validation patches',flush=True)
    except BaseException:
        if tracker:tracker.finish(exit_code=1)
        raise


if __name__=='__main__':main()
