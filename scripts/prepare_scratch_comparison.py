#!/usr/bin/env python3
"""Pair straight and spline edges with identical endpoint graphs and membership."""
import argparse
from collections import defaultdict, Counter
import csv, hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.prepare_spline_dataset import prepare_patient



def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    cache=Path('experiments/spline_learning/synthetic_mri_k8_10k_val2k').resolve()
    manifest=json.loads((cache/'manifest.json').read_text()); source=Path(manifest['source']); out=args.output
    if (out/'comparison-manifest.json').exists():
        print((out/'comparison-manifest.json').read_text()); return
    out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((cache/'patch_index.csv').open()))
    selected={r['sample_id'] for r in rows}
    all_rows=list(csv.DictReader((source/'new_patches_boundary/patch_index.csv').open()))
    patients=set(manifest['patients'])
    ranked=sorted([r for r in all_rows if r['split']=='train' and r['patient_id'] in patients],key=lambda r:hashlib.sha256(f"{manifest['seed']}\0train\0{r['sample_id']}".encode()).digest())
    prior=set(json.loads((cache/'preparation-plan.json').read_text())['selected_ids'])
    remaining=[r for r in ranked if r['sample_id'] not in prior]
    extras=out/'additional-targets'; (extras/'patients').mkdir(parents=True,exist_ok=True)
    excluded=[]; offset=0
    while sum(r['split']=='train' for r in rows)<10000:
        needed=10000-sum(r['split']=='train' for r in rows)
        candidates=remaining[offset:offset+needed+32]; offset+=len(candidates)
        assert candidates
        grouped=defaultdict(list)
        for r in candidates: grouped[r['patient_id']].append(r)
        accepted=[]
        for patient, group in grouped.items():
            cached=extras/'patients'/f'{patient}.json'
            result=json.loads(cached.read_text()) if cached.exists() else prepare_patient((str(source),manifest['graph_root'],str(extras),patient,group,manifest['crop_config'],8))
            assert {r['sample_id'] for r in result['accepted']+result['excluded']} == {r['sample_id'] for r in group}
            accepted.extend(result['accepted']); excluded.extend(result['excluded'])
        rank={r['sample_id']:i for i,r in enumerate(candidates)}
        rows.extend(sorted(accepted,key=lambda r:rank[r['sample_id']])[:needed])
    grouped=defaultdict(list)
    for r in rows: grouped[r['patient_id']].append(r)
    max_nodes=0; histogram=Counter()
    for patient, group in grouped.items():
        for r in group:
            split,sample=r['split'],r['sample_id']; origin=cache if sample in selected else extras
            for arm in ('old','spline'):
                for directory in ('raw','seg','vtp','spline'):
                    (out/arm/split/directory).mkdir(parents=True,exist_ok=True)
                for directory,suffix in (('raw','data.nii.gz'),('seg','seg.nii.gz')):
                    dest=out/arm/split/directory/f'{sample}_{suffix}'
                    if not dest.exists(): dest.symlink_to(origin/split/directory/dest.name)
            for directory,name in (('vtp',f'{sample}_graph.vtp'),('spline',f'{sample}.npz')):
                dest=out/'spline'/split/directory/name
                if not dest.exists(): dest.symlink_to(origin/split/directory/name)
            with np.load(origin/split/'spline'/f'{sample}.npz') as geometry:
                max_nodes=max(max_nodes,len(geometry['nodes'])); histogram[len(geometry['nodes'])]+=1
            dest=out/'old'/split/'vtp'/f'{sample}_graph.vtp'
            if dest.exists() or dest.is_symlink(): dest.unlink()
            dest.symlink_to(origin/split/'vtp'/dest.name)
        print(f'paired patient={patient}',flush=True)
    for arm in ('old','spline'):
        with (out/arm/'patch_index.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(sorted(rows,key=lambda r:r['sample_id']))
    result={'counts':dict(Counter(r['split'] for r in rows)), 'seed':manifest['seed'], 'train_ids':sorted(r['sample_id'] for r in rows if r['split']=='train'), 'validation_ids':sorted(r['sample_id'] for r in rows if r['split']=='val'), 'test_ids':sorted(r['sample_id'] for r in rows if r['split']=='test'), 'additional_train':291,'old_max_nodes':max_nodes,'old_node_histogram':dict(histogram),'boundary_policy':manifest['boundary_policy'],'source_manifest':str(cache/'manifest.json'),'extra_exclusions':excluded,'old_representation':'Same junction, termination, and boundary nodes as spline targets; straight edges; no intermediate degree-two nodes.'}
    assert result['counts']=={'train':10000,'val':1952,'test':13071}
    (out/'comparison-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('counts','old_max_nodes','additional_train')}))
if __name__=='__main__': main()
