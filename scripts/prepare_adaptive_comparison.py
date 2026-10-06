#!/usr/bin/env python3
"""Derive compact degree-two targets from the paired, uncertain full guides."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import pyvista as pv
from scipy.ndimage import distance_transform_edt
from scripts.ixi_vessel_graph import optimal_keep_mask, adaptive_tolerance_profile, chord_mask_fraction

ROOT=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/scratch_comparison_10k_300ep')


def derive(task):
    split, sample, output = task
    leaf=ROOT/'spline'/split
    with np.load(leaf/'spline'/f'{sample}.npz') as cache:
        anchors=cache['nodes'].copy(); branches=cache['edges'].copy()
        paths=cache['full_paths'].copy(); offsets=cache['path_offsets'].copy(); metric=cache['metric'].copy()
    mask=np.asarray(nib.load(leaf/'seg'/f'{sample}_seg.nii.gz').dataobj)>0
    shape=np.asarray(mask.shape); spacing=np.linalg.norm(metric/shape[None,:],axis=0)
    radius=distance_transform_edt(mask,sampling=spacing)
    positions=list(anchors); edges=[]; inherited_violations=0
    for (left,right), first,last in zip(branches,offsets[:-1],offsets[1:]):
        path=paths[first:last].astype(float); voxel=path*shape
        assert np.allclose(path[0],anchors[left],atol=1e-6) and np.allclose(path[-1],anchors[right],atol=1e-6)
        ids=np.clip(np.floor(voxel+.5).astype(int),0,shape-1)
        limits=adaptive_tolerance_profile(radius[tuple(ids.T)],radius_fraction=.75,minimum_tolerance_mm=.1)
        try:
            keep=optimal_keep_mask(path@metric.T,limits,voxel_polyline=voxel,mask=mask,minimum_chord_fraction=.95,preserve_adjacent=True)
        except RuntimeError as error:
            return {'sample_id':sample,'split':split,'error':str(error)}
        retained=np.flatnonzero(keep)
        inherited_violations += sum(b==a+1 and chord_mask_fraction(voxel[a],voxel[b],mask)+1e-12<.95 for a,b in zip(retained[:-1],retained[1:]))
        previous=int(left)
        for index in np.flatnonzero(keep)[1:-1]:
            current=len(positions); positions.append(path[index]); edges.append((previous,current)); previous=current
        edges.append((previous,int(right)))
    positions=np.asarray(positions).reshape(-1,3); edges=np.asarray(edges,dtype=int).reshape(-1,2)
    if output:
        target=Path(output)/split/'vtp'/f'{sample}_graph.vtp'
        target.parent.mkdir(parents=True,exist_ok=True)
        mesh=pv.PolyData(positions)
        if len(edges): mesh.lines=np.column_stack((np.full(len(edges),2),edges)).ravel()
        mesh.save(target)
    return {'sample_id':sample,'split':split,'node_count':len(positions),'anchor_count':len(anchors),'edge_count':len(edges),'inherited_mask_violations':int(inherited_violations)}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--audit',type=int,default=0); parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args(); manifest_path=ROOT/'comparison-manifest.json'; manifest=json.loads(manifest_path.read_text())
    output=ROOT/'adaptive'; tasks=[]
    for split,key in (('train','train_ids'),('val','validation_ids'),('test','test_ids')):
        ids=manifest[key]
        if args.audit:
            # Uniform deterministic coverage of the ID list; no score-based selection.
            ids=[ids[i] for i in np.linspace(0,len(ids)-1,args.audit,dtype=int)]
        tasks.extend((split,sample,None if args.audit else str(output)) for sample in ids)
    if args.workers==1: results=list(map(derive,tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results=[]
            for result in pool.map(derive,tasks,chunksize=8):
                results.append(result)
                if len(results)%500==0: print(f'prepared {len(results)}/{len(tasks)}',flush=True)
    errors=[r for r in results if 'error' in r]; good=[r for r in results if 'error' not in r]
    report={'counts':dict(Counter(r['split'] for r in results)),'failures':errors,'max_nodes':max((r['node_count'] for r in good),default=0),'nodes_p95':float(np.percentile([r['node_count'] for r in good],95)) if good else None,'over_120':sum(r['node_count']>120 for r in good),'over_256':sum(r['node_count']>256 for r in good),'inherited_mask_violations':sum(r['inherited_mask_violations'] for r in good),'added_degree_two':sum(r['node_count']-r['anchor_count'] for r in good),'policy':{'simplifier':'optimal','radius_fraction':.75,'floor_mm':.1,'minimum_chord_containment':.95,'preserve_original_adjacent_segments':True,'smoothing_iterations':0,'scope':'Existing cropped full generator guides; fixed shared endpoints; no re-extraction.'},'comparison_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    report_path=Path('/tmp/adaptive-comparison-audit.json') if args.audit else output/'manifest.json'
    report_path.parent.mkdir(parents=True,exist_ok=True); report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if errors: raise RuntimeError('Adaptive targets infeasible for original guides; no patches may be dropped or repaired silently.')
    if not args.audit:
        for split in ('train','val','test'):
            for directory in ('raw','seg'):
                dest=output/split/directory
                if not dest.exists(): dest.symlink_to(ROOT/'spline'/split/directory,target_is_directory=True)
        if not (output/'patch_index.csv').exists(): (output/'patch_index.csv').symlink_to(ROOT/'spline'/'patch_index.csv')
        (output/'sample-counts.json').write_text(json.dumps(results)+'\n')
        supervised_max=max(r['node_count'] for r in results if r['split'] in ('train','val'))
        budget=120 if supervised_max<=120 else int(np.ceil(supervised_max/32)*32)
        assert budget<=512, f'Adaptive budget {budget} exceeds audited limits; review before training'
        report['object_queries']=budget; report['supervised_max_nodes']=supervised_max
        report['test_over_budget']=sum(r['node_count']>budget for r in results if r['split']=='test')
        report_path.write_text(json.dumps(report,indent=2)+'\n')
        import yaml
        config={'defaults':[str(Path('configs/scratch_comparison_adaptive_300ep.yaml').resolve())], 'model':{'decoder':{'object_queries':budget}}}
        (output/'training-config.yaml').write_text(yaml.safe_dump(config))
        print(f'Object query budget={budget}; supervised maximum={supervised_max}',flush=True)
if __name__=='__main__':main()
