#!/usr/bin/env python3
"""Read-only descriptive split audit; no model inference or dataset modification."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_vessel_core import reference

ROOT = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/scratch_comparison_10k_300ep')


def stats(values):
    a = np.asarray(values, float)
    return dict(n=len(a), mean=float(a.mean()), p10=float(np.quantile(a,.1)),
                median=float(np.median(a)), p90=float(np.quantile(a,.9))) if len(a) else dict(n=0)


def audit(root, ids, split):
    rows=[]
    for i, name in enumerate(ids):
        leaf=root/'spline'/split
        img=nib.load(leaf/'raw'/f'{name}_data.nii.gz')
        raw=img.get_fdata(dtype=np.float32)
        mask=nib.load(leaf/'seg'/f'{name}_seg.nii.gz').get_fdata(dtype=np.float32)>0.5
        if raw.shape!=mask.shape or not np.isfinite(raw).all():
            raise ValueError(f'Invalid image/segmentation: {split}/{name}')
        gt, metric=reference(leaf/'spline'/f'{name}.npz')
        with np.load(leaf/'spline'/f'{name}.npz') as d:
            nodes=d['nodes']; boundary=d['boundary']; paths=d['full_paths']
        lengths=np.asarray([np.linalg.norm(np.diff(c,axis=0),axis=1).sum() for c in gt.curves_mm])
        chords=np.linalg.norm(gt.nodes_mm[gt.edges[:,0]]-gt.nodes_mm[gt.edges[:,1]],axis=1)
        spacing=np.asarray(img.header.get_zooms()[:3])
        # Diagnostic voxel-centre approximation, matching normalized coordinate * image size.
        vox=np.rint(nodes*np.asarray(raw.shape)).astype(int)
        vox=np.clip(vox,0,np.asarray(raw.shape)-1)
        dist=distance_transform_edt(~mask, sampling=spacing) if mask.any() else None
        row=dict(sample_id=name, split=split, source_id=name.rsplit('_',1)[0],
            image_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
            shape=str(raw.shape), spacing=str(spacing.tolist()),
            foreground_fraction=float(mask.mean()), raw_mean=float(raw.mean()), raw_std=float(raw.std()),
            input_mean=float(raw.mean()-.33335259556770325),
            intensity_p01=float(np.quantile(raw,.01)),intensity_p99=float(np.quantile(raw,.99)),
            foreground_mean=float(raw[mask].mean()) if mask.any() else None,
            nodes=len(nodes), edges=len(gt.edges), boundary_fraction=float(np.mean(boundary)) if len(nodes) else None,
            node_coordinate_min=float(nodes.min()) if nodes.size else None,
            node_coordinate_max=float(nodes.max()) if nodes.size else None,
            metric_scale_error=float(np.max(np.abs(np.linalg.norm(metric,axis=0)-spacing*np.asarray(raw.shape)))),
            metric_orientation_difference=float(np.max(np.abs(metric-np.diag(spacing*np.asarray(raw.shape))))),
            node_mask_distance_mm=float(dist[tuple(vox.T)].mean()) if dist is not None and len(vox) else None,
            branch_length_mm=float(lengths.mean()) if len(lengths) else None,
            tortuosity=float((lengths[chords>1e-8]/chords[chords>1e-8]).mean()) if np.any(chords>1e-8) else None)
        rows.append(row)
        if (i+1)%100==0: print(split,i+1,flush=True)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=512);a=p.parse_args()
    if a.samples<1:p.error('samples must be positive')
    manifest=json.loads((a.root/'comparison-manifest.json').read_text())
    membership={'train':manifest['train_ids'], 'val':sorted(p.stem for p in (a.root/'spline/val/spline').glob('*.npz'))}
    sources={s:{n.rsplit('_',1)[0] for n in ids} for s,ids in membership.items()}
    rng=np.random.default_rng(364505)
    selected={s:sorted(rng.choice(ids,min(a.samples,len(ids)),replace=False).tolist()) for s,ids in membership.items()}
    a.output.mkdir(parents=True,exist_ok=False)
    rows={s:audit(a.root,ids,s) for s,ids in selected.items()}
    summary=dict(protocol='split-audit-v2',seed=364505,dataset=str(a.root),selected_ids=selected,
        full_membership_counts={s:len(ids) for s,ids in membership.items()},
        shared_patch_ids=sorted(set(membership['train'])&set(membership['val'])),
        shared_source_ids=sorted(sources['train']&sources['val']),
        source_id_rule='sample name without final underscore-separated crop index; verify against generation provenance',
        sampled_shared_image_hashes=sorted({r['image_sha256'] for r in rows['train']}&{r['image_sha256'] for r in rows['val']}),
        numeric_summary={s:{k:stats([r[k] for r in rr if r[k] is not None]) for k in rr[0]
            if isinstance(rr[0][k],(int,float)) or rr[0][k] is None} for s,rr in rows.items()},
        notes=['Descriptive patch statistics, not independent-subject significance tests.',
               '512 deterministic random patches per split; image hashes only cover this subset.',
               'Node-mask distances approximate coordinate-to-voxel mapping and are not perfect GT checks.',
               'Scale uses column norms: source affine orientation may differ from identity patch headers. Orientation difference alone is not a scale error.'])
    with (a.output/'per-patch.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows['train'][0]));w.writeheader();w.writerows(rows['train']+rows['val'])
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(a.output,flush=True)

if __name__=='__main__':main()
