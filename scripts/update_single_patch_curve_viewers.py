#!/usr/bin/env python3
"""Add Catmull–Rom to the exact saved IXI/SyntheticMRI single-case studies.

Reuse embedded volumes, masks, guides and original fits. No dataset reads or
training changes; the original case and original numerical fits stay identical.
"""
import base64
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO.parent/'graph-representation/experiments'
SHARED = EXPERIMENTS/'ixi_patch/src'
spec = importlib.util.spec_from_file_location('single_patch_curves', SHARED/'build.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def read_data(path):
    page = path.read_text()
    marker = '<script>const DATA='
    start = page.index(marker)+len(marker)
    data, _ = json.JSONDecoder().raw_decode(page[start:])
    bundle = page.split('<script>', 1)[1].split('</script>', 1)[0]
    return data, bundle


def extend_case(name):
    results = EXPERIMENTS/name/'results'
    page_path = results/f'{name}.html'
    data, bundle = read_data(page_path)
    before = hashlib.sha256(json.dumps({k:data[k] for k in ('volume','mask','mesh','anchors')}, sort_keys=True).encode()).hexdigest()
    originals = {(b['id'],target,key):json.dumps(value,sort_keys=True) for b in data['branches']
                 for target, fits in b['fits'].items() for key,value in fits.items()
                 if not key.startswith('catmullrom_')}
    mask = np.frombuffer(base64.b64decode(data['mask']), dtype=np.uint8).reshape(64,64,64)
    inverse = np.array(data['inverse'])

    def inside(points):
        voxel = points@inverse.T
        valid = ((voxel >= -.500001)&(voxel <= 63.500001)).all(axis=1)
        indices = np.clip(np.floor(voxel+.5).astype(int),0,63)
        return valid&mask[tuple(indices.T)].astype(bool)

    def measures(guide,curve):
        g=common.sample_polyline_by_arclength(guide,.1)
        c=common.sample_polyline_by_arclength(curve,.1)
        f=common.point_to_polyline_distance(c,guide)
        r=common.point_to_polyline_distance(g,curve)
        length=common.polyline_length(curve);gl=common.polyline_length(guide)
        chord=np.linalg.norm(guide[-1]-guide[0])
        return dict(residual_mm=float((f.mean()+r.mean())/2),
                    p95_residual_mm=float(np.percentile(np.r_[f,r],95)),
                    length_mm=length,guide_length_mm=gl,tortuosity=length/chord,
                    guide_tortuosity=gl/chord,tortuosity_difference=abs(length-gl)/chord,
                    mask_inside=float(inside(c).mean()),guide_mask_inside=float(inside(g).mean()))

    added=[dict(key=f'catmullrom_{n}',family='catmullrom',label=f'Catmull–Rom: {n} interpolation points',complexity=n)
           for n in (6,8,10,12,16)]
    data['variants']=[v for v in data['variants'] if v['family']!='catmullrom']+added
    for branch in data['branches']:
        for target in ('extracted','smoothed'):
            guide=np.asarray(branch['raw' if target=='extracted' else 'smooth'])
            for variant in added:
                n=variant['complexity']; controls,curve=common.fit(guide,'catmullrom',n)
                assert np.allclose(common.catmull_rom(controls,common.catmull_rom_knots(controls)),controls,atol=1e-10)
                assert np.allclose(controls[[0,-1]],guide[[0,-1]],atol=1e-10)
                record=dict(branch=branch['id'],target=target,variant=variant['key'],family='catmullrom',
                            label=variant['label'],complexity=n,parameters=3*(n-2),**measures(guide,curve))
                branch['fits'][target][variant['key']]=dict(controls=controls.tolist(),curve=curve.tolist(),metrics=record)
    data['default_variant']='bspline_10'
    data['provenance'].setdefault('additional_families',{})['catmullrom']={
        'parameterization':'centripetal alpha=0.5; knots derived from interpolation-point distances',
        'construction':'Equal arc-length guide samples; interpolation, not least-squares fitting',
        'endpoints':'Fixed graph anchors; two reflected ghost points are derived, not parameters',
        'same_budget':'3(n-2) free scalar coordinates in 3D, like an n-control B-spline',
        'degeneracy':'Minimum knot increment 1e-9 for repeated neighboring points',
        'reference':'https://www.cemyuksel.com/research/catmullrom_param/',
    }
    data['provenance']['viewer_update']={
        'method':'Reused exact embedded original case, volume, mask, anchors, guides and non-Catmull fits',
        'preserved_case_sha256':before,
        'catmull_math_sha256':hashlib.sha256((REPO/'scripts/catmull_rom.py').read_bytes()).hexdigest(),
    }
    after=hashlib.sha256(json.dumps({k:data[k] for k in ('volume','mask','mesh','anchors')},sort_keys=True).encode()).hexdigest()
    assert before==after
    for (branch,target,key),value in originals.items():assert json.dumps(data['branches'][branch]['fits'][target][key],sort_keys=True)==value
    template=(SHARED/'viewer.html').read_text()
    viewer=(SHARED/'viewer.js').read_text()
    if name=='synthetic_mri_patch':
        template=template.replace('IXI','SyntheticMRI').replace('REAL IMAGE / REPRESENTATION STUDY','SYNTHETIC MRI / REPRESENTATION STUDY')
        template=template.replace('all 12 branches',f"all {len(data['branches'])} branches")
        template=template.replace('uncertain extracted guides','supplied generator guides, not perfect image references')
        template=template.replace('Extracted centerline (uncertain)','Generator centerline (guide)')
        template=template.replace('extracted guide','generator guide').replace('extraction noise','guide sampling or image–guide disagreement').replace('extracted graph','generator graph')
        viewer=viewer.replace('adaptive training graph','stored training graph').replace('Extracted guide','Generator guide').replace('Extracted paths (uncertain)','Generator paths (guides)').replace('ixi-patch','synthetic-mri-patch').replace('Image-derived vessel mask','Existing vessel mask')
    page=template.replace('__PLOTLY__',bundle).replace('__DATA__',json.dumps(data,separators=(',',':'))).replace('__VIEWER__',viewer)
    page_path.write_text(page)
    records=[b['fits'][target][v['key']]['metrics'] for b in data['branches']
             for target in ('extracted','smoothed') for v in data['variants']]
    with (results/'metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (results/'provenance.json').write_text(json.dumps(data['provenance'],indent=2)+'\n')
    # Compact, shareable comparison figure on the same branch selected originally.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    branch=data['branches'][data['selected']];guide=np.array(branch['raw']);dims=np.argsort(np.ptp(guide,axis=0))[-2:]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,n in zip(axes,(8,10)):
        ax.plot(guide[:,dims[0]],guide[:,dims[1]],color='#74818b',label='Uncertain guide',lw=2)
        for family,color in (('bspline','#008b80'),('catmullrom','#da7e1d')):
            model=branch['fits']['extracted'][f'{family}_{n}'];curve=np.array(model['curve']);controls=np.array(model['controls'])
            ax.plot(curve[:,dims[0]],curve[:,dims[1]],color=color,label='B-spline' if family=='bspline' else 'Catmull–Rom')
            ax.scatter(controls[1:-1,dims[0]],controls[1:-1,dims[1]],s=22,color=color,marker='s' if family=='bspline' else 'o')
            ax.plot(controls[:,dims[0]],controls[:,dims[1]],color=color,ls=':',alpha=.45)
        ax.set_aspect('equal');ax.set_title(f'{n} total points · {3*(n-2)} free 3D values')
        ax.set_xlabel('Physical mm');ax.set_ylabel('Physical mm');ax.legend(fontsize=8)
    fig.suptitle(f"{data['provenance']['sample']} · branch {data['selected']} · fitting demonstration")
    fig.tight_layout();fig.savefig(results/'catmull_rom_comparison.png',dpi=150);plt.close(fig)
    print(json.dumps({'case':name,'sample':data['provenance']['sample'],'variants':len(data['variants']),
                      'branches':len(data['branches']),'old_fits_and_image_preserved':True,'html':str(page_path)}),flush=True)


if __name__=='__main__':
    for name in ('ixi_patch','synthetic_mri_patch'):extend_case(name)
