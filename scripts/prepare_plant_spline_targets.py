#!/usr/bin/env python3
"""Contract plant annotations, preserving boundary anchors and original branch polylines.

The cache is in the original PNG/VTP normalized (y,x,z=0) coordinates, with
pixel units, NOT physical millimetres or projected64^3 coordinates. Graph-only
pretraining reads its VTP; continuous plant training requires projection of
this cache and is not enabled by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pyvista as pv
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data.spline_targets import contract_spline_graph, fit_spline, resample


def prepare(source, output, train_limit, val_limit):
    output.mkdir(parents=True,exist_ok=False)
    accepted={};excluded=[];removed=0
    for split, limit in [('train',train_limit),('val',val_limit)]:
        leaf=output/split
        for folder in ['raw','seg','vtp','spline']:(leaf/folder).mkdir(parents=True)
        candidates=sorted((source/split/'vtp').glob('*_graph.vtp'),key=lambda p:hashlib.sha256(('364505/'+split+'/'+p.name).encode()).digest())
        accepted[split]=[]
        for file in candidates:
            name=file.name[:-10];mesh=pv.read(file)
            nodes=np.asarray(mesh.points,dtype=float)
            edges=mesh.lines.reshape(-1,3)
            assert np.all(edges[:,0]==2)
            edges=edges[:,1:]
            image=source/split/'raw'/f'{name}_data.png'
            seg=source/split/'seg'/f'{name}_seg.png'
            with Image.open(image) as im:width,height=im.size
            # Original patch generator uses5-pixel padding, clipping at5 and122 in128px.
            if (height,width)!=(128,128):raise ValueError(f'Unexpected source shape {name}')
            boundary=np.any(np.isclose(nodes[:,:2],5/128,atol=1e-6)|np.isclose(nodes[:,:2],122/128,atol=1e-6),axis=1)
            graph,reason=contract_spline_graph(nodes,edges,preserve_nodes=np.flatnonzero(boundary))
            if reason:
                excluded.append(dict(split=split,sample_id=name,reason=reason));continue
            ids=sorted(graph.anchors);remap={old:new for new,old in enumerate(ids)}
            xyz=np.asarray([graph.anchors[i] for i in ids]).reshape(-1,3)
            links=[];controls=[];curves=[];guides=[];paths=[];offsets=[0]
            metric=np.diag([height,width,1.]);inverse=np.linalg.inv(metric)
            for branch in graph.branches:
                path=branch.polyline;start,end=branch.start,branch.end
                if tuple(path[0])>tuple(path[-1]):path=path[::-1];start,end=end,start
                if np.linalg.norm(np.diff(path,axis=0),axis=1).sum()<1e-10:
                    reason='degenerate_branch';break
                c,curve=fit_spline(path@metric.T,8,64)
                links.append((remap[start],remap[end]));controls.append(c@inverse.T);curves.append(curve@inverse.T)
                guides.append(resample(path@metric.T,64)@inverse.T);paths.append(path);offsets.append(offsets[-1]+len(path))
            if reason:
                excluded.append(dict(split=split,sample_id=name,reason=reason));continue
            links=np.asarray(links,dtype=np.int64).reshape(-1,2)
            target=pv.PolyData(xyz)
            if len(links):target.lines=np.column_stack((np.full(len(links),2),links)).ravel()
            target.save(leaf/'vtp'/file.name)
            for folder,src in [('raw',image),('seg',seg)]:
                if not src.is_file() or src.stat().st_size==0:raise ValueError(f'Incomplete upload: {src}')
                (leaf/folder/src.name).symlink_to(src.resolve())
            np.savez_compressed(leaf/'spline'/f'{name}.npz',nodes=xyz.astype('float32'),edges=links,
                controls=np.asarray(controls,dtype='float32').reshape(-1,8,3),curves=np.asarray(curves,dtype='float32').reshape(-1,64,3),
                guides=np.asarray(guides,dtype='float32').reshape(-1,64,3),metric=metric.astype('float32'),
                full_paths=(np.concatenate(paths) if paths else np.empty((0,3))).astype('float32'),path_offsets=np.asarray(offsets),
                boundary=boundary[ids],coordinate_system='source_png_normalized_yxz',units='pixels',family='bspline')
            removed+=len(nodes)-len(xyz)
            accepted[split].append(dict(sample_id=name,source_graph_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                original_nodes=len(nodes),nodes=len(xyz),edges=len(links),boundary_nodes=int(boundary[ids].sum())))
            if len(accepted[split])%1000==0:print(split,len(accepted[split]),flush=True)
            if len(accepted[split])==limit:break
    report=dict(source=str(source),accepted=accepted,excluded=excluded,removed_degree2_nodes=removed,
        requested=dict(train=train_limit,val=val_limit),counts={s:len(v) for s,v in accepted.items()},
        guide='original annotation polylines; previously pruned curvature cannot be recovered',
        protected_boundary='original5/128 and122/128 crop faces; original endpoints also retained',
        plant_cache_not_in_vessel_mm_coordinates=True,test_split_untouched=True)
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    if any(report['counts'][s]!=n for s,n in report['requested'].items()):
        raise ValueError(f'Insufficient accepted graphs: {report["counts"]}; inspect exclusions, no silent smaller population')
    (output/'READY').write_text('complete\n')
    print(report['counts'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/plants_3d2cut/patches_3d'))
    p.add_argument('--output',required=True,type=Path);p.add_argument('--train-limit',type=int,default=25000);p.add_argument('--val-limit',type=int,default=480)
    a=p.parse_args();prepare(a.source,a.output,a.train_limit,a.val_limit)
