#!/usr/bin/env python3
"""Save a small image/contracted-graph overlay and source membership audit."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import pyvista as pv
p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
root=a.root;manifest=json.loads((root/'leaves-k8/manifest.json').read_text())
rows=list(csv.DictReader((root/'extracted/patches_manifest.csv').open()))
parents={s:{r['leaf'] for r in rows if r['split']==s} for s in ('train','val')}
assert not parents['train']&parents['val'],'Parent leaf leakage'
assert manifest['counts']=={'train':25000,'val':480}
canvas=Image.new('RGB',(3*384,2*420),'white');draw=ImageDraw.Draw(canvas)
# Fixed quantiles avoid selecting only visually appealing samples.
records=manifest['accepted']['train'];indices=np.linspace(0,len(records)-1,6,dtype=int)
for j,i in enumerate(indices):
    record=records[i];name=record['sample_id'];image=Image.open(root/f'leaves-k8/train/raw/{name}_data.png').convert('RGB').resize((384,384))
    d=ImageDraw.Draw(image);mesh=pv.read(root/f'leaves-k8/train/vtp/{name}_graph.vtp');xy=np.asarray(mesh.points)[:,[1,0]]*384
    for u,v in mesh.lines.reshape(-1,3)[:,1:]:d.line([tuple(xy[u]),tuple(xy[v])],fill='cyan',width=1)
    for x,y in xy:d.ellipse((x-3,y-3,x+3,y+3),fill='red')
    x=(j%3)*384;y=(j//3)*420;canvas.paste(image,(x,y+32));draw.text((x+4,y+5),f'{name}: {record["nodes"]} nodes / {record["edges"]} edges',fill='black')
canvas.save(root/'leaf_graph_preview.png')
report=dict(source_parent_counts={s:len(v) for s,v in parents.items()},extracted_patch_counts={s:sum(int(r['patches']) for r in rows if r['split']==s) for s in parents},accepted_counts=manifest['counts'],exclusions=len(manifest['excluded']),median_nodes=float(np.median([v['nodes'] for v in records])),median_edges=float(np.median([v['edges'] for v in records])),preview='cyan straight contracted edges; red anchors; not predicted curves',note='HALVS red+green masks, smoothed mask inputs, graph-only pretraining; native leaf split retained')
(root/'source-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
