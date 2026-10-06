#!/usr/bin/env python3
"""Inspect cached training guides and fitted spline targets before launching."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from vessel_mesh import mask_mesh


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--targets',type=Path,default=Path('experiments/spline_learning/targets_k8'))
    args=p.parse_args();manifest=json.loads((args.targets/'manifest.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,12));checks=[];traces=[];groups=[]
    for index,(row,ax) in enumerate(zip(manifest['selected'],axes.flat)):
        sample=row['sample'];leaf=args.targets/'train'
        with np.load(leaf/'spline'/f'{sample}.npz') as f:cache={k:f[k] for k in f.files}
        image=nib.load(leaf/'raw'/f'{sample}_data.nii.gz').get_fdata()
        mask=nib.load(leaf/'seg'/f'{sample}_seg.nii.gz').get_fdata()>.5
        guides=cache['full_paths'];q=guides*64
        voxels=np.floor(q+.5).astype(int)
        assert ((voxels>=0)&(voxels<64)).all()
        support=float(mask[tuple(voxels.T)].mean())
        assert np.allclose(cache['controls'][:,0],cache['nodes'][cache['edges'][:,0]])
        assert np.allclose(cache['controls'][:,-1],cache['nodes'][cache['edges'][:,1]])
        ax.imshow(image.max(0).T,origin='lower',cmap='gray',vmin=np.percentile(image,1),vmax=np.percentile(image,99.5))
        for start,end in zip(cache['path_offsets'][:-1],cache['path_offsets'][1:]):
            a=q[start:end];ax.plot(a[:,1],a[:,2],color='#e4bd54',lw=1.5)
        for c in cache['curves']:
            c=c*64;ax.plot(c[:,1],c[:,2],color='#03a397',lw=.9)
        n=cache['nodes']*64;ax.scatter(n[:,1],n[:,2],s=35,c='#ce78c1',edgecolors='black',linewidths=.3)
        ax.set_xlim(-.5,63.5);ax.set_ylim(-.5,63.5)
        ax.set_title(f"{sample}\n{len(n)} anchors / {len(cache['edges'])} branches; guide samples in mask {support:.1%}")
        checks.append(dict(sample=sample,path_samples=len(guides),guide_vertex_mask_support=support,anchors=len(n),branches=len(cache['edges'])))
        physical=lambda points:points@cache['metric'].T
        begin=len(traces)
        surface=mask_mesh(leaf/'seg'/f'{sample}_seg.nii.gz')
        vertices=physical(np.array(surface['vertices']));faces=np.array(surface['faces'])
        traces.append(go.Mesh3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],
                               i=faces[:,0],j=faces[:,1],k=faces[:,2],opacity=.12,
                               color='#a4b9ba',name='Existing vessel-mask mesh',showlegend=True,visible=index==0))
        paths=[physical(guides[a:b]) for a,b in zip(cache['path_offsets'][:-1],cache['path_offsets'][1:])]
        for paths,name,color,width,dash in [(paths,'Full generator paths','#d3b46d',4,'dot'),
                                           ([physical(c) for c in cache['curves']],'8-control fitted targets','#56d7c3',5,'solid')]:
            joined=np.vstack([np.vstack([a,np.full((1,3),np.nan)]) for a in paths])
            traces.append(go.Scatter3d(x=joined[:,0],y=joined[:,1],z=joined[:,2],mode='lines',name=name,line=dict(color=color,width=width,dash=dash),visible=index==0))
        for boundary,color,name in [(False,'#f0b459','Original graph anchors'),(True,'#ce8ff4','Crop boundaries')]:
            ids=np.flatnonzero(cache['boundary']==boundary);n=physical(cache['nodes'][ids])
            traces.append(go.Scatter3d(x=n[:,0],y=n[:,1],z=n[:,2],mode='markers+text',text=ids.astype(str),textposition='top center',marker=dict(size=7,color=color),name=name,visible=index==0))
        groups.append((sample,list(range(begin,len(traces)))))
    fig.suptitle('Supplied generator guides (gold) and 8-control targets (cyan) on MRI MIPs\nTarget checks only: no learned predictions',fontsize=15)
    fig.tight_layout();fig.savefig(args.targets/'target_overview.png',dpi=130);plt.close(fig)
    (args.targets/'target_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    figure=go.Figure(traces)
    figure.update_layout(title='Training targets only · original generator guides are not perfect anatomical references',
                         scene=dict(aspectmode='data',xaxis_title='X (mm)',yaxis_title='Y (mm)',zaxis_title='Z (mm)'),
                         updatemenus=[dict(buttons=[dict(label=name,method='update',args=[dict(visible=[i in indices for i in range(len(traces))])]) for name,indices in groups])],
                         sliders=[dict(active=2,currentvalue=dict(prefix='Mesh opacity: '),
                                       steps=[dict(label=f'{opacity:.0%}',method='restyle',
                                                   args=[{'opacity':opacity},[indices[0] for _,indices in groups]])
                                              for opacity in (0,.06,.12,.2,.35,.5)])])
    figure.write_html(args.targets/'targets.html',include_plotlyjs=True,full_html=True)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
