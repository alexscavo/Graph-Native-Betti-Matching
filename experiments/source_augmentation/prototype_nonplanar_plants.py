#!/usr/bin/env python3
"""Two-sample research prototype: shared depth warp of raw volume, mask and graph."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import cdist
from skimage.measure import marching_cubes
from data.loaders.plants import PlantsDataset
from data.loaders.common import SamplePaths
from data.augmentations import AugmentationParameters,apply_augmentation


def mesh(image):
    v,f,_,_=marching_cubes(image,level=0.)
    return v,f


def field_for_nodes(nodes,rng):
    xy=nodes[:,:2]*64
    requested=rng.normal(0,5,len(nodes));requested-=requested.mean()
    kernel=np.exp(-cdist(xy,xy,'sqeuclidean')/(2*10.**2))
    weights=np.linalg.solve(kernel+.05*np.eye(len(xy)),requested)
    grid=np.stack(np.meshgrid(np.arange(64),np.arange(64),indexing='ij'),-1)
    height=(np.exp(-cdist(grid.reshape(-1,2),xy,'sqeuclidean')/(2*10.**2))@weights).reshape(64,64)
    slope=np.sqrt(sum(g*g for g in np.gradient(height))).max()
    scale=min(1.,10./max(abs(height).max(),1e-8),1./max(slope,1e-8))
    return height*scale,dict(gaussian_sigma_vox=5,kernel_length_vox=10,regularization=.05,global_scale=float(scale),requested_node_dz_vox=requested.tolist(),height_range_vox=[float(height.min()*scale),float(height.max()*scale)],max_slope=float(slope*scale))


def warp(image,mask,nodes,guides,height):
    grid=np.indices(image.shape,dtype=float);grid[2]-=height[:,:,None]
    raw=map_coordinates(image,grid,order=1,mode='constant',cval=-.5)
    seg=map_coordinates(mask,grid,order=0,mode='constant',cval=-.5)
    def points(p):
        q=p.copy();q[...,2]+=map_coordinates(height,(p[...,:2]*64).reshape(-1,2).T,order=1,mode='nearest').reshape(p.shape[:-1])/64
        return q
    return raw,seg,points(nodes),points(guides)


def rotate(stage,turns,flips):
    raw,mask,nodes,guides=stage;n=len(nodes)
    points=np.concatenate([nodes,guides.reshape(-1,3)])
    transformed=apply_augmentation(torch.from_numpy(raw[None].astype('float32')),torch.from_numpy(mask[None].astype('float32')),torch.from_numpy(points.astype('float32')),AugmentationParameters(quarter_turns=turns,flip_axes=flips))
    p=transformed.nodes.numpy()
    return transformed.image[0].numpy(),transformed.segmentation[0].numpy(),p[:n],p[n:].reshape(guides.shape)


def support(mask,points):
    return float((map_coordinates((mask>0).astype(float),(points.reshape(-1,3)*64).T,order=0,mode='constant',cval=0)>.5).mean())


def preview(stages,name,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(14,9))
    for i,(label,(raw,mask,nodes,guides)) in enumerate(stages.items()):
        ax=fig.add_subplot(2,3,i+1,projection='3d');v,f=mesh(raw)
        ax.add_collection3d(Poly3DCollection(v[f],alpha=.12,facecolor='#55aacc',edgecolor='none'))
        for curve in guides*64:ax.plot(*curve.T,color='#e89900',lw=1.5)
        ax.scatter(*(nodes*64).T,s=22,c='#ce2029',depthshade=False)
        ax.set(xlim=(0,64),ylim=(0,64),zlim=(0,64),xlabel='axis 0',ylabel='axis 1',zlabel='depth',title=label)
        ax.set_box_aspect((1,1,1));ax.view_init(22,-60)
        ax=fig.add_subplot(2,3,i+4)
        ax.imshow(raw.max(axis=0),origin='lower',cmap='gray',vmin=-.5,vmax=.5)
        for curve in guides*64:ax.plot(curve[:,2],curve[:,1],color='#e89900',lw=.9)
        ax.scatter(nodes[:,2]*64,nodes[:,1]*64,s=16,c='#ce2029')
        ax.set(title='Raw MIP along axis 0 + graph',xlabel='depth',ylabel='axis 1',xlim=(0,64),ylim=(0,64))
    fig.suptitle(name+' | blue: raw-image isosurface; orange: warped reference paths; red: nodes')
    fig.tight_layout();fig.savefig(path,dpi=135);plt.close(fig)


def interactive(cases,path):
    import importlib.util
    plotly_root=Path(importlib.util.find_spec('plotly').origin).parent
    bundle=(plotly_root/'package_data/plotly.min.js').read_text()
    payload={}
    for name,stages in cases.items():
        traces=[]
        for column,(label,(raw,mask,nodes,guides)) in enumerate(stages.items(),1):
            scene='scene'+(str(column) if column>1 else '')
            v,f=mesh(raw)
            traces.append(dict(type='mesh3d',x=v[:,0].tolist(),y=v[:,1].tolist(),z=v[:,2].tolist(),i=f[:,0].tolist(),j=f[:,1].tolist(),k=f[:,2].tolist(),opacity=.18,color='lightblue',name='Raw isosurface',legendgroup='raw',showlegend=column==1,scene=scene))
            points=[]
            for curve in guides:points.extend((curve*64).tolist()+[[None,None,None]])
            traces.append(dict(type='scatter3d',x=[p[0] for p in points],y=[p[1] for p in points],z=[p[2] for p in points],mode='lines',line=dict(color='orange',width=5),name='Reference paths',legendgroup='guides',showlegend=column==1,scene=scene))
            n=nodes*64
            traces.append(dict(type='scatter3d',x=n[:,0].tolist(),y=n[:,1].tolist(),z=n[:,2].tolist(),mode='markers',marker=dict(color='red',size=5),text=[f'node {i}' for i in range(len(n))],name='Nodes',legendgroup='nodes',showlegend=column==1,scene=scene))
        payload[name]=traces
    layout=dict(height=700,margin=dict(t=80,l=0,r=0,b=20),legend=dict(orientation='h'),annotations=[])
    for i,label in enumerate(['Original slab','Shared depth warp','Warp + rotations/flips']):
        scene='scene'+(str(i+1) if i else '')
        layout[scene]=dict(domain=dict(x=[i/3+.01,(i+1)/3-.01],y=[0,1]),xaxis=dict(range=[0,64],title='axis 0'),yaxis=dict(range=[0,64],title='axis 1'),zaxis=dict(range=[0,64],title='depth'),aspectmode='cube',camera=dict(eye=dict(x=1.6,y=1.6,z=.8)))
        layout['annotations'].append(dict(text=label,x=(i+.5)/3,y=1.05,xref='paper',yref='paper',showarrow=False))
    html='<!doctype html><meta charset="utf-8"><title>Nonplanar plant prototype</title><style>body{font:16px system-ui;margin:20px}</style><h2>Matched raw-image and graph deformation</h2><p>Select a sample; drag panels to rotate, scroll to zoom, click legend to hide surfaces. A warped sheet is nonplanar but does not create arbitrary 3D branching. No training data were changed.</p><select id="sample"></select><div id="plot"></div><script>'+bundle+'</script><script>'
    html+='const cases='+json.dumps(payload)+';const layout='+json.dumps(layout)+';const pick=document.getElementById("sample");Object.keys(cases).forEach(k=>pick.add(new Option(k,k)));function render(){Plotly.react("plot",cases[pick.value],layout,{responsive:true});}pick.onchange=render;render();</script>'
    path.write_text(html)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/source_nonplanar_20261016'));p.add_argument('--overwrite',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=a.overwrite)
    root=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/plant_generalization_20261014/plants-k8')
    accepted=json.loads((root/'manifest.json').read_text())['accepted']['train']
    selected=sorted([r['sample_id'] for r in accepted if r['nodes']>=6 and r['edges']>=5])[:2]
    rng=np.random.default_rng(364505);cases={};report={}
    for name in selected:
        record=SamplePaths(root/'train/raw'/(name+'_data.png'),root/'train/seg'/(name+'_seg.png'),root/'train/vtp'/(name+'_graph.vtp'),name)
        ds=PlantsDataset([record],augment=False,spline_controls=8)
        im,seg,nodes,edges,geo,_=ds[0]
        original=(im[0].squeeze().numpy(),seg[0].squeeze().numpy(),nodes[0].numpy(),geo[0]['guides'].numpy())
        height,settings=field_for_nodes(original[2],rng)
        bent=warp(*original,height);turns=tuple(int(x) for x in rng.integers(0,4,3));flips=tuple(bool(x) for x in rng.integers(0,2,3))
        rotated=rotate(bent,turns,flips)
        stages={'Original':original,'Nonplanar':bent,'Nonplanar + rotation':rotated};cases[name]=stages
        original_support=support(original[1],original[3]);warped_support=support(bent[1],bent[3]);rotation_support=support(rotated[1],rotated[3])
        assert abs(warped_support-rotation_support)<.02
        assert all(np.isfinite(x).all() for stage in stages.values() for x in stage)
        assert all(((stage[2]>=0)&(stage[2]<=63/64)).all() for stage in stages.values())
        residual=np.linalg.svd((bent[2]-bent[2].mean(0))*64,compute_uv=False)[-1]/np.sqrt(len(bent[2]))
        assert residual>.05, 'Selected deformation still effectively planar'
        settings.update(nodes=len(nodes[0]),edges=len(edges[0]),actual_node_dz_vox=((bent[2][:,2]-original[2][:,2])*64).tolist(),quarter_turns=turns,flips=flips,guide_support_original=original_support,guide_support_warped=warped_support,guide_support_rotated=rotation_support,best_fit_plane_rms_vox=float(residual))
        report[name]=settings
        np.savez_compressed(a.output/(name+'.npz'),height=height,edges=edges[0].numpy(),**{f'{i}_{key}':value for i,stage in enumerate(stages.values()) for key,value in zip(['raw','seg','nodes','guides'],stage)})
        preview(stages,name,a.output/(name+'.png'))
    interactive(cases,a.output/'nonplanar-plants.html')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
