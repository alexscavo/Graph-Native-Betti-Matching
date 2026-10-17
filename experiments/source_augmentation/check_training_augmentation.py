"""CPU loader check on two actual plants and two leaves; no training or writes to datasets."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scipy.ndimage import map_coordinates
from data.loaders.plants import PlantsDataset
from data.loaders.common import SamplePaths, image_graph_collate
from data.spline_targets import resample


def preview(original, augmented, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(9,9))
    for col,(label,item) in enumerate([('Original embedding',original),('Augmented loader output',augmented)]):
        raw=item[0][0][0].numpy();nodes=item[2][0].numpy()*64;guides=item[4][0]['guides'].numpy()*64
        for row,axis in enumerate([2,0]):
            dims=[j for j in range(3) if j!=axis]
            ax=axes[row,col];ax.imshow(raw.max(axis=axis).T,origin='lower',cmap='gray',vmin=-.5,vmax=.5)
            for curve in guides:ax.plot(curve[:,dims[0]],curve[:,dims[1]],color='orange',lw=.7)
            ax.scatter(nodes[:,dims[0]],nodes[:,dims[1]],color='red',s=12)
            ax.set_title(f'{label}, MIP axis {axis}');ax.set_xlim(0,63);ax.set_ylim(0,63)
    fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    roots=dict(plants=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/plant_generalization_20261014/plants-k8'),
        leaves=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/HALVS_DATASET/patches_3d_transfer_20261015/leaves-k8'))
    results={};batch=[]
    for domain,root in roots.items():
        records=json.loads((root/'manifest.json').read_text())['accepted']['train']
        names=sorted(r['sample_id'] for r in records if r['nodes']>=6 and r['edges']>=5)[:2]
        for index,name in enumerate(names):
            record=SamplePaths(root/'train/raw'/f'{name}_data.png',root/'train/seg'/f'{name}_seg.png',root/'train/vtp'/f'{name}_graph.vtp',name)
            policy=dict(enabled=True,nonplanar=True,textured_soft=True)
            old=PlantsDataset([record],augment=False,spline_controls=8)[0]
            validation=PlantsDataset([record],augment=False,spline_controls=8,source_augmentation=policy)[0]
            for component in [0,1,2,3]:torch.testing.assert_close(old[component][0],validation[component][0],atol=0,rtol=0)
            for k in old[4][0]:torch.testing.assert_close(old[4][0][k],validation[4][0][k],atol=0,rtol=0)
            ds=PlantsDataset([record],augment=True,spline_controls=8,source_augmentation=policy,rng=random.Random(364507+index))
            start=time.perf_counter();new=ds[0];elapsed=time.perf_counter()-start
            replay=PlantsDataset([record],augment=True,spline_controls=8,source_augmentation=policy,rng=random.Random(364507+index))[0]
            torch.testing.assert_close(new[0][0],replay[0][0],atol=0,rtol=0)
            torch.testing.assert_close(new[3][0],old[3][0],atol=0,rtol=0)
            xyz=new[2][0];geo=new[4][0];edges=new[3][0]
            assert new[0][0].shape==(1,64,64,64) and torch.isfinite(new[0][0]).all()
            assert (xyz>=0).all() and (xyz<=63/64).all()
            torch.testing.assert_close(geo['controls'][:,[0,-1]],xyz[edges],atol=2e-6,rtol=2e-6)
            torch.testing.assert_close(geo['guides'][:,[0,-1]],xyz[edges],atol=2e-6,rtol=2e-6)
            def support(item):
                mask=item[1][0][0].numpy()>0
                points=item[4][0]['guides'].numpy().reshape(-1,3)*64
                return float(map_coordinates(mask.astype(float),points.T,order=0,mode='constant',cval=0).mean())
            fitted=np.stack([resample(c.numpy()*64,64) for c in geo['curves']])
            error=np.linalg.norm(fitted-geo['guides'].numpy()*64,axis=-1)
            key=domain+'_'+name
            results[key]=dict(loader_seconds=elapsed,nodes=len(xyz),edges=len(edges),
                original_guide_mask_support=support(old),augmented_guide_mask_support=support(new),
                refit_mean_error_vox=float(error.mean()),refit_max_error_vox=float(error.max()),
                same_seed_exact=True,validation_unchanged=True,endpoints_match=True)
            preview(old,new,a.output/(key+'.png'));batch.append(new)
            print(key,results[key],flush=True)
    collated=image_graph_collate(batch)
    assert len(collated)==6
    # Also exercise the graph-only pretraining route used by the original source comparison.
    graph_only=PlantsDataset([record],augment=True,source_augmentation=policy,rng=random.Random(2))[0]
    assert isinstance(graph_only[4][0],float) and torch.isfinite(graph_only[0][0]).all()
    (a.output/'report.json').write_text(json.dumps(results,indent=2)+'\n')
    print('Four real samples, collation and graph-only route passed.',flush=True)


if __name__=='__main__':main()
