#!/usr/bin/env python3
"""Immutable balanced source manifest referencing existing plant/leaf targets."""
import hashlib,json
from pathlib import Path

sources={
 'plants':Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/plant_generalization_20261014/plants-k8'),
 'leaves':Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/HALVS_DATASET/patches_3d_transfer_20261015/leaves-k8'),
}
base=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/biological_mixture_20261015')

def main():
    assert all((root/'READY').is_file() for root in sources.values())
    if base.exists():raise FileExistsError(base)
    base.mkdir()
    manifests={}
    for name,limits in [('combined-k8',{'train':12500,'val':240}),('plants-smoke',{'train':2,'val':2})]:
        output=base/name;rows=[]
        for split,count in limits.items():
            for kind in ('raw','seg','vtp','spline'):(output/split/kind).mkdir(parents=True,exist_ok=True)
            for source,root in sources.items():
                files=sorted((root/split/'raw').glob('*_data.png'),key=lambda p:hashlib.sha256(('364505:'+source+':'+p.name).encode()).hexdigest())
                assert len(files)>=count,(source,split,len(files),count)
                for raw in files[:count]:
                    original=raw.name.removesuffix('_data.png');sample=source+'_'+original
                    for kind,suffix in [('raw','_data.png'),('seg','_seg.png'),('vtp','_graph.vtp'),('spline','.npz')]:
                        target=root/split/kind/(original+suffix)
                        assert target.is_file(),target
                        (output/split/kind/(sample+suffix)).symlink_to(target.resolve())
                    rows.append(dict(source=source,split=split,sample_id=sample,original_id=original,original_root=str(root)))
        manifest=dict(seed=364505,source_counts_per_split=limits,rows=rows)
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (output/'READY').write_text('complete\n')
        manifests[name]={'train':sum(r['split']=='train' for r in rows),'val':sum(r['split']=='val' for r in rows)}
    print(json.dumps(manifests),flush=True)

if __name__=='__main__':main()
