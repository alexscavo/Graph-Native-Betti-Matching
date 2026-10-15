#!/usr/bin/env python3
"""Expose leaf datasets now; physically relocate after all leaf training jobs end."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');p.add_argument('--finish',action='store_true');a=p.parse_args()
    if not a.apply and not a.finish:p.error('Choose --apply (links) or --finish (move after jobs finish)')
    source=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/leaf_generalization_20261015')
    target=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/HALVS_DATASET/patches_3d_transfer_20261015');target.mkdir(exist_ok=True)
    names=('extracted','leaves-k8','plants-smoke','extraction-smoke','contraction-smoke')
    for name in names:
        old,new=source/name,target/name
        if a.finish:
            if old.is_symlink() and old.resolve()==new.resolve():continue
            if new.is_symlink() and new.resolve()==old.resolve():new.unlink()
            if new.exists():raise FileExistsError(new)
            old.rename(new);old.symlink_to(new)
        elif not new.exists() and not new.is_symlink():new.symlink_to(old)
    for name in ('source-audit.json','leaf_graph_preview.png'):
        (target/name).write_bytes((source/name).read_bytes())
    (target/'README.md').write_text('# HALVS transfer patches\n\n`extracted`: raw/seg/VTP pool. `leaves-k8`: selected25k train/480val graphs and spline caches. PNG patches become64³ slabs in PlantsDataset. While training runs, these dataset entries link to the experiment storage; a dependent CPU job moves the actual directories here after training and reverses the links to preserve old paths. Original HALVS images/annotations are untouched.\n')
    (target/'relocation.json').write_text(json.dumps(dict(original=str(source),destination=str(target),directories=names,physically_moved=a.finish,old_paths_preserved=True),indent=2)+'\n')
    print(target)
if __name__=='__main__':main()
