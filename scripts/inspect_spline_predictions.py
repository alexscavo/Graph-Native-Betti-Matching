#!/usr/bin/env python3
"""Inspect actual decoded curves; topology counts use spatial node assignment."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def inspect(path):
    rows=json.loads((path/'predictions.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,11));checks=[]
    for row,ax in zip(rows,axes.flat):
        metric=np.array(row['metric'])
        gt=np.array(row['target_nodes']).reshape(-1,3)@metric.T
        pred=np.array(row['decoded']['nodes']).reshape(-1,3)@metric.T
        distance=np.linalg.norm(gt[:,None]-pred[None],axis=-1)
        target,predicted=linear_sum_assignment(distance)
        mapping=dict(zip(predicted,target))
        true={tuple(sorted(x)) for x in row['target_edges']}
        edges=row['decoded']['edges']
        mapped={tuple(sorted((mapping[x],mapping[y]))) for x,y in edges if x in mapping and y in mapping}
        unmatched=sum(x not in mapping or y not in mapping for x,y in edges)
        errors=np.linalg.norm(gt[target]-pred[predicted],axis=1)
        learned=np.array(row['matched_curves'])@metric.T
        fitted=np.array(row['target_curves'])@metric.T
        t=np.linspace(0,1,learned.shape[1])[None,:,None]
        chord=(1-t)*learned[:,:1]+t*learned[:,-1:]
        learned_errors=np.linalg.norm(learned-fitted,axis=-1).mean(1)
        chord_errors=np.linalg.norm(chord-fitted,axis=-1).mean(1)
        checks.append(dict(sample=row['sample'],target_nodes=len(gt),predicted_nodes=len(pred),
                           target_edges=len(true),predicted_edges=len(edges),missing_edges=len(true-mapped),
                           extra_edges=len(mapped-true)+unmatched,
                           mean_matched_node_error_mm=float(errors.mean()) if len(errors) else None,
                           max_matched_node_error_mm=float(errors.max()) if len(errors) else None,
                           learned_curve_point_error_mm=float(learned_errors.mean()),
                           same_endpoint_chord_point_error_mm=float(chord_errors.mean()),
                           branches_better_than_chord=int((learned_errors<chord_errors).sum()),
                           branch_count=len(learned_errors),
                           convention='Minimum-distance node assignment; topology counts are diagnostic, not thresholded benchmark F1.'))
        for curves,label,color,width in [(row['target_curves'],'Fitted target','#727a84',2),
                                          (row['decoded']['spline_curves'],'Decoded spline','#009f85',1.3)]:
            for index,curve in enumerate(curves):
                c=np.array(curve)@metric.T
                ax.plot(c[:,0],c[:,2],c=color,lw=width,label=label if index==0 else None)
        ax.scatter(gt[:,0],gt[:,2],s=20,c='#bf6dbe');ax.set_aspect('equal')
        ax.set_title(row['sample']);ax.legend(fontsize=8)
        ax.set_xlabel('Physical X (mm)');ax.set_ylabel('Physical Z (mm)')
    fig.suptitle(f'{path.name}: decoded splines versus fitted targets · four training patches only')
    fig.tight_layout();fig.savefig(path/'decoded_overview.png',dpi=130);plt.close(fig)
    (path/'decoded_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(dict(run=path.name,checks=checks),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    args=p.parse_args()
    for path in sorted(args.run_root.glob('v*/predictions.json')):inspect(path.parent)


if __name__=='__main__':main()
