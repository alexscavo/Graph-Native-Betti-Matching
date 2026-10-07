#!/usr/bin/env python3
"""Score saved predictions with the same topology/geometry protocol for all arms."""
import argparse,csv,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.build_spline_test_viewers import prediction_records
from metrics.common_vessel_metrics import evaluate_common_vessels,aggregate_common_vessels
ROOT=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/scratch_comparison_10k_300ep')

def score(record):
    sample=record['source_sample_id']; prediction={'nodes':record['nodes_dhw'],'edges':record['edges']}
    if 'spline_controls' in record:prediction['spline_controls']=record['spline_controls']
    return {'sample_id':sample,**evaluate_common_vessels(prediction,ROOT/'spline/test/spline'/f'{sample}.npz')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    expected=json.loads((ROOT/'comparison-manifest.json').read_text())['test_ids']
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows=list(pool.map(score,prediction_records(a.output/'predictions.json'),chunksize=32))
    assert sorted(r['sample_id'] for r in rows)==expected
    summary=aggregate_common_vessels(rows)
    (a.output/'common-v2-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (a.output/'common-v2-per-patch.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Common representation-independent evaluation complete:',len(rows),'patches',flush=True)
if __name__=='__main__':main()
