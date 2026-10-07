#!/usr/bin/env python3
"""Save common-population provenance and log completed held-out results offline."""
import argparse,json,subprocess
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from configs import load_config
from training.tracking import build_tracker
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--selection',choices=('best','final'),required=True);a=p.parse_args()
config=load_config(a.run/'resolved-config.yaml')
root=Path(config['data']['datasets']['synthetic_mri']['root']).parent
shared=json.loads((root/'comparison-manifest.json').read_text())
source=json.loads(Path(shared['source_manifest']).read_text())
output=a.run/'test'/a.selection
subprocess.run([sys.executable,'scripts/evaluate_common_vessel_predictions.py','--output',str(output),'--workers','4'],check=True)
summary=json.loads((output/'summary.json').read_text())
common=json.loads((output/'common-v2-summary.json').read_text())
assert summary['samples']==len(shared['test_ids'])==13071
geometry=('Predicted cubic B-splines, with predicted endpoints and internal controls.' if config['model'].get('spline',{}).get('enabled',False) else 'Piecewise straight segments with predicted degree-two nodes.' if Path(config['data']['datasets']['synthetic_mri']['root']).name=='adaptive' else 'Straight chords decoded from predicted endpoints.')
population={'evaluated_test_patches':summary['samples'],'available_source_test_patches':source['available_counts']['test'],'original_test_patches':source['source_counts']['test'],'excluded_by_reason':source['excluded_counts']['test'],'selection':a.selection,'prediction_geometry':geometry+' No GT geometry is used to construct predictions.','geometry_reference':'Shared uncertain generator guides; matched geometry errors must accompany branch coverage.'}
(a.run/'test'/a.selection/'population.json').write_text(json.dumps(population,indent=2)+'\n')
config['tracking']['mode']='offline'
tracker=build_tracker(config,a.run,resume=True,launch_metadata={'purpose':f'Held-out {a.selection} checkpoint test'})
if tracker:
    tracker.run.log({f'test/{a.selection}/'+k:v for k,v in summary.items() if isinstance(v,(int,float))})
    tracker.run.log({f'test/{a.selection}/'+k:v for k,v in common.items() if isinstance(v,(int,float))})
    tracker.run.summary[f'test/{a.selection}/population']=population
    tracker.finish()
