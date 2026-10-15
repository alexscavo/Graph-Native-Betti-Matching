#!/usr/bin/env python3
"""Six held-out evaluations with a small GPU gate and automatic comparison."""
import json,subprocess
from pathlib import Path
path=Path('experiments/leaf_pretraining/evaluation/submission.json')
if path.exists():raise FileExistsError(path)
state={'jobs':{}}
path.write_text(json.dumps(state,indent=2)+'\n')
def submit(name,args):
 cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes',*args]
 job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
 state['jobs'][name]={'id':job,'command':cmd};path.write_text(json.dumps(state,indent=2)+'\n')
 print(name,job,flush=True);return job
script='cluster/jean_zay/source_transfer_test.slurm'
v=submit('verify',['--array=0,1%2','--time=00:30:00',script,'verify'])
t=submit('test',['--array=0-5%3',f'--dependency=afterok:{v}',script,'test'])
submit('comparison',[f'--dependency=afterok:{t}','cluster/jean_zay/collect_source_transfer_test.slurm'])
