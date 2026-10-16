#!/usr/bin/env python3
import json,subprocess
from pathlib import Path
p=Path('experiments/ixi_overfit/submission.json')
if p.exists():raise FileExistsError(p)
s={'jobs':{}}
p.write_text(json.dumps(s,indent=2)+'\n')
def submit(name,args):
 cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes','--array=0-1','--exclude=jzxh163',*args]
 job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
 s['jobs'][name]={'id':job,'command':cmd};p.write_text(json.dumps(s,indent=2)+'\n');print(name,job,flush=True);return job
v=submit('smoke',['--time=00:30:00','cluster/jean_zay/ixi_overfit.slurm','smoke'])
submit('overfit',[f'--dependency=aftercorr:{v}','cluster/jean_zay/ixi_overfit.slurm','train'])
