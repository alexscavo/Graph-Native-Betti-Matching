#!/usr/bin/env python3
"""Add delayed-spline arms without replacing any submitted IXI jobs."""
import json
import subprocess
from pathlib import Path

original=Path('experiments/ixi_transfer/submission-20261015.json')
previous=json.loads(original.read_text())['jobs']
path=Path('experiments/ixi_transfer/delayed-submission-20261015.json')
if path.exists():
    raise FileExistsError(path)
state=dict(existing_submission=str(original),arms={0:'ixi_delayed_spline',2:'plants_delayed_spline',4:'synthetic_delayed_spline'},jobs={},status='submitting')
path.write_text(json.dumps(state,indent=2)+'\n')

def submit(name,args):
    cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes','--array=0,2,4%3',*args]
    job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    state['jobs'][name]=dict(id=job,command=cmd)
    path.write_text(json.dumps(state,indent=2)+'\n')
    print(name,job,flush=True)
    return job

script='cluster/jean_zay/ixi_delayed_spline.slurm'
verify=submit('verify',['--time=01:00:00',f"--dependency=aftercorr:{previous['verify']['id']}",script,'verify'])
f50=submit('finetune50',[f"--dependency=aftercorr:{previous['pretrain50']['id']},aftercorr:{verify}",script,'finetune','50'])
f100=submit('finetune100',[f'--dependency=aftercorr:{f50}',script,'finetune','100'])
submit('test_best',[f'--dependency=aftercorr:{f100}',script,'test'])
state['status']='queued'
path.write_text(json.dumps(state,indent=2)+'\n')
