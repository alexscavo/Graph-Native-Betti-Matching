#!/usr/bin/env python3
"""Six paired IXI strategies, with CPU/GPU gates and paired continuations."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--smoke-job',required=True);a=p.parse_args()
path=Path('experiments/ixi_transfer/submission-20261015.json')
if path.exists():raise FileExistsError(path)
state=dict(smoke_job=a.smoke_job,arms=['ixi_straight','ixi_spline','plants_straight','plants_spline','synthetic_straight','synthetic_spline'],jobs={},status='submitting')
path.write_text(json.dumps(state,indent=2)+'\n')
def submit(name,args):
 cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes',*args]
 job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
 state['jobs'][name]=dict(id=job,command=cmd);path.write_text(json.dumps(state,indent=2)+'\n');print(name,job,flush=True);return job
prep=[]
for kind in ('ixi','synthetic'):
 prep.append(submit('prepare_'+kind,[f'--dependency=afterok:{a.smoke_job}','cluster/jean_zay/prepare_ixi_transfer.slurm',kind]))
script='cluster/jean_zay/ixi_transfer.slurm';arrays=['--array=0-5%3']
verify=submit('verify',[*arrays,'--time=01:00:00','--dependency=afterok:'+':'.join(prep),script,'verify'])
p25=submit('pretrain25',[*arrays,f'--dependency=afterok:{verify}',script,'pretrain','25'])
p50=submit('pretrain50',[*arrays,f'--dependency=aftercorr:{p25}',script,'pretrain','50'])
f50=submit('finetune50',[*arrays,f'--dependency=aftercorr:{p50}',script,'finetune','50'])
f100=submit('finetune100',[*arrays,f'--dependency=aftercorr:{f50}',script,'finetune','100'])
submit('test_best',[*arrays,f'--dependency=aftercorr:{f100}',script,'test'])
state['status']='queued';path.write_text(json.dumps(state,indent=2)+'\n')
