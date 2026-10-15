#!/usr/bin/env python3
"""Restore Slurm-cancelled chains after the batch-one geometry probe fix."""
import json
import subprocess
from pathlib import Path
path=Path('experiments/ixi_transfer/probe-recovery-20261015.json')
if path.exists() and json.loads(path.read_text())['jobs']:raise FileExistsError(path)
old=['1126577','1126578','1126579','1126580','1126581','1128483','1128484','1128485']
active=subprocess.check_output(['squeue','-h','-j',','.join(old),'-o','%i'],text=True).strip()
if active:raise RuntimeError('Old dependent tasks still active: '+active)
required=['1126571','1126575','1126576_0','1126576_1','1126576_2','1126576_4','1128482_0','1128482_2','1128482_4']
accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(required),'--format=JobID,State'],text=True)
finished=dict(line.strip().split('|')[:2] for line in accounting.splitlines() if line.strip())
assert all(finished.get(job)=='COMPLETED' for job in required),finished
state=dict(completed_prerequisites=required,reason='Single-sample smoke geometry probe reached training-mode BatchNorm; production training had not started.',replaces=old,jobs={},status='submitting')
path.write_text(json.dumps(state,indent=2)+'\n')
def submit(name,array,dependency,script,*args):
 cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes',f'--array={array}']
 if dependency:cmd += [f'--dependency={dependency}']
 if name=='verify':cmd+=['--time=01:00:00']
 cmd += [script,*args]
 job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
 state['jobs'][name]=dict(id=job,command=cmd)
 path.write_text(json.dumps(state,indent=2)+'\n');print(name,job,flush=True)
 return job
script='cluster/jean_zay/ixi_transfer.slurm'
v=submit('verify','3,5%2',None,script,'verify')
p25=submit('pretrain25','0-5%3','afterok:'+v,script,'pretrain','25')
p50=submit('pretrain50','0-5%3','aftercorr:'+p25,script,'pretrain','50')
f50=submit('finetune50','0-5%3','aftercorr:'+p50,script,'finetune','50')
f100=submit('finetune100','0-5%3','aftercorr:'+f50,script,'finetune','100')
submit('test_best','0-5%3','aftercorr:'+f100,script,'test')
delayed='cluster/jean_zay/ixi_delayed_spline.slurm'
d50=submit('delayed_finetune50','0,2,4%3',f'aftercorr:{p50}',delayed,'finetune','50')
d100=submit('delayed_finetune100','0,2,4%3','aftercorr:'+d50,delayed,'finetune','100')
submit('delayed_test_best','0,2,4%3','aftercorr:'+d100,delayed,'test')
state['status']='queued';path.write_text(json.dumps(state,indent=2)+'\n')
