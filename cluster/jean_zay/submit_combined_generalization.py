#!/usr/bin/env python3
"""Balanced biological-source mixture, matched fine-tunings and held-out tests."""
import json,subprocess
from pathlib import Path
record=Path('experiments/combined_pretraining/submission.json')
if record.exists():raise FileExistsError(record)
state=dict(config='configs/combined_generalization_da_b32.yaml',jobs={},source_budget={'plants':12500,'leaves':12500},status='submitting')
record.write_text(json.dumps(state,indent=2)+'\n')
def submit(name,args):
 cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes',*args]
 job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
 state['jobs'][name]={'id':job,'command':cmd};record.write_text(json.dumps(state,indent=2)+'\n');print(name,job,flush=True);return job
prep=submit('prepare',['cluster/jean_zay/prepare_combined_sources.slurm'])
export='--export=ALL,GENERALIZATION_CONFIG=configs/combined_generalization_da_b32.yaml'
script='cluster/jean_zay/plant_generalization.slurm'
v=submit('verify',[export,'--job-name=combined-verify','--time=01:00:00',f'--dependency=afterok:{prep}',script,'verify'])
p25=submit('pretrain25',[export,'--job-name=combined-pretrain','--array=0',f'--dependency=afterok:{v}',script,'pretrain','25'])
p50=submit('pretrain50',[export,'--job-name=combined-pretrain','--array=0',f'--dependency=aftercorr:{p25}',script,'pretrain','50'])
f50=submit('finetune50',[export,'--job-name=combined-finetune','--array=0-1',f'--dependency=afterok:{p50}',script,'finetune','50'])
f100=submit('finetune100',[export,'--job-name=combined-finetune','--array=0-1',f'--dependency=aftercorr:{f50}',script,'finetune','100'])
t=submit('test',['--array=0-1',f'--dependency=aftercorr:{f100}','cluster/jean_zay/combined_transfer_test.slurm','test'])
submit('comparison',[f'--dependency=afterok:{t}:1145698','cluster/jean_zay/collect_combined_transfer_test.slurm'])
state['status']='queued';record.write_text(json.dumps(state,indent=2)+'\n')
