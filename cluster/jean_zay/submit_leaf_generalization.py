#!/usr/bin/env python3
"""Submit leaf replacement only; reuse the completed MRI-only control."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--smoke-job',required=True);a=p.parse_args()
record=Path('experiments/leaf_pretraining/submission-20261015.json')
if record.exists():raise FileExistsError(record)
state=dict(config='configs/leaf_generalization_da_b32.yaml',cpu_smoke=a.smoke_job,jobs={},mri_control='/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/plant_generalization_20261014_da_b32_w1_restart/runs',status='submitting')
record.write_text(json.dumps(state,indent=2)+'\n')
def submit(name,args):
    command=['sbatch','--parsable','--kill-on-invalid-dep=yes',*args]
    job=subprocess.check_output(command,text=True).strip().split(';')[0]
    state['jobs'][name]={'id':job,'command':command};record.write_text(json.dumps(state,indent=2)+'\n');print(name,job,flush=True);return job
prepare=submit('prepare',[f'--dependency=afterok:{a.smoke_job}','cluster/jean_zay/prepare_leaf_targets.slurm'])
export='--export=ALL,GENERALIZATION_CONFIG=configs/leaf_generalization_da_b32.yaml'
script='cluster/jean_zay/plant_generalization.slurm'
verify=submit('verify',[export,'--job-name=leaf-verify',f'--dependency=afterok:{prepare}',script,'verify'])
p25=submit('pretrain25',[export,'--job-name=leaf-pretrain','--array=0',f'--dependency=afterok:{verify}',script,'pretrain','25'])
p50=submit('pretrain50',[export,'--job-name=leaf-pretrain','--array=0',f'--dependency=aftercorr:{p25}',script,'pretrain','50'])
f50=submit('finetune50',[export,'--job-name=leaf-finetune','--array=0-1',f'--dependency=afterok:{p50}',script,'finetune','50'])
f100=submit('finetune100',[export,'--job-name=leaf-finetune','--array=0-1',f'--dependency=aftercorr:{f50}',script,'finetune','100'])
state['status']='queued';record.write_text(json.dumps(state,indent=2)+'\n')
