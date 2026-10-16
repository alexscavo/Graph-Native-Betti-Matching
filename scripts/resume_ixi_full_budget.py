#!/usr/bin/env python3
"""Preserve an early-stopped result, resume latest state to 100, then retest."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import yaml
p=argparse.ArgumentParser();p.add_argument('--arm',required=True);a=p.parse_args()
config=Path('configs/ixi_transfer.yaml');plan=yaml.safe_load(config.read_text())
assert plan['early_stopping'] is False
run=Path(plan['output_root'])/'runs'/a.arm
assert (run/'COMPLETE').is_file(), 'Only resume a completed, inactive run'
summary=json.loads((run/'summary.json').read_text())
assert summary['early_stopped'] and summary['epoch']<plan['finetune_epochs']
archive=run/f"early-stopped-epoch{summary['epoch']}"
archive.mkdir(exist_ok=False)
# Checkpoints are replaced atomically by the runner: preserve the old best
# without copying its payload or changing its contents.
os.link(run/'best.pt',archive/'best.pt')
for name in ['COMPLETE','summary.json','provenance.json','resolved-config.yaml','full-validation','best-full-validation','final-full-validation.json','best-full-validation.json','held-out-test']:
 path=run/name
 if path.exists():path.rename(archive/name)
(archive/'continuation.json').write_text(json.dumps(dict(resume_checkpoint=str(run/'latest.pt'),previous_epoch=summary['epoch'],target_epoch=plan['finetune_epochs'],reason='User requested disabling early stopping; all comparison fine-tunings complete 100 epochs',optimizer_and_scheduler='Restored unchanged from latest checkpoint'),indent=2)+'\n')
subprocess.run([sys.executable,'scripts/train_ixi_transfer.py','--config',str(config),'--arm',a.arm,'--until-epoch',str(plan['finetune_epochs'])],check=True)
final=json.loads((run/'summary.json').read_text())
assert final['epoch']==plan['finetune_epochs'] and not final['early_stopped']
subprocess.run([sys.executable,'scripts/evaluate_ixi_transfer.py','--config',str(config),'--arm',a.arm],check=True)
