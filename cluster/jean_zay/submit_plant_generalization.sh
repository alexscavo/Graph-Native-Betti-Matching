#!/usr/bin/env bash
# Run from repository root after the separately submitted target preparation and GPU verification.
set -euo pipefail
prepare_job="${1:?plant target job}"
verify_job="${2:?GPU verification job}"
submission="${3:-experiments/plant_pretraining/submissions/20261014.json}"
if [[ -e "$submission" ]]; then echo "Submission record already exists: $submission" >&2; exit 1; fi
runner=cluster/jean_zay/plant_generalization.slurm
dependency="afterok:$verify_job"
if [[ "$prepare_job" != ready ]]; then dependency="$dependency:$prepare_job"; fi
p25=$(sbatch --parsable --array=0-1 --dependency="$dependency" --kill-on-invalid-dep=yes "$runner" pretrain 25)
p50=$(sbatch --parsable --array=0-1 --dependency="aftercorr:$p25" --kill-on-invalid-dep=yes "$runner" pretrain 50)
f50=$(sbatch --parsable --array=0-3 --dependency="afterok:$p50" --kill-on-invalid-dep=yes "$runner" finetune 50)
f100=$(sbatch --parsable --array=0-3 --dependency="aftercorr:$f50" --kill-on-invalid-dep=yes "$runner" finetune 100)
python - "$prepare_job" "$verify_job" "$p25" "$p50" "$f50" "$f100" "$submission" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[7])
out.parent.mkdir(parents=True,exist_ok=True)
if out.exists():raise FileExistsError(out)
names=['prepare','verify','pretrain_25','pretrain_50','finetune_50','finetune_100']
record=dict(zip(names,sys.argv[1:7]));out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
PY
