#!/usr/bin/env bash
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib" TMPDIR=/tmp WANDB_MODE=offline
export GNBM_OUTPUT_DIR=/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs
comparison=/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/scratch_comparison_10k_300ep
export SYNTHETIC_MRI_DATASET="$comparison/old"
source cluster/jean_zay/env.sh
source cluster/jean_zay/stage_spline_runtime.sh
set -u
run="$GNBM_OUTPUT_DIR/scratch_old_graph_10k_300ep"
python - "$run" "$comparison" <<'PY'
import json,sys,yaml
from pathlib import Path
run,root=map(Path,sys.argv[1:]); status=json.loads((run/'training-status.json').read_text())
assert status['epoch']==300 and status['reason']=='max_epochs',status
m=json.loads((root/'comparison-manifest.json').read_text()); assert len(m['test_ids'])==13071
config=yaml.safe_load((run/'resolved-config.yaml').read_text())
config['evaluation']['guide_reference_root']=str(root/'spline')
(run/'test').mkdir(exist_ok=True)
(run/'test/evaluation-config.yaml').write_text(yaml.safe_dump(config))
(run/'test/sample-ids.txt').write_text('\n'.join(m['test_ids'])+'\n')
PY
for selection in best final; do
    checkpoint=best_checkpoint.pt
    [[ "$selection" == final ]] && checkpoint=latest_checkpoint.pt
    python -u evaluate.py --config "$run/test/evaluation-config.yaml" --checkpoint "$run/models/$checkpoint" \
      --output-dir "$run/test/$selection" --dataset synthetic_mri --split test --sample-list "$run/test/sample-ids.txt" \
      --batch-size 4 --workers 4 --device cuda
    python scripts/log_scratch_comparison_test.py --run "$run" --selection "$selection"
done
