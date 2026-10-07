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
#SBATCH --time=20:00:00
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
comparison=/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/scratch_comparison_10k_300ep
arm="${COMPARISON_ARM:?Set COMPARISON_ARM to old, spline or adaptive}"
[[ "$arm" == old || "$arm" == spline || "$arm" == adaptive ]]
export GNBM_OUTPUT_DIR=/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs
export SYNTHETIC_MRI_DATASET="$comparison/$arm"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib"
export TMPDIR=/tmp WANDB_MODE=offline
source cluster/jean_zay/env.sh
source cluster/jean_zay/stage_spline_runtime.sh
set -u
python -c 'import json,os; from pathlib import Path; p=Path(os.environ["SYNTHETIC_MRI_DATASET"]).parent; m=json.load(open(p/"comparison-manifest.json")); assert m["counts"]=={"train":10000,"val":1952,"test":13071}; print("Paired population:",m["counts"],flush=True)'
run_name=scratch_spline_k8_10k_300ep
[[ "$arm" == old ]] && run_name=scratch_old_graph_10k_300ep
[[ "$arm" == adaptive ]] && run_name=scratch_adaptive_degree2_10k_300ep
resume_checkpoint="${COMPARISON_RESUME_CHECKPOINT:-}"
if [[ -z "$resume_checkpoint" && -e "$GNBM_OUTPUT_DIR/$run_name/models/latest_checkpoint.pt" ]]; then
    echo "Refusing to overwrite an existing scratch run: $run_name" >&2
    exit 2
fi
# Initial submissions start from random weights. Recovery restores complete state.
config="configs/scratch_comparison_${arm}_300ep.yaml"
if [[ "$arm" == adaptive ]]; then
    config="$comparison/adaptive/training-config.yaml"
    python -c 'import json,os; from pathlib import Path; p=Path(os.environ["SYNTHETIC_MRI_DATASET"]); m=json.load(open(p/"manifest.json")); assert not m["failures"]; assert m["counts"]=={"train":10000,"val":1952,"test":13071}; print("Adaptive object queries:",m["object_queries"],flush=True)'
fi
if [[ -n "$resume_checkpoint" ]]; then
    [[ "$resume_checkpoint" == "$GNBM_OUTPUT_DIR/$run_name/models/latest_checkpoint.pt" && -f "$resume_checkpoint" ]]
    python -u train.py --config "$config" --resume "$resume_checkpoint"
else
    python -u train.py --config "$config"
fi
