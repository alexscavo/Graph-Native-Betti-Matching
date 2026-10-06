#!/usr/bin/env bash
#SBATCH --job-name=spline-k8-eval-four
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export GNBM_OUTPUT_DIR="$GNBM_REPO_DIR/experiments/spline_learning/runs"
export SYNTHETIC_MRI_DATASET="$GNBM_REPO_DIR/experiments/spline_learning/targets_k8"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv"
export TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache"
export MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib"
export TMPDIR=/tmp WANDB_MODE=disabled
source "$GNBM_REPO_DIR/cluster/jean_zay/env.sh"
set -u
for variant in v0_pretrained v0_scratch v1_pretrained; do
  run="$GNBM_OUTPUT_DIR/job-612351/$variant"
  python -u evaluate.py --config "$run/resolved-config.yaml" \
    --checkpoint "$run/model.pt" --output-dir "$run/evaluation_four_patches" \
    --dataset synthetic_mri --split val --max-samples 4 \
    --batch-size 4 --workers 0 --device cuda --visualizations 0
done
