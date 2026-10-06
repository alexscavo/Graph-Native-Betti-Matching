#!/usr/bin/env bash
#SBATCH --job-name=spline-v0-test
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
export GNBM_OUTPUT_DIR="$GNBM_REPO_DIR/experiments/spline_learning/runs"
export SYNTHETIC_MRI_DATASET="$GNBM_REPO_DIR/experiments/spline_learning/synthetic_mri_k8_10k_val2k"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib"
# Compute-node online initialization has failed; retain test logs offline and
# upload from a login node after evaluation.
export TMPDIR=/tmp WANDB_MODE="${WANDB_MODE:-offline}"
source cluster/jean_zay/env.sh
source cluster/jean_zay/stage_spline_runtime.sh
set -u
run="$GNBM_OUTPUT_DIR/spline_v0_k8_10k_val2k_50ep"
python -u evaluate.py --config "$run/resolved-config.yaml" --checkpoint "$run/models/best_checkpoint.pt" \
  --output-dir "$run/test" --dataset synthetic_mri --split test --batch-size 4 --workers 4 --device cuda
python scripts/log_spline_test.py --run "$run"
