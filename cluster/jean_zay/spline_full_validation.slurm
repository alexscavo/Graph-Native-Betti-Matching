#!/usr/bin/env bash
#SBATCH --job-name=spline-full-val
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --array=0-2
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib" TMPDIR=/tmp IDR_DEBUG=WARN WANDB_MODE=offline
# Use node-local temporary storage: JOBSCRATCH was inaccessible in job 876621.
unset JOBSCRATCH
source cluster/jean_zay/env.sh
source cluster/jean_zay/stage_spline_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
config="${1:-${DIAGNOSTIC_CONFIG:-configs/spline_full_validation.yaml}}"
mapfile -t arms < <(python -c 'import sys,yaml; print("\n".join(yaml.safe_load(open(sys.argv[1]))["checkpoints"]))' "$config")
arm="${arms[$SLURM_ARRAY_TASK_ID]}"
python -u scripts/evaluate_spline_validation.py --config "$config" --arm "$arm" --smoke
python -u scripts/evaluate_spline_validation.py --config "$config" --arm "$arm"
