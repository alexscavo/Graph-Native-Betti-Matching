#!/usr/bin/env bash
#SBATCH --job-name=scratch-extend300
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=0-1
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib" TMPDIR=/tmp IDR_DEBUG=WARN WANDB_MODE=offline
unset JOBSCRATCH
source cluster/jean_zay/env.sh
source cluster/jean_zay/stage_spline_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
arms=(graph_control detached_spline)
arm="${arms[$SLURM_ARRAY_TASK_ID]}"
stage="${1:?Specify 150 or 300 as the first argument}"
python -u scripts/extend_scratch_geometry.py --arm "$arm" --stage "$stage" --smoke
if [[ "${SMOKE_ONLY:-0}" != 1 ]]; then
    python -u scripts/extend_scratch_geometry.py --arm "$arm" --stage "$stage"
fi
