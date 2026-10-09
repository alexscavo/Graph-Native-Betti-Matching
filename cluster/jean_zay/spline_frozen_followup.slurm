#!/usr/bin/env bash
#SBATCH --job-name=spline-frozen-followup
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --array=0-1
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib" TMPDIR=/tmp IDR_DEBUG=WARN WANDB_MODE=offline
source cluster/jean_zay/env.sh
source cluster/jean_zay/stage_spline_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
arms=(residual_continue absolute_full)
configs=(configs/spline_frozen_residual_continue.yaml configs/spline_frozen_absolute_10k_pilot.yaml)
arm="${arms[$SLURM_ARRAY_TASK_ID]}"
config="${configs[$SLURM_ARRAY_TASK_ID]}"
python -m unittest -v tests.test_frozen_spline_diagnostic
python -u scripts/train_spline_weight_pilot.py --config "$config" --arm "$arm" --smoke
python -u scripts/train_spline_weight_pilot.py --config "$config" --arm "$arm"
