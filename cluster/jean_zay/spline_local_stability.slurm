#!/usr/bin/env bash
#SBATCH --job-name=spline-local-stability
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib" TMPDIR=/tmp IDR_DEBUG=WARN WANDB_MODE=offline
source cluster/jean_zay/env.sh
source cluster/jean_zay/stage_spline_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u scripts/select_spline_stability.py
root=/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/local_stability_20261010
config="$root/training-config.yaml"
arm=$(cat "$root/selected-arm.txt")
python -m unittest -v tests.test_controlled_spline_unfreeze.ControlledUnfreezeTests
python -u scripts/controlled_spline_unfreeze.py --config "$config" --arm "$arm" --smoke
python -u scripts/controlled_spline_unfreeze.py --config "$config" --arm "$arm"
