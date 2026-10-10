#!/usr/bin/env bash
#SBATCH --job-name=spline-local-geometry
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
source cluster/jean_zay/env.sh
source cluster/jean_zay/stage_spline_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
arms=(token_control local_features zero_features)
arm="${arms[$SLURM_ARRAY_TASK_ID]}"
python -m unittest -v tests.test_local_spline_geometry.LocalGeometryTests tests.test_frozen_spline_diagnostic.FrozenSplineTests
python -u scripts/train_spline_weight_pilot.py --config configs/spline_local_geometry_pilot.yaml --arm "$arm" --smoke
python -u scripts/train_spline_weight_pilot.py --config configs/spline_local_geometry_pilot.yaml --arm "$arm"
