#!/usr/bin/env bash
#SBATCH --job-name=relation-distance
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
#SBATCH --output=experiments/spline_learning/logs/relation-distance-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/relation-distance-%A_%a.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib"
export TMPDIR=/tmp IDR_DEBUG=WARN WANDB_MODE=offline
unset JOBSCRATCH
source cluster/jean_zay/env.sh
source cluster/jean_zay/stage_spline_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
config=configs/relation_distance_pilot.yaml
arms=(relation_control relation_near node_control node_near)
if [[ "${1:-}" == verify ]]; then
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/vessel/test_relation_distance.py
    for arm in "${arms[@]}"; do
        python -u scripts/train_relation_distance_pilot.py --config "$config" --arm "$arm" --smoke
    done
else
    arm="${arms[$SLURM_ARRAY_TASK_ID]}"
    python -u scripts/train_relation_distance_pilot.py --config "$config" --arm "$arm"
fi
