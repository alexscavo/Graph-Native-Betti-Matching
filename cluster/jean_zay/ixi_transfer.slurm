#!/usr/bin/env bash
#SBATCH --job-name=ixi-transfer
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
#SBATCH --output=experiments/spline_learning/logs/ixi-transfer-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/ixi-transfer-%A_%a.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib" TMPDIR=/tmp IDR_DEBUG=WARN WANDB_MODE=offline
unset JOBSCRATCH
source cluster/jean_zay/env.sh
source cluster/jean_zay/stage_spline_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
config=configs/ixi_transfer.yaml
mode="${1:?verify, pretrain or finetune}"
arms=(ixi_straight ixi_spline plants_straight plants_spline synthetic_straight synthetic_spline)
if [[ "$mode" == verify ]]; then
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/vessel/test_plant_spline_projection.py tests/vessel/test_spline_augmentation.py tests/vessel/test_plant_contraction.py
    arm="${arms[$SLURM_ARRAY_TASK_ID]}"
    python -u scripts/train_ixi_transfer.py --config "$config" --arm "pretrain_$arm" --until-epoch 1 --smoke
    python -u scripts/train_ixi_transfer.py --config "$config" --arm "$arm" --until-epoch 1 --smoke
elif [[ "$mode" == pretrain ]]; then
    python -u scripts/train_ixi_transfer.py --config "$config" --arm "pretrain_${arms[$SLURM_ARRAY_TASK_ID]}" --until-epoch "${2:?epoch}"
elif [[ "$mode" == finetune ]]; then
    python -u scripts/train_ixi_transfer.py --config "$config" --arm "${arms[$SLURM_ARRAY_TASK_ID]}" --until-epoch "${2:?epoch}"
elif [[ "$mode" == test ]]; then
    python -u scripts/evaluate_ixi_transfer.py --config "$config" --arm "${arms[$SLURM_ARRAY_TASK_ID]}"
else
    exit 2
fi
