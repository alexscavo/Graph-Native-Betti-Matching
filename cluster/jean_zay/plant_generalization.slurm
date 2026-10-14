#!/usr/bin/env bash
#SBATCH --job-name=plant-generalization
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
#SBATCH --output=experiments/spline_learning/logs/plant-generalization-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/plant-generalization-%A_%a.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib" TMPDIR=/tmp IDR_DEBUG=WARN WANDB_MODE=offline
unset JOBSCRATCH
source cluster/jean_zay/env.sh
source cluster/jean_zay/stage_spline_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mode="${1:?verify, pretrain or finetune}"
if [[ "$mode" == verify ]]; then
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/vessel/test_spline_augmentation.py tests/vessel/test_plant_contraction.py
    for arm in pretrain_mixed pretrain_vessels mixed_straight mixed_spline; do
        python -u scripts/train_plant_generalization.py --arm "$arm" --until-epoch 1 --smoke
    done
elif [[ "$mode" == pretrain ]]; then
    arms=(pretrain_mixed pretrain_vessels)
    python -u scripts/train_plant_generalization.py --arm "${arms[$SLURM_ARRAY_TASK_ID]}" --until-epoch "${2:?epoch}"
elif [[ "$mode" == finetune ]]; then
    arms=(mixed_straight mixed_spline vessels_straight vessels_spline)
    python -u scripts/train_plant_generalization.py --arm "${arms[$SLURM_ARRAY_TASK_ID]}" --until-epoch "${2:?epoch}"
else
    exit 2
fi
