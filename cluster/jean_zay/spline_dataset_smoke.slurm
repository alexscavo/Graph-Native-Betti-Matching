#!/usr/bin/env bash
#SBATCH --job-name=spline-dataset-smoke
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export GNBM_OUTPUT_DIR="$GNBM_REPO_DIR/experiments/spline_learning/runs"
export SYNTHETIC_MRI_DATASET="$GNBM_REPO_DIR/experiments/spline_learning/synthetic_mri_k8_10k"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib"
export TMPDIR=/tmp WANDB_MODE="${WANDB_MODE:-online}"
source cluster/jean_zay/env.sh
set -u
prior=/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/finetune_vessels_full_300ep
pilot="$GNBM_REPO_DIR/experiments/spline_learning/dataset_pilot_complete_sources"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/prepare_spline_dataset.py \
  --output "$pilot" --train-limit 32 --val-limit 16 --test-limit 16 --workers 2 --pilot-available-subjects --initial-manifest "$prior/dataset-manifest.json"
python -m unittest -v tests.test_spline tests.test_tracking
python scripts/smoke_spline_dataset.py --data-root "$pilot" \
  --initial-weights "$prior/models/best_edge_f1_checkpoint.pt" --output "$GNBM_OUTPUT_DIR/dataset-smoke-$SLURM_JOB_ID"
