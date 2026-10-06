#!/usr/bin/env bash
#SBATCH --job-name=spline-v0-10k
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export GNBM_OUTPUT_DIR="$GNBM_REPO_DIR/experiments/spline_learning/runs"
export SYNTHETIC_MRI_DATASET="$GNBM_REPO_DIR/experiments/spline_learning/synthetic_mri_k8_10k_val2k"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib"
export TMPDIR=/tmp WANDB_MODE="${WANDB_MODE:-online}"
source cluster/jean_zay/env.sh
source cluster/jean_zay/stage_spline_runtime.sh
set -u
prior=/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/finetune_vessels_full_300ep/models/best_edge_f1_checkpoint.pt
python -c 'import json, os; m=json.load(open(os.environ["SYNTHETIC_MRI_DATASET"]+"/manifest.json")); assert not m.get("pilot_available_subjects",False), "Pilot cache cannot be used for the long run"; assert m["available_subjects_only"]; assert m["requested_counts"]["test"]==m["available_counts"]["test"], "Available test population must be uncapped"; assert m["requested_counts"]["train"]<=10000; assert m["requested_counts"]["val"]==2000'
# The real-data GPU smoke already passed (615708); no repeated smoke run.
if [[ -n "${SPLINE_RESUME_CHECKPOINT:-}" ]]; then
    python -u train.py --config configs/finetune_spline_v0_10k.yaml --resume "$SPLINE_RESUME_CHECKPOINT"
else
    python -u train.py --config configs/finetune_spline_v0_10k.yaml --initial-weights "$prior"
fi
