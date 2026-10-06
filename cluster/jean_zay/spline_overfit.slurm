#!/usr/bin/env bash
#SBATCH --job-name=spline-k8-overfit
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=00:45:00
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export GNBM_OUTPUT_DIR="$GNBM_REPO_DIR/experiments/spline_learning/runs"
export SYNTHETIC_MRI_DATASET="$GNBM_REPO_DIR/experiments/spline_learning/targets_k8"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv"
export TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache"
export MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib"
export TMPDIR=/tmp
export WANDB_MODE=disabled
source "$GNBM_REPO_DIR/cluster/jean_zay/env.sh"
set -u
checkpoint=/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/finetune_vessels_full_300ep/models/best_edge_f1_checkpoint.pt
run_root="$GNBM_OUTPUT_DIR/job-$SLURM_JOB_ID"
mkdir -p "$run_root"
echo "job=$SLURM_JOB_ID stages=calibration,smoke-pretrained,smoke-scratch,V0-pretrained,V0-scratch,conditional-V1"
python -c 'from models.ops.functions.ms_deform_attn_func import cuda_extension_available; assert cuda_extension_available(), "Missing CUDA operator"'
python scripts/overfit_spline.py --action calibrate --initial-weights "$checkpoint" --calibration "$run_root/calibration_v0.json"
# Both forward/backward smokes must pass before either 500-update run.
python scripts/overfit_spline.py --action smoke --initial-weights "$checkpoint" --calibration "$run_root/calibration_v0.json" --output "$run_root/smoke_pretrained"
python scripts/overfit_spline.py --action smoke --calibration "$run_root/calibration_v0.json" --output "$run_root/smoke_scratch"
python scripts/overfit_spline.py --action train --steps 500 --initial-weights "$checkpoint" --calibration "$run_root/calibration_v0.json" --output "$run_root/v0_pretrained"
python scripts/overfit_spline.py --action train --steps 500 --calibration "$run_root/calibration_v0.json" --output "$run_root/v0_scratch"
python scripts/summarize_spline_learning.py --run-root "$run_root"
# Gate is diagnostic only: >=50% reduction AND <=0.5 mm mean fitted-curve
# point deviation on the four training patches, not a generalization claim.
choice=$(python scripts/summarize_spline_learning.py --run-root "$run_root" --gate)
if [[ "$choice" == none ]]; then
  echo 'V1 deferred: neither V0 initialization passed the recorded geometry-learning gate.'
else
  echo "V1 diagnostic initialization=$choice (same original initialization, no V0 weight reuse)"
  python scripts/overfit_spline.py --action calibrate --config configs/overfit_spline_v1.yaml --initial-weights "$checkpoint" --calibration "$run_root/calibration_v1.json"
  weights=()
  if [[ "$choice" == pretrained ]]; then weights=(--initial-weights "$checkpoint"); fi
  python scripts/overfit_spline.py --action train --steps 500 --config configs/overfit_spline_v1.yaml --calibration "$run_root/calibration_v1.json" --output "$run_root/v1_$choice" "${weights[@]}"
fi
python scripts/summarize_spline_learning.py --run-root "$run_root"
