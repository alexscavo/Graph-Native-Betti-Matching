#!/usr/bin/env bash
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
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib" TMPDIR=/tmp IDR_DEBUG=WARN WANDB_MODE=offline
source cluster/jean_zay/env.sh
source cluster/jean_zay/stage_spline_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ "${VERIFY_ONLY:-0}" == 1 ]]; then
  python -m unittest -v tests.test_curve_evidence tests.test_spline tests.test_inference
  previous=/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/staged_pilot_20261008
  targets=/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/scratch_comparison_10k_300ep/spline/val/spline
  for arm in graph_control staged_spline; do
    python scripts/diagnose_graph_connectivity.py --predictions "$previous/$arm/validation-005000/predictions.json" \
      --targets "$targets" --output "experiments/spline_learning/metadata/curve-evidence/$arm-connectivity.json"
  done
  python -u scripts/train_spline_weight_pilot.py --config configs/spline_curve_evidence_pilot.yaml --arm image_evidence --smoke
else
  arms=(token_control image_evidence)
  arm="${arms[$SLURM_ARRAY_TASK_ID]}"
  python -u scripts/train_spline_weight_pilot.py --config configs/spline_curve_evidence_pilot.yaml --arm "$arm"
  run=/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/curve_evidence_pilot_20261008
  targets=/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/scratch_comparison_10k_300ep/spline/val/spline
  python scripts/diagnose_graph_connectivity.py --predictions "$run/$arm/validation-005000/predictions.json" \
    --targets "$targets" --output "$run/$arm/connectivity-diagnostic.json"
fi
