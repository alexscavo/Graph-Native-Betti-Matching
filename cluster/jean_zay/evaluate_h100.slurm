#!/usr/bin/env bash
# Evaluate one frozen checkpoint on a complete dataset split using one H100.
#SBATCH --job-name=gnbm-eval
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread

set -eo pipefail

repo_dir="${GNBM_REPO_DIR:-$SLURM_SUBMIT_DIR}"
source "$repo_dir/cluster/jean_zay/env.sh"
set -u

for name in GNBM_EVAL_CONFIG GNBM_EVAL_CHECKPOINT GNBM_EVAL_OUTPUT GNBM_OUTPUT_DIR SYNTHETIC_MRI_DATASET; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing batch environment variable: $name" >&2
    exit 2
  fi
done

echo "job=$SLURM_JOB_ID host=$(hostname) checkpoint=$GNBM_EVAL_CHECKPOINT"
echo "dataset=$SYNTHETIC_MRI_DATASET split=${GNBM_EVAL_SPLIT:-test} output=$GNBM_EVAL_OUTPUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

eval_args=(
  --config "$GNBM_EVAL_CONFIG"
  --checkpoint "$GNBM_EVAL_CHECKPOINT"
  --dataset synthetic_mri
  --split "${GNBM_EVAL_SPLIT:-test}"
  --batch-size "${GNBM_EVAL_BATCH_SIZE:-8}"
  --workers "${GNBM_EVAL_WORKERS:-4}"
  --device cuda
  --visualizations "${GNBM_EVAL_VISUALIZATIONS:-0}"
)
if [[ -n "${GNBM_EVAL_DEGREE_PENALTIES:-}" ]]; then
  read -r -a degree_penalties <<< "$GNBM_EVAL_DEGREE_PENALTIES"
  for penalty in "${degree_penalties[@]}"; do
    echo "degree penalty=$penalty"
    "$GNBM_VENV/bin/python" -u "$repo_dir/evaluate.py" \
      "${eval_args[@]}" \
      --output-dir "$GNBM_EVAL_OUTPUT/penalty_$penalty" \
      --node-type-degree-penalty "$penalty"
  done
  if [[ -n "${GNBM_EVAL_EDGE_THRESHOLDS:-}" ]]; then
    read -r -a edge_thresholds <<< "$GNBM_EVAL_EDGE_THRESHOLDS"
    for threshold in "${edge_thresholds[@]}"; do
      echo "plain edge threshold=$threshold"
      "$GNBM_VENV/bin/python" -u "$repo_dir/evaluate.py" \
        "${eval_args[@]}" \
        --output-dir "$GNBM_EVAL_OUTPUT/edge_threshold_$threshold" \
        --edge-threshold "$threshold"
    done
  fi
else
  "$GNBM_VENV/bin/python" -u "$repo_dir/evaluate.py" \
    "${eval_args[@]}" --output-dir "$GNBM_EVAL_OUTPUT"
fi
