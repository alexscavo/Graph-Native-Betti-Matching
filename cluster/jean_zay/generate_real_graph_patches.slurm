#!/bin/bash
# Generate exact-boundary 64^3 patches only after every selected graph is complete.
#SBATCH --job-name=vessel-patches
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread
set -euo pipefail

: "${REAL_PATCH_REPO:?set REAL_PATCH_REPO}"
: "${REAL_PATCH_PYTHON:?set REAL_PATCH_PYTHON}"
: "${REAL_PATCH_MANIFEST:?set REAL_PATCH_MANIFEST}"
: "${REAL_PATCH_DATASET_ROOT:?set REAL_PATCH_DATASET_ROOT}"
: "${REAL_PATCH_DATASET:?set REAL_PATCH_DATASET}"
: "${REAL_PATCH_SOURCES:?set REAL_PATCH_SOURCES}"
: "${REAL_PATCH_OUTPUT:?set REAL_PATCH_OUTPUT}"
: "${REAL_PATCH_SPLIT:?set REAL_PATCH_SPLIT}"
: "${REAL_PATCH_EVIDENCE:?set REAL_PATCH_EVIDENCE}"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}}"
mkdir -p "$MPLCONFIGDIR" "$REAL_PATCH_OUTPUT" "$REAL_PATCH_EVIDENCE"
cd "$REAL_PATCH_REPO"

echo "[1/3] validating and staging selected full-volume graphs for $REAL_PATCH_DATASET"
srun --ntasks=1 --cpus-per-task=1 "$REAL_PATCH_PYTHON" -u scripts/prepare_real_graph_patch_sources.py \
  --manifest "$REAL_PATCH_MANIFEST" \
  --dataset-root "$REAL_PATCH_DATASET_ROOT" --dataset "$REAL_PATCH_DATASET" \
  --output "$REAL_PATCH_SOURCES" \
  --split-output "$REAL_PATCH_SPLIT"

echo "[2/3] producing exact-boundary 64^3 patches"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" "$REAL_PATCH_PYTHON" -u \
  scripts/generate_synthetic_mri_dataset.py \
  --root "$REAL_PATCH_SOURCES" \
  --output-dir "$REAL_PATCH_OUTPUT" \
  --split-output "$REAL_PATCH_SPLIT" \
  --patch-size 64 64 64 --pad 0 0 0 --maximum-stride 40 40 40 \
  --workers "${SLURM_CPUS_PER_TASK}" --resume --allow-existing-split-proportions

echo "[3/3] classifying empty and graph-positive vessel patches"
srun --ntasks=1 --cpus-per-task=1 "$REAL_PATCH_PYTHON" -u scripts/audit_real_patch_eligibility.py \
  --patch-root "$REAL_PATCH_OUTPUT" \
  --output "$REAL_PATCH_EVIDENCE/$REAL_PATCH_DATASET"
