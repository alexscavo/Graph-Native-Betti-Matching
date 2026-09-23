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
: "${REAL_PATCH_CENTERLINE_BACKEND:=legacy}"
: "${REAL_PATCH_REUSE_PATCHES_FROM:=}"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}}"
mkdir -p "$MPLCONFIGDIR" "$REAL_PATCH_OUTPUT" "$REAL_PATCH_EVIDENCE"
cd "$REAL_PATCH_REPO"

reuse_args=()
if [ -n "$REAL_PATCH_REUSE_PATCHES_FROM" ]; then
  reuse_args=(--reuse-patches-from "$REAL_PATCH_REUSE_PATCHES_FROM")
fi

echo "extracting $REAL_PATCH_DATASET patches from completed $REAL_PATCH_CENTERLINE_BACKEND graphs"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" "$REAL_PATCH_PYTHON" -u \
  scripts/extract_vessel_patches.py \
  --manifest "$REAL_PATCH_MANIFEST" \
  --dataset-root "$REAL_PATCH_DATASET_ROOT" --dataset "$REAL_PATCH_DATASET" \
  --centerline-backend "$REAL_PATCH_CENTERLINE_BACKEND" \
  --sources "$REAL_PATCH_SOURCES" --output "$REAL_PATCH_OUTPUT" \
  --split-output "$REAL_PATCH_SPLIT" --evidence "$REAL_PATCH_EVIDENCE" \
  "${reuse_args[@]}" \
  --workers "${SLURM_CPUS_PER_TASK}"
