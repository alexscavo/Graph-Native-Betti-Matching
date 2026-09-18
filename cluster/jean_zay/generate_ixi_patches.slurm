#!/bin/bash
#SBATCH --job-name=ixi-patches
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
set -euo pipefail

PYTHON="${IXI_PYTHON:?}"
REPO="${IXI_REPO:?}"
SOURCES="${IXI_SOURCES:?}"
PATCHES="${IXI_PATCHES:?}"
SPLIT="${IXI_SPLIT:?}"

export OMP_NUM_THREADS=1          # the generator parallelises over patients itself
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRATCH/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

cd "$REPO"
echo "generating patches on $(hostname) with ${SLURM_CPUS_PER_TASK:-1} workers"
srun "$PYTHON" -u scripts/generate_synthetic_mri_dataset.py \
  --root "$SOURCES" \
  --output-dir "$PATCHES" \
  --split-output "$SPLIT" \
  --workers "${SLURM_CPUS_PER_TASK:-1}" \
  --resume \
  ${IXI_PATCH_ARGS:-}
