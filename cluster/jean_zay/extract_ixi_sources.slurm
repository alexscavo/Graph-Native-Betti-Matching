#!/bin/bash
#SBATCH --job-name=ixi-sources
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
set -euo pipefail

# The project venv's interpreter runs standalone on cpu_p1 -- verified by srun.
# No modules are loaded: this stage needs numpy/scipy/skimage/nibabel, not torch.
PYTHON="${IXI_PYTHON:?IXI_PYTHON must point at the venv python}"
REPO="${IXI_REPO:?IXI_REPO must be set}"
OUTPUT="${IXI_SOURCES:?IXI_SOURCES must be set}"
# IXI_NUM_SHARDS must be the size of the ORIGINAL array. Re-running a single failed
# task as --array=14-14 sets SLURM_ARRAY_TASK_COUNT=1, which would silently change
# which subjects that shard owns.
SHARDS="${IXI_NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRATCH/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

echo "shard $SHARD/$SHARDS on $(hostname), ${SLURM_CPUS_PER_TASK:-1} cpus"
cd "$REPO"
srun "$PYTHON" -u scripts/prepare_ixi_sources.py \
  --output-dir "$OUTPUT" \
  --shard "$SHARD" \
  --num-shards "$SHARDS" \
  ${IXI_EXTRACT_ARGS:-}
echo "shard $SHARD finished with status $?"
