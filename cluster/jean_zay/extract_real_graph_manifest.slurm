#!/bin/bash
# One process and one volume per shard. On Jean Zay, memory allowance scales
# with --cpus-per-task; --exclusive alone does not increase a 1-CPU task's 4 GB.
# Submit as --array=0-219%12 for the current 220-volume manifest.
#SBATCH --job-name=real-graphs
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
set -euo pipefail

: "${REAL_GRAPH_REPO:?set REAL_GRAPH_REPO}"
: "${REAL_GRAPH_PYTHON:?set REAL_GRAPH_PYTHON}"
: "${REAL_GRAPH_MANIFEST:?set REAL_GRAPH_MANIFEST}"
: "${REAL_GRAPH_DATASET_OUTPUT_ROOT:?set REAL_GRAPH_DATASET_OUTPUT_ROOT}"
: "${REAL_GRAPH_NUM_SHARDS:=220}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}}"
mkdir -p "$MPLCONFIGDIR"
cd "$REAL_GRAPH_REPO"
if [ -n "${REAL_GRAPH_OUTPUT:-}" ]; then
  destination=(--output "$REAL_GRAPH_OUTPUT")
else
  destination=(--dataset-root "$REAL_GRAPH_DATASET_OUTPUT_ROOT")
fi
srun "$REAL_GRAPH_PYTHON" -u scripts/extract_vessel_graphs.py \
  --manifest "$REAL_GRAPH_MANIFEST" "${destination[@]}" \
  --shard "${SLURM_ARRAY_TASK_ID}" --num-shards "$REAL_GRAPH_NUM_SHARDS" \
  ${REAL_GRAPH_ARGS:-}
