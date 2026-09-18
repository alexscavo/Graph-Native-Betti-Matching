#!/bin/bash
#SBATCH --job-name=graph-sweep
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
: "${REAL_SWEEP_REPO:?}" "${REAL_SWEEP_PYTHON:?}" "${REAL_SWEEP_MANIFEST:?}" "${REAL_SWEEP_OUTPUT:?}"
: "${REAL_SWEEP_NUM_SHARDS:=15}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}}"
mkdir -p "$MPLCONFIGDIR"
cd "$REAL_SWEEP_REPO"
srun "$REAL_SWEEP_PYTHON" -u scripts/sweep_real_graph_representations.py \
  --manifest "$REAL_SWEEP_MANIFEST" --output "$REAL_SWEEP_OUTPUT" \
  --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$REAL_SWEEP_NUM_SHARDS" ${REAL_SWEEP_ARGS:-}
