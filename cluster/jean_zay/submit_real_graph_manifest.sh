#!/bin/bash
# Validate the IXI + TopBrain inventory and optionally submit full-volume graph extraction.
set -euo pipefail

REPO="${REAL_GRAPH_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${REAL_GRAPH_DATA_ROOT:-/lustre/fsn1/projects/rech/vnc/upz25mj/datasets}"
PYTHON="${REAL_GRAPH_PYTHON:-/lustre/fsn1/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python}"
MANIFEST="${REAL_GRAPH_MANIFEST:-$REPO/docs/research/vedo_original_extraction_manifest.csv}"
BACKEND="${REAL_GRAPH_CENTERLINE_BACKEND:-vedo_original}"
case "$BACKEND" in
  legacy|vedo_original) ;;
  *) echo "Unsupported REAL_GRAPH_CENTERLINE_BACKEND: $BACKEND" >&2; exit 1 ;;
esac
export REAL_GRAPH_DATASET_OUTPUT_ROOT="${REAL_GRAPH_DATASET_OUTPUT_ROOT:-$DATA_ROOT}"
export REAL_GRAPH_OUTPUT="${REAL_GRAPH_OUTPUT:-}"
LOGS="${REAL_GRAPH_LOGS:-${WORK:-$REPO}/logs/graph-native-betti-matching}"
if [ -n "${REAL_GRAPH_ARRAY:-}" ]; then
  ARRAYS=("$REAL_GRAPH_ARRAY")
  CPU_REQUESTS=("${REAL_GRAPH_CPUS:-1}")
elif [ "$BACKEND" = vedo_original ]; then
  # IXI fits the 4 GB/core task allowance; larger TopBrain volumes do not.
  ARRAYS=("0-169%8" "170-219%2")
  CPU_REQUESTS=(1 16)
else
  ARRAYS=("0-219%12")
  CPU_REQUESTS=(1)
fi
for CPUS in "${CPU_REQUESTS[@]}"; do
case "$CPUS" in
  ''|*[!0-9]*) echo "REAL_GRAPH_CPUS must be a positive integer" >&2; exit 1 ;;
esac
if [ "$CPUS" -lt 1 ]; then
  echo "REAL_GRAPH_CPUS must be a positive integer" >&2
  exit 1
fi
done
export REAL_GRAPH_NUM_SHARDS="${REAL_GRAPH_NUM_SHARDS:-220}"
export REAL_GRAPH_REPO="$REPO" REAL_GRAPH_PYTHON="$PYTHON"
export REAL_GRAPH_MANIFEST="$MANIFEST"
export REAL_GRAPH_ARGS="--centerline-backend $BACKEND ${REAL_GRAPH_ARGS:-}"

[ -x "$PYTHON" ] || { echo "Python is not executable: $PYTHON" >&2; exit 1; }
[ -d "$DATA_ROOT/IXI" ] || { echo "IXI dataset missing below $DATA_ROOT" >&2; exit 1; }
[ -d "$DATA_ROOT/TopBrain_Data_Release_Batches1n2_081425" ] || {
  echo "TopBrain dataset missing below $DATA_ROOT" >&2; exit 1;
}
mkdir -p "$(dirname "$MANIFEST")" "$LOGS"

# This checks every image/label pairing, shape and affine, then replaces the CSV
# atomically. A stale manifest can therefore never be submitted silently.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PYTHON" \
  "$REPO/scripts/make_real_graph_manifest.py" \
  --data-root "$DATA_ROOT" --output "$MANIFEST"

row_count=$(awk 'END {print NR-1}' "$MANIFEST")
if [ "$row_count" -ne "$REAL_GRAPH_NUM_SHARDS" ]; then
  echo "Manifest has $row_count rows but REAL_GRAPH_NUM_SHARDS=$REAL_GRAPH_NUM_SHARDS" >&2
  echo "Use one shard per volume and update REAL_GRAPH_ARRAY consistently." >&2
  exit 1
fi

echo "repo       : $REPO"
echo "data       : $DATA_ROOT"
echo "manifest   : $MANIFEST ($row_count volumes)"
echo "output     : ${REAL_GRAPH_OUTPUT:-$REAL_GRAPH_DATASET_OUTPUT_ROOT (per-dataset vascular_graphs)}"
for index in "${!ARRAYS[@]}"; do
  CPUS="${CPU_REQUESTS[index]}"
  echo "array      : ${ARRAYS[index]} (one volume per shard)"
  echo "resources  : $CPUS CPU(s), about $((CPUS * 4)) GB Slurm memory allowance per task, 00:30:00"
done
echo "backend    : $BACKEND"
echo "extra args : ${REAL_GRAPH_ARGS:-<none>}"

if [ "${REAL_GRAPH_DRY_RUN:-0}" = "1" ]; then
  echo "REAL_GRAPH_DRY_RUN=1 -- validated manifest; not submitting"
  exit 0
fi

for index in "${!ARRAYS[@]}"; do
  job=$(sbatch --parsable \
    --array="${ARRAYS[index]}" \
    --cpus-per-task="${CPU_REQUESTS[index]}" \
    --output="$LOGS/real-graphs-%A_%a.out" \
    --error="$LOGS/real-graphs-%A_%a.err" \
    --export=ALL,REAL_GRAPH_REPO,REAL_GRAPH_PYTHON,REAL_GRAPH_MANIFEST,REAL_GRAPH_OUTPUT,REAL_GRAPH_DATASET_OUTPUT_ROOT,REAL_GRAPH_NUM_SHARDS,REAL_GRAPH_ARGS \
    "$REPO/cluster/jean_zay/extract_real_graph_manifest.slurm")
  echo "submitted array job $job (${ARRAYS[index]}; ${CPU_REQUESTS[index]} CPUs/task)"
  echo "logs: $LOGS/real-graphs-${job}_*.out"
done
