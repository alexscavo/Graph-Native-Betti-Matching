#!/bin/bash
# Validate the IXI + TopBrain inventory and optionally submit full-volume graph extraction.
set -euo pipefail

REPO="${REAL_GRAPH_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${REAL_GRAPH_DATA_ROOT:-/lustre/fsn1/projects/rech/vnc/upz25mj/datasets}"
PYTHON="${REAL_GRAPH_PYTHON:-/lustre/fsn1/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python}"
MANIFEST="${REAL_GRAPH_MANIFEST:-$REPO/docs/research/artifacts/real_graph_dataset/extraction_manifest.csv}"
export REAL_GRAPH_DATASET_OUTPUT_ROOT="${REAL_GRAPH_DATASET_OUTPUT_ROOT:-$DATA_ROOT}"
export REAL_GRAPH_OUTPUT="${REAL_GRAPH_OUTPUT:-}"
LOGS="${REAL_GRAPH_LOGS:-${WORK:-$REPO}/logs/graph-native-betti-matching}"
ARRAY="${REAL_GRAPH_ARRAY:-0-219%12}"
export REAL_GRAPH_NUM_SHARDS="${REAL_GRAPH_NUM_SHARDS:-220}"
export REAL_GRAPH_REPO="$REPO" REAL_GRAPH_PYTHON="$PYTHON"
export REAL_GRAPH_MANIFEST="$MANIFEST"
export REAL_GRAPH_ARGS="${REAL_GRAPH_ARGS:-}"

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
echo "array      : $ARRAY (one volume per shard)"
echo "resources  : 1 CPU, partition-default memory, 00:20:00; concurrency capped by array suffix"
echo "extra args : ${REAL_GRAPH_ARGS:-<none>}"

if [ "${REAL_GRAPH_DRY_RUN:-0}" = "1" ]; then
  echo "REAL_GRAPH_DRY_RUN=1 -- validated manifest; not submitting"
  exit 0
fi

job=$(sbatch --parsable \
  --array="$ARRAY" \
  --output="$LOGS/real-graphs-%A_%a.out" \
  --error="$LOGS/real-graphs-%A_%a.err" \
  --export=ALL,REAL_GRAPH_REPO,REAL_GRAPH_PYTHON,REAL_GRAPH_MANIFEST,REAL_GRAPH_OUTPUT,REAL_GRAPH_DATASET_OUTPUT_ROOT,REAL_GRAPH_NUM_SHARDS,REAL_GRAPH_ARGS \
  "$REPO/cluster/jean_zay/extract_real_graph_manifest.slurm")
echo "submitted array job $job"
echo "logs: $LOGS/real-graphs-${job}_*.out"
