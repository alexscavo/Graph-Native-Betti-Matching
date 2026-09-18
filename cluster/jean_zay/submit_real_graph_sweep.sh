#!/bin/bash
set -euo pipefail
REPO="${REAL_SWEEP_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${REAL_SWEEP_PYTHON:-/lustre/fsn1/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python}"
DATA="${REAL_SWEEP_DATA_ROOT:-/lustre/fsn1/projects/rech/vnc/upz25mj/datasets}"
OUTPUT="${REAL_SWEEP_OUTPUT:-$REPO/docs/research/artifacts/real_graph_sweep}"
FULL="${REAL_SWEEP_FULL_MANIFEST:-$OUTPUT/full_manifest.csv}"
SUBSET="${REAL_SWEEP_MANIFEST:-$OUTPUT/subset_manifest.csv}"
LOGS="${REAL_SWEEP_LOGS:-${WORK:-$REPO}/logs/graph-native-betti-matching}"
ARRAY="${REAL_SWEEP_ARRAY:-0-14%4}"
export REAL_SWEEP_REPO="$REPO" REAL_SWEEP_PYTHON="$PYTHON" REAL_SWEEP_OUTPUT="$OUTPUT"
export REAL_SWEEP_MANIFEST="$SUBSET" REAL_SWEEP_NUM_SHARDS=15 REAL_SWEEP_ARGS="${REAL_SWEEP_ARGS:-}"
[ -x "$PYTHON" ] || { echo "Python not executable: $PYTHON" >&2; exit 1; }
mkdir -p "$OUTPUT" "$LOGS"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PYTHON" "$REPO/scripts/make_real_graph_manifest.py" --data-root "$DATA" --output "$FULL"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-real-sweep" \
  "$PYTHON" "$REPO/scripts/select_real_graph_subset.py" --manifest "$FULL" --output "$SUBSET"
rows=$(awk 'END {print NR-1}' "$SUBSET"); [ "$rows" -eq 15 ] || { echo "expected 15 rows, got $rows" >&2; exit 1; }
echo "subset=$SUBSET output=$OUTPUT array=$ARRAY resources='1 CPU, partition-default memory, 30m'"
if [ "${REAL_SWEEP_DRY_RUN:-0}" = 1 ]; then echo "dry run: not submitting"; exit 0; fi
job=$(sbatch --parsable --array="$ARRAY" --output="$LOGS/graph-sweep-%A_%a.out" \
  --error="$LOGS/graph-sweep-%A_%a.err" \
  --export=ALL,REAL_SWEEP_REPO,REAL_SWEEP_PYTHON,REAL_SWEEP_MANIFEST,REAL_SWEEP_OUTPUT,REAL_SWEEP_NUM_SHARDS,REAL_SWEEP_ARGS \
  "$REPO/cluster/jean_zay/sweep_real_graph_representations.slurm")
echo "submitted $job"
