#!/bin/bash
# Launch IXI source extraction (raw + seg + graphs) as a CPU array job, then
# print the follow-up commands for the split and the patch generation.
set -euo pipefail

REPO="${IXI_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCRATCH_DIR="${SCRATCH:-/lustre/fsn1/projects/rech/vnc/upz25mj}"
export IXI_REPO="$REPO"
export IXI_PYTHON="${IXI_PYTHON:-$SCRATCH_DIR/venvs/vascular-graph-extraction-h100-torch231/bin/python}"
export IXI_SOURCES="${IXI_SOURCES:-$SCRATCH_DIR/datasets/IXI_sources}"
IXI_ARRAY="${IXI_ARRAY:-0-39}"
IXI_CPUS="${IXI_CPUS:-4}"
IXI_TIME="${IXI_TIME:-04:00:00}"
LOGS="${IXI_LOGS:-$WORK/logs/graph-native-betti-matching}"

[ -x "$IXI_PYTHON" ] || { echo "venv python not executable: $IXI_PYTHON" >&2; exit 1; }
[ -d "${IXI_DATASET:-$SCRATCH_DIR/datasets/IXI_dataset}" ] || { echo "IXI_dataset missing" >&2; exit 1; }
mkdir -p "$LOGS" "$IXI_SOURCES"

echo "repo     : $REPO"
echo "python   : $IXI_PYTHON"
echo "sources  : $IXI_SOURCES"
echo "array    : $IXI_ARRAY   cpus/task: $IXI_CPUS   walltime: $IXI_TIME   shards: ${IXI_NUM_SHARDS:-auto}"
echo "extra    : ${IXI_EXTRACT_ARGS:-<none>}"

echo "--- preflight: how many subjects are complete ---"
"$IXI_PYTHON" "$REPO/scripts/prepare_ixi_sources.py" \
  --output-dir "$IXI_SOURCES" --plan-only ${IXI_EXTRACT_ARGS:-}

if [ "${IXI_DRY_RUN:-0}" = "1" ]; then
  echo "IXI_DRY_RUN=1 -- not submitting"; exit 0
fi

JOB=$(sbatch --parsable \
  --array="$IXI_ARRAY" \
  --cpus-per-task="$IXI_CPUS" \
  --time="$IXI_TIME" \
  --output="$LOGS/ixi-sources-%A_%a.out" \
  --error="$LOGS/ixi-sources-%A_%a.err" \
  --export=ALL,IXI_REPO,IXI_PYTHON,IXI_SOURCES,IXI_EXTRACT_ARGS,IXI_NUM_SHARDS \
  "$REPO/cluster/jean_zay/extract_ixi_sources.slurm")

echo "submitted array job $JOB"
echo "logs: $LOGS/ixi-sources-${JOB}_*.out"
cat <<NEXT

Once every shard is COMPLETED:

  # 1. site-stratified subject split
  "$IXI_PYTHON" "$REPO/scripts/make_ixi_split.py" \\
      --sources "$IXI_SOURCES" --output "$IXI_SOURCES/ixi_split.csv"

  # 2. patch generation (own array job)
  IXI_SOURCES="$IXI_SOURCES" bash "$REPO/cluster/jean_zay/submit_ixi_patches.sh"
NEXT
