#!/bin/bash
# Stage 2: cut 64^3 patches from the IXI source tree using the existing generator.
# The source layout produced by prepare_ixi_sources.py is exactly what
# discover_sources() expects, so the audited generator is reused unchanged.
set -euo pipefail

REPO="${IXI_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCRATCH_DIR="${SCRATCH:-/lustre/fsn1/projects/rech/vnc/upz25mj}"
export IXI_REPO="$REPO"
export IXI_PYTHON="${IXI_PYTHON:-$SCRATCH_DIR/venvs/vascular-graph-extraction-h100-torch231/bin/python}"
export IXI_SOURCES="${IXI_SOURCES:-$SCRATCH_DIR/datasets/IXI_sources}"
export IXI_PATCHES="${IXI_PATCHES:-$SCRATCH_DIR/datasets/IXI_patches}"
export IXI_SPLIT="${IXI_SPLIT:-$IXI_SOURCES/ixi_split.csv}"
CPUS="${IXI_PATCH_CPUS:-20}"
TIME="${IXI_PATCH_TIME:-10:00:00}"
LOGS="${IXI_LOGS:-$WORK/logs/graph-native-betti-matching}"

for path in "$IXI_SOURCES/raw" "$IXI_SOURCES/seg" "$IXI_SOURCES/graphs/adaptive"; do
  [ -d "$path" ] || { echo "missing source dir: $path (run submit_ixi_extraction.sh first)" >&2; exit 1; }
done
if [ ! -f "$IXI_SPLIT" ]; then
  echo "split file missing: $IXI_SPLIT" >&2
  echo "create it with: $IXI_PYTHON $REPO/scripts/make_ixi_split.py --sources $IXI_SOURCES --output $IXI_SPLIT" >&2
  exit 1
fi
mkdir -p "$LOGS" "$IXI_PATCHES"

echo "sources : $IXI_SOURCES  ($(ls "$IXI_SOURCES/graphs/adaptive" | wc -l) adaptive graphs)"
echo "split   : $IXI_SPLIT"
echo "patches : $IXI_PATCHES"
echo "cpus    : $CPUS   walltime: $TIME"

echo "--- preflight: candidate patch count (no data written) ---"
cd "$REPO"
"$IXI_PYTHON" scripts/generate_synthetic_mri_dataset.py \
  --root "$IXI_SOURCES" --output-dir "$IXI_PATCHES" \
  --split-output "$IXI_SPLIT" --resume --plan-only ${IXI_PATCH_ARGS:-}

if [ "${IXI_DRY_RUN:-0}" = "1" ]; then
  echo "IXI_DRY_RUN=1 -- not submitting"; exit 0
fi

JOB=$(sbatch --parsable \
  --cpus-per-task="$CPUS" \
  --time="$TIME" \
  --output="$LOGS/ixi-patches-%j.out" \
  --error="$LOGS/ixi-patches-%j.err" \
  --export=ALL,IXI_REPO,IXI_PYTHON,IXI_SOURCES,IXI_PATCHES,IXI_SPLIT,IXI_PATCH_ARGS \
  "$REPO/cluster/jean_zay/generate_ixi_patches.slurm")
echo "submitted job $JOB"
echo "logs: $LOGS/ixi-patches-${JOB}.out"
