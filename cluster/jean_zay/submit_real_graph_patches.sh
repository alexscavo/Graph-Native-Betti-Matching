#!/bin/bash
# Submit production patch generation after the full-volume extraction array.
set -euo pipefail

REPO="${REAL_PATCH_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCRATCH_DIR="${SCRATCH:-/lustre/fsn1/projects/rech/vnc/upz25mj}"
export REAL_PATCH_REPO="$REPO"
export REAL_PATCH_PYTHON="${REAL_PATCH_PYTHON:-$SCRATCH_DIR/venvs/vascular-graph-extraction-h100-torch231/bin/python}"
export REAL_PATCH_DATASET="${REAL_PATCH_DATASET:-ixi}"
export REAL_PATCH_CENTERLINE_BACKEND="${REAL_PATCH_CENTERLINE_BACKEND:-vedo_original}"
case "$REAL_PATCH_CENTERLINE_BACKEND" in
  legacy) policy=optimal_radius_0p75x_c095 ;;
  vedo_original) policy=optimal_radius_0p75x_c095_vedo_original ;;
  *) echo "Unsupported backend: $REAL_PATCH_CENTERLINE_BACKEND" >&2; exit 1 ;;
esac
case "$REAL_PATCH_DATASET" in
  ixi) dataset_folder=IXI ;;
  topbrain) dataset_folder=TopBrain_Data_Release_Batches1n2_081425 ;;
  *) echo "Unsupported REAL_PATCH_DATASET: $REAL_PATCH_DATASET" >&2; exit 1 ;;
esac
export REAL_PATCH_DATASET_ROOT="${REAL_PATCH_DATASET_ROOT:-$SCRATCH_DIR/datasets}"
export REAL_PATCH_MANIFEST="${REAL_PATCH_MANIFEST:-$REPO/docs/research/vedo_original_extraction_manifest.csv}"
export REAL_PATCH_SOURCES="${REAL_PATCH_SOURCES:-$REAL_PATCH_DATASET_ROOT/$dataset_folder/vascular_graph_sources/$policy}"
export REAL_PATCH_OUTPUT="${REAL_PATCH_OUTPUT:-$REAL_PATCH_DATASET_ROOT/$dataset_folder/vascular_patches/$policy}"
export REAL_PATCH_SPLIT="${REAL_PATCH_SPLIT:-$REAL_PATCH_OUTPUT/patient_split.csv}"
export REAL_PATCH_REUSE_PATCHES_FROM="${REAL_PATCH_REUSE_PATCHES_FROM:-}"
export REAL_PATCH_EVIDENCE="${REAL_PATCH_EVIDENCE:-$REPO/docs/research/evidence/full_dataset_production}"
DEPENDENCY="${REAL_PATCH_DEPENDENCY:-${1:-}}"
LOGS="${REAL_PATCH_LOGS:-${WORK:-$REPO}/logs/graph-native-betti-matching}"
CPUS="${REAL_PATCH_CPUS:-4}"
TIME="${REAL_PATCH_TIME:-00:20:00}"

[ -x "$REAL_PATCH_PYTHON" ] || { echo "Python is not executable: $REAL_PATCH_PYTHON" >&2; exit 1; }
[ -f "$REAL_PATCH_MANIFEST" ] || { echo "Manifest is missing: $REAL_PATCH_MANIFEST" >&2; exit 1; }
mkdir -p "$LOGS"

SBATCH_ARGS=(
  --parsable
  --cpus-per-task="$CPUS"
  --time="$TIME"
  --output="$LOGS/vessel-patches-%j.out"
  --error="$LOGS/vessel-patches-%j.err"
  --export=ALL,REAL_PATCH_REPO,REAL_PATCH_PYTHON,REAL_PATCH_MANIFEST,REAL_PATCH_DATASET,REAL_PATCH_DATASET_ROOT,REAL_PATCH_SOURCES,REAL_PATCH_SPLIT,REAL_PATCH_OUTPUT,REAL_PATCH_EVIDENCE,REAL_PATCH_CENTERLINE_BACKEND,REAL_PATCH_REUSE_PATCHES_FROM
)
if [ -n "$DEPENDENCY" ]; then
  SBATCH_ARGS+=(--dependency="afterok:$DEPENDENCY")
fi

echo "graphs     : $REAL_PATCH_DATASET_ROOT/$dataset_folder/vascular_graphs"
echo "sources    : $REAL_PATCH_SOURCES"
echo "patches    : $REAL_PATCH_OUTPUT"
echo "evidence   : $REAL_PATCH_EVIDENCE"
echo "dependency : ${DEPENDENCY:-none}"
echo "resources  : $CPUS CPUs, $TIME"
echo "backend    : $REAL_PATCH_CENTERLINE_BACKEND"
echo "reuse      : ${REAL_PATCH_REUSE_PATCHES_FROM:-none}"

job=$(sbatch "${SBATCH_ARGS[@]}" "$REPO/cluster/jean_zay/generate_real_graph_patches.slurm")
echo "submitted patch job $job"
echo "logs: $LOGS/vessel-patches-${job}.out"
