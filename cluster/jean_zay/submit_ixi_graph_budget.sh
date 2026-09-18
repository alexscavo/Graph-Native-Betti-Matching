#!/bin/bash
set -euo pipefail

REPO="${IXI_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCRATCH_DIR="${SCRATCH:-/lustre/fsn1/projects/rech/vnc/upz25mj}"
export IXI_REPO="$REPO"
export IXI_PYTHON="${IXI_PYTHON:-$SCRATCH_DIR/venvs/vascular-graph-extraction-h100-torch231/bin/python}"
export IXI_BUDGET_OUTPUT="${IXI_BUDGET_OUTPUT:-$REPO/docs/research/evidence/real_ixi_patch_budget}"
export IXI_AUDIT_SUBJECTS="${IXI_AUDIT_SUBJECTS:-IXI002-Guys-0828,IXI609-HH-2600,IXI290-IOP-0874}"
LOGS="${IXI_LOGS:-$WORK/logs/graph-native-betti-matching}"

IFS=',' read -r -a subjects <<< "$IXI_AUDIT_SUBJECTS"
last_index=$((${#subjects[@]} - 1))
[ "$last_index" -ge 0 ] || { echo "No subjects configured" >&2; exit 1; }
[ -x "$IXI_PYTHON" ] || { echo "Python is not executable: $IXI_PYTHON" >&2; exit 1; }
mkdir -p "$LOGS" "$IXI_BUDGET_OUTPUT"

echo "subjects : $IXI_AUDIT_SUBJECTS"
echo "output   : $IXI_BUDGET_OUTPUT"
echo "extra    : ${IXI_BUDGET_ARGS:-<none>}"
if [ "${IXI_DRY_RUN:-0}" = "1" ]; then
  echo "IXI_DRY_RUN=1 -- not submitting"
  exit 0
fi

job=$(sbatch --parsable \
  --array="0-$last_index" \
  --output="$LOGS/ixi-graph-budget-%A_%a.out" \
  --error="$LOGS/ixi-graph-budget-%A_%a.err" \
  --export=ALL,IXI_REPO,IXI_PYTHON,IXI_BUDGET_OUTPUT,IXI_AUDIT_SUBJECTS,IXI_BUDGET_ARGS \
  "$REPO/cluster/jean_zay/audit_ixi_graph_budget.slurm")
echo "submitted array job $job"
echo "logs: $LOGS/ixi-graph-budget-${job}_*.out"

