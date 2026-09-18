#!/bin/bash
#SBATCH --job-name=ixi-graph-budget
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
set -euo pipefail

PYTHON="${IXI_PYTHON:?IXI_PYTHON must point at the project interpreter}"
REPO="${IXI_REPO:?IXI_REPO must be set}"
OUTPUT="${IXI_BUDGET_OUTPUT:?IXI_BUDGET_OUTPUT must be set}"
SUBJECTS="${IXI_AUDIT_SUBJECTS:?IXI_AUDIT_SUBJECTS must be comma-separated}"

IFS=',' read -r -a subject_array <<< "$SUBJECTS"
index="${SLURM_ARRAY_TASK_ID:-0}"
subject="${subject_array[$index]:?No subject for array index $index}"

# The extraction is process-level parallel across Slurm array tasks. Prevent
# NumPy/SciPy from spawning large hidden BLAS thread pools inside each task.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRATCH/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT"

cd "$REPO"
echo "auditing $subject on $(hostname)"
srun "$PYTHON" -u scripts/audit_ixi_graph_budget.py "$subject" \
  --output-dir "$OUTPUT" \
  ${IXI_BUDGET_ARGS:-}

