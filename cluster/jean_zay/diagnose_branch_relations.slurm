#!/usr/bin/env bash
#SBATCH --job-name=branch-relations
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:15:00
#SBATCH --output=experiments/spline_learning/logs/branch-relations-%j.out
#SBATCH --error=experiments/spline_learning/logs/branch-relations-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
metrics_python=/lustre/fshomisc/sup/pub/anaconda-py3/h100/2024.06/envs/pytorch-gpu-2.3.1+py3.11.9/bin/python
exec "$metrics_python" -u scripts/diagnose_branch_relations.py "$@"
