#!/usr/bin/env bash
#SBATCH --job-name=leaf-relocate
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:05:00
#SBATCH --output=experiments/spline_learning/logs/leaf-relocate-%j.out
#SBATCH --error=experiments/spline_learning/logs/leaf-relocate-%j.err
set -euo pipefail
python3 scripts/relocate_leaf_patches.py --finish
