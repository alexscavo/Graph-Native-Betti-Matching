#!/usr/bin/env bash
#SBATCH --job-name=combined-prepare
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=experiments/spline_learning/logs/combined-prepare-%j.out
#SBATCH --error=experiments/spline_learning/logs/combined-prepare-%j.err
set -euo pipefail
python3 scripts/prepare_combined_sources.py
