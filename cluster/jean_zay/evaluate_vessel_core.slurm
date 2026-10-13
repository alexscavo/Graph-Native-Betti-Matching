#!/usr/bin/env bash
#SBATCH --job-name=vessel-core-metrics
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-1
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib" TMPDIR=/tmp IDR_DEBUG=WARN
unset JOBSCRATCH
source cluster/jean_zay/env.sh
# Only NumPy/SciPy are needed. CPU node /tmp cannot hold the GPU runtime archive.
metrics_python_prefix="$(python -c 'import sys; print(sys.base_prefix)')"
metrics_python="$metrics_python_prefix/bin/python"
export PYTHONPATH="$GNBM_REPO_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
arms=(straight_final bspline_detached_final)
arm="${arms[$SLURM_ARRAY_TASK_ID]}"
"$metrics_python" -u scripts/evaluate_vessel_core.py --arm "$arm" --max-samples 4
"$metrics_python" -u scripts/evaluate_vessel_core.py --arm "$arm"
