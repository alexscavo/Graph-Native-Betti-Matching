#!/usr/bin/env bash
#SBATCH --job-name=spline-prepare-10k
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib"
export TMPDIR=/tmp IDR_DEBUG=WARN
cd "$SLURM_SUBMIT_DIR"
source cluster/jean_zay/env.sh
set -u
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python_bin="$GNBM_VENV/bin/python"
"$python_bin" -u scripts/prepare_spline_dataset.py --output experiments/spline_learning/synthetic_mri_k8_10k_val2k \
  --source /lustre/fsn1/projects/rech/vnc/upz25mj/datasets/syntheticMRI \
  --train-limit 10000 --val-limit 2000 --workers 8 --available-subjects-only \
  --initial-manifest /lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/finetune_vessels_full_300ep/dataset-manifest.json
