#!/usr/bin/env bash
#SBATCH --job-name=plant-targets
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=experiments/spline_learning/logs/plant-targets-%j.out
#SBATCH --error=experiments/spline_learning/logs/plant-targets-%j.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" IDR_DEBUG=WARN
source cluster/jean_zay/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
root=/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/plant_generalization_20261014
python -u scripts/prepare_plant_spline_targets.py --output "$root/plants-smoke" --train-limit 2 --val-limit 2
python -u scripts/prepare_plant_spline_targets.py --output "$root/plants-k8" --train-limit 25000 --val-limit 480
