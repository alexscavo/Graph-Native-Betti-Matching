#!/usr/bin/env bash
#SBATCH --job-name=leaf-targets
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=experiments/spline_learning/logs/leaf-targets-%j.out
#SBATCH --error=experiments/spline_learning/logs/leaf-targets-%j.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" IDR_DEBUG=WARN
source cluster/jean_zay/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
root=/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/leaf_generalization_20261015
source=/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/HALVS_DATASET
python -u experiments/leaf_pretraining/reference/generate_halvs_patches.py --source "$source" --target "$root/extracted" --seed 364505 --overlap 0.875 --downscale-factor 5.42 --veins red,green --debug
python -u scripts/prepare_plant_spline_targets.py --source "$root/extracted" --output "$root/plants-smoke" --train-limit 2 --val-limit 2
python -u scripts/prepare_plant_spline_targets.py --source "$root/extracted" --output "$root/leaves-k8" --train-limit 25000 --val-limit 480
python -u scripts/preview_leaf_targets.py --root "$root"
