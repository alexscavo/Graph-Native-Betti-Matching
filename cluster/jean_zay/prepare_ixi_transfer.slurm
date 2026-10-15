#!/usr/bin/env bash
#SBATCH --job-name=ixi-targets
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=experiments/spline_learning/logs/ixi-targets-%j.out
#SBATCH --error=experiments/spline_learning/logs/ixi-targets-%j.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" IDR_DEBUG=WARN
source cluster/jean_zay/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
kind="${1:?ixi or synthetic}"
mode="${2:-full}"
base=/lustre/fsn1/projects/rech/vnc/upz25mj/datasets
if [[ "$kind" == ixi ]]; then
    source="$base/IXI/vascular_graph_sources/optimal_radius_0p75x_c095_vedo_original"
    patches="$base/IXI/vascular_patches/optimal_radius_0p75x_c095_vedo_original"
    output="$base/IXI/transfer_splines_k8_20261015"
    options=(--train-limit 10000 --test)
else
    source="$base/syntheticMRI"
    patches="$source/new_patches_boundary"
    output="$source/transfer_source_k8_25k_20261015"
    options=(--train-limit 25000 --val-limit 480)
fi
if [[ "$mode" == smoke ]]; then
    output="${output}_smoke_v2"
    options+=(--smoke)
fi
python -u scripts/prepare_transfer_splines.py --source "$source" --patches "$patches" --output "$output" "${options[@]}"
if [[ "$kind" == ixi ]]; then
    python -u scripts/prepare_ixi_monitoring.py --root "$output"
fi
