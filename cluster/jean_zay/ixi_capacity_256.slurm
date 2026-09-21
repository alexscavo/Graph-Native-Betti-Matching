#!/usr/bin/env bash
# One IXI-only paired 192/256-query timing run; submit with --time=00:08:00.
#SBATCH --job-name=gnbm-ixi-q256
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread

set -eo pipefail
repo_dir="${GNBM_REPO_DIR:-$SLURM_SUBMIT_DIR}"
source "$repo_dir/cluster/jean_zay/env.sh"
set -u

source_root="/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/IXI/vascular_patches/optimal_radius_0p75x_c095"
paired="$repo_dir/runs/ixi_capacity_192_256/paired"
report="$repo_dir/docs/research/evidence/ixi_capacity_192_256"
mkdir -p "$report"
cd "$repo_dir"
"$GNBM_VENV/bin/python" -u -m scripts.benchmark_ixi_object_queries \
  --config configs/finetune_synthetic_mri.yaml \
  --dataset "$paired" \
  --overflow-dataset "$source_root" \
  --overflow-manifest "$source_root/patch_index.csv" \
  --output "$report" \
  --batch-size 8 --workers 2 --warmup-steps 5 --measured-steps 60 \
  --query-order 256 192
