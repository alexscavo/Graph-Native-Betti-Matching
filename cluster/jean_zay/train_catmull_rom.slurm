#!/usr/bin/env bash
#SBATCH --job-name=scratch-catmull-rom
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export GNBM_OUTPUT_DIR=/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs
export SYNTHETIC_MRI_DATASET=/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/scratch_comparison_10k_300ep/catmull_rom
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib" TMPDIR=/tmp IDR_DEBUG=WARN WANDB_MODE=offline
source cluster/jean_zay/env.sh
source cluster/jean_zay/stage_spline_runtime.sh
set -u
config=configs/scratch_comparison_catmull_rom_300ep.yaml
run="$GNBM_OUTPUT_DIR/scratch_catmull_rom_k8_10k_300ep"
if [[ "${VERIFY_ONLY:-0}" == 1 ]]; then
    python -u scripts/check_catmull_rom_comparison.py
    python -u scripts/smoke_spline_dataset.py --config "$config" --data-root "$SYNTHETIC_MRI_DATASET" \
        --output "$GNBM_OUTPUT_DIR/catmull-rom-smoke-$SLURM_JOB_ID"
else
    # Fresh training never silently resumes or overwrites an existing run.
    checkpoint="${COMPARISON_RESUME_CHECKPOINT:-}"
    if [[ -n "$checkpoint" ]]; then
        [[ "$checkpoint" == "$run/models/latest_checkpoint.pt" && -f "$checkpoint" ]]
        python -u train.py --config "$config" --resume "$checkpoint"
    else
        [[ ! -e "$run" ]] || { echo "Refusing to overwrite $run" >&2; exit 2; }
        python -u train.py --config "$config"
    fi
fi
