#!/usr/bin/env bash
#SBATCH --job-name=restore-syntheticMRI
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache"
export TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib"
export TMPDIR=/tmp IDR_DEBUG=WARN
cd "$SLURM_SUBMIT_DIR"
source cluster/jean_zay/env.sh
set -u
archive="${SYNTHETIC_MRI_RESTORE_ARCHIVE:-$GNBM_REPO_DIR/experiments/spline_learning/datasets-restoration.tar}"
if ! test -f "$archive"; then
    echo "Stage the STORE backup on WORK/SCRATCH from a login node and set SYNTHETIC_MRI_RESTORE_ARCHIVE." >&2
    echo "Backup: /lustre/fsstor/projects/rech/vnc/upz25mj/archives/2026-10-06/source-datasets-2026-10-05.tar" >&2
    exit 1
fi
parent=/lustre/fsn1/projects/rech/vnc/upz25mj/datasets
staging="$parent/.syntheticMRI-restore-$SLURM_JOB_ID"
backup="$parent/syntheticMRI.before-restore-$SLURM_JOB_ID"
mkdir "$staging"
echo "Extracting SyntheticMRI only to $staging"
# Fresh timestamps prevent inherited archive dates from making restored files
# immediately eligible for scratch cleanup. File contents are unchanged.
tar --extract --file="$archive" --directory="$staging" --strip-components=2 \
    --no-same-owner --no-same-permissions --touch datasets/syntheticMRI
python -u scripts/verify_synthetic_mri_restore.py --root "$staging" \
    --snapshot experiments/spline_learning/source_snapshot --archive "$archive"
test ! -e "$backup"
if test -e "$parent/syntheticMRI"; then
    mv "$parent/syntheticMRI" "$backup"
fi
if ! mv "$staging" "$parent/syntheticMRI"; then
    if test -e "$backup"; then mv "$backup" "$parent/syntheticMRI"; fi
    exit 1
fi
cp "$parent/syntheticMRI/restore-verification.json" experiments/spline_learning/source_snapshot/restore-verification.json
echo "Restore complete: $parent/syntheticMRI; previous folder retained at $backup"
