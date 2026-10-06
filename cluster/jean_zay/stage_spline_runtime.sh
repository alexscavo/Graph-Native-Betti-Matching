#!/usr/bin/env bash
# Source after env.sh. Preserve system PyTorch/CUDA while using a fixed local
# copy of the project's Python packages for the duration of this job.
spline_runtime_archive=/lustre/fswork/projects/rech/vnc/upz25mj/experiments/gnbm/spline-runtime-20261006/site-packages.tar
sha256sum -c "${spline_runtime_archive%.tar}.sha256"
spline_runtime_dir="${JOBSCRATCH:-/tmp}/gnbm-spline-runtime-${SLURM_JOB_ID:-$$}"
mkdir -p "$spline_runtime_dir"
tar -xf "$spline_runtime_archive" -C "$spline_runtime_dir"
export PYTHONPATH="$spline_runtime_dir${PYTHONPATH:+:$PYTHONPATH}"
echo "Fixed job-local Python package copy: $spline_runtime_dir"
