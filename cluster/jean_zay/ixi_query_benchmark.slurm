#!/usr/bin/env bash
# submit_ixi_query_benchmark.sh supplies QoS, walltime, paths, and log files.
#SBATCH --job-name=gnbm-ixi-q192
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

set -eo pipefail

repo_dir="${GNBM_REPO_DIR:-$SLURM_SUBMIT_DIR}"
source "$repo_dir/cluster/jean_zay/env.sh"
set -u

dataset="${IXI_QUERY_BENCHMARK_DATASET:?}"
overflow_dataset="${IXI_QUERY_OVERFLOW_DATASET:?}"
overflow_manifest="${IXI_QUERY_OVERFLOW_MANIFEST:?}"
report_dir="${IXI_QUERY_BENCHMARK_REPORT_DIR:?}"
selection_dir="${IXI_QUERY_SELECTION_DIR:?}"
mkdir -p "$report_dir/results"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "host=$(hostname)"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git -C "$repo_dir" rev-parse HEAD)"
  echo "python=$GNBM_VENV/bin/python"
  echo "dataset=$dataset"
  echo "overflow_dataset=$overflow_dataset"
  echo "config=configs/finetune_synthetic_mri.yaml"
  echo "batch_size=8"
  echo "warmup_steps=5"
  echo "measured_steps=30"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
  "$GNBM_VENV/bin/python" -c 'import torch; print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda)); print("cudnn=" + str(torch.backends.cudnn.version()))'
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader
} > "$report_dir/system-info.txt" 2>&1

git -C "$repo_dir" status --short > "$report_dir/git-status.txt"
cp "$repo_dir/configs/finetune_synthetic_mri.yaml" "$report_dir/base-config.yaml"
cp "$selection_dir/selection_summary.json" "$report_dir/selection-summary.json"
cp "$selection_dir/paired_patches.csv" "$report_dir/paired-patches.csv"
cp "$selection_dir/overflow_patches.csv" "$report_dir/overflow-patches.csv"
sha256sum \
  "$report_dir/base-config.yaml" \
  "$report_dir/selection-summary.json" \
  "$report_dir/paired-patches.csv" \
  "$report_dir/overflow-patches.csv" > "$report_dir/input-checksums.sha256"

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv,noheader,nounits \
  -l 1 > "$report_dir/gpu-telemetry.csv" &
telemetry_pid=$!
cleanup_telemetry() {
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup_telemetry EXIT

started_seconds=$(date +%s)
set +e
/usr/bin/time -v "$GNBM_VENV/bin/python" -u scripts/benchmark_ixi_object_queries.py \
  --config configs/finetune_synthetic_mri.yaml \
  --dataset "$dataset" \
  --overflow-dataset "$overflow_dataset" \
  --overflow-manifest "$overflow_manifest" \
  --output "$report_dir/results" \
  --batch-size 8 \
  --workers 4 \
  --warmup-steps 5 \
  --measured-steps 30 \
  > "$report_dir/benchmark.log" 2> "$report_dir/time-and-errors.log"
exit_code=$?
set -e
finished_seconds=$(date +%s)
cleanup_telemetry
trap - EXIT

{
  echo "job_id=$SLURM_JOB_ID"
  echo "wall_seconds=$((finished_seconds - started_seconds))"
  awk -v seconds="$((finished_seconds - started_seconds))" 'BEGIN {printf "allocated_gpu_hours=%.6f\n", seconds / 3600}'
  echo "exit_code=$exit_code"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$report_dir/job-summary.txt"

if [[ "$exit_code" -ne 0 ]]; then
  echo "IXI query benchmark failed; see $report_dir/time-and-errors.log" >&2
  exit "$exit_code"
fi
cat "$report_dir/results/summary.json"
echo "Benchmark report: $report_dir"
