#!/usr/bin/env bash
# Submit one short, IXI-only H100 benchmark comparing 120 and 192 queries.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this on a Jean Zay login node." >&2
  exit 2
fi

dataset="$repo_dir/docs/research/artifacts/ixi_query_benchmark/paired"
overflow_dataset="$repo_dir/docs/research/artifacts/ixi_query_benchmark/overflow"
selection_dir="$repo_dir/docs/research/evidence/ixi_query_benchmark/selection"
overflow_manifest="$selection_dir/overflow_patches.csv"
for path in \
  "$dataset/train/raw" \
  "$dataset/val/raw" \
  "$overflow_dataset/train/raw" \
  "$selection_dir/selection_summary.json" \
  "$overflow_manifest"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing IXI benchmark input: $path" >&2
    exit 2
  fi
done

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
report_dir="$repo_dir/docs/research/evidence/ixi_query_benchmark/runs/$stamp"
mkdir -p "$report_dir"

submission="$(sbatch \
  --parsable \
  --chdir="$repo_dir" \
  --qos=qos_gpu_h100-dev \
  --time="${IXI_QUERY_BENCHMARK_WALLTIME:-00:30:00}" \
  --output="$report_dir/slurm-%j.out" \
  --error="$report_dir/slurm-%j.err" \
  --export=ALL,GNBM_REPO_DIR="$repo_dir",IXI_QUERY_BENCHMARK_DATASET="$dataset",IXI_QUERY_OVERFLOW_DATASET="$overflow_dataset",IXI_QUERY_OVERFLOW_MANIFEST="$overflow_manifest",IXI_QUERY_SELECTION_DIR="$selection_dir",IXI_QUERY_BENCHMARK_REPORT_DIR="$report_dir" \
  "$repo_dir/cluster/jean_zay/ixi_query_benchmark.slurm")"

job_id="${submission%%;*}"
{
  echo "job_id=$job_id"
  echo "submitted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "qos=qos_gpu_h100-dev"
  echo "walltime=00:30:00"
  echo "gpus=1"
  echo "gpu_type=H100"
  echo "cpus=8"
  echo "dataset=IXI only"
} > "$report_dir/submission.txt"

echo "Submitted job $job_id"
echo "Queue:  squeue -j $job_id"
echo "Report: $report_dir"
echo "Result: $report_dir/results/summary.json"
