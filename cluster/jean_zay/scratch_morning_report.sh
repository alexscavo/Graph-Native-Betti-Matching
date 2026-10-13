#!/usr/bin/env bash
# Bounded morning review. A second timer firing handles maintenance/queue delays.
set -euo pipefail
projects_root=/lustre/fswork/projects/rech/vnc/upz25mj/projects
report_repo="$projects_root/Graph-Native-Betti-Matching-Representation"
report_logs="$report_repo/experiments/spline_learning/logs"
report_done="$report_logs/morning-results-20261013.done"
mkdir -p "$report_logs"
[[ ! -f "$report_done" ]] || exit 0
exec 9>"$report_logs/morning-results-20261013.lock"
flock -n 9 || exit 0
cd "$projects_root"
report_stamp=$(date +%Y%m%d-%H%M%S)
/linkhome/rech/gendkt01/upz25mj/.local/bin/codex exec \
    --approve-for-me \
    -c sandbox_workspace_write.network_access=true \
    --cd "$projects_root" --skip-git-repo-check \
    --output-last-message "$report_logs/morning-results-$report_stamp.md" \
    resume 01a0fbdc-f0f9-7490-bd07-8da288759f27 - \
    < "$report_repo/cluster/jean_zay/scratch_morning_report_prompt.md" \
    > "$report_logs/morning-results-$report_stamp.log" 2>&1

# Completion requires both final runs, all four evaluations, a committed report,
# and the exact current branch head on GitHub. An incomplete review can retry.
cd "$report_repo"
python3 - <<'PY'
import json
from pathlib import Path
root = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/scratch_extend300_20261012')
for arm in ('graph_control', 'detached_spline'):
    path = root / arm / 'epoch300/summary.json'
    if not path.exists() or json.loads(path.read_text())['provenance']['end_update'] != 750000:
        raise SystemExit(1)
for arm in ('graph_control_final', 'graph_control_best', 'detached_spline_final', 'detached_spline_best'):
    path = root / 'full_validation' / arm / 'summary.json'
    if not path.exists() or json.loads(path.read_text())['common']['samples'] != 1952:
        raise SystemExit(1)
evidence = Path('experiments/spline_learning/metadata/local-geometry/scratch-extension-final-results.json')
if not evidence.exists(): raise SystemExit(1)
report = json.loads(evidence.read_text())
if report.get('status') != 'complete' or report.get('validation_samples') != 1952:
    raise SystemExit(1)
if report.get('training_end_updates') != {'graph_control': 750000, 'detached_spline': 750000}:
    raise SystemExit(1)
if set(report.get('evaluated_arms', [])) != {'graph_control_final', 'graph_control_best', 'detached_spline_final', 'detached_spline_best'}:
    raise SystemExit(1)
PY
[[ "$(git branch --show-current)" == representation ]]
git ls-files --error-unmatch docs/research/spline_learning/scratch-extension-300.md >/dev/null
git ls-files --error-unmatch experiments/spline_learning/metadata/local-geometry/scratch-extension-final-results.json >/dev/null
git diff --quiet HEAD -- docs/research/spline_learning/scratch-extension-300.md experiments/spline_learning/metadata/local-geometry/scratch-extension-final-results.json
report_local_head=$(git rev-parse HEAD)
report_remote_head=$(git ls-remote --heads origin representation | awk '{print $1}')
[[ "$report_local_head" == "$report_remote_head" ]]
date -Iseconds > "$report_done"
