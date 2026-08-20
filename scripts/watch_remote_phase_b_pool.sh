#!/bin/bash
set -euo pipefail

project_root="${OSS36_PROJECT_ROOT:-/code/openstatesearch}"
python_bin="${OSS36_PYTHON:-/root/miniforge3/bin/python}"
remote_host="${1:?usage: watch_remote_phase_b_pool.sh REMOTE_HOST REMOTE_ROOT LAUNCHER_PID TAG}"
remote_root="${2:?usage: watch_remote_phase_b_pool.sh REMOTE_HOST REMOTE_ROOT LAUNCHER_PID TAG}"
launcher_pid="${3:?usage: watch_remote_phase_b_pool.sh REMOTE_HOST REMOTE_ROOT LAUNCHER_PID TAG}"
tag="${4:?usage: watch_remote_phase_b_pool.sh REMOTE_HOST REMOTE_ROOT LAUNCHER_PID TAG}"

if ! [[ "$launcher_pid" =~ ^[1-9][0-9]*$ ]]; then
  echo "LAUNCHER_PID must be a positive integer" >&2
  exit 2
fi
if ! [[ "$tag" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "TAG contains unsupported characters" >&2
  exit 2
fi

cd "$project_root"
echo "watching ${remote_host}:${remote_root} launcher ${launcher_pid}"
while ssh "$remote_host" "kill -0 ${launcher_pid} 2>/dev/null"; do
  sleep 30
done

remote_eval="${remote_root}/artifacts/eval/${tag}.jsonl"
remote_hard="${remote_root}/data/processed/full_context_seed36/rl_phase_b_hard.jsonl"
ready=0
for _ in $(seq 1 60); do
  if ssh "$remote_host" \
    "test -f '${remote_eval}' && test -f '${remote_eval}.metrics.json' && test -f '${remote_eval}.eval_metrics.json' && test -f '${remote_eval}.manifest.json' && test -f '${remote_hard}' && test -f '${remote_hard}.manifest.json'"; then
    ready=1
    break
  fi
  sleep 5
done
if (( ready == 0 )); then
  echo "remote Phase-B launcher exited without all expected outputs" >&2
  exit 1
fi

mkdir -p artifacts/eval data/processed/full_context_seed36
for suffix in "" .metrics.json .eval_metrics.json .manifest.json; do
  rsync -a \
    "${remote_host}:${remote_eval}${suffix}" \
    "artifacts/eval/${tag}.jsonl${suffix}"
done
for suffix in "" .manifest.json; do
  rsync -a \
    "${remote_host}:${remote_hard}${suffix}" \
    "data/processed/full_context_seed36/rl_phase_b_hard.jsonl${suffix}"
done

"$python_bin" scripts/verify_phase_b_pool.py \
  --pool data/processed/full_context_seed36/rl_phase_b_pool.jsonl \
  --merged "artifacts/eval/${tag}.jsonl" \
  --hard data/processed/full_context_seed36/rl_phase_b_hard.jsonl \
  --phase-a data/processed/full_context_seed36/rl_train.jsonl \
  --output artifacts/audits/phase_b_hard_pool_verification.json
