#!/bin/bash
set -euo pipefail

project_root="${OSS36_PROJECT_ROOT:-/code/openstatesearch}"
experiment="${1:?usage: watch_periodic_grpo_eval.sh EXPERIMENT TRIAL TRAIN_PID [INTERVAL [LAST_STEP]]}"
trial="${2:?usage: watch_periodic_grpo_eval.sh EXPERIMENT TRIAL TRAIN_PID [INTERVAL [LAST_STEP]]}"
train_pid="${3:?usage: watch_periodic_grpo_eval.sh EXPERIMENT TRIAL TRAIN_PID [INTERVAL [LAST_STEP]]}"
interval="${4:-10}"
last_step="${5:-375}"

for value in "$train_pid" "$interval" "$last_step"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "TRAIN_PID, INTERVAL, and LAST_STEP must be positive integers" >&2
    exit 2
  fi
done
if ! [[ "$experiment" =~ ^[A-Za-z0-9_.-]+$ && "$trial" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "EXPERIMENT and TRIAL may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
fi

cd "$project_root"
main_log="artifacts/areal/logs/root/${experiment}/${trial}/main.log"
runtime_root="artifacts/runtime/periodic_grpo"
mkdir -p "$runtime_root"

training_is_alive() {
  kill -0 "$train_pid" 2>/dev/null || return 1
  local command_line
  command_line=$(tr '\0' ' ' < "/proc/${train_pid}/cmdline")
  [[ "$command_line" == *"scripts/train_grpo.py"* && \
     "$command_line" == *"--experiment-name ${experiment}"* && \
     "$command_line" == *"--trial-name ${trial}"* ]]
}

next_step="$interval"
while (( next_step <= last_step )); do
  done_marker="Step ${next_step}/[0-9]+ Train step ${next_step}/[0-9]+ done"
  while ! rg -q "$done_marker" "$main_log"; do
    if ! training_is_alive; then
      echo "training stopped before periodic evaluation step ${next_step}" >&2
      exit 1
    fi
    sleep 30
  done

  step_root="artifacts/eval/periodic_grpo/step_$(printf '%06d' "$next_step")"
  if [[ ! -f "${step_root}/predictions.jsonl.manifest.json" ]]; then
    success=0
    for attempt in 1 2 3; do
      echo "starting periodic evaluation step=${next_step} attempt=${attempt}"
      if /bin/bash scripts/run_periodic_grpo_eval.sh \
        "$experiment" "$trial" "$next_step" \
        > "${runtime_root}/step_$(printf '%06d' "$next_step")_watch.log" 2>&1; then
        success=1
        break
      fi
      sleep 60
    done
    if (( success == 0 )); then
      echo "periodic evaluation failed three times at step ${next_step}" \
        > "${runtime_root}/step_$(printf '%06d' "$next_step").failed"
    fi
  fi
  next_step=$((next_step + interval))
done

echo "periodic GRPO evaluation watcher completed through step ${last_step}"
