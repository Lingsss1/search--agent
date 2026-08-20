#!/bin/bash
set -euo pipefail

project_root="${OSS36_PROJECT_ROOT:-/code/openstatesearch}"
experiment="${1:?usage: watch_phasea_checkpoint.sh EXPERIMENT TRIAL STEP TRAIN_WRAPPER_PID}"
trial="${2:?usage: watch_phasea_checkpoint.sh EXPERIMENT TRIAL STEP TRAIN_WRAPPER_PID}"
step="${3:?usage: watch_phasea_checkpoint.sh EXPERIMENT TRIAL STEP TRAIN_WRAPPER_PID}"
train_pid="${4:?usage: watch_phasea_checkpoint.sh EXPERIMENT TRIAL STEP TRAIN_WRAPPER_PID}"
continue_after_archive="${OSS36_CONTINUE_AFTER_ARCHIVE:-false}"

if ! [[ "$step" =~ ^[1-9][0-9]*$ && "$train_pid" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEP and TRAIN_WRAPPER_PID must be positive integers" >&2
  exit 2
fi
case "$continue_after_archive" in
  true|false) ;;
  *)
    echo "OSS36_CONTINUE_AFTER_ARCHIVE must be true or false" >&2
    exit 2
    ;;
esac
cd "$project_root"
log_path="artifacts/areal/logs/root/${experiment}/${trial}/main.log"
adapter_dir="artifacts/areal/checkpoints/root/${experiment}/${trial}/default/weight_update_v${step}"

verify_training_process() {
  if ! kill -0 "$train_pid" 2>/dev/null; then
    echo "training wrapper ${train_pid} is no longer running" >&2
    return 1
  fi
  command_line=$(tr '\0' ' ' <"/proc/${train_pid}/cmdline")
  if [[ "$command_line" != *"scripts/train_grpo.py"* || \
        "$command_line" != *"--experiment-name ${experiment}"* || \
        "$command_line" != *"--trial-name ${trial}"* ]]; then
    echo "PID ${train_pid} is not the expected GRPO training wrapper" >&2
    return 1
  fi
}

verify_training_process
echo "watching ${experiment}/${trial} for completed step ${step}"
while ! rg -q "Step ${step}/[0-9]+ Train step ${step}/[0-9]+ done" "$log_path"; do
  verify_training_process
  sleep 30
done

for _ in $(seq 1 60); do
  if [[ -f "${adapter_dir}/adapter_config.json" && \
        -f "${adapter_dir}/adapter_model.safetensors" ]]; then
    break
  fi
  sleep 2
done
if [[ ! -f "${adapter_dir}/adapter_config.json" || \
      ! -f "${adapter_dir}/adapter_model.safetensors" ]]; then
  echo "step ${step} completed but its LoRA weight update is missing" >&2
  exit 1
fi

/bin/bash scripts/audit_phasea_checkpoint.sh "$experiment" "$trial" "$step"
checkpoint_manifest="artifacts/policy_grpo_phasea_step${step}_adapter_seed36/checkpoint_manifest.json"
if [[ ! -f "$checkpoint_manifest" ]]; then
  echo "checkpoint audit completed without an archived manifest" >&2
  exit 1
fi

if [[ "$continue_after_archive" == "true" ]]; then
  echo "checkpoint ${step} archived and audited; continuing training by request"
  exit 0
fi

verify_training_process
process_group=$(ps -o pgid= -p "$train_pid" | tr -d ' ')
if [[ "$process_group" != "$train_pid" ]]; then
  echo "refusing to signal unexpected process group ${process_group}" >&2
  exit 1
fi
echo "checkpoint ${step} archived and audited; sending SIGINT to process group ${process_group}"
kill -INT -- "-${process_group}"

for _ in $(seq 1 30); do
  if ! kill -0 "$train_pid" 2>/dev/null; then
    echo "training stopped after audited checkpoint ${step}"
    exit 0
  fi
  sleep 10
done
echo "SIGINT was sent, but training wrapper ${train_pid} is still alive" >&2
exit 1
