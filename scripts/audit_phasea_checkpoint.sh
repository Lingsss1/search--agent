#!/bin/bash
set -euo pipefail

project_root="${OSS36_PROJECT_ROOT:-/code/openstatesearch}"
python_bin="${OSS36_PYTHON:-/root/miniforge3/bin/python}"
experiment="${1:?usage: audit_phasea_checkpoint.sh EXPERIMENT TRIAL STEP [RAW_REWARD_AUDIT]}"
trial="${2:?usage: audit_phasea_checkpoint.sh EXPERIMENT TRIAL STEP [RAW_REWARD_AUDIT]}"
step="${3:?usage: audit_phasea_checkpoint.sh EXPERIMENT TRIAL STEP [RAW_REWARD_AUDIT]}"
raw_audit="${4:-artifacts/runtime/${experiment}_${trial}_reward_audit.jsonl}"
rollout_artifact_source="${OSS36_ROLLOUT_ARTIFACT_SOURCE:-h800}"
remote_project_root="${OSS36_ROLLOUT_REMOTE_ROOT:-/code/openstatesearch}"

if ! [[ "$step" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEP must be a positive integer" >&2
  exit 2
fi
if ! [[ "$experiment" =~ ^[A-Za-z0-9_.-]+$ && "$trial" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "EXPERIMENT and TRIAL contain unsupported characters" >&2
  exit 2
fi

cd "$project_root"
run_root="artifacts/areal/logs/root/${experiment}/${trial}"
rollout_root="${run_root}/rollout"
log_path="${run_root}/main.log"
prompts="data/processed/full_context_seed36/rl_train.jsonl"
prefix="artifacts/audits/${experiment}_${trial}"

if [[ -n "$rollout_artifact_source" ]]; then
  if ! [[ "$rollout_artifact_source" =~ ^[A-Za-z0-9_.@-]+$ ]]; then
    echo "OSS36_ROLLOUT_ARTIFACT_SOURCE contains unsupported characters" >&2
    exit 2
  fi
  if [[ "$remote_project_root" != /* || "$remote_project_root" == *".."* ]]; then
    echo "OSS36_ROLLOUT_REMOTE_ROOT must be an absolute normalized path" >&2
    exit 2
  fi
  remote_run_root="${remote_project_root}/${run_root}"
  remote_rollout_step="${remote_run_root}/rollout/${step}"
  remote_named_audit="${remote_project_root}/${raw_audit}"
  remote_default_audit="${remote_project_root}/artifacts/runtime/grpo_reward_audit.jsonl"
  mkdir -p "${rollout_root}/${step}" "$(dirname "$raw_audit")"
  evidence_ready=0
  for attempt in $(seq 1 120); do
    remote_raw_audit=""
    if ssh "$rollout_artifact_source" test -f "$remote_named_audit"; then
      remote_raw_audit="$remote_named_audit"
    elif ssh "$rollout_artifact_source" test -f "$remote_default_audit"; then
      remote_raw_audit="$remote_default_audit"
    fi
    if [[ -n "$remote_raw_audit" ]] && \
      ssh "$rollout_artifact_source" test -d "$remote_rollout_step" && \
      rsync -a \
        "${rollout_artifact_source}:${remote_rollout_step}/" \
        "${rollout_root}/${step}/" && \
      rsync -a \
        "${rollout_artifact_source}:${remote_raw_audit}" \
        "$raw_audit" && \
      "$python_bin" -c \
        'import json,sys
from pathlib import Path
from openstatesearch.eval.full_reward_audit import load_rollout_trajectories
from openstatesearch.eval.grpo_rollout_trend import select_latest_complete_rollout_batch
raw,root,step=Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3])
records=[json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line]
assert sum(record.get("step") == step for record in records) >= 50
trajectories,sources=load_rollout_trajectories(root, step)
selected,_=select_latest_complete_rollout_batch(trajectories,sources,64)
assert len(selected) == 64' \
        "$raw_audit" "$rollout_root" "$step"; then
      evidence_ready=1
      break
    fi
    if (( attempt % 10 == 0 )); then
      echo "waiting for complete remote rollout/audit evidence: attempt ${attempt}/120" >&2
    fi
    sleep 30
  done
  if (( evidence_ready == 0 )); then
    echo "remote rollout/audit evidence did not become complete within one hour" >&2
    exit 2
  fi
  history_synced=0
  for attempt in 1 2 3; do
    if rsync -a \
      "${rollout_artifact_source}:${remote_run_root}/rollout/" \
      "${rollout_root}/"; then
      history_synced=1
      break
    fi
    echo "full rollout-history sync attempt ${attempt}/3 failed" >&2
    sleep 30
  done
  if (( history_synced == 0 )); then
    echo "failed to sync complete rollout history after three attempts" >&2
    exit 2
  fi
fi

for path in "$log_path" "$prompts" "$raw_audit"; do
  if [[ ! -f "$path" ]]; then
    echo "required checkpoint-audit input is missing: $path" >&2
    exit 2
  fi
done
if [[ ! -d "${rollout_root}/${step}" ]]; then
  echo "rollout version ${step} is not present under ${rollout_root}" >&2
  exit 2
fi
if ! rg -q "Step ${step}/[0-9]+ Train step ${step}/[0-9]+ done" "$log_path"; then
  echo "training log does not prove that step ${step} completed" >&2
  exit 2
fi

normalized="${prefix}_reward_audit_through_step${step}_normalized.jsonl"
normalized_summary="${prefix}_reward_audit_through_step${step}_summary.json"
full_output="${prefix}_reward_audit_step${step}_full.jsonl"
full_summary="${prefix}_reward_audit_step${step}_full_summary.json"
curve="${prefix}_curve_through_step${step}.json"
trend="${prefix}_rollout_trend_through_step${step}.json"
adapter_source="artifacts/areal/checkpoints/root/${experiment}/${trial}/default/weight_update_v${step}"
adapter_archive="artifacts/policy_grpo_phasea_step${step}_adapter_seed36"
base_model="artifacts/policy_sft_step200_legalspace20_merged_seed36"

"$python_bin" scripts/normalize_reward_audit.py \
  --source "$raw_audit" \
  --output "$normalized" \
  --summary "$normalized_summary" \
  --sample-size 50 \
  --seed 36

"$python_bin" scripts/build_full_reward_audit.py \
  --rollout-root "$rollout_root" \
  --reward-audit "$raw_audit" \
  --prompts "$prompts" \
  --output "$full_output" \
  --summary "$full_summary" \
  --version "$step" \
  --sample-size 50 \
  --seed 36

"$python_bin" scripts/summarize_grpo_log.py \
  --log "$log_path" \
  --output "$curve" \
  --through-step "$step" \
  --window-size 25

"$python_bin" scripts/summarize_grpo_rollouts.py \
  --rollout-root "$rollout_root" \
  --through-version "$step" \
  --output "$trend" \
  --expected-episodes 64 \
  --window-size 25

"$python_bin" scripts/archive_grpo_adapter.py \
  --source "$adapter_source" \
  --output "$adapter_archive" \
  --base "$base_model" \
  --experiment "$experiment" \
  --trial "$trial" \
  --step "$step" \
  --training-log "$log_path" \
  --training-curve "$curve" \
  --grpo-config configs/grpo_a.yaml \
  --areal-config configs/areal_grpo_lora.yaml \
  --reward-audit-summary "$full_summary" \
  --rollout-trend "$trend"

"$python_bin" -c \
  'import json,sys; full=json.load(open(sys.argv[1])); curve=json.load(open(sys.argv[2])); trend=json.load(open(sys.argv[3])); checkpoint=json.load(open(sys.argv[5])); step=int(sys.argv[4]); assert full["step"] == step and full["sample_size"] == 50; assert full["matching"]["unmatched_raw_records"] == 0; assert full["selected_metrics"]["trajectories"] == 50; assert curve["last_completed_step"] == step and curve["through_step"] == step; assert trend["through_version"] == step; assert checkpoint["step"] == step; print(json.dumps({"step":step,"valid_rate":full["selected_metrics"]["valid_rate"],"mean_reward":full["selected_metrics"]["reward_components"]["total"]["mean"],"curve":sys.argv[2],"trend":sys.argv[3],"reward_audit":sys.argv[1],"checkpoint_manifest":sys.argv[5]}, sort_keys=True))' \
  "$full_summary" "$curve" "$trend" "$step" "$adapter_archive/checkpoint_manifest.json"
