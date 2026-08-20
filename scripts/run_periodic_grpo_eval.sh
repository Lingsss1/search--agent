#!/bin/bash
set -euo pipefail

project_root="${OSS36_PROJECT_ROOT:-/code/openstatesearch}"
python_bin="${OSS36_PYTHON:-/root/miniforge3/bin/python}"
base_model="${OSS36_PERIODIC_EVAL_BASE_MODEL:-artifacts/policy_grpo_r21_step1_merged_seed36}"
prompts="${OSS36_PERIODIC_EVAL_PROMPTS:-data/processed/full_context_seed36/rl_dev.jsonl}"
retriever_url="${OSS36_PERIODIC_EVAL_RETRIEVER_URL:-http://10.82.124.28:8036}"
retriever_name="${OSS36_PERIODIC_EVAL_RETRIEVER_NAME:-lrat_hybrid}"
count="${OSS36_PERIODIC_EVAL_COUNT:-50}"
temperature="${OSS36_PERIODIC_EVAL_TEMPERATURE:-1.0}"
server_url="${OSS36_PERIODIC_EVAL_SERVER_URL:-http://10.48.41.83:18084}"
remote_root="${OSS36_PERIODIC_EVAL_REMOTE_ROOT:-/code/openstatesearch}"

experiment="${1:?usage: run_periodic_grpo_eval.sh EXPERIMENT TRIAL STEP}"
trial="${2:?usage: run_periodic_grpo_eval.sh EXPERIMENT TRIAL STEP}"
step="${3:?usage: run_periodic_grpo_eval.sh EXPERIMENT TRIAL STEP}"

if ! [[ "$experiment" =~ ^[A-Za-z0-9_.-]+$ && "$trial" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "EXPERIMENT and TRIAL may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
fi
if ! [[ "$step" =~ ^[1-9][0-9]*$ && "$count" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEP and OSS36_PERIODIC_EVAL_COUNT must be positive integers" >&2
  exit 2
fi

cd "$project_root"
adapter_rel="artifacts/areal/checkpoints/root/${experiment}/${trial}/default/weight_update_v${step}"
adapter_dir="${project_root}/${adapter_rel}"
remote_adapter="${remote_root}/${adapter_rel}"
adapter_weights="${adapter_dir}/adapter_model.safetensors"
adapter_config="${adapter_dir}/adapter_config.json"
tag="grpo_phasea_step${step}_temp10_seed36"
run_root="artifacts/eval/periodic_grpo/step_$(printf '%06d' "$step")"
shard_dir="${run_root}/shards"
log_dir="artifacts/runtime/periodic_grpo/step_$(printf '%06d' "$step")"
output="${run_root}/predictions.jsonl"
lora_name="oss36-phasea-step-${step}"

for path in "$base_model" "$prompts" "$adapter_weights" "$adapter_config"; do
  if [[ ! -e "$path" ]]; then
    echo "periodic-eval input is missing: $path" >&2
    exit 1
  fi
done
mkdir -p "$shard_dir" "$log_dir"

post_json() {
  local endpoint="$1"
  local payload="$2"
  curl -fsS -X POST "${server_url%/}${endpoint}" \
    -H 'Content-Type: application/json' \
    --data "$payload"
}

curl -fsS "${server_url%/}/health" >/dev/null
# A retry of an interrupted evaluation may find the adapter already resident.
post_json /v1/unload_lora_adapter "{\"lora_name\":\"${lora_name}\"}" >/dev/null 2>&1 || true
post_json /v1/load_lora_adapter \
  "{\"lora_name\":\"${lora_name}\",\"lora_path\":\"${remote_adapter}\"}" >/dev/null

cleanup() {
  post_json /v1/unload_lora_adapter "{\"lora_name\":\"${lora_name}\"}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

pids=()
base=$((count / 8))
remainder=$((count % 8))
cursor=0
for shard in $(seq 0 7); do
  size="$base"
  if (( shard < remainder )); then size=$((size + 1)); fi
  start="$cursor"
  stop=$((cursor + size))
  cursor="$stop"
  if (( start == stop )); then continue; fi
  CUDA_VISIBLE_DEVICES="" \
  HF_HOME=/code/hf_cache \
  TOKENIZERS_PARALLELISM=false \
  "$python_bin" scripts/run_sft_gate.py \
    --model "$base_model" \
    --prompts "$prompts" \
    --retriever-url "$retriever_url" \
    --retriever-name "$retriever_name" \
    --require-retriever-provenance \
    --generation-url "$server_url" \
    --generation-backend vllm \
    --generation-model "$lora_name" \
    --model-provenance-manifest "$adapter_weights" \
    --require-model-provenance \
    --output "${shard_dir}/part_${start}_${stop}.jsonl" \
    --count "$count" --start-index "$start" --stop-index "$stop" \
    --state-mode external_state \
    --memory-token-budget 8192 \
    --generation-token-budget 8192 \
    --max-action-tokens 256 \
    --temperature "$temperature" --top-p 0.95 --top-k 20 \
    --sampling-seed 36 --sampling-scheme per_prompt_turn_v1 \
    --experiment-name "$tag" \
    --skip-gate-enforcement \
    > "${log_dir}/part_${start}_${stop}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if (( status != 0 )); then
  echo "one or more periodic-eval shards failed; inspect ${log_dir}" >&2
  exit 1
fi

"$python_bin" scripts/merge_sft_gate_shards.py \
  --prompts "$prompts" \
  --inputs "${shard_dir}"/part_*.jsonl \
  --output "$output" \
  --count "$count" \
  --expected-experiment "$tag" \
  --expected-state-mode external_state \
  --expected-retriever "$retriever_name" \
  --expected-memory-token-budget 8192 \
  --require-retriever-provenance \
  --require-model-provenance \
  --require-run-config \
  > "${log_dir}/merge.log" 2>&1

previous_step=0
previous_output=""
for candidate in artifacts/eval/periodic_grpo/step_*/predictions.jsonl; do
  [[ -f "$candidate" && -f "${candidate}.manifest.json" ]] || continue
  candidate_dir="$(basename "$(dirname "$candidate")")"
  candidate_step="${candidate_dir#step_}"
  candidate_step=$((10#$candidate_step))
  if (( candidate_step < step && candidate_step > previous_step )); then
    previous_step="$candidate_step"
    previous_output="$candidate"
  fi
done
if (( previous_step > 0 )); then
  comparison="${run_root}/paired_vs_step_$(printf '%06d' "$previous_step").json"
  "$python_bin" scripts/compare_gate_runs.py \
    --baseline "$previous_output" \
    --target "$output" \
    --output "$comparison" \
    --bootstrap-samples 20000 \
    > "${log_dir}/paired_comparison.log" 2>&1
fi

"$python_bin" scripts/summarize_periodic_grpo_evals.py \
  --root artifacts/eval/periodic_grpo \
  --output artifacts/eval/periodic_grpo/index.json

echo "periodic GRPO evaluation complete: step=${step} output=${output}"
