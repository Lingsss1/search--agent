#!/bin/bash
set -euo pipefail

project_root="${OSS36_PROJECT_ROOT:-/code/openstatesearch}"
python_bin="${OSS36_PYTHON:-/root/miniforge3/bin/python}"
hf_home="${OSS36_HF_HOME:-/code/hf_cache}"
gpu_devices_raw="${OSS36_GPU_DEVICES:-0,1,2,3,4,5,6,7}"
generation_url="${OSS36_GENERATION_URL:-}"
generation_backend="${OSS36_GENERATION_BACKEND:-vllm}"
cd "$project_root"

experiment="${1:?usage: run_parallel_eval_experiment.sh EXPERIMENT MODEL PROMPTS RETRIEVER_URL MEMORY_BUDGET COUNT TAG}"
model="${2:?usage: run_parallel_eval_experiment.sh EXPERIMENT MODEL PROMPTS RETRIEVER_URL MEMORY_BUDGET COUNT TAG}"
prompts="${3:?usage: run_parallel_eval_experiment.sh EXPERIMENT MODEL PROMPTS RETRIEVER_URL MEMORY_BUDGET COUNT TAG}"
retriever_url="${4:?usage: run_parallel_eval_experiment.sh EXPERIMENT MODEL PROMPTS RETRIEVER_URL MEMORY_BUDGET COUNT TAG}"
memory_budget="${5:?usage: run_parallel_eval_experiment.sh EXPERIMENT MODEL PROMPTS RETRIEVER_URL MEMORY_BUDGET COUNT TAG}"
count="${6:?usage: run_parallel_eval_experiment.sh EXPERIMENT MODEL PROMPTS RETRIEVER_URL MEMORY_BUDGET COUNT TAG}"
tag="${7:?usage: run_parallel_eval_experiment.sh EXPERIMENT MODEL PROMPTS RETRIEVER_URL MEMORY_BUDGET COUNT TAG}"

case "$experiment" in
  A) state_mode=transcript; retriever_name=base_hybrid ;;
  B) state_mode=external_state; retriever_name=base_hybrid ;;
  C) state_mode=transcript; retriever_name=lrat_hybrid ;;
  D) state_mode=external_state; retriever_name=lrat_hybrid ;;
  E) state_mode=transcript; retriever_name=lrat_hybrid ;;
  F) state_mode=external_state; retriever_name=lrat_hybrid ;;
  *) echo "EXPERIMENT must be one of A,B,C,D,E,F" >&2; exit 2 ;;
esac
case "$memory_budget" in
  4096|8192) ;;
  *) echo "MEMORY_BUDGET must be 4096 or 8192" >&2; exit 2 ;;
esac
if ! [[ "$count" =~ ^[1-9][0-9]*$ ]]; then
  echo "COUNT must be a positive integer" >&2
  exit 2
fi
if ! [[ "$tag" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "TAG may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
fi
if [[ ! -f "$prompts" || ! -d "$model" ]]; then
  echo "MODEL directory and PROMPTS file must exist" >&2
  exit 2
fi
model_manifest="$model/merge_manifest.json"
if [[ ! -f "$model_manifest" ]]; then
  model_manifest="$model/model_provenance.json"
fi
if [[ ! -f "$model_manifest" ]]; then
  echo "MODEL requires merge_manifest.json or model_provenance.json" >&2
  exit 2
fi

run_root="artifacts/eval/matrix/${experiment}/${tag}/budget_${memory_budget}"
shard_dir="${run_root}/shards"
log_dir="artifacts/runtime/eval_matrix/${experiment}/${tag}/budget_${memory_budget}"
mkdir -p "$shard_dir" "$log_dir"

base=$((count / 8))
remainder=$((count % 8))
cursor=0
starts=()
stops=()
for shard in {0..7}; do
  size="$base"
  if (( shard < remainder )); then
    size=$((size + 1))
  fi
  starts+=("$cursor")
  next=$((cursor + size))
  stops+=("$next")
  cursor="$next"
done

status=0
generation_args=()
if [[ -n "$generation_url" ]]; then
  generation_args=(
    --generation-url "$generation_url"
    --generation-backend "$generation_backend"
  )
fi
IFS=',' read -r -a gpu_devices <<< "$gpu_devices_raw"
if (( ${#gpu_devices[@]} == 0 )); then
  echo "OSS36_GPU_DEVICES must contain at least one GPU" >&2
  exit 2
fi
for wave_start in $(seq 0 "${#gpu_devices[@]}" 7); do
  pids=()
  for slot in "${!gpu_devices[@]}"; do
    shard=$((wave_start + slot))
    if (( shard >= 8 )); then
      break
    fi
    device="${gpu_devices[$slot]}"
    start="${starts[$shard]}"
    stop="${stops[$shard]}"
    if (( start == stop )); then
      continue
    fi
    CUDA_VISIBLE_DEVICES="$device" \
    HF_HOME="$hf_home" \
    TOKENIZERS_PARALLELISM=false \
    "$python_bin" scripts/run_sft_gate.py \
      --model "$model" \
      --prompts "$prompts" \
      --retriever-url "$retriever_url" \
      --retriever-name "$retriever_name" \
      --require-retriever-provenance \
      --model-provenance-manifest "$model_manifest" \
      --require-model-provenance \
      --experiment-name "$experiment" \
      --state-mode "$state_mode" \
      --memory-token-budget "$memory_budget" \
      --output "${shard_dir}/part_${start}_${stop}.jsonl" \
      --count "$count" --start-index "$start" --stop-index "$stop" \
      --skip-gate-enforcement \
      "${generation_args[@]}" \
      > "${log_dir}/part_${start}_${stop}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
done
if (( status != 0 )); then
  echo "one or more evaluation shards failed; inspect ${log_dir}" >&2
  exit "$status"
fi

"$python_bin" scripts/merge_sft_gate_shards.py \
  --prompts "$prompts" \
  --inputs "${shard_dir}"/part_*.jsonl \
  --output "${run_root}/predictions.jsonl" \
  --count "$count" \
  --expected-experiment "$experiment" \
  --expected-state-mode "$state_mode" \
  --expected-retriever "$retriever_name" \
  --expected-memory-token-budget "$memory_budget" \
  --require-retriever-provenance \
  --require-model-provenance \
  --require-run-config
