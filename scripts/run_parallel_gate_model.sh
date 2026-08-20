#!/bin/bash
set -u

cd /code/openstatesearch || exit 1
model="${1:?usage: run_parallel_gate_model.sh MODEL TAG [COUNT [RETRIEVER_URL [TEMPERATURE [RETRIEVER_NAME [GENERATION_URL [GENERATION_BACKEND]]]]]]}"
tag="${2:?usage: run_parallel_gate_model.sh MODEL TAG [COUNT [RETRIEVER_URL [TEMPERATURE [RETRIEVER_NAME [GENERATION_URL [GENERATION_BACKEND]]]]]]}"
count="${3:-100}"
retriever_url="${4:-http://127.0.0.1:8036}"
temperature="${5:-0.0}"
retriever_name="${6:-unspecified}"
generation_url="${7:-${OSS36_GENERATION_URL:-}}"
generation_backend="${8:-${OSS36_GENERATION_BACKEND:-vllm}}"
sampling_scheme="${OSS36_SAMPLING_SCHEME:-per_prompt_turn_v1}"
model_manifest="$model/merge_manifest.json"
shard_dir="artifacts/eval/${tag}_shards"
log_dir="artifacts/runtime/${tag}_shards"
mkdir -p "$shard_dir" "$log_dir"

pids=()
model_provenance_args=()
if [[ -f "$model_manifest" ]]; then
  model_provenance_args=(
    --model-provenance-manifest "$model_manifest"
    --require-model-provenance
  )
fi
retriever_provenance_args=(--retriever-name "$retriever_name")
if [[ "$retriever_name" != "unspecified" ]]; then
  retriever_provenance_args+=(--require-retriever-provenance)
fi
linear_attention_args=()
if [[ "${OSS36_DISABLE_FLA:-0}" == "1" ]]; then
  linear_attention_args=(--disable-flash-linear-attention)
fi
generation_args=()
if [[ -n "$generation_url" ]]; then
  generation_args=(
    --generation-url "$generation_url"
    --generation-backend "$generation_backend"
  )
fi

for device in {0..7}; do
  start=$((device * count / 8))
  stop=$(((device + 1) * count / 8))
  CUDA_VISIBLE_DEVICES="$device" \
  HF_HOME=/code/hf_cache \
  TOKENIZERS_PARALLELISM=false \
  /root/miniforge3/bin/python scripts/run_sft_gate.py \
    --model "$model" \
    --prompts data/processed/full_context_seed36/rl_dev.jsonl \
    --retriever-url "$retriever_url" \
    --output "$shard_dir/part_${start}_${stop}.jsonl" \
    --count "$count" --start-index "$start" --stop-index "$stop" \
    --temperature "$temperature" --top-p 0.95 \
    --sampling-scheme "$sampling_scheme" \
    --experiment-name "$tag" \
    --skip-gate-enforcement \
    "${model_provenance_args[@]}" \
    "${retriever_provenance_args[@]}" \
    "${linear_attention_args[@]}" \
    "${generation_args[@]}" \
    > "$log_dir/part_${start}_${stop}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
