#!/bin/bash
set -u

cd /code/openstatesearch || exit 1
mkdir -p artifacts/runtime/sft_gate_shards artifacts/eval/sft_gate_shards

devices=(0 1 2 3 5 6 7)
starts=(14 27 40 52 64 76 88)
stops=(27 40 52 64 76 88 100)
pids=()

for i in "${!devices[@]}"; do
  device="${devices[$i]}"
  start="${starts[$i]}"
  stop="${stops[$i]}"
  CUDA_VISIBLE_DEVICES="$device" \
  HF_HOME=/code/hf_cache \
  TOKENIZERS_PARALLELISM=false \
  /root/miniforge3/bin/python scripts/run_sft_gate.py \
    --model artifacts/policy_sft_merged_seed36 \
    --prompts data/processed/full_context_seed36/rl_dev.jsonl \
    --retriever-url http://127.0.0.1:8036 \
    --output "artifacts/eval/sft_gate_shards/part_${start}_${stop}.jsonl" \
    --count 100 --start-index "$start" --stop-index "$stop" \
    > "artifacts/runtime/sft_gate_shards/part_${start}_${stop}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
