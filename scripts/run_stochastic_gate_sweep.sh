#!/usr/bin/env bash
set -euo pipefail

model="artifacts/policy_sft_step200_legalspace20_merged_seed36"
prompts="data/processed/full_context_seed36/rl_dev.jsonl"
retriever_url="http://10.82.124.28:8036"
count=32
temperatures=(0.2 0.4 0.6 0.8 1.0)
labels=(02 04 06 08 10)
pids=()

mkdir -p artifacts/eval artifacts/runtime

for index in "${!temperatures[@]}"; do
  temperature="${temperatures[$index]}"
  label="${labels[$index]}"
  gpu="$index"
  output="artifacts/eval/stochastic_gate_sft_step200_temp${label}_seed36.jsonl"
  log="artifacts/runtime/stochastic_gate_sft_step200_temp${label}_seed36.log"
  CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HOME=/code/hf_cache \
    TOKENIZERS_PARALLELISM=false \
    /root/miniforge3/bin/python scripts/run_sft_gate.py \
      --model "$model" \
      --prompts "$prompts" \
      --retriever-url "$retriever_url" \
      --output "$output" \
      --count "$count" \
      --temperature "$temperature" \
      --top-p 0.95 \
      --sampling-seed 36 \
      --state-mode external_state \
      --memory-token-budget 8192 \
      --generation-token-budget 8192 \
      --max-action-tokens 256 \
      --experiment-name "stochastic_gate_temp${label}" \
      --retriever-name R4_lrat_hybrid \
      --skip-gate-enforcement \
      >>"$log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
