#!/bin/bash
set -u

project_root="${OSS36_PROJECT_ROOT:-/code/openstatesearch}"
python_bin="${OSS36_PYTHON:-/root/miniforge3/bin/python}"
hf_home="${OSS36_HF_HOME:-/code/hf_cache}"
gpu_devices_raw="${OSS36_GPU_DEVICES:-0,1,2,3,4,5,6,7}"
cd "$project_root" || exit 1
model="${1:?usage: run_parallel_phase_b_pool.sh MODEL RETRIEVER_URL TAG}"
retriever_url="${2:?usage: run_parallel_phase_b_pool.sh MODEL RETRIEVER_URL TAG}"
tag="${3:?usage: run_parallel_phase_b_pool.sh MODEL RETRIEVER_URL TAG}"
prompts="data/processed/full_context_seed36/rl_phase_b_pool.jsonl"
shard_dir="artifacts/eval/${tag}_shards"
log_dir="artifacts/runtime/${tag}_shards"
mkdir -p "$shard_dir" "$log_dir"

starts=(0 563 1126 1689 2252 2814 3376 3938)
stops=(563 1126 1689 2252 2814 3376 3938 4500)
status=0
IFS=',' read -r -a gpu_devices <<< "$gpu_devices_raw"
if [ "${#gpu_devices[@]}" -eq 0 ]; then
  echo "OSS36_GPU_DEVICES must contain at least one GPU" >&2
  exit 2
fi
for wave_start in $(seq 0 "${#gpu_devices[@]}" 7); do
  pids=()
  for slot in "${!gpu_devices[@]}"; do
    shard=$((wave_start + slot))
    if [ "$shard" -ge 8 ]; then
      break
    fi
    device="${gpu_devices[$slot]}"
    start="${starts[$shard]}"
    stop="${stops[$shard]}"
    CUDA_VISIBLE_DEVICES="$device" \
    HF_HOME="$hf_home" \
    TOKENIZERS_PARALLELISM=false \
    "$python_bin" scripts/run_sft_gate.py \
      --model "$model" \
      --prompts "$prompts" \
      --retriever-url "$retriever_url" \
      --output "$shard_dir/part_${start}_${stop}.jsonl" \
      --count 4500 --start-index "$start" --stop-index "$stop" \
      --state-mode external_state \
      --memory-token-budget 8192 \
      --generation-token-budget 8192 \
      --max-action-tokens 256 \
      --experiment-name phase_b_hard_pool_sft \
      --retriever-name R4_lrat_hybrid \
      --skip-gate-enforcement \
      > "$log_dir/part_${start}_${stop}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
done
if [ "$status" -ne 0 ]; then
  exit "$status"
fi

"$python_bin" scripts/merge_sft_gate_shards.py \
  --prompts "$prompts" \
  --inputs "$shard_dir"/part_*.jsonl \
  --output "artifacts/eval/${tag}.jsonl" \
  --count 4500 \
  --expected-experiment phase_b_hard_pool_sft \
  --expected-state-mode external_state \
  --expected-retriever R4_lrat_hybrid \
  --expected-memory-token-budget 8192

"$python_bin" scripts/select_phase_b_hard.py \
  --pool "$prompts" \
  --trajectories "artifacts/eval/${tag}.jsonl" \
  --output data/processed/full_context_seed36/rl_phase_b_hard.jsonl \
  --per-dataset 1000 \
  --seed 36
