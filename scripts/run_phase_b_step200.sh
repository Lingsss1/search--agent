#!/bin/bash
set -euo pipefail

project_root="${OSS36_PROJECT_ROOT:-/code/openstatesearch}"
python_bin="${OSS36_PYTHON:-/root/miniforge3/bin/python}"
retriever_url="${OSS36_PHASE_B_RETRIEVER_URL:-http://10.82.124.28:8036}"
cd "$project_root"

base_model="artifacts/policy_sft_step200_legalspace20_merged_seed36"
adapter="artifacts/policy_grpo_phasea_step200_adapter_seed36"
merged_model="artifacts/policy_grpo_a_merged"
checkpoint_manifest="${adapter}/checkpoint_manifest.json"
pool="data/processed/full_context_seed36/rl_phase_b_pool.jsonl"
hard="data/processed/full_context_seed36/rl_phase_b_hard.jsonl"
merged_pool="artifacts/eval/phase_b_hard_pool_sft_seed36.jsonl"

for path in "$checkpoint_manifest" "$pool" "$hard" "${hard}.manifest.json"; do
  if [[ ! -f "$path" ]]; then
    echo "required audited Phase-B input is missing: $path" >&2
    exit 2
  fi
done

"$python_bin" scripts/verify_phase_b_pool.py \
  --pool "$pool" \
  --merged "$merged_pool" \
  --hard "$hard" \
  --phase-a data/processed/full_context_seed36/rl_train.jsonl \
  --output artifacts/audits/phase_b_hard_pool_verification.json

"$python_bin" -c \
  'import json,sys; value=json.load(open(sys.argv[1])); assert value["kind"] == "grpo_lora_checkpoint" and value["step"] == 200; assert value["output"]["path"] == sys.argv[2]' \
  "$checkpoint_manifest" "$adapter"

if [[ ! -f "${merged_model}/merge_manifest.json" ]]; then
  "$python_bin" scripts/merge_adapter.py \
    --base "$base_model" \
    --adapter "$adapter" \
    --output "$merged_model"
fi

"$python_bin" -c \
  'import json,sys; value=json.load(open(sys.argv[1])); assert value["adapter"]["path"] == sys.argv[2]; assert value["base"]["path"] == sys.argv[3]; assert value["output"]["path"] == sys.argv[4]' \
  "${merged_model}/merge_manifest.json" "$adapter" "$base_model" "$merged_model"

exec "$python_bin" scripts/train_grpo.py \
  --config configs/grpo_b.yaml \
  --areal-config configs/areal_grpo_lora.yaml \
  --corpus data/processed/full_context_seed36/in_domain_corpus.jsonl \
  --dense-model artifacts/retriever_lrat_seed36 \
  --dense-index artifacts/indexes/full_context_seed36/r3_trained_lrat_seed36.npz \
  --retriever-url "$retriever_url" \
  --retriever-name lrat_hybrid \
  --require-retriever-provenance \
  --experiment-name oss36-grpo-b \
  --trial-name phaseb-step200
