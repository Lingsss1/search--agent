#!/bin/bash
set -euo pipefail

project_root="${OSS36_PROJECT_ROOT:-/code/openstatesearch}"
python_bin="${OSS36_PYTHON:-/root/miniforge3/bin/python}"
# GPU 0 hosts the three frozen retriever services on the evaluation machine.
# Keep policy inference on GPUs 1-7 unless the caller explicitly supplies a
# different non-overlapping device list.
export OSS36_GPU_DEVICES="${OSS36_GPU_DEVICES:-1,2,3,4,5,6,7}"
sft_generation_url="${OSS36_SFT_GENERATION_URL:-}"
grpo_generation_url="${OSS36_GRPO_GENERATION_URL:-}"
cd "$project_root"

sft_model="${1:?usage: run_formal_eval_suite.sh SFT_MODEL GRPO_MODEL MAIN_PROMPTS BASE_URL LRAT_URL BROWSECOMP_PROMPTS BROWSECOMP_URL XBENCH_PROMPTS BROWSECOMP_ZH_PROMPTS LIVE_URL TAG}"
grpo_model="${2:?usage: run_formal_eval_suite.sh SFT_MODEL GRPO_MODEL MAIN_PROMPTS BASE_URL LRAT_URL BROWSECOMP_PROMPTS BROWSECOMP_URL XBENCH_PROMPTS BROWSECOMP_ZH_PROMPTS LIVE_URL TAG}"
main_prompts="${3:?usage: run_formal_eval_suite.sh SFT_MODEL GRPO_MODEL MAIN_PROMPTS BASE_URL LRAT_URL BROWSECOMP_PROMPTS BROWSECOMP_URL XBENCH_PROMPTS BROWSECOMP_ZH_PROMPTS LIVE_URL TAG}"
base_url="${4:?usage: run_formal_eval_suite.sh SFT_MODEL GRPO_MODEL MAIN_PROMPTS BASE_URL LRAT_URL BROWSECOMP_PROMPTS BROWSECOMP_URL XBENCH_PROMPTS BROWSECOMP_ZH_PROMPTS LIVE_URL TAG}"
lrat_url="${5:?usage: run_formal_eval_suite.sh SFT_MODEL GRPO_MODEL MAIN_PROMPTS BASE_URL LRAT_URL BROWSECOMP_PROMPTS BROWSECOMP_URL XBENCH_PROMPTS BROWSECOMP_ZH_PROMPTS LIVE_URL TAG}"
browsecomp_prompts="${6:?usage: run_formal_eval_suite.sh SFT_MODEL GRPO_MODEL MAIN_PROMPTS BASE_URL LRAT_URL BROWSECOMP_PROMPTS BROWSECOMP_URL XBENCH_PROMPTS BROWSECOMP_ZH_PROMPTS LIVE_URL TAG}"
browsecomp_url="${7:?usage: run_formal_eval_suite.sh SFT_MODEL GRPO_MODEL MAIN_PROMPTS BASE_URL LRAT_URL BROWSECOMP_PROMPTS BROWSECOMP_URL XBENCH_PROMPTS BROWSECOMP_ZH_PROMPTS LIVE_URL TAG}"
xbench_prompts="${8:?usage: run_formal_eval_suite.sh SFT_MODEL GRPO_MODEL MAIN_PROMPTS BASE_URL LRAT_URL BROWSECOMP_PROMPTS BROWSECOMP_URL XBENCH_PROMPTS BROWSECOMP_ZH_PROMPTS LIVE_URL TAG}"
browsecomp_zh_prompts="${9:?usage: run_formal_eval_suite.sh SFT_MODEL GRPO_MODEL MAIN_PROMPTS BASE_URL LRAT_URL BROWSECOMP_PROMPTS BROWSECOMP_URL XBENCH_PROMPTS BROWSECOMP_ZH_PROMPTS LIVE_URL TAG}"
live_url="${10:?usage: run_formal_eval_suite.sh SFT_MODEL GRPO_MODEL MAIN_PROMPTS BASE_URL LRAT_URL BROWSECOMP_PROMPTS BROWSECOMP_URL XBENCH_PROMPTS BROWSECOMP_ZH_PROMPTS LIVE_URL TAG}"
tag="${11:?usage: run_formal_eval_suite.sh SFT_MODEL GRPO_MODEL MAIN_PROMPTS BASE_URL LRAT_URL BROWSECOMP_PROMPTS BROWSECOMP_URL XBENCH_PROMPTS BROWSECOMP_ZH_PROMPTS LIVE_URL TAG}"

for path in "$sft_model" "$grpo_model"; do
  if [[ ! -d "$path" ]]; then
    echo "model directory is missing: $path" >&2
    exit 2
  fi
done
for path in "$main_prompts" "$browsecomp_prompts" "$xbench_prompts" "$browsecomp_zh_prompts"; do
  if [[ ! -f "$path" ]]; then
    echo "prompt file is missing: $path" >&2
    exit 2
  fi
done
for url in "$base_url" "$lrat_url" "$browsecomp_url" "$live_url"; do
  curl -fsS "${url%/}/provenance" >/dev/null
done
for url in "$sft_generation_url" "$grpo_generation_url"; do
  if [[ -n "$url" ]]; then
    curl -fsS "${url%/}/health" >/dev/null
  fi
done

for experiment in A B C D E F; do
  case "$experiment" in
    A|B) model="$sft_model"; retriever_url="$base_url"; generation_url="$sft_generation_url" ;;
    C|D) model="$sft_model"; retriever_url="$lrat_url"; generation_url="$sft_generation_url" ;;
    E|F) model="$grpo_model"; retriever_url="$lrat_url"; generation_url="$grpo_generation_url" ;;
  esac
  for budget in 4096 8192; do
    OSS36_GENERATION_URL="$generation_url" /bin/bash scripts/run_parallel_eval_experiment.sh \
      "$experiment" "$model" "$main_prompts" "$retriever_url" \
      "$budget" 1500 "$tag"
  done
done

OSS36_GENERATION_URL="$grpo_generation_url" /bin/bash scripts/run_parallel_eval_dataset.sh \
  browsecomp_plus "$grpo_model" "$browsecomp_prompts" "$browsecomp_url" \
  browsecomp_plus_lrat_hybrid external_state 8192 830 "$tag"

OSS36_GENERATION_URL="$grpo_generation_url" /bin/bash scripts/run_parallel_eval_dataset.sh \
  xbench_deepsearch "$grpo_model" "$xbench_prompts" "$live_url" \
  live_web_duckduckgo external_state 8192 100 "$tag"

OSS36_GENERATION_URL="$grpo_generation_url" /bin/bash scripts/run_parallel_eval_dataset.sh \
  browsecomp_zh "$grpo_model" "$browsecomp_zh_prompts" "$live_url" \
  live_web_duckduckgo external_state 8192 289 "$tag"

xbench_manifest="artifacts/eval/datasets/xbench_deepsearch/${tag}/budget_8192/predictions.jsonl.manifest.json"
browsecomp_zh_manifest="artifacts/eval/datasets/browsecomp_zh/${tag}/budget_8192/predictions.jsonl.manifest.json"
"$python_bin" scripts/aggregate_eval_datasets.py \
  --input "xbench_deepsearch=${xbench_manifest}" \
  --input "browsecomp_zh=${browsecomp_zh_manifest}" \
  --output "artifacts/eval/datasets/chinese_test/${tag}/budget_8192/predictions.jsonl" \
  --expected-rows 389

"$python_bin" scripts/summarize_eval_matrix.py \
  --matrix-root artifacts/eval/matrix \
  --tag "$tag" \
  --output-dir "artifacts/eval/formal_summary/${tag}" \
  --failure-limit 50 \
  --seed 36

echo "formal evaluation suite complete: ${tag}"
