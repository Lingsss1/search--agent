#!/bin/bash
set -euo pipefail

project_root="${OSS36_PROJECT_ROOT:-/code/oss36_eval}"
retriever_root="${OSS36_RETRIEVER_ROOT:-/code/oss36_retriever}"
python_bin="${OSS36_PYTHON:-/root/miniforge3/envs/agent_env/bin/python}"
hf_home="${OSS36_HF_HOME:-/code/hf_cache}"
service_gpu="${OSS36_RETRIEVER_GPU:-0}"
listen_host="${OSS36_RETRIEVER_HOST:-0.0.0.0}"
base_model="${OSS36_BASE_EMBEDDING_MODEL:-/code/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3}"
lrat_model="${OSS36_LRAT_MODEL:-${retriever_root}/model}"
runtime_dir="${project_root}/artifacts/runtime/formal_retriever_services"

base_corpus="${retriever_root}/data/corpus.jsonl"
base_index="${project_root}/artifacts/indexes/full_context_seed36/r1_base.npz"
lrat_corpus="${retriever_root}/data/corpus.jsonl"
lrat_index="${retriever_root}/artifacts/index.npz"
browsecomp_corpus="${project_root}/data/processed/browsecomp_plus/corpus.jsonl"
browsecomp_index="${project_root}/artifacts/indexes/browsecomp_plus/r3_trained_lrat_seed36.npz"

required_files=(
  "$base_corpus"
  "$base_index"
  "${base_index}.manifest.json"
  "$lrat_corpus"
  "$lrat_index"
  "${lrat_index}.manifest.json"
  "$browsecomp_corpus"
  "$browsecomp_index"
  "${browsecomp_index}.manifest.json"
)
for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "required formal-evaluation input is missing: $path" >&2
    exit 2
  fi
done
for path in "$base_model" "$lrat_model"; do
  if [[ ! -d "$path" ]]; then
    echo "required dense model directory is missing: $path" >&2
    exit 2
  fi
done

mkdir -p "$runtime_dir"
started_pids=()
cleanup_failed_start() {
  status=$?
  if (( status != 0 )); then
    for pid in "${started_pids[@]}"; do
      kill -TERM "$pid" 2>/dev/null || true
    done
  fi
  exit "$status"
}
trap cleanup_failed_start EXIT

for port in 8035 8036 8037; do
  if curl -sS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "port ${port} already has a retriever service; refusing to mix runs" >&2
    exit 2
  fi
done

start_service() {
  local name="$1"
  local port="$2"
  local corpus="$3"
  local model="$4"
  local index="$5"
  local log_path="${runtime_dir}/${name}.log"
  local pid_path="${runtime_dir}/${name}.pid"

  nohup env \
    CUDA_VISIBLE_DEVICES="$service_gpu" \
    HF_HOME="$hf_home" \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="$project_root" \
    "$python_bin" -m openstatesearch.retriever.service \
      --corpus "$corpus" \
      --host "$listen_host" \
      --port "$port" \
      --name "$name" \
      --dense-model "$model" \
      --dense-index "$index" \
      --device cuda \
      --dtype bfloat16 \
      >"$log_path" 2>&1 </dev/null &
  local pid=$!
  started_pids+=("$pid")
  printf '%s\n' "$pid" >"$pid_path"

  local ready=0
  for _ in $(seq 1 180); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "${name} exited during startup; inspect ${log_path}" >&2
      return 1
    fi
    if "$python_bin" -c \
      'import json,sys,urllib.request; value=json.load(urllib.request.urlopen(sys.argv[1], timeout=3)); assert value["name"] == sys.argv[2]; assert len(value["provenance_sha256"]) == 64' \
      "http://127.0.0.1:${port}/provenance" "$name" \
      >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 2
  done
  if (( ready == 0 )); then
    echo "${name} did not become ready; inspect ${log_path}" >&2
    return 1
  fi

  curl -fsS "http://127.0.0.1:${port}/health" \
    >"${runtime_dir}/${name}.health.json"
  curl -fsS "http://127.0.0.1:${port}/provenance" \
    >"${runtime_dir}/${name}.provenance.json"
  echo "started ${name} pid=${pid} url=http://127.0.0.1:${port}"
}

start_service base_hybrid 8035 "$base_corpus" "$base_model" "$base_index"
start_service lrat_hybrid 8036 "$lrat_corpus" "$lrat_model" "$lrat_index"
start_service browsecomp_plus_lrat_hybrid 8037 "$browsecomp_corpus" "$lrat_model" "$browsecomp_index"

trap - EXIT
echo "formal retrievers ready; reserve GPU ${service_gpu} and run policy workers on the other GPUs"
