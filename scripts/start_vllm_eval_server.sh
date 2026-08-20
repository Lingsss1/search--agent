#!/bin/bash
set -euo pipefail

project_root="${OSS36_PROJECT_ROOT:-/code/openstatesearch}"
python_bin="${OSS36_VLLM_PYTHON:-$project_root/vendor/AReaL/.venv-vllm/bin/python}"
hf_home="${OSS36_HF_HOME:-/code/hf_cache}"
cd "$project_root"

model="${1:?usage: start_vllm_eval_server.sh MODEL PORT [GPU_DEVICES [TP_SIZE [GPU_MEMORY_FRACTION [SERVED_MODEL_NAME]]]]}"
port="${2:?usage: start_vllm_eval_server.sh MODEL PORT [GPU_DEVICES [TP_SIZE [GPU_MEMORY_FRACTION [SERVED_MODEL_NAME]]]]}"
gpu_devices="${3:-0,1,2,3}"
tp_size="${4:-4}"
memory_fraction="${5:-0.80}"
served_model_name="${6:-$model}"
disable_fi_allreduce_fusion="${OSS36_DISABLE_FLASHINFER_ALLREDUCE_FUSION:-1}"

if [[ ! -d "$model" ]]; then
  echo "model directory is missing: $model" >&2
  exit 2
fi
if ! [[ "$port" =~ ^[1-9][0-9]*$ && "$tp_size" =~ ^[1-9][0-9]*$ ]]; then
  echo "PORT and TP_SIZE must be positive integers" >&2
  exit 2
fi
if [[ "$disable_fi_allreduce_fusion" != "0" && "$disable_fi_allreduce_fusion" != "1" ]]; then
  echo "OSS36_DISABLE_FLASHINFER_ALLREDUCE_FUSION must be 0 or 1" >&2
  exit 2
fi
IFS=',' read -r -a devices <<< "$gpu_devices"
if (( ${#devices[@]} != tp_size )); then
  echo "GPU_DEVICES count must equal TP_SIZE" >&2
  exit 2
fi
model_manifest="$model/merge_manifest.json"
if [[ ! -f "$model_manifest" ]]; then
  model_manifest="$model/model_provenance.json"
fi
if [[ ! -f "$model_manifest" ]]; then
  echo "model requires merge_manifest.json or model_provenance.json" >&2
  exit 2
fi

runtime_dir="artifacts/runtime/vllm_eval_servers"
mkdir -p "$runtime_dir"
launch_manifest="$runtime_dir/port_${port}_launch.json"
"$python_bin" - "$launch_manifest" "$model" "$model_manifest" "$port" \
  "$gpu_devices" "$tp_size" "$memory_fraction" "$served_model_name" \
  "$disable_fi_allreduce_fusion" <<'PY'
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import vllm

(
    output,
    model,
    model_manifest,
    port,
    devices,
    tp_size,
    memory_fraction,
    served_model_name,
    disable_fi_allreduce_fusion,
) = sys.argv[1:]
manifest_path = Path(model_manifest)
digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
value = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "model": model,
    "model_manifest": str(manifest_path),
    "model_manifest_sha256": digest,
    "served_model_name": served_model_name,
    "port": int(port),
    "gpu_devices": devices.split(","),
    "tensor_parallel_size": int(tp_size),
    "gpu_memory_fraction": float(memory_fraction),
    "gdn_prefill_backend": "triton",
    "vllm_plugins": [],
    "flashinfer_allreduce_fusion_enabled": disable_fi_allreduce_fusion == "0",
    "python": platform.python_version(),
    "vllm": vllm.__version__,
}
Path(output).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
PY

compilation_args=()
if [[ "$disable_fi_allreduce_fusion" == "1" ]]; then
  compilation_args=(
    --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'
  )
fi

venv_bin="$(dirname "$python_bin")"
exec env \
  PATH="$venv_bin:$PATH" \
  CUDA_VISIBLE_DEVICES="$gpu_devices" \
  HF_HOME="$hf_home" \
  TOKENIZERS_PARALLELISM=false \
  VLLM_PLUGINS="" \
  "$python_bin" -m vllm.entrypoints.openai.api_server \
    --model "$model" \
    --served-model-name "$served_model_name" \
    --tensor-parallel-size "$tp_size" \
    --host 0.0.0.0 \
    --port "$port" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization "$memory_fraction" \
    --trust-remote-code \
    --generation-config vllm \
    --gdn-prefill-backend triton \
    "${compilation_args[@]}"
