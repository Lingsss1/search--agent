#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
ray_bin="/code/openstatesearch/vendor/AReaL/.venv-vllm/bin/ray"
project_root="/code/openstatesearch"
head_ip="${OSS36_RAY_HEAD_IP:-10.82.123.139}"
head_port="${OSS36_RAY_HEAD_PORT:-26379}"
h800_ip="${OSS36_H800_NODE_IP:-10.48.41.83}"

if [[ ! -x "${ray_bin}" ]]; then
  echo "Ray executable is missing: ${ray_bin}" >&2
  exit 2
fi

# Ray launchers and the RPC commands they fork must resolve the same Python and
# editable source tree on both machines.
export PATH="${project_root}/vendor/AReaL/.venv-vllm/bin:${PATH}"
export PYTHONPATH="${project_root}:${project_root}/vendor/AReaL${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="/code/hf_cache"
export TOKENIZERS_PARALLELISM="false"

case "${mode}" in
  head)
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    exec "${ray_bin}" start \
      --head \
      --node-ip-address="${head_ip}" \
      --port="${head_port}" \
      --num-gpus=8 \
      --include-dashboard=false \
      --disable-usage-stats \
      --block
    ;;
  worker)
    # Only H800 0-3 are admitted to this cluster. GPU4 belongs to another
    # process and the remaining cards stay outside the formal TP4 allocation.
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    export AREAL_DISK_WEIGHT_SOURCE="root@${head_ip}"
    export AREAL_DISK_WEIGHT_SSH_PORT="22222"
    export AREAL_DISK_WEIGHT_SYNC_TIMEOUT="300"
    export NO_PROXY="127.0.0.1,localhost,${head_ip},10.82.124.28,${h800_ip}"
    export no_proxy="${NO_PROXY}"
    exec "${ray_bin}" start \
      --address="${head_ip}:${head_port}" \
      --node-ip-address="${h800_ip}" \
      --num-gpus=4 \
      --disable-usage-stats \
      --block
    ;;
  status)
    exec "${ray_bin}" status --address="${head_ip}:${head_port}"
    ;;
  *)
    echo "usage: $0 {head|worker|status}" >&2
    exit 2
    ;;
esac
