#!/bin/bash
# Shared env for 2-node DeepSeek-V4 PD disaggregation.
# Source before launch scripts on either node.
#
#   prefiller+proxy: PREFILL_HOST (default 192.168.0.223)
#   decoder:         DECODE_HOST (default 192.168.0.110)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Load pinned versions if present.
if [[ -f "${REPO_ROOT}/versions.env" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/versions.env"
fi

export PREFILL_HOST="${PREFILL_HOST:-192.168.0.223}"
export DECODE_HOST="${DECODE_HOST:-192.168.0.110}"
export PROXY_HOST="${PROXY_HOST:-${PREFILL_HOST}}"
export DECODE_BIND_HOST="${DECODE_BIND_HOST:-${DECODE_HOST}}"

export NIC_NAME="${NIC_NAME:-enp23s0f3}"

export MODEL_PATH="${MODEL_PATH:-/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
export MODEL_NAME="${MODEL_NAME:-dsv4}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export PREFILL_NPUS="${PREFILL_NPUS:-0,1,2,3,4,5,6,7}"
export DECODE_NPUS="${DECODE_NPUS:-0,1,2,3,4,5,6,7}"

export PREFILL_PORT="${PREFILL_PORT:-7100}"
export DECODE_PORT="${DECODE_PORT:-7200}"
export PROXY_PORT="${PROXY_PORT:-9100}"
export PROXY_ZMQ_PORT="${PROXY_ZMQ_PORT:-7500}"

export PD_BUFFER_SIZE="${PD_BUFFER_SIZE:-1073741824}"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-1024}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
export SAVE_ONLY_FIRST_RANK="${SAVE_ONLY_FIRST_RANK:-true}"
export SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-360}"

# LMCache examples (disagg proxy) live in the submodule.
export LMCACHE_ROOT="${LMCACHE_ROOT:-${CONTAINER_LMCACHE_MOUNT:-/workspace/LMCache}}"
