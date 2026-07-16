#!/bin/bash
# Single-node DeepSeek-V4-Flash with LMCache L1 (CPU) + L2 (LocalDisk).
#
# Usage:
#   ./scripts/serving/single_node_disk.sh
#
# Optional env:
#   MODEL_PATH, PORT, ASCEND_RT_VISIBLE_DEVICES, TENSOR_PARALLEL_SIZE
#   LMCACHE_LOCAL_DISK, LMCACHE_MAX_LOCAL_DISK_SIZE
#   LMCACHE_MAX_LOCAL_CPU_SIZE, LMCACHE_CHUNK_SIZE, LMCACHE_LOG_LEVEL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env/common_env.sh"

export LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU:-True}"
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-70}"

LMCACHE_LOCAL_DISK="${LMCACHE_LOCAL_DISK:-file:///tmp/lmcache_disk}"
export LMCACHE_LOCAL_DISK
export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-500}"
mkdir -p "${LMCACHE_LOCAL_DISK#file://}"

MODEL_PATH="${MODEL_PATH:-/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
MODEL_NAME="${MODEL_NAME:-dsv4}"
PORT="${PORT:-8008}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

echo "Single-node DSv4 + LMCache L2 disk"
echo "  model: ${MODEL_PATH}"
echo "  L1 CPU: max=${LMCACHE_MAX_LOCAL_CPU_SIZE}GB"
echo "  L2 disk: ${LMCACHE_LOCAL_DISK} max=${LMCACHE_MAX_LOCAL_DISK_SIZE}GB"
echo "  NPUs: ${ASCEND_RT_VISIBLE_DEVICES}  TP=${TENSOR_PARALLEL_SIZE}  port=${PORT}"

vllm serve "$MODEL_PATH" \
    --no-enable-prefix-caching \
    --max_model_len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 8192 \
    --served-model-name "$MODEL_NAME" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --api-server-count 1 \
    --max-num-seqs 16 \
    --data-parallel-size 1 \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 128}' \
    --safetensors-load-strategy prefetch \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 1,"method": "mtp", "enforce_eager": true}' \
    --port "$PORT" \
    --no-disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both","kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector"}' \
    --block-size 128 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --async-scheduling \
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}'
