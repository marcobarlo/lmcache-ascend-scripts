#!/bin/bash
# Single-node DeepSeek-V4-Flash with LMCache L1 (CPU) only — no LocalDisk L2.
#
# Usage:
#   ./scripts/serving/single_node.sh
#
# Optional env: MODEL_PATH, PORT, ASCEND_RT_VISIBLE_DEVICES, TENSOR_PARALLEL_SIZE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env/common_env.sh"

# L1 CPU only (no LMCACHE_LOCAL_DISK)
export LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU:-True}"
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-70}"
unset LMCACHE_LOCAL_DISK 2>/dev/null || true
unset LMCACHE_MAX_LOCAL_DISK_SIZE 2>/dev/null || true

MODEL_PATH="${MODEL_PATH:-/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
MODEL_NAME="${MODEL_NAME:-dsv4}"
PORT="${PORT:-8008}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

echo "Single-node DSv4 + LMCache L1 CPU only"
echo "  model: ${MODEL_PATH}"
echo "  L1 CPU: max=${LMCACHE_MAX_LOCAL_CPU_SIZE}GB"
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
