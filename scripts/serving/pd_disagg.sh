#!/bin/bash
# Ascend PD disaggregation smoke test — DeepSeek-V4-Flash multi-group (1p1d).
#
# Based on:
#   - tests/run_deepseek_lmc20.sh (model + LMCache env)
#   - LMCache-Ascend/examples/disagg_prefill/1p1d (LMCache PD proxy flow)
#   - vLLM Ascend DeepSeek-V4-Flash §5.2 PD separation (prefill vs decode tuning)
#     https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Flash.html#5-online-service-deployment
#
# Single-node layout: two TP8 instances (16 NPUs total).
#   prefiller: ASCEND_RT_VISIBLE_DEVICES=0-7
#   decoder:   ASCEND_RT_VISIBLE_DEVICES=8-15
#
# Multi-node example (reachable NIC IPs + RoCE/HCCL):
#   # Machine A (prefiller + proxy)
#   PREFILL_HOST=10.0.0.1 DECODE_HOST=10.0.0.2 PROXY_HOST=10.0.0.1 \
#     NIC_NAME=eth0 LOCAL_IP=10.0.0.1 PREFILL_NPUS=0,1,2,3,4,5,6,7 \
#     PD_INSTANCE=prefill LAUNCH_PROXY=1 ./run_pd_disagg_deepseek_v4.sh launch
#   # Machine B (decoder)
#   PREFILL_HOST=10.0.0.1 DECODE_HOST=10.0.0.2 PROXY_HOST=10.0.0.1 \
#     NIC_NAME=eth0 LOCAL_IP=10.0.0.2 DECODE_NPUS=0,1,2,3,4,5,6,7 \
#     PD_INSTANCE=decode ./run_pd_disagg_deepseek_v4.sh launch
#   # Client: URL=http://10.0.0.1:9100/v1/chat/completions ./query_example.sh
#
# Uses LMCacheAscendConnector (kv_producer / kv_consumer) instead of MooncakeHybridConnector.
#
# Usage:
#   ./run_pd_disagg_deepseek_v4.sh [all|launch|test|stop] [--debug]
#
# Optional env:
#   MODEL_PATH, PREFILL_NPUS, DECODE_NPUS, TENSOR_PARALLEL_SIZE (default 8)
#   PREFILL_HOST, DECODE_HOST, PROXY_HOST (default localhost; use NIC IPs for multi-node)
#   LAUNCH_PROXY=0|1 — with PD_INSTANCE=prefill|decode, also start the disagg proxy
#   NIC_NAME, LOCAL_IP (RoCE/HCCL networking — set for multi-node)
#   PD_BUFFER_SIZE, LMCACHE_CHUNK_SIZE, MAX_MODEL_LEN (default 131072), PROXY_PORT
#   PD_INSTANCE: prefill | decode | both (default prefill — single instance for debugging)
#   GPU_MEMORY_UTILIZATION (default 0.85), LMCACHE_LOG_LEVEL, VLLM_LOGGING_LEVEL
#   PREFILL_LMCACHE_CONFIG / DECODE_LMCACHE_CONFIG — skip auto-generated YAML if set
#
# LMCache is configured via YAML (see write_lmcache_configs) and loaded with
# LMCACHE_CONFIG_FILE in launch_prefiller / launch_decoder.
# Pass --debug for Ascend slog-to-stdout, ASCEND_GLOBAL_LOG_LEVEL=0, and faulthandler.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEBUG_MODE=0
MODE="all"
for arg in "$@"; do
    case "$arg" in
        --debug) DEBUG_MODE=1 ;;
        all|launch|test|stop) MODE="$arg" ;;
        -h|--help)
            echo "Usage: $0 [all|launch|test|stop] [--debug]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [all|launch|test|stop] [--debug]"
            exit 1
            ;;
    esac
done

MODEL_PATH="${MODEL_PATH:-/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
MODEL_NAME="${MODEL_NAME:-dsv4}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
PREFILL_NPUS="${PREFILL_NPUS:-0,1,2,3,4,5,6,7}"
DECODE_NPUS="${DECODE_NPUS:-8,9,10,11,12,13,14,15}"

# Reachable hosts (use NIC IPs for multi-node; localhost for single-node).
PREFILL_HOST="${PREFILL_HOST:-localhost}"
DECODE_HOST="${DECODE_HOST:-localhost}"
PROXY_HOST="${PROXY_HOST:-localhost}"
# ZMQ bind address on the decoder for pd alloc/init listeners.
# Defaults to DECODE_HOST; set to 0.0.0.0 to listen on all interfaces while
# still advertising DECODE_HOST via the proxy to the prefiller.
DECODE_BIND_HOST="${DECODE_BIND_HOST:-${DECODE_HOST}}"
LAUNCH_PROXY="${LAUNCH_PROXY:-0}"

PREFILL_PORT="${PREFILL_PORT:-7100}"
DECODE_PORT="${DECODE_PORT:-7200}"
PROXY_PORT="${PROXY_PORT:-9100}"
PROXY_ZMQ_PORT="${PROXY_ZMQ_PORT:-7500}"

INIT_PORT_BASE="${INIT_PORT_BASE:-7300}"
ALLOC_PORT_BASE="${ALLOC_PORT_BASE:-7400}"
DONE_PORT_BASE="${DONE_PORT_BASE:-7600}"

PD_BUFFER_SIZE="${PD_BUFFER_SIZE:-1073741824}"   # 1 GiB — leave headroom for model + KV on 64 GiB NPUs
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-1024}"
LMCACHE_USE_LAYERWISE="${LMCACHE_USE_LAYERWISE:-False}"
LMCACHE_NUMA_MODE="${LMCACHE_NUMA_MODE:-auto}"
SAVE_ONLY_FIRST_RANK="${SAVE_ONLY_FIRST_RANK:-true}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"   # 128K tokens
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
PD_INSTANCE="${PD_INSTANCE:-prefill}"   # prefill | decode | both

# Cross-host: each machine typically owns devices 0..TP-1 (not 8-15).
if [[ "$PREFILL_HOST" != "$DECODE_HOST" && "$DECODE_NPUS" == "8,9,10,11,12,13,14,15" ]]; then
    DECODE_NPUS="0,1,2,3,4,5,6,7"
fi

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/pd_disagg_dsv4}"
CFG_DIR="${CFG_DIR:-${LOG_DIR}/configs}"
PID_FILE="${PID_FILE:-${LOG_DIR}/pd_disagg_dsv4.pids}"
PREFILLER_LOG="${PREFILLER_LOG:-${LOG_DIR}/prefiller.log}"
DECODER_LOG="${DECODER_LOG:-${LOG_DIR}/decoder.log}"
PROXY_LOG="${PROXY_LOG:-${LOG_DIR}/proxy.log}"

LMCACHE_ROOT="${LMCACHE_ROOT:-${CONTAINER_LMCACHE_MOUNT:-/workspace/LMCache}}"
PROXY_SCRIPT="${LMCACHE_ROOT}/examples/disagg_prefill/disagg_proxy_server.py"

PREFILL_CFG="${PREFILL_LMCACHE_CONFIG:-${CFG_DIR}/lmcache-dsv4-prefiller-config.yaml}"
DECODE_CFG="${DECODE_LMCACHE_CONFIG:-${CFG_DIR}/lmcache-dsv4-decoder-config.yaml}"

PIDS=()

setup_network_env() {
    # §5.2.1: set when PD prefiller/decoder run on separate nodes over RoCE.
    if [[ -n "${NIC_NAME:-}" ]]; then
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"
    fi
    if [[ -n "${LOCAL_IP:-}" ]]; then
        export HCCL_IF_IP="$LOCAL_IP"
    fi
}

setup_common_env() {
    export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
    export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
    export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
    export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
    export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
    export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"

    export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-3600000}"
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-30000}"
    export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-204}"

    export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
    export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-1}"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

    export LMCACHE_TRACK_USAGE="${LMCACHE_TRACK_USAGE:-false}"

    if [[ "$DEBUG_MODE" -eq 1 ]]; then
        export LMCACHE_LOG_LEVEL="${LMCACHE_LOG_LEVEL:-DEBUG}"
        export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
        export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
        export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-1}"
        export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-0}"
    else
        export LMCACHE_LOG_LEVEL="${LMCACHE_LOG_LEVEL:-INFO}"
        export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
        # Quiet Ascend slog by default (was flooding prefiller/decoder logs).
        unset PYTHONFAULTHANDLER ASCEND_SLOG_PRINT_TO_STDOUT 2>/dev/null || true
        export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"  # ERROR
    fi

    setup_network_env
}

setup_prefill_env() {
    setup_common_env
    export HCCL_BUFFSIZE="${PREFILL_HCCL_BUFFSIZE:-1024}"
    export HCCL_CONNECT_TIMEOUT="${PREFILL_HCCL_CONNECT_TIMEOUT:-120}"
}

setup_decoder_env() {
    setup_common_env
    # Decode node tuning from §5.2.1 (A3 PD separation).
    export HCCL_BUFFSIZE="${DECODE_HCCL_BUFFSIZE:-1024}"
    export HCCL_CONNECT_TIMEOUT="${DECODE_HCCL_CONNECT_TIMEOUT:-1200}"
}

yaml_port_list() {
    local start=$1
    local count=$2
    local ports=""
    local i
    for ((i = 0; i < count; i++)); do
        [[ -n "$ports" ]] && ports+=", "
        ports+=$((start + i))
    done
    echo "[${ports}]"
}

port_csv() {
    local start=$1
    local count=$2
    local ports=()
    local i
    for ((i = 0; i < count; i++)); do
        ports+=("$((start + i))")
    done
    local IFS=,
    echo "${ports[*]}"
}

write_lmcache_configs() {
    mkdir -p "$CFG_DIR"

    if [[ -z "${PREFILL_LMCACHE_CONFIG:-}" ]]; then
        cat > "$PREFILL_CFG" <<EOF
chunk_size: ${LMCACHE_CHUNK_SIZE}
use_layerwise: ${LMCACHE_USE_LAYERWISE}
numa_mode: "${LMCACHE_NUMA_MODE}"

extra_config:
  save_only_first_rank: ${SAVE_ONLY_FIRST_RANK}

local_cpu: False

enable_pd: True
transfer_channel: "hccl"
pd_role: "sender"
pd_pull_mode: False
pd_delay_pull: False
pd_pull_done_port: $(yaml_port_list "$DONE_PORT_BASE" "$TENSOR_PARALLEL_SIZE")
pd_use_cpu_offload: False
pd_cpu_buffer_size: 21474836480
pd_peer_host: "${PREFILL_HOST}"
pd_proxy_host: "${PROXY_HOST}"
pd_proxy_port: ${PROXY_ZMQ_PORT}
pd_buffer_size: ${PD_BUFFER_SIZE}
pd_buffer_device: "npu"
save_unfull_chunk: False
EOF
    fi

    if [[ -z "${DECODE_LMCACHE_CONFIG:-}" ]]; then
        cat > "$DECODE_CFG" <<EOF
chunk_size: ${LMCACHE_CHUNK_SIZE}
use_layerwise: ${LMCACHE_USE_LAYERWISE}
numa_mode: "${LMCACHE_NUMA_MODE}"

extra_config:
  save_only_first_rank: ${SAVE_ONLY_FIRST_RANK}

local_cpu: False

enable_pd: True
transfer_channel: "hccl"
pd_role: "receiver"
# Bind/advertise decoder alloc+init listeners (must be reachable from prefiller).
pd_peer_host: "${DECODE_BIND_HOST}"
pd_pull_mode: False
pd_delay_pull: False
pd_peer_init_port: $(yaml_port_list "$INIT_PORT_BASE" "$TENSOR_PARALLEL_SIZE")
pd_peer_alloc_port: $(yaml_port_list "$ALLOC_PORT_BASE" "$TENSOR_PARALLEL_SIZE")
pd_buffer_size: ${PD_BUFFER_SIZE}
pd_buffer_device: "npu"
EOF
    fi

    echo "LMCache config files:"
    echo "  prefill: ${PREFILL_CFG}"
    echo "  decode:  ${DECODE_CFG}"
    echo "  hosts: prefill=${PREFILL_HOST} decode=${DECODE_HOST} proxy=${PROXY_HOST} decode_bind=${DECODE_BIND_HOST}"
}

cleanup() {
    echo "Stopping PD DeepSeek-V4 processes..."
    trap - INT TERM USR1

    if [[ -f "$PID_FILE" ]]; then
        while read -r pid; do
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi

    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done

    sleep 2
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    echo "Stopped."
}

record_pid() {
    mkdir -p "$LOG_DIR"
    echo "$1" >> "$PID_FILE"
}

wait_for_server() {
    local port=$1
    # Fail fast by default (override with SERVER_WAIT_TIMEOUT or explicit arg).
    local timeout_seconds="${2:-${SERVER_WAIT_TIMEOUT:-360}}"
    local host="${3:-127.0.0.1}"
    local start_time
    start_time=$(date +%s)

    # Remote role checks: use the advertised host. Local role: 127.0.0.1 is fine
    # even when PREFILL_HOST/DECODE_HOST are NIC IPs for peer discovery.
    echo "Waiting for server at http://${host}:${port}/v1/models ..."
    while true; do
        if curl -sf "http://${host}:${port}/v1/models" >/dev/null 2>&1; then
            echo "Server ready at ${host}:${port}"
            return 0
        fi
        if (( $(date +%s) - start_time >= timeout_seconds )); then
            echo "Timeout waiting for server at ${host}:${port}"
            return 1
        fi
        sleep 5
    done
}

# vLLM args aligned with tests/run_deepseek_lmc20.sh (PD role/kv-transfer differ).
VLLM_LMC20_ADDITIONAL_CONFIG='{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}'

VLLM_COMMON_ARGS=(
    --host 0.0.0.0
    --model "$MODEL_PATH"
    --served-model-name "$MODEL_NAME"
    --data-parallel-size 1
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --enable-expert-parallel
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens 8192
    --max-num-seqs 16
    --api-server-count 1
    --no-disable-hybrid-kv-cache-manager
    --no-enable-prefix-caching
    --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 128}'
    --safetensors-load-strategy prefetch
    --speculative-config '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}'
    --block-size 128
    --tokenizer-mode deepseek_v4
    --tool-call-parser deepseek_v4
    --enable-auto-tool-choice
    --reasoning-parser deepseek_v4
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --quantization ascend
    --async-scheduling
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
    --additional-config "$VLLM_LMC20_ADDITIONAL_CONFIG"
)

VLLM_PREFILL_ARGS=("${VLLM_COMMON_ARGS[@]}")
VLLM_DECODE_ARGS=("${VLLM_COMMON_ARGS[@]}")

launch_proxy() {
    if [[ ! -f "$PROXY_SCRIPT" ]]; then
        echo "Proxy script not found: $PROXY_SCRIPT"
        exit 1
    fi

    # MLA + save_only_first_rank: only TP0 stores/transfers and sends one
    # ProxyNotif. wait_decode_kv_ready waits for len(init_ports) signals.
    local proxy_init_ports proxy_alloc_ports
    if [[ "${SAVE_ONLY_FIRST_RANK}" == "true" ]]; then
        proxy_init_ports="${INIT_PORT_BASE}"
        proxy_alloc_ports="${ALLOC_PORT_BASE}"
    else
        proxy_init_ports="$(port_csv "$INIT_PORT_BASE" "$TENSOR_PARALLEL_SIZE")"
        proxy_alloc_ports="$(port_csv "$ALLOC_PORT_BASE" "$TENSOR_PARALLEL_SIZE")"
    fi

    python3 "$PROXY_SCRIPT" \
        --host 0.0.0.0 \
        --port "$PROXY_PORT" \
        --prefiller-host "$PREFILL_HOST" \
        --prefiller-port "$PREFILL_PORT" \
        --num-prefillers 1 \
        --decoder-host "$DECODE_HOST" \
        --decoder-port "$DECODE_PORT" \
        --decoder-init-port "$proxy_init_ports" \
        --decoder-alloc-port "$proxy_alloc_ports" \
        --proxy-host "$PROXY_HOST" \
        --proxy-port "$PROXY_ZMQ_PORT" \
        --num-decoders 1 \
        --model "$MODEL_PATH" \
        --pd-buffer-size "$PD_BUFFER_SIZE" \
        --chunk-size "$LMCACHE_CHUNK_SIZE" \
        >"${PROXY_LOG}" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    record_pid "$pid"
    echo "Proxy pid=${pid} http=0.0.0.0:${PROXY_PORT} advertise=${PROXY_HOST} prefill=${PREFILL_HOST}:${PREFILL_PORT} decode=${DECODE_HOST}:${DECODE_PORT} log=${PROXY_LOG} init_ports=${proxy_init_ports}"
}

launch_prefiller() {
    setup_prefill_env
    export LMCACHE_CONFIG_FILE="$PREFILL_CFG"
    export ASCEND_RT_VISIBLE_DEVICES="$PREFILL_NPUS"

    echo "Prefiller LMCACHE_CONFIG_FILE=${LMCACHE_CONFIG_FILE}"

    vllm serve "${VLLM_PREFILL_ARGS[@]}" \
        --port "$PREFILL_PORT" \
        --kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_producer","kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector","kv_connector_extra_config":{"discard_partial_chunks":true,"lmcache_rpc_port":"producer1"}}' \
        >"${PREFILLER_LOG}" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    record_pid "$pid"
    echo "Prefiller pid=${pid} port=${PREFILL_PORT} NPUs=${PREFILL_NPUS} log=${PREFILLER_LOG}"
}

launch_decoder() {
    setup_decoder_env
    export LMCACHE_CONFIG_FILE="$DECODE_CFG"
    export ASCEND_RT_VISIBLE_DEVICES="$DECODE_NPUS"

    echo "Decoder LMCACHE_CONFIG_FILE=${LMCACHE_CONFIG_FILE}"

    vllm serve "${VLLM_DECODE_ARGS[@]}" \
        --port "$DECODE_PORT" \
        --kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_consumer","kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector","kv_connector_extra_config":{"discard_partial_chunks":true,"lmcache_rpc_port":"consumer1","skip_last_n_tokens":1}}' \
        >"${DECODER_LOG}" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    record_pid "$pid"
    echo "Decoder pid=${pid} port=${DECODE_PORT} NPUs=${DECODE_NPUS} log=${DECODER_LOG}"
}

run_smoke_test() {
    # Chunk-aligned prompt (discard_partial_chunks=true requires full 1024-token chunks).
    local prompt
    prompt="$(printf 'Summarize the role of multi-group KV cache in DeepSeek-V4 disaggregated prefill. %.0s' {1..200})"
    local smoke_url="${SMOKE_URL:-http://${PROXY_HOST}:${PROXY_PORT}/v1/chat/completions}"

    echo "Sending DSv4 PD smoke request to ${smoke_url}..."
    curl -sf -X POST "$smoke_url" \
        -H "Content-Type: application/json" \
        -d "$(python3 - <<PY
import json
print(json.dumps({
    "model": "${MODEL_NAME}",
    "messages": [{"role": "user", "content": """${prompt}"""}],
    "max_tokens": 32,
    "temperature": 0,
}))
PY
)" | tee "${LOG_DIR}/smoke_response.json"

    echo
    echo "Smoke test completed."
    echo "Verify PD transfer in:"
    echo "  ${PREFILLER_LOG}"
    echo "  ${DECODER_LOG}"
}

launch_all() {
    mkdir -p "$LOG_DIR"
    : > "$PID_FILE"

    write_lmcache_configs
    setup_common_env
    trap cleanup INT TERM USR1

    echo "Model: ${MODEL_PATH}"
    echo "PD_INSTANCE: ${PD_INSTANCE}  LAUNCH_PROXY: ${LAUNCH_PROXY}"
    echo "Hosts: prefill=${PREFILL_HOST} decode=${DECODE_HOST} proxy=${PROXY_HOST} decode_bind=${DECODE_BIND_HOST}"
    echo "TP: ${TENSOR_PARALLEL_SIZE} per instance"
    echo "max_model_len: ${MAX_MODEL_LEN}  chunk_size: ${LMCACHE_CHUNK_SIZE}  pd_buffer_size: ${PD_BUFFER_SIZE}"
    echo "gpu_mem_util: ${GPU_MEMORY_UTILIZATION}  LMCACHE_LOG_LEVEL: ${LMCACHE_LOG_LEVEL}"
    echo "save_only_first_rank: ${SAVE_ONLY_FIRST_RANK}  (via LMCache YAML extra_config)"
    echo "debug: ${DEBUG_MODE}  Prefill NPUs: ${PREFILL_NPUS}  Decode NPUs: ${DECODE_NPUS}"
    echo "NIC_NAME: ${NIC_NAME:-}  LOCAL_IP: ${LOCAL_IP:-}"
    echo "Prefiller log: ${PREFILLER_LOG}"
    echo "Decoder log:   ${DECODER_LOG}"
    echo "Logs dir: ${LOG_DIR}"

    if [[ "$PD_INSTANCE" == "both" && "$PREFILL_HOST" != "$DECODE_HOST" ]]; then
        echo "WARNING: PD_INSTANCE=both with different PREFILL_HOST/DECODE_HOST will try to start both roles on this machine."
        echo "         Prefer PD_INSTANCE=prefill on the prefill node and PD_INSTANCE=decode on the decode node."
    fi

    maybe_launch_proxy() {
        if [[ "$LAUNCH_PROXY" == "1" || "$LAUNCH_PROXY" == "true" ]]; then
            launch_proxy
            echo "Waiting for proxy listen on port ${PROXY_PORT}..."
            start_time=$(date +%s)
            while true; do
                code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 \
                    "http://127.0.0.1:${PROXY_PORT}/v1/chat/completions" || true)
                if [[ "$code" != "000" && -n "$code" ]]; then
                    echo "Proxy ready on port ${PROXY_PORT} (HTTP ${code})"
                    break
                fi
                if (( $(date +%s) - start_time >= 120 )); then
                    echo "Timeout waiting for proxy on port ${PROXY_PORT}"
                    return 1
                fi
                sleep 2
            done
        fi
    }

    case "$PD_INSTANCE" in
        prefill)
            maybe_launch_proxy
            launch_prefiller
            wait_for_server "$PREFILL_PORT" "${SERVER_WAIT_TIMEOUT:-360}" "127.0.0.1"
            echo "Prefiller is up (host advertise ${PREFILL_HOST}:${PREFILL_PORT})."
            ;;
        decode)
            maybe_launch_proxy
            launch_decoder
            wait_for_server "$DECODE_PORT" "${SERVER_WAIT_TIMEOUT:-360}" "127.0.0.1"
            echo "Decoder is up (host advertise ${DECODE_HOST}:${DECODE_PORT})."
            ;;
        both)
            launch_proxy
            launch_decoder
            launch_prefiller
            wait_for_server "$DECODE_PORT" "${SERVER_WAIT_TIMEOUT:-360}" "127.0.0.1"
            wait_for_server "$PREFILL_PORT" "${SERVER_WAIT_TIMEOUT:-360}" "127.0.0.1"
            # Proxy only exposes /v1/completions and /v1/chat/completions (no /v1/models).
            echo "Waiting for proxy listen on port ${PROXY_PORT}..."
            start_time=$(date +%s)
            while true; do
                code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 \
                    "http://127.0.0.1:${PROXY_PORT}/v1/chat/completions" || true)
                # 405/422/404 mean the HTTP server is up; 000 means connection failed.
                if [[ "$code" != "000" && -n "$code" ]]; then
                    echo "Proxy ready on port ${PROXY_PORT} (HTTP ${code})"
                    break
                fi
                if (( $(date +%s) - start_time >= 120 )); then
                    echo "Timeout waiting for proxy on port ${PROXY_PORT}"
                    return 1
                fi
                sleep 2
            done
            echo "All PD DeepSeek-V4 servers are up."
            ;;
        *)
            echo "Invalid PD_INSTANCE=${PD_INSTANCE} (use prefill, decode, or both)"
            exit 1
            ;;
    esac
}

case "$MODE" in
    launch)
        launch_all
        echo "Press Ctrl-C to stop."
        while true; do sleep 1; done
        ;;
    test)
        run_smoke_test
        ;;
    stop)
        cleanup
        ;;
    all)
        launch_all
        if [[ "$PD_INSTANCE" == "both" ]]; then
            run_smoke_test
        else
            echo "Skipping smoke test (PD_INSTANCE=${PD_INSTANCE}; set PD_INSTANCE=both for full PD test)."
        fi
        echo "Press Ctrl-C to stop servers."
        while true; do sleep 1; done
        ;;
    *)
        echo "Usage: $0 [all|launch|test|stop] [--debug]"
        exit 1
        ;;
esac
