#!/bin/bash
# 2P + 2D DeepSeek-V4 Ascend serving with Prefill↔Prefill P2P KV sharing.
#
# Topology:
#   Machine A (${HOST_A}): controller + proxy + P0 (NPUs 0-7) + D0 (NPUs 8-15)
#   Machine B (${HOST_B}):                    P1 (NPUs 0-7) + D1 (NPUs 8-15)
#   P2P between P0 and P1 (cross-host HCCL AscendDirect / p2p_use_npu)
#   PD pairs: P0↔D0 and P1↔D1 (local on each machine)
#
# Usage (do not run from here until ready):
#   # Machine A
#   NODE_ROLE=a LOCAL_IP=192.168.0.223 ./scripts/serving/pd_2p2d_p2p_node_a.sh launch
#   # Machine B
#   NODE_ROLE=b LOCAL_IP=192.168.0.110 ./scripts/serving/pd_2p2d_p2p_node_b.sh launch
#
# Or call this script directly:
#   NODE_ROLE=a|b ./scripts/serving/pd_2p2d_p2p.sh [launch|stop|test] [--debug]
#
# Upstream constraint: LMCache asserts enable_pd XOR enable_p2p. Prefillers
# that need both require ALLOW_PD_P2P=1 *and* that assert removed. Default
# with ENABLE_PD=1 ENABLE_P2P=1 ALLOW_PD_P2P=0 keeps PD on Prefill/Decode and
# still emits P2P YAML fields but sets enable_p2p=False until ALLOW_PD_P2P=1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env/pd_2p2d_p2p_env.sh"

DEBUG_MODE=0
MODE="launch"
for arg in "$@"; do
    case "$arg" in
        --debug) DEBUG_MODE=1 ;;
        launch|stop|test) MODE="$arg" ;;
        -h|--help)
            echo "Usage: $0 [launch|stop|test] [--debug]"
            echo "  NODE_ROLE=a|b  (required for launch/stop)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

NODE_ROLE="${NODE_ROLE:-}"
if [[ "$MODE" != "test" && -z "$NODE_ROLE" ]]; then
    echo "NODE_ROLE=a|b is required"
    exit 1
fi

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/pd_2p2d_p2p_${NODE_ROLE}}"
CFG_DIR="${CFG_DIR:-${LOG_DIR}/configs}"
PID_FILE="${PID_FILE:-${LOG_DIR}/pids}"

PROXY_SCRIPT="${LMCACHE_ROOT}/examples/disagg_prefill/disagg_proxy_server.py"

PIDS=()

yaml_port_list() {
    local start=$1
    local count=$2
    local ports="" i
    for ((i = 0; i < count; i++)); do
        [[ -n "$ports" ]] && ports+=", "
        ports+=$((start + i))
    done
    echo "[${ports}]"
}

port_csv() {
    local start=$1
    local count=$2
    local ports=() i
    for ((i = 0; i < count; i++)); do
        ports+=("$((start + i))")
    done
    local IFS=,
    echo "${ports[*]}"
}

resolve_prefill_flags() {
    # Returns via globals: PREFILL_ENABLE_PD / PREFILL_ENABLE_P2P (True/False YAML)
    PREFILL_ENABLE_PD="False"
    PREFILL_ENABLE_P2P="False"
    if [[ "${ENABLE_PD}" == "1" ]]; then
        PREFILL_ENABLE_PD="True"
    fi
    if [[ "${ENABLE_P2P}" == "1" ]]; then
        PREFILL_ENABLE_P2P="True"
    fi
    if [[ "$PREFILL_ENABLE_PD" == "True" && "$PREFILL_ENABLE_P2P" == "True" ]]; then
        if [[ "${ALLOW_PD_P2P}" != "1" ]]; then
            echo "WARNING: enable_pd + enable_p2p on Prefill is rejected by upstream LMCache"
            echo "         (assert: PD only supports enable_p2p=False)."
            echo "         Keeping enable_pd=True, enable_p2p=False."
            echo "         To force both in YAML: ALLOW_PD_P2P=1 (and patch the assert)."
            PREFILL_ENABLE_P2P="False"
        else
            echo "WARNING: ALLOW_PD_P2P=1 — Prefill YAML has enable_pd+enable_p2p."
            echo "         Ensure LMCache config assert is patched or launch will fail."
        fi
    fi
}

setup_network_env() {
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
    export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-1}"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    export LMCACHE_TRACK_USAGE="${LMCACHE_TRACK_USAGE:-false}"
    # P2P peers must share PYTHONHASHSEED
    export PYTHONHASHSEED="${PYTHONHASHSEED:-123}"

    if [[ "$DEBUG_MODE" -eq 1 ]]; then
        export LMCACHE_LOG_LEVEL="${LMCACHE_LOG_LEVEL:-DEBUG}"
        export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
        export PYTHONFAULTHANDLER=1
        export ASCEND_SLOG_PRINT_TO_STDOUT=1
        export ASCEND_GLOBAL_LOG_LEVEL=0
    else
        export LMCACHE_LOG_LEVEL="${LMCACHE_LOG_LEVEL:-INFO}"
        export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
        unset PYTHONFAULTHANDLER ASCEND_SLOG_PRINT_TO_STDOUT 2>/dev/null || true
        export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
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
    export HCCL_BUFFSIZE="${DECODE_HCCL_BUFFSIZE:-1024}"
    export HCCL_CONNECT_TIMEOUT="${DECODE_HCCL_CONNECT_TIMEOUT:-1200}"
}

record_pid() {
    mkdir -p "$LOG_DIR"
    echo "$1" >> "$PID_FILE"
}

cleanup() {
    echo "Stopping 2P2D+P2P processes (NODE_ROLE=${NODE_ROLE})..."
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
        kill "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${PIDS[@]:-}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    echo "Stopped."
}

wait_for_server() {
    local port=$1
    local timeout_seconds="${2:-${SERVER_WAIT_TIMEOUT:-360}}"
    local host="${3:-127.0.0.1}"
    local start_time
    start_time=$(date +%s)
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

write_lmcache_configs() {
    mkdir -p "$CFG_DIR"
    resolve_prefill_flags

    local local_host instance_id
    if [[ "$NODE_ROLE" == "a" ]]; then
        local_host="$HOST_A"
        instance_id="lmcache_prefill_a"
    else
        local_host="$HOST_B"
        instance_id="lmcache_prefill_b"
    fi

    local p2p_init p2p_lookup worker_ports done_ports init_ports alloc_ports
    p2p_init="$(yaml_port_list "$P2P_INIT_PORT_BASE" "$TENSOR_PARALLEL_SIZE")"
    p2p_lookup="$(yaml_port_list "$P2P_LOOKUP_PORT_BASE" "$TENSOR_PARALLEL_SIZE")"
    worker_ports="$(yaml_port_list "$LMCACHE_WORKER_PORT_BASE" "$TENSOR_PARALLEL_SIZE")"
    done_ports="$(yaml_port_list "$DONE_PORT_BASE" "$TENSOR_PARALLEL_SIZE")"
    init_ports="$(yaml_port_list "$INIT_PORT_BASE" "$TENSOR_PARALLEL_SIZE")"
    alloc_ports="$(yaml_port_list "$ALLOC_PORT_BASE" "$TENSOR_PARALLEL_SIZE")"

    local ctrl_pull="${CONTROLLER_HOST}:${CONTROLLER_PULL_PORT}"
    local ctrl_reply="${CONTROLLER_HOST}:${CONTROLLER_REPLY_PORT}"

    # Prefill: PD sender (+ optional P2P peer for cross-host Prefill sharing)
    cat > "${CFG_DIR}/lmcache-prefill.yaml" <<EOF
chunk_size: ${LMCACHE_CHUNK_SIZE}
use_layerwise: False
numa_mode: "auto"
local_cpu: True
max_local_cpu_size: ${LMCACHE_MAX_LOCAL_CPU_SIZE}
enable_async_loading: True
save_unfull_chunk: False

extra_config:
  save_only_first_rank: ${SAVE_ONLY_FIRST_RANK}
  lookup_backoff_time: 0.001

# --- PD (local P↔D on this machine) ---
enable_pd: ${PREFILL_ENABLE_PD}
transfer_channel: "hccl"
pd_role: "sender"
pd_pull_mode: False
pd_delay_pull: False
pd_pull_done_port: ${done_ports}
pd_use_cpu_offload: False
pd_cpu_buffer_size: 21474836480
pd_peer_host: "${local_host}"
pd_proxy_host: "${PROXY_HOST}"
pd_proxy_port: ${PROXY_ZMQ_PORT}
pd_buffer_size: ${PD_BUFFER_SIZE}
pd_buffer_device: "npu"

# --- P2P (Prefill↔Prefill across HOST_A / HOST_B) ---
enable_p2p: ${PREFILL_ENABLE_P2P}
p2p_host: "${local_host}"
p2p_init_ports: ${p2p_init}
p2p_lookup_ports: ${p2p_lookup}
p2p_use_npu: ${P2P_USE_NPU}
p2p_pull_mode: ${P2P_PULL_MODE}
p2p_delay_pull: ${P2P_DELAY_PULL}
p2p_npu_buffer_size: ${P2P_NPU_BUFFER_SIZE}

enable_controller: True
lmcache_instance_id: "${instance_id}"
controller_pull_url: "${ctrl_pull}"
controller_reply_url: "${ctrl_reply}"
lmcache_worker_ports: ${worker_ports}
EOF

    # Decode: PD receiver only (no P2P)
    cat > "${CFG_DIR}/lmcache-decode.yaml" <<EOF
chunk_size: ${LMCACHE_CHUNK_SIZE}
use_layerwise: False
numa_mode: "auto"
local_cpu: False

extra_config:
  save_only_first_rank: ${SAVE_ONLY_FIRST_RANK}

enable_pd: True
transfer_channel: "hccl"
pd_role: "receiver"
pd_peer_host: "${local_host}"
pd_pull_mode: False
pd_delay_pull: False
pd_peer_init_port: ${init_ports}
pd_peer_alloc_port: ${alloc_ports}
pd_buffer_size: ${PD_BUFFER_SIZE}
pd_buffer_device: "npu"
EOF

    echo "Wrote LMCache configs under ${CFG_DIR}"
    echo "  prefill enable_pd=${PREFILL_ENABLE_PD} enable_p2p=${PREFILL_ENABLE_P2P}"
    echo "  p2p_host=${local_host} controller=${ctrl_pull}/${ctrl_reply}"
}

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

launch_controller() {
    local log="${LOG_DIR}/controller.log"
    local ports_json
    ports_json=$(printf '{"pull": %s, "reply": %s}' "$CONTROLLER_PULL_PORT" "$CONTROLLER_REPLY_PORT")
    echo "Launching lmcache_controller on 0.0.0.0:${CONTROLLER_PORT} monitor=${ports_json}"
    PYTHONHASHSEED="$PYTHONHASHSEED" lmcache_controller \
        --host 0.0.0.0 \
        --port "$CONTROLLER_PORT" \
        --monitor-ports "$ports_json" \
        >"$log" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    record_pid "$pid"
    echo "Controller pid=${pid} log=${log}"
    sleep 2
}

launch_proxy() {
    if [[ ! -f "$PROXY_SCRIPT" ]]; then
        echo "Proxy script not found: $PROXY_SCRIPT"
        exit 1
    fi
    local proxy_init_ports proxy_alloc_ports
    if [[ "${SAVE_ONLY_FIRST_RANK}" == "true" ]]; then
        proxy_init_ports="${INIT_PORT_BASE}"
        proxy_alloc_ports="${ALLOC_PORT_BASE}"
    else
        proxy_init_ports="$(port_csv "$INIT_PORT_BASE" "$TENSOR_PARALLEL_SIZE")"
        proxy_alloc_ports="$(port_csv "$ALLOC_PORT_BASE" "$TENSOR_PARALLEL_SIZE")"
    fi

    # Two hosts, same ports: one-to-one pairing → (A:7100),(B:7100) and (A:7200),(B:7200)
    python3 "$PROXY_SCRIPT" \
        --host 0.0.0.0 \
        --port "$PROXY_PORT" \
        --prefiller-host "${HOST_A},${HOST_B}" \
        --prefiller-port "${PREFILL_PORT},${PREFILL_PORT}" \
        --num-prefillers 2 \
        --decoder-host "${HOST_A},${HOST_B}" \
        --decoder-port "${DECODE_PORT},${DECODE_PORT}" \
        --decoder-init-port "$proxy_init_ports" \
        --decoder-alloc-port "$proxy_alloc_ports" \
        --proxy-host "$PROXY_HOST" \
        --proxy-port "$PROXY_ZMQ_PORT" \
        --num-decoders 2 \
        --model "$MODEL_PATH" \
        --pd-buffer-size "$PD_BUFFER_SIZE" \
        --chunk-size "$LMCACHE_CHUNK_SIZE" \
        >"${LOG_DIR}/proxy.log" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    record_pid "$pid"
    echo "Proxy pid=${pid} http=0.0.0.0:${PROXY_PORT} P=[${HOST_A},${HOST_B}]:${PREFILL_PORT} D=[${HOST_A},${HOST_B}]:${DECODE_PORT}"
}

launch_prefiller() {
    setup_prefill_env
    export LMCACHE_CONFIG_FILE="${CFG_DIR}/lmcache-prefill.yaml"
    export ASCEND_RT_VISIBLE_DEVICES="$PREFILL_NPUS"
    local rpc="producer_${NODE_ROLE}"
    echo "Prefiller LMCACHE_CONFIG_FILE=${LMCACHE_CONFIG_FILE} NPUs=${PREFILL_NPUS}"
    vllm serve "${VLLM_COMMON_ARGS[@]}" \
        --port "$PREFILL_PORT" \
        --kv-transfer-config "{\"kv_connector\":\"LMCacheAscendConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_module_path\":\"lmcache_ascend.integration.vllm.lmcache_ascend_connector\",\"kv_connector_extra_config\":{\"discard_partial_chunks\":true,\"lmcache_rpc_port\":\"${rpc}\"}}" \
        >"${LOG_DIR}/prefiller.log" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    record_pid "$pid"
    echo "Prefiller pid=${pid} port=${PREFILL_PORT} log=${LOG_DIR}/prefiller.log"
}

launch_decoder() {
    setup_decoder_env
    export LMCACHE_CONFIG_FILE="${CFG_DIR}/lmcache-decode.yaml"
    export ASCEND_RT_VISIBLE_DEVICES="$DECODE_NPUS"
    local rpc="consumer_${NODE_ROLE}"
    echo "Decoder LMCACHE_CONFIG_FILE=${LMCACHE_CONFIG_FILE} NPUs=${DECODE_NPUS}"
    vllm serve "${VLLM_COMMON_ARGS[@]}" \
        --port "$DECODE_PORT" \
        --kv-transfer-config "{\"kv_connector\":\"LMCacheAscendConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_module_path\":\"lmcache_ascend.integration.vllm.lmcache_ascend_connector\",\"kv_connector_extra_config\":{\"discard_partial_chunks\":true,\"lmcache_rpc_port\":\"${rpc}\",\"skip_last_n_tokens\":1}}" \
        >"${LOG_DIR}/decoder.log" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    record_pid "$pid"
    echo "Decoder pid=${pid} port=${DECODE_PORT} log=${LOG_DIR}/decoder.log"
}

wait_proxy_ready() {
    echo "Waiting for proxy listen on port ${PROXY_PORT}..."
    local start_time code
    start_time=$(date +%s)
    while true; do
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 \
            "http://127.0.0.1:${PROXY_PORT}/v1/chat/completions" || true)
        if [[ "$code" != "000" && -n "$code" ]]; then
            echo "Proxy ready on port ${PROXY_PORT} (HTTP ${code})"
            return 0
        fi
        if (( $(date +%s) - start_time >= 120 )); then
            echo "Timeout waiting for proxy on port ${PROXY_PORT}"
            return 1
        fi
        sleep 2
    done
}

launch_all() {
    mkdir -p "$LOG_DIR"
    : > "$PID_FILE"
    write_lmcache_configs
    setup_common_env
    trap cleanup INT TERM USR1

    echo "=== 2P2D + Prefill P2P ==="
    echo "NODE_ROLE=${NODE_ROLE} LOCAL_IP=${LOCAL_IP:-} NIC=${NIC_NAME}"
    echo "HOST_A=${HOST_A} HOST_B=${HOST_B} PROXY=${PROXY_HOST}:${PROXY_PORT}"
    echo "P NPUs=${PREFILL_NPUS} D NPUs=${DECODE_NPUS} TP=${TENSOR_PARALLEL_SIZE}"
    echo "ENABLE_PD=${ENABLE_PD} ENABLE_P2P=${ENABLE_P2P} ALLOW_PD_P2P=${ALLOW_PD_P2P}"
    echo "Logs: ${LOG_DIR}"

    case "$NODE_ROLE" in
        a)
            launch_controller
            launch_proxy
            wait_proxy_ready
            launch_decoder
            launch_prefiller
            wait_for_server "$DECODE_PORT"
            wait_for_server "$PREFILL_PORT"
            echo "Node A up: controller + proxy + P0 + D0"
            ;;
        b)
            # Controller + proxy live on A; B only runs local P1+D1
            launch_decoder
            launch_prefiller
            wait_for_server "$DECODE_PORT"
            wait_for_server "$PREFILL_PORT"
            echo "Node B up: P1 + D1 (P2P peers to ${HOST_A})"
            ;;
        *)
            echo "Invalid NODE_ROLE=${NODE_ROLE} (use a or b)"
            exit 1
            ;;
    esac
}

run_smoke_test() {
    local prompt
    prompt="$(printf 'Summarize the role of multi-group KV cache in DeepSeek-V4 disaggregated prefill. %.0s' {1..200})"
    local smoke_url="${SMOKE_URL:-http://${PROXY_HOST}:${PROXY_PORT}/v1/chat/completions}"
    echo "Sending smoke request to ${smoke_url}..."
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
)" | tee "${LOG_DIR:-/tmp}/smoke_2p2d_p2p.json"
    echo
    echo "Check Prefill logs for P2P retrieve / PD Stored lines."
}

case "$MODE" in
    launch)
        launch_all
        echo "Press Ctrl-C to stop."
        while true; do sleep 1; done
        ;;
    stop)
        cleanup
        ;;
    test)
        run_smoke_test
        ;;
    *)
        echo "Usage: $0 [launch|stop|test] [--debug]"
        exit 1
        ;;
esac
