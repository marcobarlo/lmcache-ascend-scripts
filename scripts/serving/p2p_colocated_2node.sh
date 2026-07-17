#!/bin/bash
# Two colocated (kv_both) DeepSeek-V4 instances with Prefill/Decode on the same
# engine, sharing KV via LMCache P2P across hosts.
#
# Hosts (LAN):
#   192.168.0.223  (NODE_ROLE=a, controller + instance)
#   192.168.0.110  (NODE_ROLE=b, peer instance)
#
# Free NPUs (probed 2026-07-16):
#   A (.223): cards 4-7 free → ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
#   B (.110): cards 0-3 free → ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#   (card 7 on B is busy; cards 0-3 on A are busy)
#
# Usage (inside container vllm-ascend-latest-lmcache_dsv4_marco):
#   NODE_ROLE=a LOCAL_IP=192.168.0.223 ./p2p_colocated_2node.sh launch
#   NODE_ROLE=b LOCAL_IP=192.168.0.110 ./p2p_colocated_2node.sh launch
#   NODE_ROLE=a|b ./p2p_colocated_2node.sh stop
#   ./p2p_colocated_2node.sh test   # hit instance A then B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEBUG_MODE=0
MODE="launch"
for arg in "$@"; do
    case "$arg" in
        --debug) DEBUG_MODE=1 ;;
        launch|stop|test) MODE="$arg" ;;
        -h|--help)
            echo "Usage: NODE_ROLE=a|b $0 [launch|stop|test] [--debug]"
            exit 0
            ;;
        *) echo "Unknown: $arg"; exit 1 ;;
    esac
done

# --- topology ---
HOST_A="${HOST_A:-192.168.0.223}"
HOST_B="${HOST_B:-192.168.0.110}"
NIC_NAME="${NIC_NAME:-enp23s0f3}"

CONTROLLER_HOST="${CONTROLLER_HOST:-${HOST_A}}"
CONTROLLER_PORT="${CONTROLLER_PORT:-9000}"
CONTROLLER_PULL_PORT="${CONTROLLER_PULL_PORT:-9800}"
CONTROLLER_REPLY_PORT="${CONTROLLER_REPLY_PORT:-9900}"

PORT_A="${PORT_A:-8010}"
PORT_B="${PORT_B:-8011}"

# Free 8 logical devices per host (see header)
NPUS_A="${NPUS_A:-8,9,10,11,12,13,14,15}"
NPUS_B="${NPUS_B:-0,1,2,3,4,5,6,7}"

P2P_INIT_PORT_BASE="${P2P_INIT_PORT_BASE:-8200}"
P2P_LOOKUP_PORT_BASE="${P2P_LOOKUP_PORT_BASE:-8210}"
LMCACHE_WORKER_PORT_BASE="${LMCACHE_WORKER_PORT_BASE:-8500}"
P2P_NPU_BUFFER_SIZE="${P2P_NPU_BUFFER_SIZE:-134217728}"

MODEL_PATH="${MODEL_PATH:-/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
MODEL_NAME="${MODEL_NAME:-dsv4}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-16}"
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-1024}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-360}"
PYTHONHASHSEED="${PYTHONHASHSEED:-123}"

NODE_ROLE="${NODE_ROLE:-}"
if [[ "$MODE" != "test" && -z "$NODE_ROLE" ]]; then
    echo "NODE_ROLE=a|b required"
    exit 1
fi

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/p2p_colocated_${NODE_ROLE:-x}}"
CFG_DIR="${CFG_DIR:-${LOG_DIR}/configs}"
PID_FILE="${PID_FILE:-${LOG_DIR}/pids}"
PIDS=()

yaml_port_list() {
    local start=$1 count=$2 ports="" i
    for ((i = 0; i < count; i++)); do
        [[ -n "$ports" ]] && ports+=", "
        ports+=$((start + i))
    done
    echo "[${ports}]"
}

setup_env() {
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
    export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
    export HCCL_OP_EXPANSION_MODE=AIV
    export TASK_QUEUE_ENABLE=1
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export VLLM_ENABLE_V1_MULTIPROCESSING=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export PYTHONHASHSEED
    export LMCACHE_TRACK_USAGE=false
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    if [[ -n "${LOCAL_IP:-}" ]]; then
        export HCCL_IF_IP="$LOCAL_IP"
    fi
    if [[ "$DEBUG_MODE" -eq 1 ]]; then
        export LMCACHE_LOG_LEVEL=DEBUG
        export VLLM_LOGGING_LEVEL=DEBUG
    else
        export LMCACHE_LOG_LEVEL=INFO
        export VLLM_LOGGING_LEVEL=INFO
        export ASCEND_GLOBAL_LOG_LEVEL=3
    fi
}

record_pid() { mkdir -p "$LOG_DIR"; echo "$1" >> "$PID_FILE"; }

cleanup() {
    echo "Stopping p2p colocated (NODE_ROLE=${NODE_ROLE})..."
    trap - INT TERM USR1
    if [[ -f "$PID_FILE" ]]; then
        while read -r pid; do
            [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
    for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
    sleep 2
    for pid in "${PIDS[@]:-}"; do kill -9 "$pid" 2>/dev/null || true; done
    # also kill orphaned vllm on our port
    local port
    if [[ "$NODE_ROLE" == "a" ]]; then port=$PORT_A; else port=$PORT_B; fi
    pkill -f "vllm serve.*--port ${port}" 2>/dev/null || true
    if [[ "$NODE_ROLE" == "a" ]]; then
        pkill -f "lmcache_controller.*--port ${CONTROLLER_PORT}" 2>/dev/null || true
    fi
    echo "Stopped."
}

wait_for_server() {
    local port=$1 host="${2:-127.0.0.1}" timeout="${3:-$SERVER_WAIT_TIMEOUT}"
    local start
    start=$(date +%s)
    echo "Waiting for http://${host}:${port}/v1/models ..."
    while true; do
        if curl -sf "http://${host}:${port}/v1/models" >/dev/null 2>&1; then
            echo "Ready ${host}:${port}"
            return 0
        fi
        if (( $(date +%s) - start >= timeout )); then
            echo "Timeout ${host}:${port}"
            return 1
        fi
        sleep 5
    done
}

write_config() {
    mkdir -p "$CFG_DIR"
    local local_host instance_id
    if [[ "$NODE_ROLE" == "a" ]]; then
        local_host="$HOST_A"
        instance_id="lmcache_colocated_a"
    else
        local_host="$HOST_B"
        instance_id="lmcache_colocated_b"
    fi
    cat > "${CFG_DIR}/lmcache-p2p.yaml" <<EOF
chunk_size: ${LMCACHE_CHUNK_SIZE}
local_cpu: True
max_local_cpu_size: ${LMCACHE_MAX_LOCAL_CPU_SIZE}
enable_async_loading: True
use_layerwise: False
numa_mode: "auto"
save_unfull_chunk: False

# P2P KV sharing (no enable_pd — colocated kv_both)
enable_p2p: True
p2p_host: "${local_host}"
p2p_init_ports: $(yaml_port_list "$P2P_INIT_PORT_BASE" "$TENSOR_PARALLEL_SIZE")
p2p_lookup_ports: $(yaml_port_list "$P2P_LOOKUP_PORT_BASE" "$TENSOR_PARALLEL_SIZE")
transfer_channel: "hccl"
p2p_use_npu: True
p2p_pull_mode: True
p2p_delay_pull: ${P2P_DELAY_PULL:-False}
p2p_npu_buffer_size: ${P2P_NPU_BUFFER_SIZE}

enable_controller: True
lmcache_instance_id: "${instance_id}"
controller_pull_url: "${CONTROLLER_HOST}:${CONTROLLER_PULL_PORT}"
controller_reply_url: "${CONTROLLER_HOST}:${CONTROLLER_REPLY_PORT}"
lmcache_worker_ports: $(yaml_port_list "$LMCACHE_WORKER_PORT_BASE" "$TENSOR_PARALLEL_SIZE")

extra_config:
  save_only_first_rank: true
  lookup_backoff_time: 0.001
EOF
    echo "Wrote ${CFG_DIR}/lmcache-p2p.yaml (host=${local_host} id=${instance_id})"
}

launch_controller() {
    local log="${LOG_DIR}/controller.log"
    local ports_json
    ports_json=$(printf '{"pull": %s, "reply": %s}' "$CONTROLLER_PULL_PORT" "$CONTROLLER_REPLY_PORT")
    echo "lmcache_controller 0.0.0.0:${CONTROLLER_PORT} ${ports_json}"
    PYTHONHASHSEED="$PYTHONHASHSEED" lmcache_controller \
        --host 0.0.0.0 \
        --port "$CONTROLLER_PORT" \
        --monitor-ports "$ports_json" \
        >"$log" 2>&1 &
    local pid=$!
    PIDS+=("$pid"); record_pid "$pid"
    echo "Controller pid=${pid} log=${log}"
    sleep 2
}

launch_instance() {
    local port npus
    if [[ "$NODE_ROLE" == "a" ]]; then
        port=$PORT_A; npus=$NPUS_A
    else
        port=$PORT_B; npus=$NPUS_B
    fi
    export LMCACHE_CONFIG_FILE="${CFG_DIR}/lmcache-p2p.yaml"
    export ASCEND_RT_VISIBLE_DEVICES="$npus"
    local addcfg='{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}'
    local kvcfg='{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both","kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector","kv_connector_extra_config":{"discard_partial_chunks":true}}'

    echo "vllm serve port=${port} NPUs=${npus} role=kv_both"
    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$port" \
        --served-model-name "$MODEL_NAME" \
        --no-enable-prefix-caching \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 16 \
        --api-server-count 1 \
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
        --speculative-config '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}' \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --block-size 128 \
        --no-disable-hybrid-kv-cache-manager \
        --async-scheduling \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --additional-config "$addcfg" \
        --kv-transfer-config "$kvcfg" \
        >"${LOG_DIR}/instance.log" 2>&1 &
    local pid=$!
    PIDS+=("$pid"); record_pid "$pid"
    echo "Instance pid=${pid} log=${LOG_DIR}/instance.log"
}

launch_all() {
    mkdir -p "$LOG_DIR"
    : > "$PID_FILE"
    setup_env
    write_config
    trap cleanup INT TERM USR1

    echo "=== P2P colocated 2-node ==="
    echo "NODE_ROLE=${NODE_ROLE} LOCAL_IP=${LOCAL_IP:-} NIC=${NIC_NAME}"
    echo "A ${HOST_A} NPUs=${NPUS_A} port=${PORT_A}"
    echo "B ${HOST_B} NPUs=${NPUS_B} port=${PORT_B}"
    echo "controller ${CONTROLLER_HOST}:${CONTROLLER_PORT} pull=${CONTROLLER_PULL_PORT} reply=${CONTROLLER_REPLY_PORT}"
    echo "Logs ${LOG_DIR}"

    if [[ "$NODE_ROLE" == "a" ]]; then
        launch_controller
        launch_instance
        wait_for_server "$PORT_A"
    elif [[ "$NODE_ROLE" == "b" ]]; then
        launch_instance
        wait_for_server "$PORT_B"
    else
        echo "NODE_ROLE must be a or b"; exit 1
    fi
    echo "Node ${NODE_ROLE} ready."
}

run_smoke() {
    local prompt
    prompt="$(printf 'Explain KV cache P2P sharing across Ascend instances. %.0s' {1..80})"
    local body
    body=$(python3 - <<PY
import json
print(json.dumps({
  "model": "${MODEL_NAME}",
  "messages": [{"role":"user","content":"""${prompt}"""}],
  "max_tokens": 16,
  "temperature": 0,
}))
PY
)
    echo "=== hit A ${HOST_A}:${PORT_A} ==="
    curl -sf -X POST "http://${HOST_A}:${PORT_A}/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$body" | tee /tmp/p2p_colocated_a.json | tail -c 400
    echo
    echo "=== hit B ${HOST_B}:${PORT_B} (should P2P-retrieve from A) ==="
    curl -sf -X POST "http://${HOST_B}:${PORT_B}/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$body" | tee /tmp/p2p_colocated_b.json | tail -c 400
    echo
}

case "$MODE" in
    launch)
        launch_all
        echo "Press Ctrl-C to stop."
        while true; do sleep 1; done
        ;;
    stop) cleanup ;;
    test) run_smoke ;;
    *) echo "Usage: $0 [launch|stop|test]"; exit 1 ;;
esac
