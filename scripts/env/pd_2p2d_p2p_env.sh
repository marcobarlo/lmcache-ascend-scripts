#!/bin/bash
# Shared env for 2P+2D DeepSeek-V4 with Prefill↔Prefill P2P KV sharing.
#
# Topology (16 logical NPUs per machine = 8 cards × 2 dies):
#   Machine A (192.168.0.223): P0 NPUs 0-7  + D0 NPUs 8-15
#   Machine B (192.168.0.110): P1 NPUs 0-7  + D1 NPUs 8-15
#   P2P:        P0 ↔ P1 (cross-host, via lmcache_controller on A)
#   PD pairs:   P0 ↔ D0 (local), P1 ↔ D1 (local)
#   Proxy:      Machine A :9100 (2 prefillers + 2 decoders)
#
# IMPORTANT — upstream LMCache constraint:
#   lmcache/v1/config.py asserts enable_pd XOR enable_p2p
#   ("PD only supports enable_p2p=False"). Prefillers therefore cannot
#   legally enable both PD and P2P until that assert is lifted/patched.
#   This env still configures both; see ENABLE_P2P / ENABLE_PD on Prefill
#   in pd_2p2d_p2p.sh. Default keeps PD on Prefill+Decode and P2P on
#   Prefill only when ALLOW_PD_P2P=1 (user acknowledges the assert).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${REPO_ROOT}/versions.env" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/versions.env"
fi

# --- hosts ---
export HOST_A="${HOST_A:-192.168.0.223}"
export HOST_B="${HOST_B:-192.168.0.110}"
export PROXY_HOST="${PROXY_HOST:-${HOST_A}}"
export CONTROLLER_HOST="${CONTROLLER_HOST:-${HOST_A}}"
export NIC_NAME="${NIC_NAME:-enp23s0f3}"

# --- model ---
export MODEL_PATH="${MODEL_PATH:-/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
export MODEL_NAME="${MODEL_NAME:-dsv4}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
export SAVE_ONLY_FIRST_RANK="${SAVE_ONLY_FIRST_RANK:-true}"
export SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-360}"

# --- NPUs (16 logical devices per host) ---
export PREFILL_NPUS="${PREFILL_NPUS:-0,1,2,3,4,5,6,7}"
export DECODE_NPUS="${DECODE_NPUS:-8,9,10,11,12,13,14,15}"

# --- HTTP / PD ports (same on each host; distinct roles) ---
export PREFILL_PORT="${PREFILL_PORT:-7100}"
export DECODE_PORT="${DECODE_PORT:-7200}"
export PROXY_PORT="${PROXY_PORT:-9100}"
export PROXY_ZMQ_PORT="${PROXY_ZMQ_PORT:-7500}"
export INIT_PORT_BASE="${INIT_PORT_BASE:-7300}"
export ALLOC_PORT_BASE="${ALLOC_PORT_BASE:-7400}"
export DONE_PORT_BASE="${DONE_PORT_BASE:-7600}"

# --- PD buffer / LMCache ---
export PD_BUFFER_SIZE="${PD_BUFFER_SIZE:-1073741824}"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-1024}"
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-5}"

# --- controller (runs on HOST_A) ---
export CONTROLLER_PORT="${CONTROLLER_PORT:-9000}"
export CONTROLLER_PULL_PORT="${CONTROLLER_PULL_PORT:-9800}"
export CONTROLLER_REPLY_PORT="${CONTROLLER_REPLY_PORT:-9900}"

# --- P2P ports (per TP rank; same numbers OK across hosts) ---
export P2P_INIT_PORT_BASE="${P2P_INIT_PORT_BASE:-8200}"
export P2P_LOOKUP_PORT_BASE="${P2P_LOOKUP_PORT_BASE:-8210}"
export LMCACHE_WORKER_PORT_BASE="${LMCACHE_WORKER_PORT_BASE:-8500}"
export P2P_NPU_BUFFER_SIZE="${P2P_NPU_BUFFER_SIZE:-134217728}"
export P2P_USE_NPU="${P2P_USE_NPU:-True}"
export P2P_PULL_MODE="${P2P_PULL_MODE:-True}"
export P2P_DELAY_PULL="${P2P_DELAY_PULL:-True}"

# Consistent token hashing across P2P peers (required).
export PYTHONHASHSEED="${PYTHONHASHSEED:-123}"

# Feature flags (see header comment).
export ENABLE_PD="${ENABLE_PD:-1}"
export ENABLE_P2P="${ENABLE_P2P:-1}"
# Set ALLOW_PD_P2P=1 only if LMCache assert has been removed/patched.
export ALLOW_PD_P2P="${ALLOW_PD_P2P:-0}"

export LMCACHE_ROOT="${LMCACHE_ROOT:-${CONTAINER_LMCACHE_MOUNT:-/workspace/LMCache}}"
