# Shared helpers for HCCL PD + Mooncake L2. Sourced by pd_mooncake_*_node.sh.
# shellcheck shell=bash

MOONCAKE_RPC_PORT="${MOONCAKE_RPC_PORT:-50051}"
MOONCAKE_HTTP_PORT="${MOONCAKE_HTTP_PORT:-8080}"
MOONCAKE_HOST="${MOONCAKE_HOST:-${PREFILL_HOST}}"

_pd_mooncake_subst() {
    local src=$1
    local dst=$2
    sed \
        -e "s/__MOONCAKE_HOST__/${MOONCAKE_HOST}/g" \
        -e "s/__PREFILL_HOST__/${PREFILL_HOST}/g" \
        -e "s/__DECODE_HOST__/${DECODE_HOST}/g" \
        -e "s/__PROXY_HOST__/${PROXY_HOST}/g" \
        "$src" >"$dst"
}

pd_mooncake_write_configs() {
    local cfg_dir="${1:-${LOG_DIR:-${REPO_ROOT}/logs}/configs}"
    local tpl_dir="${REPO_ROOT}/scripts/serving/configs"
    mkdir -p "$cfg_dir"
    _pd_mooncake_subst \
        "${tpl_dir}/lmcache-dsv4-prefiller-mooncake.yaml" \
        "${cfg_dir}/lmcache-dsv4-prefiller-mooncake.yaml"
    _pd_mooncake_subst \
        "${tpl_dir}/lmcache-dsv4-decoder-mooncake.yaml" \
        "${cfg_dir}/lmcache-dsv4-decoder-mooncake.yaml"
    export PREFILL_LMCACHE_CONFIG="${cfg_dir}/lmcache-dsv4-prefiller-mooncake.yaml"
    export DECODE_LMCACHE_CONFIG="${cfg_dir}/lmcache-dsv4-decoder-mooncake.yaml"
    echo "Mooncake LMCache configs:"
    echo "  prefill: ${PREFILL_LMCACHE_CONFIG}"
    echo "  decode:  ${DECODE_LMCACHE_CONFIG}"
    echo "  mooncake_host=${MOONCAKE_HOST} rpc=${MOONCAKE_RPC_PORT} http=${MOONCAKE_HTTP_PORT}"
}

pd_mooncake_start_master() {
    local log="${LOG_DIR:-.}/mooncake_master.log"
    mkdir -p "$(dirname "$log")"
    if command -v mooncake_master >/dev/null 2>&1; then
        echo "Starting mooncake_master rpc=${MOONCAKE_RPC_PORT} http=${MOONCAKE_HTTP_PORT} log=${log}"
        mooncake_master \
            --enable_http_metadata_server=true \
            --http_metadata_server_port="${MOONCAKE_HTTP_PORT}" \
            --rpc_port="${MOONCAKE_RPC_PORT}" \
            -v=1 \
            >"${log}" 2>&1 &
        echo $! >"${LOG_DIR:-.}/mooncake_master.pid"
    else
        echo "WARNING: mooncake_master not on PATH; start it before serving."
        echo "  mooncake_master --enable_http_metadata_server=true --http_metadata_server_port=${MOONCAKE_HTTP_PORT} --rpc_port=${MOONCAKE_RPC_PORT}"
    fi
}
