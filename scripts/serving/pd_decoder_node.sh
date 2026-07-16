#!/bin/bash
# Launch decoder on this node (default: 192.168.0.110).
#
# Usage:
#   ./scripts/serving/pd_decoder_node.sh launch
#   ./scripts/serving/pd_decoder_node.sh --disk launch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

USE_DISK=0
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --disk) USE_DISK=1 ;;
        -h|--help)
            echo "Usage: $0 [--disk] [launch|test|stop|all] [--debug]"
            exit 0
            ;;
        *) ARGS+=("$arg") ;;
    esac
done

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env/pd_dsv4_env.sh"

export LOCAL_IP="${LOCAL_IP:-${DECODE_HOST}}"
export PD_INSTANCE=decode
export LAUNCH_PROXY=0

if [[ "$USE_DISK" -eq 1 ]]; then
    export LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/pd_disagg_dsv4_disk_decode_node}"
    export LMCACHE_LOCAL_DISK="${LMCACHE_LOCAL_DISK:-/tmp/lmcache_disk_pd}"
    LAUNCHER="${SCRIPT_DIR}/pd_disagg_disk.sh"
    echo "=== Decoder node (PD + LocalDisk L2) ==="
else
    export LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/pd_disagg_dsv4_decode_node}"
    LAUNCHER="${SCRIPT_DIR}/pd_disagg.sh"
    echo "=== Decoder node (HCCL PD) ==="
fi

echo "LOCAL_IP=${LOCAL_IP} PREFILL_HOST=${PREFILL_HOST} DECODE_HOST=${DECODE_HOST} PROXY_HOST=${PROXY_HOST}"
echo "MODEL_PATH=${MODEL_PATH}"
if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "WARNING: model not found at ${MODEL_PATH}."
    echo "Ensure the model is mounted under /workspace/models before serving."
fi

exec "$LAUNCHER" "${ARGS[@]}"
