#!/bin/bash
# Launch prefiller + proxy on this node (default: 192.168.0.223).
#
# Usage:
#   ./scripts/serving/pd_prefiller_node.sh launch
#   ./scripts/serving/pd_prefiller_node.sh --disk launch

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

export LOCAL_IP="${LOCAL_IP:-${PREFILL_HOST}}"
export PD_INSTANCE=prefill
export LAUNCH_PROXY=1

if [[ "$USE_DISK" -eq 1 ]]; then
    export LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/pd_disagg_dsv4_disk_prefill_node}"
    export LMCACHE_LOCAL_DISK="${LMCACHE_LOCAL_DISK:-/tmp/lmcache_disk_pd}"
    LAUNCHER="${SCRIPT_DIR}/pd_disagg_disk.sh"
    echo "=== Prefiller node (PD + LocalDisk L2) ==="
else
    export LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/pd_disagg_dsv4_prefill_node}"
    LAUNCHER="${SCRIPT_DIR}/pd_disagg.sh"
    echo "=== Prefiller node (HCCL PD) ==="
fi

echo "LOCAL_IP=${LOCAL_IP} PREFILL_HOST=${PREFILL_HOST} DECODE_HOST=${DECODE_HOST} PROXY_HOST=${PROXY_HOST}"
exec "$LAUNCHER" "${ARGS[@]}"
