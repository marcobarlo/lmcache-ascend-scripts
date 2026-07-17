#!/bin/bash
# Machine B (default 192.168.0.110): P1 + D1 (P2P peer of Node A Prefill)
#
# Usage (prepare only — do not run until ready):
#   # After Node A is up (controller listening on HOST_A:9800/9900):
#   ./scripts/serving/pd_2p2d_p2p_node_b.sh launch
#   ./scripts/serving/pd_2p2d_p2p_node_b.sh stop

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env/pd_2p2d_p2p_env.sh"

export NODE_ROLE=b
export LOCAL_IP="${LOCAL_IP:-${HOST_B}}"
export LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/pd_2p2d_p2p_b}"

echo "=== Node B: P1 (NPUs ${PREFILL_NPUS}) + D1 (NPUs ${DECODE_NPUS}) ==="
echo "LOCAL_IP=${LOCAL_IP} HOST_A=${HOST_A} HOST_B=${HOST_B} controller=${CONTROLLER_HOST}:${CONTROLLER_PULL_PORT}"
exec "${SCRIPT_DIR}/pd_2p2d_p2p.sh" "$@"
