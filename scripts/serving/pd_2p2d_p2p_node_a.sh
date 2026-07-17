#!/bin/bash
# Machine A (default 192.168.0.223): controller + proxy + P0 + D0
#
# Usage (prepare only — do not run until ready):
#   ./scripts/serving/pd_2p2d_p2p_node_a.sh launch
#   ./scripts/serving/pd_2p2d_p2p_node_a.sh stop
#
# Optional: ALLOW_PD_P2P=1 ENABLE_P2P=1 ENABLE_PD=1 (needs LMCache assert patch)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env/pd_2p2d_p2p_env.sh"

export NODE_ROLE=a
export LOCAL_IP="${LOCAL_IP:-${HOST_A}}"
export LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/pd_2p2d_p2p_a}"

echo "=== Node A: controller + proxy + P0 (NPUs ${PREFILL_NPUS}) + D0 (NPUs ${DECODE_NPUS}) ==="
echo "LOCAL_IP=${LOCAL_IP} HOST_A=${HOST_A} HOST_B=${HOST_B}"
exec "${SCRIPT_DIR}/pd_2p2d_p2p.sh" "$@"
