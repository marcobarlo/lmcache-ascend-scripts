#!/bin/bash
# Wrapper: Node B = 192.168.0.110 (colocated peer instance)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NODE_ROLE=b
export LOCAL_IP="${LOCAL_IP:-192.168.0.110}"
export HOST_A="${HOST_A:-192.168.0.223}"
export HOST_B="${HOST_B:-192.168.0.110}"
export LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/p2p_colocated_b}"
exec "${SCRIPT_DIR}/p2p_colocated_2node.sh" "$@"
