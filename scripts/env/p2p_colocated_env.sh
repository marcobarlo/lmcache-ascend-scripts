#!/bin/bash
# Shared env for two colocated (kv_both) DeepSeek-V4 instances with LMCache P2P.
#
# Topology:
#   Machine A (192.168.0.223): controller + instance :8010, NPUs 8-15
#   Machine B (192.168.0.110): peer instance :8011, NPUs 0-7
#   P2P: A ↔ B via lmcache_controller on A (no enable_pd)
#
# Launch (inside container):
#   source scripts/env/p2p_colocated_env.sh   # optional overrides
#   ./scripts/serving/p2p_colocated_node_a.sh launch
#   ./scripts/serving/p2p_colocated_node_b.sh launch
# Smoke:
#   ./scripts/benchmark/query4.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${REPO_ROOT}/versions.env" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/versions.env"
fi

export HOST_A="${HOST_A:-192.168.0.223}"
export HOST_B="${HOST_B:-192.168.0.110}"
export CONTROLLER_HOST="${CONTROLLER_HOST:-${HOST_A}}"
export NIC_NAME="${NIC_NAME:-enp23s0f3}"

export PORT_A="${PORT_A:-8010}"
export PORT_B="${PORT_B:-8011}"
export NPUS_A="${NPUS_A:-8,9,10,11,12,13,14,15}"
export NPUS_B="${NPUS_B:-0,1,2,3,4,5,6,7}"

export MODEL_PATH="${MODEL_PATH:-/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
export MODEL_NAME="${MODEL_NAME:-dsv4}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"

# Required for save_only_first_rank + P2P (delay-pull crashes MLA broadcast)
export P2P_DELAY_PULL="${P2P_DELAY_PULL:-False}"
