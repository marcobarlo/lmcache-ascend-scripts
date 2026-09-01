#!/bin/bash
# Prefiller + proxy with LMCache-Ascend HCCL PD and Mooncake as L2 store.
#
# Usage:
#   PREFILL_HOST=... DECODE_HOST=... ./scripts/serving/pd_mooncake_prefiller_node.sh launch
#
# Starts mooncake_master (RPC 50051, HTTP metadata 8080) unless
# START_MOONCAKE_MASTER=0. Instantiates LMCache YAML from configs/ templates
# using PREFILL_HOST / DECODE_HOST / PROXY_HOST / MOONCAKE_HOST.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env/pd_dsv4_env.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/pd_mooncake_common.sh"

export LOCAL_IP="${LOCAL_IP:-${PREFILL_HOST}}"
export LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/pd_mooncake_prefill_node}"
export START_MOONCAKE_MASTER="${START_MOONCAKE_MASTER:-1}"

mkdir -p "$LOG_DIR"
pd_mooncake_write_configs "${LOG_DIR}/configs"

if [[ "${START_MOONCAKE_MASTER}" == "1" || "${START_MOONCAKE_MASTER}" == "true" ]]; then
    pd_mooncake_start_master
fi

echo "=== Prefiller node (HCCL PD + Mooncake L2) ==="
echo "LOCAL_IP=${LOCAL_IP} PREFILL_HOST=${PREFILL_HOST} DECODE_HOST=${DECODE_HOST} PROXY_HOST=${PROXY_HOST}"
exec "${SCRIPT_DIR}/pd_prefiller_node.sh" "$@"
