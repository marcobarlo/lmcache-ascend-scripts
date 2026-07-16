#!/bin/bash
# Parameterized Ascend vLLM container bootstrap for DSv4 serving.
#
# Run on each Ascend host before install_editable.sh (inside container).
#
# Env (override on host):
#   REPO_ROOT          — path to this repo on host
#   MODEL_MOUNT        — host path to model directory parent (contains DeepSeek-V4-Flash-w8a8-mtp)
#   CACHE_MOUNT        — host path for container /root/.cache
#   IMAGE, NAME        — docker image and container name
#   NPU_COUNT          — number of davinci devices to pass through (default 16)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

if [[ -f "${REPO_ROOT}/versions.env" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/versions.env"
fi

export IMAGE="${IMAGE:-${VLLM_ASCEND_IMAGE:-quay.io/ascend/vllm-ascend:v0.20.2rc1-a3}}"
export NAME="${NAME:-vllm-ascend-dsv4-serving}"
NPU_COUNT="${NPU_COUNT:-16}"

CONTAINER_REPO_MOUNT="${CONTAINER_REPO_MOUNT:-/workspace/dsv4-serving}"
CONTAINER_LMCACHE_MOUNT="${CONTAINER_LMCACHE_MOUNT:-/workspace/LMCache}"
CONTAINER_LMCACHE_ASCEND_MOUNT="${CONTAINER_LMCACHE_ASCEND_MOUNT:-/workspace/LMCache-Ascend}"
CONTAINER_MODEL_MOUNT="${CONTAINER_MODEL_MOUNT:-/workspace/models}"

MODEL_MOUNT="${MODEL_MOUNT:-/mnt/sdb/models}"
CACHE_MOUNT="${CACHE_MOUNT:-${HOME}/.cache}"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "Removing existing container $NAME ..."
    docker rm -f "$NAME" >/dev/null
fi

mkdir -p "$CACHE_MOUNT"

DEVICES=()
for ((i = 0; i < NPU_COUNT; i++)); do
    DEVICES+=(--device "/dev/davinci${i}")
done

NPU_LIST=$(seq -s, 0 $((NPU_COUNT - 1)))

docker run -dit --privileged \
    --name "$NAME" \
    --net=host \
    --shm-size=512g \
    -e "ASCEND_VISIBLE_DEVICES=${NPU_LIST}" \
    "${DEVICES[@]}" \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v "${REPO_ROOT}:${CONTAINER_REPO_MOUNT}" \
    -v "${REPO_ROOT}/LMCache:${CONTAINER_LMCACHE_MOUNT}" \
    -v "${REPO_ROOT}/LMCache-Ascend:${CONTAINER_LMCACHE_ASCEND_MOUNT}" \
    -v "${MODEL_MOUNT}:${CONTAINER_MODEL_MOUNT}" \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v "${CACHE_MOUNT}:/root/.cache" \
    -it "$IMAGE" bash

echo "Started $NAME"
echo "  repo:      ${REPO_ROOT} -> ${CONTAINER_REPO_MOUNT}"
echo "  LMCache:   ${REPO_ROOT}/LMCache -> ${CONTAINER_LMCACHE_MOUNT}"
echo "  Ascend:    ${REPO_ROOT}/LMCache-Ascend -> ${CONTAINER_LMCACHE_ASCEND_MOUNT}"
echo "  models:    ${MODEL_MOUNT} -> ${CONTAINER_MODEL_MOUNT}"
docker ps --filter "name=$NAME" --format '{{.Names}} {{.Status}}'
