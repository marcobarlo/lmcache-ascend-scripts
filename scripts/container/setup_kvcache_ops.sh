#!/bin/bash
# Wire top-level third_party/kvcache-ops into LMCache-Ascend before build.
# Run from repo root (host or container).
#
# Replaces LMCache-Ascend/third_party/kvcache-ops with a symlink to the
# pinned submodule at third_party/kvcache-ops (no kernel copy).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

if [[ -f "${REPO_ROOT}/versions.env" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/versions.env"
fi

CONTAINER_LMCACHE_ASCEND_MOUNT="${CONTAINER_LMCACHE_ASCEND_MOUNT:-/workspace/LMCache-Ascend}"
LMCACHE_ASCEND_DIR="${LMCACHE_ASCEND_DIR:-${CONTAINER_LMCACHE_ASCEND_MOUNT}}"

# When invoked on host, paths are under REPO_ROOT.
if [[ -d "${REPO_ROOT}/LMCache-Ascend" ]]; then
    LMCACHE_ASCEND_DIR="${REPO_ROOT}/LMCache-Ascend"
fi

KVCACHE_OPS_SRC="${REPO_ROOT}/third_party/kvcache-ops"
KVCACHE_OPS_DST="${LMCACHE_ASCEND_DIR}/third_party/kvcache-ops"

if [[ ! -d "${KVCACHE_OPS_SRC}/.git" && ! -f "${KVCACHE_OPS_SRC}/CMakeLists.txt" ]]; then
    echo "ERROR: kvcache-ops submodule missing at ${KVCACHE_OPS_SRC}"
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

if [[ -n "${KVCACHE_OPS_COMMIT:-}" && -d "${KVCACHE_OPS_SRC}/.git" ]]; then
    echo "Checking out kvcache-ops @ ${KVCACHE_OPS_COMMIT} ..."
    git -C "${KVCACHE_OPS_SRC}" fetch --depth 1 origin "${KVCACHE_OPS_COMMIT}" 2>/dev/null || true
    git -C "${KVCACHE_OPS_SRC}" checkout "${KVCACHE_OPS_COMMIT}"
fi

mkdir -p "${LMCACHE_ASCEND_DIR}/third_party"

if [[ -L "${KVCACHE_OPS_DST}" ]]; then
    rm -f "${KVCACHE_OPS_DST}"
elif [[ -d "${KVCACHE_OPS_DST}/.git" ]]; then
    echo "Deinitializing nested kvcache-ops submodule ..."
    git -C "${LMCACHE_ASCEND_DIR}" submodule deinit -f third_party/kvcache-ops 2>/dev/null || true
    rm -rf "${KVCACHE_OPS_DST}"
elif [[ -e "${KVCACHE_OPS_DST}" ]]; then
    rm -rf "${KVCACHE_OPS_DST}"
fi

# Relative symlink: LMCache-Ascend/third_party/kvcache-ops -> ../../third_party/kvcache-ops
ln -sfn ../../third_party/kvcache-ops "${KVCACHE_OPS_DST}"

echo "Linked ${KVCACHE_OPS_DST} -> ${KVCACHE_OPS_SRC}"
ls -la "${KVCACHE_OPS_DST}"
