#!/bin/bash
# In-container editable install of LMCache + LMCache-Ascend.
#
# Usage (on host):
#   docker exec -u root "$NAME" bash /workspace/dsv4-serving/scripts/container/install_editable.sh
#
# Or inside container:
#   bash scripts/container/install_editable.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

if [[ -f "${REPO_ROOT}/versions.env" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/versions.env"
fi

CONTAINER_REPO_MOUNT="${CONTAINER_REPO_MOUNT:-/workspace/dsv4-serving}"
CONTAINER_LMCACHE_MOUNT="${CONTAINER_LMCACHE_MOUNT:-/workspace/LMCache}"
CONTAINER_LMCACHE_ASCEND_MOUNT="${CONTAINER_LMCACHE_ASCEND_MOUNT:-/workspace/LMCache-Ascend}"

# Detect container vs host layout.
if [[ -d "${CONTAINER_LMCACHE_MOUNT}" ]]; then
    LMCACHE_DIR="${CONTAINER_LMCACHE_MOUNT}"
    LMCACHE_ASCEND_DIR="${CONTAINER_LMCACHE_ASCEND_MOUNT}"
    REPO_FOR_SETUP="${CONTAINER_REPO_MOUNT}"
else
    LMCACHE_DIR="${REPO_ROOT}/LMCache"
    LMCACHE_ASCEND_DIR="${REPO_ROOT}/LMCache-Ascend"
    REPO_FOR_SETUP="${REPO_ROOT}"
fi

export REPO_ROOT="${REPO_FOR_SETUP}"

echo "=== setup_kvcache_ops ==="
bash "${REPO_FOR_SETUP}/scripts/container/setup_kvcache_ops.sh"

git config --global --add safe.directory "${LMCACHE_DIR}" || true
git config --global --add safe.directory "${LMCACHE_ASCEND_DIR}" || true

if [[ -n "${LMCACHE_COMMIT:-}" && -d "${LMCACHE_DIR}/.git" ]]; then
    git -C "${LMCACHE_DIR}" checkout "${LMCACHE_COMMIT}"
fi
if [[ -n "${LMCACHE_ASCEND_COMMIT:-}" && -d "${LMCACHE_ASCEND_DIR}/.git" ]]; then
    git -C "${LMCACHE_ASCEND_DIR}" checkout "${LMCACHE_ASCEND_COMMIT}"
fi

echo "=== pip install LMCache (pure Python, no CUDA ext) ==="
cd "${LMCACHE_DIR}"
NO_CUDA_EXT=1 pip install -e . --no-build-isolation --no-deps
pip install -q aiofile aiofiles awscrt nvtx \
  "opentelemetry-api<=1.40.0,>=1.20.0" \
  "opentelemetry-exporter-prometheus<=0.61b0,>=0.50b0" \
  opentelemetry-sdk opentelemetry-exporter-otlp \
  "prometheus_client<=0.24.1"

echo "=== install LMCache-Ascend (builds kvcache-ops) ==="
if [[ -x "${LMCACHE_ASCEND_DIR}/install.sh" ]]; then
    bash "${LMCACHE_ASCEND_DIR}/install.sh"
else
    pip install -e "${LMCACHE_ASCEND_DIR}" --no-build-isolation
fi

python3 -c "import lmcache, lmcache_ascend; print('OK', lmcache.__file__, lmcache_ascend.__file__)"
echo "Install complete."
