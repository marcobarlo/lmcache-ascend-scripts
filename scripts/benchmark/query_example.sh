#!/bin/bash
# PD disagg smoke request — uses payload.json as the request body.
# Targets the disagg proxy (default :9100), not a standalone vLLM port.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROXY_PORT="${PROXY_PORT:-9100}"
URL="${URL:-http://127.0.0.1:${PROXY_PORT}/v1/chat/completions}"
OUT="${OUT:-${REPO_ROOT}/artifacts/smoke/smoke_response.json}"
PAYLOAD="${PAYLOAD:-${SCRIPT_DIR}/payload.json}"

mkdir -p "$(dirname "$OUT")"

if [[ ! -f "$PAYLOAD" ]]; then
  echo "Payload file not found: ${PAYLOAD}" >&2
  exit 1
fi

echo "Sending DSv4 PD smoke request to ${URL} (payload=${PAYLOAD})..."
curl -sf -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d @"${PAYLOAD}" | tee "$OUT"

echo
echo "Smoke test completed. Response saved to ${OUT}"
