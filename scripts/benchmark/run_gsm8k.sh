#!/bin/bash
set -euo pipefail

# Run from inside the container:
#   cd /workspace/dsv4-serving && ./scripts/benchmark/run_gsm8k.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# vLLM serve settings (see scripts/serving/single_node.sh)
MODEL_NAME="${MODEL_NAME:-dsv4}"
BASE_URL="${BASE_URL:-http://localhost:8008/v1/chat/completions}"
NUM_CONCURRENT="${NUM_CONCURRENT:-8}"
# Limit eval to N test examples (unset = full 1319). Same subset every run.
NUM_SAMPLES="${NUM_SAMPLES:-200}"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/gsm8k}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/gsm8k_results.json}"

GSM8K_TRAIN_URL="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
GSM8K_TEST_URL="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

ensure_gsm8k_data() {
    mkdir -p "${DATA_DIR}"

    if [[ ! -f "${DATA_DIR}/train.jsonl" ]]; then
        echo "Downloading GSM8K train split from GitHub..."
        curl -fsSL "${GSM8K_TRAIN_URL}" -o "${DATA_DIR}/train.jsonl"
    fi

    if [[ ! -f "${DATA_DIR}/test.jsonl" ]]; then
        echo "Downloading GSM8K test split from GitHub..."
        curl -fsSL "${GSM8K_TEST_URL}" -o "${DATA_DIR}/test.jsonl"
    fi
}

if ! python3 -c "import lm_eval" 2>/dev/null; then
    echo "Installing lm-eval..."
    pip install -q lm-eval
fi

ensure_gsm8k_data
mkdir -p "${OUTPUT_DIR}"

TASK_YAML="${OUTPUT_DIR}/gsm8k_local.runtime.yaml"
sed \
    -e "s|__GSM8K_TRAIN__|${DATA_DIR}/train.jsonl|g" \
    -e "s|__GSM8K_TEST__|${DATA_DIR}/test.jsonl|g" \
    "${SCRIPT_DIR}/gsm8k_local.yaml" > "${TASK_YAML}"

# HuggingFace is unreachable in this environment; load GSM8K from local JSONL.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

LIMIT_ARGS=()
if [[ -n "${NUM_SAMPLES}" ]]; then
    LIMIT_ARGS=(--limit "${NUM_SAMPLES}")
fi

lm_eval \
  --model local-chat-completions \
  --model_args "model=${MODEL_NAME},base_url=${BASE_URL},num_concurrent=${NUM_CONCURRENT},tokenized_requests=False,max_retries=3" \
  --include_path "${OUTPUT_DIR}" \
  --tasks gsm8k_local \
  --num_fewshot 5 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --gen_kwargs "do_sample=False,temperature=0.0" \
  --batch_size 1 \
  "${LIMIT_ARGS[@]}" \
  --output_path "${OUTPUT_PATH}"

echo "Results written to ${OUTPUT_PATH}"
