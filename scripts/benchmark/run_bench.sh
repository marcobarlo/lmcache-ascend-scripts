#!/bin/bash
# Run from inside the container:
#   cd /workspace/dsv4-serving && ./scripts/benchmark/run_bench.sh
#
# --model is the served-model-name on the running vLLM instance.
# --tokenizer must point at the actual model directory for prompt tokenization.

MODEL_NAME="${MODEL_NAME:-dsv4}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8008}"

vllm bench serve \
    --host "${HOST}" \
    --port "${PORT}" \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --model "${MODEL_NAME}" \
    --tokenizer "${MODEL_PATH}" \
    --tokenizer-mode deepseek_v4 \
    --dataset-name prefix_repetition \
    --num-prompts 100 \
    --request-rate 0.5 \
    --prefix-repetition-num-prefixes 30 \
    --prefix-repetition-prefix-len 30000 \
    --prefix-repetition-suffix-len 10000 \
    --prefix-repetition-output-len 1000 \
    --ignore-eos \
    --temperature 0
