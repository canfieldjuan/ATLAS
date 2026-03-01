#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT_DIR/.env.local" ]]; then
  set -a
  . "$ROOT_DIR/.env.local"
  set +a
elif [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  . "$ROOT_DIR/.env"
  set +a
fi

: "${VLLM_PYTHON:?}"
: "${VLLM_MODEL:?}"
: "${VLLM_HOST:?}"
: "${VLLM_PORT:?}"
: "${VLLM_QUANTIZATION:?}"
: "${VLLM_DTYPE:?}"
: "${VLLM_GPU_MEMORY_UTILIZATION:?}"
: "${VLLM_MAX_NUM_BATCHED_TOKENS:?}"
: "${VLLM_MAX_NUM_SEQS:?}"
: "${VLLM_ENABLE_PREFIX_CACHING:?}"

"$VLLM_PYTHON" -c "import vllm" >/dev/null 2>&1 || { echo "vllm not available in VLLM_PYTHON"; exit 1; }

ARGS=(--model "$VLLM_MODEL" --host "$VLLM_HOST" --port "$VLLM_PORT" --quantization "$VLLM_QUANTIZATION" --dtype "$VLLM_DTYPE" --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" --max-num-seqs "$VLLM_MAX_NUM_SEQS")
if [[ "$VLLM_ENABLE_PREFIX_CACHING" == "true" ]]; then
  ARGS+=(--enable-prefix-caching)
fi

exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
