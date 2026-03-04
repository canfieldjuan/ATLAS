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
: "${VLLM_DTYPE:?}"
: "${VLLM_GPU_MEMORY_UTILIZATION:?}"
: "${VLLM_MAX_NUM_BATCHED_TOKENS:?}"
: "${VLLM_MAX_NUM_SEQS:?}"
: "${VLLM_MAX_MODEL_LEN:?}"
: "${VLLM_KV_CACHE_DTYPE:=auto}"
: "${VLLM_ENABLE_PREFIX_CACHING:?}"

# Ensure venv bin is in PATH (FlashInfer JIT needs ninja)
export PATH="$(dirname "$VLLM_PYTHON"):$PATH"

"$VLLM_PYTHON" -c "import vllm" >/dev/null 2>&1 || { echo "vllm not available in VLLM_PYTHON"; exit 1; }

# Auto-detect quantization method from model config.json
if [[ -z "${VLLM_QUANTIZATION:-}" || "${VLLM_QUANTIZATION:-}" == "auto" ]]; then
  VLLM_QUANTIZATION=$("$VLLM_PYTHON" -c "
import json, glob, sys
paths = glob.glob('$HOME/.cache/huggingface/hub/models--${VLLM_MODEL//\//--}/snapshots/*/config.json')
if not paths:
    print('', end='')
    sys.exit(0)
with open(paths[0]) as f:
    qm = json.load(f).get('quantization_config', {}).get('quant_method', '')
print(qm, end='')
")
  if [[ -z "$VLLM_QUANTIZATION" ]]; then
    echo "ERROR: Could not detect quantization method from model config. Set VLLM_QUANTIZATION explicitly."
    exit 1
  fi
  echo "Auto-detected quantization: $VLLM_QUANTIZATION"
fi

ARGS=(--model "$VLLM_MODEL" --host "$VLLM_HOST" --port "$VLLM_PORT" --quantization "$VLLM_QUANTIZATION" --dtype "$VLLM_DTYPE" --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" --max-num-seqs "$VLLM_MAX_NUM_SEQS" --max-model-len "$VLLM_MAX_MODEL_LEN" --kv-cache-dtype "$VLLM_KV_CACHE_DTYPE")
if [[ "$VLLM_ENABLE_PREFIX_CACHING" == "true" ]]; then
  ARGS+=(--enable-prefix-caching)
fi

exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
