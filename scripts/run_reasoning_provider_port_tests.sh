#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec('pytest_asyncio') else 1)
PY
then
  pytest -q \
    tests/test_b2b_reasoning_consumer_adapter.py \
    tests/test_b2b_mcp_signals_overlay_contract.py \
    tests/test_extracted_campaign_reasoning_data.py \
    tests/test_extracted_campaign_generation_example.py \
    tests/test_extracted_campaign_postgres_generation.py
else
  echo "SKIP: pytest_asyncio is not installed; skipping pytest-based provider-port tests"
fi
