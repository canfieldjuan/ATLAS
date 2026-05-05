#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m py_compile \
  extracted_content_pipeline/services/reasoning_provider_port.py \
  extracted_content_pipeline/services/__init__.py \
  extracted_content_pipeline/campaign_reasoning_data.py \
  extracted_content_pipeline/campaign_example.py \
  extracted_content_pipeline/campaign_postgres_generation.py \
  scripts/run_extracted_campaign_generation_example.py \
  scripts/run_extracted_campaign_generation_postgres.py \
  tests/test_extracted_campaign_reasoning_data.py \
  tests/test_extracted_campaign_generation_example.py \
  tests/test_extracted_campaign_postgres_generation.py

echo "reasoning provider-port compatibility py_compile checks passed"
