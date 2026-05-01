#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/validate_extracted_content_pipeline.sh
bash scripts/check_ascii_python.sh
python scripts/check_extracted_imports.py
python scripts/smoke_extracted_pipeline_imports.py
python scripts/audit_extracted_standalone.py
pytest \
  tests/test_extracted_campaign_analytics.py \
  tests/test_extracted_campaign_generation.py \
  tests/test_extracted_campaign_llm_bridge.py \
  tests/test_extracted_campaign_suppression.py \
  tests/test_extracted_campaign_sequence_context.py \
  tests/test_extracted_campaign_sequence_progression.py \
  tests/test_extracted_campaign_sender.py \
  tests/test_extracted_campaign_send.py \
  tests/test_extracted_campaign_webhooks.py

echo "All extracted_content_pipeline checks completed"
