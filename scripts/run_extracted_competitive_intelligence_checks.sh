#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/validate_extracted_competitive_intelligence.sh
bash scripts/check_ascii_python_competitive_intelligence.sh
python scripts/check_extracted_competitive_intelligence_imports.py
python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_competitive_intelligence
python scripts/smoke_extracted_competitive_intelligence_imports.py
python scripts/smoke_extracted_competitive_intelligence_standalone.py
python -m pytest -q \
  tests/test_extracted_competitive_manifest.py \
  tests/test_extracted_competitive_cross_vendor_selection.py \
  tests/test_extracted_competitive_prompt_contracts.py \
  tests/test_extracted_competitive_ecosystem_port.py \
  tests/test_extracted_competitive_challenger_claims_port.py \
  tests/test_extracted_competitive_product_claim.py \
  tests/test_extracted_competitive_crm_provider_port.py \
  tests/test_extracted_competitive_email_provider_port.py \
  tests/test_extracted_competitive_pdf_renderer_port.py \
  tests/test_extracted_competitive_llm_exact_cache_bridge.py \
  tests/test_extracted_competitive_anthropic_batch_bridge.py \
  tests/test_extracted_competitive_batch_utils.py \
  tests/test_extracted_competitive_llm_router_bridge.py \
  tests/test_extracted_competitive_battle_card_ports.py \
  tests/test_extracted_competitive_vendor_briefing_ports.py \
  tests/test_extracted_competitive_sets.py \
  tests/test_extracted_competitive_synthesis_packets.py \
  tests/test_extracted_competitive_vendor_briefing_renderer.py \
  tests/test_b2b_reasoning_consumer_adapter.py \
  tests/test_b2b_mcp_signals_overlay_contract.py

echo "All extracted_competitive_intelligence checks passed"
