#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/validate_extracted_content_pipeline.sh
bash scripts/check_ascii_python.sh
python scripts/check_extracted_imports.py
python scripts/smoke_extracted_pipeline_imports.py
python scripts/audit_extracted_standalone.py --fail-on-debt
pytest \
  tests/test_extracted_campaign_analytics.py \
  tests/test_extracted_campaign_manifest.py \
  tests/test_extracted_campaign_generation_seams.py \
  tests/test_extracted_campaign_generation.py \
  tests/test_extracted_campaign_reasoning_data.py \
  tests/test_extracted_campaign_skill_registry.py \
  tests/test_extracted_campaign_generation_example.py \
  tests/test_extracted_campaign_customer_data.py \
  tests/test_extracted_campaign_postgres_generation.py \
  tests/test_extracted_campaign_postgres_export.py \
  tests/test_extracted_campaign_postgres_review.py \
  tests/test_extracted_campaign_postgres_send.py \
  tests/test_extracted_campaign_postgres_analytics.py \
  tests/test_extracted_campaign_postgres_sequence_progression.py \
  tests/test_extracted_campaign_postgres_import.py \
  tests/test_extracted_content_pipeline_migration_runner.py \
  tests/test_extracted_pipeline_notify.py \
  tests/test_extracted_content_pipeline_reasoning_archetypes.py \
  tests/test_extracted_content_pipeline_reasoning_temporal.py \
  tests/test_extracted_content_pipeline_reasoning_evidence_engine.py \
  tests/test_extracted_reasoning_core_api.py \
  tests/test_extracted_reasoning_core_archetypes.py \
  tests/test_extracted_reasoning_core_evidence_engine.py \
  tests/test_extracted_reasoning_core_event_trace_ports.py \
  tests/test_extracted_reasoning_core_pack_registry.py \
  tests/test_extracted_reasoning_core_semantic_cache_keys.py \
  tests/test_extracted_reasoning_core_temporal.py \
  tests/test_extracted_reasoning_core_types.py \
  tests/test_extracted_reasoning_core_wedge_registry.py \
  tests/test_extracted_product_utilities.py \
  tests/test_extracted_b2b_batch_utils.py \
  tests/test_extracted_blog_matching.py \
  tests/test_extracted_campaign_audit.py \
  tests/test_extracted_campaign_llm_client.py \
  tests/test_extracted_campaign_llm_bridge.py \
  tests/test_extracted_vendor_briefing_seams.py \
  tests/test_extracted_campaign_postgres.py \
  tests/test_extracted_campaign_suppression.py \
  tests/test_extracted_campaign_sequence_context.py \
  tests/test_extracted_campaign_sequence_progression.py \
  tests/test_extracted_campaign_sender.py \
  tests/test_extracted_campaign_send.py \
  tests/test_extracted_campaign_webhooks.py \
  tests/test_extracted_podcast_transcript_import.py \
  tests/test_extracted_podcast_extraction.py \
  tests/test_extracted_podcast_repurpose_generation.py \
  tests/test_extracted_podcast_quality.py

echo "All extracted_content_pipeline checks completed"
