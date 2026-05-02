from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CORE_CAMPAIGN_MIGRATIONS = {
    "066_b2b_campaigns.sql",
    "068_campaign_sequences.sql",
    "069_campaign_analytics.sql",
    "070_campaign_suppressions.sql",
    "073_campaign_sequence_fixes.sql",
    "074_campaign_target_modes.sql",
    "075_amazon_seller_campaigns.sql",
    "090_audit_log_metadata_index.sql",
    "104_campaign_outcomes.sql",
    "146_campaign_score_components.sql",
    "150_campaign_engagement_timing.sql",
}

DEFERRED_CROSS_PRODUCT_MIGRATIONS = {
    "080_b2b_alert_baselines.sql",
    "106_score_calibration.sql",
    "235_vendor_targets_account_scope.sql",
    "255_anthropic_message_batches.sql",
}


def _manifest_targets() -> set[str]:
    data = json.loads((ROOT / "extracted_content_pipeline/manifest.json").read_text())
    return {
        Path(mapping["target"]).name
        for mapping in data["mappings"]
        if "/storage/migrations/" in mapping["target"]
    }


def _owned_targets() -> set[str]:
    data = json.loads((ROOT / "extracted_content_pipeline/manifest.json").read_text())
    return {entry["target"] for entry in data.get("owned", [])}


def _mapped_targets() -> set[str]:
    data = json.loads((ROOT / "extracted_content_pipeline/manifest.json").read_text())
    return {entry["target"] for entry in data.get("mappings", [])}


def test_manifest_syncs_core_campaign_schema_migrations() -> None:
    targets = _manifest_targets()

    assert CORE_CAMPAIGN_MIGRATIONS <= targets
    for migration in CORE_CAMPAIGN_MIGRATIONS:
        assert (ROOT / "extracted_content_pipeline/storage/migrations" / migration).exists()


def test_deferred_cross_product_migrations_stay_out_of_core_manifest() -> None:
    targets = _manifest_targets()

    assert not (DEFERRED_CROSS_PRODUCT_MIGRATIONS & targets)


def test_core_campaign_migrations_define_product_tables() -> None:
    migrations_dir = ROOT / "extracted_content_pipeline/storage/migrations"

    campaign_schema = (migrations_dir / "066_b2b_campaigns.sql").read_text()
    sequence_schema = (migrations_dir / "068_campaign_sequences.sql").read_text()
    suppression_schema = (migrations_dir / "070_campaign_suppressions.sql").read_text()
    target_schema = (migrations_dir / "074_campaign_target_modes.sql").read_text()
    seller_schema = (migrations_dir / "075_amazon_seller_campaigns.sql").read_text()

    assert "CREATE TABLE IF NOT EXISTS b2b_campaigns" in campaign_schema
    assert "CREATE TABLE IF NOT EXISTS campaign_sequences" in sequence_schema
    assert "CREATE TABLE IF NOT EXISTS campaign_audit_log" in sequence_schema
    assert "CREATE TABLE IF NOT EXISTS campaign_suppressions" in suppression_schema
    assert "CREATE TABLE IF NOT EXISTS vendor_targets" in target_schema
    assert "CREATE TABLE IF NOT EXISTS seller_targets" in seller_schema


def test_manifest_tracks_product_owned_adapter_files() -> None:
    owned = _owned_targets()

    assert "extracted_content_pipeline/pipelines/notify.py" in owned
    assert "extracted_content_pipeline/campaign_llm_client.py" in owned
    assert "extracted_content_pipeline/campaign_ports.py" in owned
    assert "extracted_content_pipeline/campaign_generation.py" in owned
    assert "extracted_content_pipeline/campaign_example.py" in owned
    assert "extracted_content_pipeline/campaign_customer_data.py" in owned
    assert "extracted_content_pipeline/campaign_opportunities.py" in owned
    assert "extracted_content_pipeline/settings.py" in owned
    assert "extracted_content_pipeline/reasoning/archetypes.py" in owned
    assert "extracted_content_pipeline/reasoning/temporal.py" in owned
    assert "extracted_content_pipeline/reasoning/evidence_engine.py" in owned
    assert "extracted_content_pipeline/autonomous/tasks/_execution_progress.py" in owned
    assert "extracted_content_pipeline/autonomous/tasks/_google_news.py" in owned
    assert "extracted_content_pipeline/autonomous/tasks/_blog_ts.py" in owned
    assert "extracted_content_pipeline/autonomous/tasks/_blog_deploy.py" in owned
    assert "extracted_content_pipeline/autonomous/tasks/_b2b_batch_utils.py" in owned
    assert "extracted_content_pipeline/autonomous/tasks/_blog_matching.py" in owned
    assert "extracted_content_pipeline/campaign_sequence_context.py" in owned
    assert "extracted_content_pipeline/autonomous/tasks/_campaign_sequence_context.py" in owned
    assert "extracted_content_pipeline/autonomous/tasks/campaign_audit.py" in owned
    assert "extracted_content_pipeline/services/campaign_sender.py" in owned
    assert "extracted_content_pipeline/services/vendor_target_selection.py" in owned
    assert "extracted_content_pipeline/services/vendor_registry.py" in owned
    assert "extracted_content_pipeline/autonomous/tasks/campaign_suppression.py" in owned
    assert "extracted_content_pipeline/autonomous/visibility.py" in owned
    assert "extracted_content_pipeline/services/b2b/account_opportunity_claims.py" in owned
    assert "extracted_content_pipeline/services/campaign_reasoning_context.py" in owned
    assert "extracted_content_pipeline/services/campaign_quality.py" in owned
    assert "extracted_content_pipeline/templates/email/vendor_briefing.py" in owned


def test_product_owned_utility_helpers_are_not_manifest_synced() -> None:
    mapped = _mapped_targets()

    assert "extracted_content_pipeline/campaign_ports.py" not in mapped
    assert "extracted_content_pipeline/campaign_generation.py" not in mapped
    assert "extracted_content_pipeline/campaign_example.py" not in mapped
    assert "extracted_content_pipeline/campaign_customer_data.py" not in mapped
    assert "extracted_content_pipeline/campaign_opportunities.py" not in mapped
    assert "extracted_content_pipeline/autonomous/tasks/_execution_progress.py" not in mapped
    assert "extracted_content_pipeline/autonomous/tasks/_google_news.py" not in mapped
    assert "extracted_content_pipeline/autonomous/tasks/_blog_ts.py" not in mapped
    assert "extracted_content_pipeline/autonomous/tasks/_blog_deploy.py" not in mapped
    assert "extracted_content_pipeline/autonomous/tasks/_b2b_batch_utils.py" not in mapped
    assert "extracted_content_pipeline/autonomous/tasks/_blog_matching.py" not in mapped
    assert "extracted_content_pipeline/campaign_sequence_context.py" not in mapped
    assert "extracted_content_pipeline/autonomous/tasks/_campaign_sequence_context.py" not in mapped
    assert "extracted_content_pipeline/autonomous/tasks/campaign_audit.py" not in mapped
    assert "extracted_content_pipeline/services/campaign_sender.py" not in mapped
    assert "extracted_content_pipeline/services/vendor_target_selection.py" not in mapped
    assert "extracted_content_pipeline/services/vendor_registry.py" not in mapped
    assert "extracted_content_pipeline/autonomous/tasks/campaign_suppression.py" not in mapped
    assert "extracted_content_pipeline/autonomous/visibility.py" not in mapped
    assert "extracted_content_pipeline/services/b2b/account_opportunity_claims.py" not in mapped
    assert "extracted_content_pipeline/services/campaign_reasoning_context.py" not in mapped
    assert "extracted_content_pipeline/services/campaign_quality.py" not in mapped
    assert "extracted_content_pipeline/templates/email/vendor_briefing.py" not in mapped
