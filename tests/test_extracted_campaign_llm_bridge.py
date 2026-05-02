from __future__ import annotations

from pathlib import Path

from extracted_content_pipeline.skills.registry import get_skill_registry


ROOT = Path(__file__).resolve().parents[1]

LLM_BRIDGE_PATHS = [
    "extracted_content_pipeline/pipelines/llm.py",
    "extracted_content_pipeline/services/b2b/anthropic_batch.py",
    "extracted_content_pipeline/services/llm/anthropic.py",
]

LOCAL_UTILITY_SHIM_PATHS = [
    "extracted_content_pipeline/pipelines/notify.py",
    "extracted_content_pipeline/reasoning/archetypes.py",
    "extracted_content_pipeline/reasoning/evidence_engine.py",
    "extracted_content_pipeline/reasoning/temporal.py",
    "extracted_content_pipeline/reasoning/wedge_registry.py",
    "extracted_content_pipeline/config.py",
    "extracted_content_pipeline/autonomous/tasks/_blog_matching.py",
    "extracted_content_pipeline/autonomous/tasks/_campaign_sequence_context.py",
    "extracted_content_pipeline/autonomous/tasks/campaign_audit.py",
    "extracted_content_pipeline/autonomous/tasks/campaign_suppression.py",
    "extracted_content_pipeline/autonomous/visibility.py",
    "extracted_content_pipeline/campaign_sequence_context.py",
    "extracted_content_pipeline/skills/registry.py",
    "extracted_content_pipeline/services/__init__.py",
    "extracted_content_pipeline/services/apollo_company_overrides.py",
    "extracted_content_pipeline/services/b2b/corrections.py",
    "extracted_content_pipeline/services/b2b/cache_runner.py",
    "extracted_content_pipeline/services/b2b/account_opportunity_claims.py",
    "extracted_content_pipeline/services/b2b/enrichment_contract.py",
    "extracted_content_pipeline/services/blog_quality.py",
    "extracted_content_pipeline/services/campaign_quality.py",
    "extracted_content_pipeline/services/campaign_reasoning_context.py",
    "extracted_content_pipeline/services/campaign_sender.py",
    "extracted_content_pipeline/services/company_normalization.py",
    "extracted_content_pipeline/services/protocols.py",
    "extracted_content_pipeline/services/scraping/sources.py",
    "extracted_content_pipeline/services/scraping/universal/html_cleaner.py",
    "extracted_content_pipeline/services/tracing.py",
    "extracted_content_pipeline/services/vendor_target_selection.py",
    "extracted_content_pipeline/services/vendor_registry.py",
    "extracted_content_pipeline/storage/database.py",
    "extracted_content_pipeline/storage/models.py",
    "extracted_content_pipeline/storage/repositories/scheduled_task.py",
    "extracted_content_pipeline/templates/email/vendor_briefing.py",
]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_content_llm_bridges_target_extracted_llm_infrastructure() -> None:
    for path in LLM_BRIDGE_PATHS:
        text = _read(path)
        assert "extracted_llm_infrastructure" in text
        assert "from atlas_brain" not in text


def test_local_utility_shims_do_not_delegate_to_atlas_brain() -> None:
    for path in LOCAL_UTILITY_SHIM_PATHS:
        text = _read(path)
        assert "from atlas_brain" not in text
        assert "import atlas_brain" not in text


def test_extracted_pipeline_checks_enforce_zero_atlas_runtime_imports() -> None:
    text = _read("scripts/run_extracted_pipeline_checks.sh")

    assert "python scripts/audit_extracted_standalone.py --fail-on-debt" in text


def test_local_skill_registry_exposes_campaign_prompt_contracts() -> None:
    registry = get_skill_registry()
    expected_skills = [
        "digest/b2b_campaign_generation",
        "digest/b2b_vendor_outreach",
        "digest/b2b_challenger_outreach",
        "digest/b2b_campaign_sequence",
        "digest/b2b_vendor_sequence",
        "digest/b2b_challenger_sequence",
        "digest/b2b_onboarding_sequence",
        "digest/amazon_seller_campaign_generation",
        "digest/amazon_seller_campaign_sequence",
    ]

    for skill_name in expected_skills:
        prompt = registry.get_prompt(skill_name)
        assert prompt
        assert registry.get(skill_name).content == prompt
