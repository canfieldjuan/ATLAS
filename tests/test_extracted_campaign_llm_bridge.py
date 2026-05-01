from __future__ import annotations

from pathlib import Path


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
    "extracted_content_pipeline/skills/registry.py",
    "extracted_content_pipeline/services/__init__.py",
    "extracted_content_pipeline/services/apollo_company_overrides.py",
    "extracted_content_pipeline/services/b2b/corrections.py",
    "extracted_content_pipeline/services/b2b/cache_runner.py",
    "extracted_content_pipeline/services/b2b/enrichment_contract.py",
    "extracted_content_pipeline/services/blog_quality.py",
    "extracted_content_pipeline/services/company_normalization.py",
    "extracted_content_pipeline/services/protocols.py",
    "extracted_content_pipeline/services/scraping/sources.py",
    "extracted_content_pipeline/services/scraping/universal/html_cleaner.py",
    "extracted_content_pipeline/services/tracing.py",
    "extracted_content_pipeline/services/vendor_registry.py",
    "extracted_content_pipeline/storage/database.py",
    "extracted_content_pipeline/storage/models.py",
    "extracted_content_pipeline/storage/repositories/scheduled_task.py",
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
