import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4


if "asyncpg" not in sys.modules:
    asyncpg_module = types.ModuleType("asyncpg")
    asyncpg_module.connect = MagicMock()
    asyncpg_module.Connection = object
    asyncpg_module.Record = dict
    asyncpg_exceptions = types.ModuleType("asyncpg.exceptions")
    asyncpg_exceptions.UndefinedTableError = Exception
    asyncpg_module.exceptions = asyncpg_exceptions
    sys.modules["asyncpg"] = asyncpg_module
    sys.modules["asyncpg.exceptions"] = asyncpg_exceptions

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from backfill_low_substance_enriched_reviews import (  # noqa: E402
    _default_sources,
    _metadata_patch,
    _parse_sources,
    _plan_row_backfill,
)
from atlas_brain.config import B2BChurnConfig  # noqa: E402


def _row(**overrides):
    row = {
        "id": uuid4(),
        "source": "reddit",
        "vendor_name": "HubSpot",
        "product_name": "HubSpot",
        "summary": "Anyone switch?",
        "review_text": "Thinking about options.",
        "pros": "",
        "cons": "",
        "enrichment_status": "enriched",
        "raw_metadata": {},
        "enrichment": {"urgency_score": 2, "competitors_mentioned": []},
    }
    row.update(overrides)
    return row


def test_b2b_churn_config_default_low_fidelity_sources_include_reddit_and_software_advice():
    raw_default = B2BChurnConfig.model_fields["enrichment_low_fidelity_noisy_sources"].default
    values = {part.strip() for part in str(raw_default).split(",") if part.strip()}

    assert "reddit" in values
    assert "software_advice" in values


def test_b2b_churn_config_default_repair_strict_discussion_sources_include_hackernews():
    raw_default = B2BChurnConfig.model_fields["enrichment_repair_strict_discussion_sources"].default
    values = {part.strip() for part in str(raw_default).split(",") if part.strip()}

    assert "reddit" in values
    assert "hackernews" in values


def test_parse_sources_normalizes_values():
    assert _parse_sources(" Reddit, software_advice ,, HackerNews ") == [
        "hackernews",
        "reddit",
        "software_advice",
    ]


def test_default_sources_include_trustpilot_and_configured_noisy_sources():
    values = set(_default_sources())

    assert "trustpilot" in values
    assert "reddit" in values
    assert "software_advice" in values


def test_plan_row_backfill_returns_low_fidelity_quarantine_for_thin_reddit_context():
    planned = _plan_row_backfill(_row())

    assert planned is not None
    assert planned["source"] == "reddit"
    assert "thin_social_context" in planned["low_fidelity_reasons"]


def test_plan_row_backfill_skips_rows_without_low_fidelity_reasons():
    row = _row(
        summary="HubSpot pricing doubled before renewal",
        review_text="We are evaluating alternatives because HubSpot pricing doubled for our team before renewal.",
        enrichment={"urgency_score": 8, "competitors_mentioned": [{"name": "Salesforce"}]},
    )

    planned = _plan_row_backfill(row)

    assert planned is None


def test_metadata_patch_preserves_prior_status_and_backfill_scope():
    detected_at = datetime(2026, 4, 13, 12, 0, tzinfo=timezone.utc)

    metadata = _metadata_patch(
        _row(raw_metadata={"source_fit": "commercial"}),
        low_fidelity_reasons=["thin_social_context"],
        detected_at=detected_at,
    )

    assert metadata["source_fit"] == "commercial"
    assert metadata["prior_enrichment_status"] == "enriched"
    assert metadata["low_fidelity_backfill"]["scope"] == "historical_low_substance_enriched_reviews"
    assert metadata["low_fidelity_backfill"]["reasons"] == ["thin_social_context"]
