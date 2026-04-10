import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.modules.setdefault("torch", MagicMock())
_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)
_curl_cffi = MagicMock()
_curl_cffi_requests = MagicMock()
_curl_cffi_requests.AsyncSession = MagicMock()
_curl_cffi_requests.Response = MagicMock()
_curl_cffi.requests = _curl_cffi_requests
sys.modules.setdefault("curl_cffi", _curl_cffi)
sys.modules.setdefault("curl_cffi.requests", _curl_cffi_requests)


@pytest.mark.asyncio
async def test_apply_missing_core_targets_inserts_empty_metadata_json():
    from atlas_brain.services.scraping.target_provisioning import apply_missing_core_targets

    pool = AsyncMock()
    pool.fetchrow = AsyncMock(return_value={"id": "target-1"})

    candidate = {
        "vendor_name": "HubSpot",
        "source": "g2",
        "suggested_product_slug": "hubspot",
        "product_category": "CRM",
    }

    with patch(
        "atlas_brain.services.scraping.target_provisioning.resolve_vendor_name",
        new=AsyncMock(return_value="HubSpot"),
    ):
        actions = await apply_missing_core_targets(
            pool,
            existing_targets=[],
            candidates=[candidate],
            dry_run=False,
        )

    args = pool.fetchrow.await_args.args
    assert args[-1] == "{}"
    assert actions[0]["target_id"] == "target-1"
    assert "metadata" not in actions[0]


@pytest.mark.asyncio
async def test_fetch_review_funnel_audit_includes_recent_scrape_intake_counts():
    from atlas_brain.autonomous.tasks import _b2b_shared as mod

    pool = SimpleNamespace(
        fetch=AsyncMock(
            side_effect=[
                [
                    {"enrichment_status": "enriched", "ct": 4},
                    {"enrichment_status": "raw_only", "ct": 1},
                ],
                [
                    {
                        "result_text": json.dumps(
                            {
                                "funnel": {
                                    "found": 12,
                                    "filtered": 5,
                                    "short_flagged": 2,
                                    "quality_gated": 1,
                                    "duplicate_or_existing": 3,
                                    "retained_pending": 4,
                                    "retained_raw_only": 2,
                                    "inserted": 6,
                                    "company_signal_eligible": 1,
                                }
                            }
                        )
                    },
                    {"result_text": "not-json"},
                ],
            ]
        ),
        fetchrow=AsyncMock(
            return_value={
                "intelligence_eligible_reviews": 4,
                "company_signal_eligible_reviews": 2,
                "high_confidence_named_account_reviews": 1,
            }
        ),
    )

    with patch.object(mod, "_intelligence_source_allowlist", return_value=["g2"]):
        result = await mod._fetch_review_funnel_audit(pool, 30)

    assert result["found"] == 5
    assert result["scrape_runs"] == 1
    assert result["scrape_found"] == 12
    assert result["scrape_filtered"] == 5
    assert result["scrape_short_flagged"] == 2
    assert result["scrape_quality_gated"] == 1
    assert result["scrape_duplicate_or_existing"] == 3
    assert result["scrape_retained_pending"] == 4
    assert result["scrape_retained_raw_only"] == 2
    assert result["scrape_inserted"] == 6
    assert result["scrape_company_signal_eligible"] == 1
