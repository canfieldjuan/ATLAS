"""Tests for shared producer migration onto enrichment_contract helpers.

Phase 2.2: producers in _b2b_shared.py that feed b2b_company_signals and
b2b_vendor_pain_points must apply pain_category_for_bucket() gating.

Strategy: mock the upstream identity/source gates and the SQL fetch so the
new pain-gating logic is exercised in isolation. Each fixture row carries
the full SQL projection shape that _fetch_high_intent_companies expects.
"""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)


def _row(
    *,
    review_id: str = "00000000-0000-0000-0000-000000000001",
    source: str = "reddit",
    reviewer_company: str = "Acme Corp",
    vendor_name: str = "Shopify",
    enrichment: dict | None = None,
    intent_to_leave: bool = True,
    indicator_cancel: bool = False,
    pain_category_sql: str | None = None,
    urgency: float = 8.5,
) -> dict[str, Any]:
    """Build a row matching the _fetch_high_intent_companies SELECT projection.

    enrichment: the JSONB dict that will be serialized to enrichment_raw.
                Defaults to a v4 strong-pricing row.
    """
    if enrichment is None:
        enrichment = {
            "enrichment_schema_version": 4,
            "pain_category": "pricing",
            "pain_confidence": "strong",
            "urgency_score": urgency,
            "churn_signals": {"intent_to_leave": intent_to_leave},
        }
    pain_for_sql = pain_category_sql
    if pain_for_sql is None:
        pain_for_sql = enrichment.get("pain_category")
    return {
        "review_id": review_id,
        "source": source,
        "reviewer_company": reviewer_company,
        "raw_reviewer_company": reviewer_company,
        "resolution_confidence": "high",
        "vendor_name": vendor_name,
        "product_category": "ecommerce",
        "reviewer_title": "Director of Operations",
        "company_size_raw": "Mid-Market",
        "industry": "Retail",
        "verified_employee_count": 250,
        "company_country": "US",
        "company_domain": "acme.example.com",
        "revenue_range": "$10M-$50M",
        "founded_year": 2010,
        "total_funding": None,
        "latest_funding_stage": None,
        "headcount_growth_6m": None,
        "headcount_growth_12m": None,
        "headcount_growth_24m": None,
        "publicly_traded_exchange": None,
        "publicly_traded_symbol": None,
        "company_description": None,
        "role_level": "director",
        "is_dm": True,
        "urgency": urgency,
        "pain": pain_for_sql,
        "enrichment_raw": json.dumps(enrichment) if enrichment is not None else None,
        "alternatives": json.dumps([]),
        "quotes": json.dumps([]),
        "value_signal": None,
        "seat_count": None,
        "lock_in_level": None,
        "contract_end": None,
        "buying_stage": None,
        "relevance_score": 0.9,
        "author_churn_score": None,
        "intent_to_leave": intent_to_leave,
        "actively_evaluating": False,
        "contract_renewal_mentioned": False,
        "indicator_cancel": indicator_cancel,
        "indicator_migration": False,
        "indicator_evaluation": False,
        "indicator_switch": False,
        "indicator_named_alternative": False,
        "indicator_timeline": False,
        "indicator_price_pressure": False,
        "has_budget_authority": True,
    }


class _FakePool:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    async def fetch(self, _sql: str, *_args: Any) -> list[dict[str, Any]]:
        return list(self._rows)


@pytest.fixture(autouse=True)
def _bypass_identity_gates(monkeypatch):
    """Always pass identity/exclusion/signal gates so tests focus on pain logic."""
    from atlas_brain.autonomous.tasks import _b2b_shared

    monkeypatch.setattr(
        _b2b_shared, "_company_signal_exclusion_reason", lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        _b2b_shared, "_company_signal_named_account_gap_reason", lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        _b2b_shared, "_compute_company_signal_confidence", lambda _payload: 0.9,
    )
    # Disable the optional signal-evidence gate so the only churn check is
    # whatever the test explicitly sets via the row's intent flags.
    monkeypatch.setattr(
        _b2b_shared, "_high_intent_signal_evidence_enabled", lambda: False,
    )
    yield


@pytest.mark.asyncio
async def test_v4_strong_pricing_row_keeps_pain_with_strong_confidence():
    from atlas_brain.autonomous.tasks._b2b_shared import _fetch_high_intent_companies

    pool = _FakePool([_row(
        enrichment={
            "enrichment_schema_version": 4,
            "pain_category": "pricing",
            "pain_confidence": "strong",
        },
    )])
    out = await _fetch_high_intent_companies(pool, urgency_threshold=5.0, window_days=30)

    assert len(out) == 1
    assert out[0]["pain"] == "pricing"
    assert out[0]["pain_confidence"] == "strong"


@pytest.mark.asyncio
async def test_v4_weak_specific_pain_is_kept_at_default_min_confidence():
    """min_confidence='weak' is the producer policy; weak pricing must survive."""
    from atlas_brain.autonomous.tasks._b2b_shared import _fetch_high_intent_companies

    pool = _FakePool([_row(
        enrichment={
            "enrichment_schema_version": 4,
            "pain_category": "pricing",
            "pain_confidence": "weak",
        },
    )])
    out = await _fetch_high_intent_companies(pool, urgency_threshold=5.0, window_days=30)

    assert len(out) == 1
    assert out[0]["pain"] == "pricing"
    assert out[0]["pain_confidence"] == "weak"


@pytest.mark.asyncio
async def test_v4_explicit_none_confidence_keeps_row_with_pain_none():
    """Critical: pain_confidence='none' must NOT drop the row when other
    high-intent gates pass. Pain is annotation, not admission."""
    from atlas_brain.autonomous.tasks._b2b_shared import _fetch_high_intent_companies

    pool = _FakePool([_row(
        enrichment={
            "enrichment_schema_version": 4,
            "pain_category": "overall_dissatisfaction",
            "pain_confidence": "none",
            "churn_signals": {"intent_to_leave": True},
        },
        intent_to_leave=True,
    )])
    out = await _fetch_high_intent_companies(pool, urgency_threshold=5.0, window_days=30)

    assert len(out) == 1, "row with confidence=none + real churn signal must be retained"
    assert out[0]["pain"] is None, "gated pain must be None for confidence=none"
    assert out[0]["pain_confidence"] == "none"


@pytest.mark.asyncio
async def test_v4_strong_generic_pain_is_kept_default_allows_generic():
    """Bucket policy default allows generic categories with strong confidence."""
    from atlas_brain.autonomous.tasks._b2b_shared import _fetch_high_intent_companies

    pool = _FakePool([_row(
        enrichment={
            "enrichment_schema_version": 4,
            "pain_category": "overall_dissatisfaction",
            "pain_confidence": "strong",
        },
    )])
    out = await _fetch_high_intent_companies(pool, urgency_threshold=5.0, window_days=30)

    assert len(out) == 1
    assert out[0]["pain"] == "overall_dissatisfaction"
    assert out[0]["pain_confidence"] == "strong"


@pytest.mark.asyncio
async def test_legacy_v3_row_resolves_via_recompute_path():
    """Legacy v3 row missing pain_confidence should produce a resolved tier."""
    from atlas_brain.autonomous.tasks._b2b_shared import _fetch_high_intent_companies

    pool = _FakePool([_row(
        enrichment={
            "pain_category": "pricing",
            "specific_complaints": ["too expensive for what you get"],
            "churn_signals": {"intent_to_leave": True},
        },
    )])
    out = await _fetch_high_intent_companies(pool, urgency_threshold=5.0, window_days=30)

    assert len(out) == 1
    # Resolved confidence comes from _compute_pain_confidence -- one of
    # the canonical tiers, never None or 'unknown' for a real legacy row
    assert out[0]["pain_confidence"] in ("strong", "weak", "none")


@pytest.mark.asyncio
async def test_malformed_enrichment_resolves_to_unknown_with_pain_none():
    """Row with un-parseable enrichment_raw -> pain=None, confidence='unknown'."""
    from atlas_brain.autonomous.tasks._b2b_shared import _fetch_high_intent_companies

    row = _row()
    row["enrichment_raw"] = None  # producer returned no enrichment
    pool = _FakePool([row])
    out = await _fetch_high_intent_companies(pool, urgency_threshold=5.0, window_days=30)

    assert len(out) == 1
    assert out[0]["pain"] is None
    assert out[0]["pain_confidence"] == "unknown"


@pytest.mark.asyncio
async def test_pain_confidence_is_not_a_high_intent_admission_gate():
    """Two rows with identical urgency + signal evidence; one has
    confidence=none, the other has confidence=strong. Both must be kept."""
    from atlas_brain.autonomous.tasks._b2b_shared import _fetch_high_intent_companies

    rows = [
        _row(
            review_id="00000000-0000-0000-0000-00000000000a",
            reviewer_company="Strong Corp",
            enrichment={
                "enrichment_schema_version": 4,
                "pain_category": "pricing",
                "pain_confidence": "strong",
            },
        ),
        _row(
            review_id="00000000-0000-0000-0000-00000000000b",
            reviewer_company="None Corp",
            enrichment={
                "enrichment_schema_version": 4,
                "pain_category": "overall_dissatisfaction",
                "pain_confidence": "none",
                "churn_signals": {"intent_to_leave": True},
            },
            intent_to_leave=True,
        ),
    ]
    pool = _FakePool(rows)
    out = await _fetch_high_intent_companies(pool, urgency_threshold=5.0, window_days=30)

    assert len(out) == 2
    by_company = {r["company"]: r for r in out}
    assert by_company["Strong Corp"]["pain"] == "pricing"
    assert by_company["Strong Corp"]["pain_confidence"] == "strong"
    assert by_company["None Corp"]["pain"] is None
    assert by_company["None Corp"]["pain_confidence"] == "none"


@pytest.mark.asyncio
async def test_result_row_preserves_existing_keys():
    """Migration must not drop any existing keys from the result dict."""
    from atlas_brain.autonomous.tasks._b2b_shared import _fetch_high_intent_companies

    pool = _FakePool([_row()])
    out = await _fetch_high_intent_companies(pool, urgency_threshold=5.0, window_days=30)

    assert len(out) == 1
    keys = set(out[0].keys())
    expected_keys = {
        "company", "raw_company", "resolution_confidence", "vendor",
        "category", "title", "company_size", "industry",
        "verified_employee_count", "company_country", "company_domain",
        "revenue_range", "founded_year", "total_funding", "funding_stage",
        "headcount_growth_6m", "headcount_growth_12m", "headcount_growth_24m",
        "publicly_traded", "ticker", "company_description", "role_level",
        "decision_maker", "urgency", "pain", "pain_confidence",
        "alternatives", "quotes", "contract_signal", "review_id", "source",
        "seat_count", "lock_in_level", "contract_end", "buying_stage",
        "relevance_score", "author_churn_score", "intent_signals",
    }
    missing = expected_keys - keys
    assert not missing, f"migration dropped keys: {missing}"


# ---------------------------------------------------------------------------
# _fetch_pain_provenance (Phase 2.2.B): SQL aggregation moved to Python so
# pain_category_for_bucket can gate per-row before bucketing.
# ---------------------------------------------------------------------------


def _provenance_row(
    *,
    vendor: str = "Shopify",
    review_id: str = "00000000-0000-0000-0000-000000000001",
    source: str = "g2",
    rating: float | None = 3.0,
    urgency: float | None = 7.0,
    enrichment: dict | None = None,
) -> dict[str, Any]:
    if enrichment is None:
        enrichment = {
            "enrichment_schema_version": 4,
            "pain_category": "pricing",
            "pain_confidence": "strong",
            "urgency_score": urgency,
        }
    return {
        "vendor_name": vendor,
        "review_id": review_id,
        "source": source,
        "rating": rating,
        "enrichment_raw": json.dumps(enrichment) if enrichment is not None else None,
        "urgency_score": urgency,
    }


class _ProvenancePool:
    """Fake pool that returns core_rows on the first fetch and severity_rows on the second."""

    def __init__(self, core_rows: list[dict], severity_rows: list[dict] | None = None):
        self._core = core_rows
        self._severity = severity_rows or []
        self._calls = 0

    async def fetch(self, _sql: str, *_args: Any) -> list[dict[str, Any]]:
        self._calls += 1
        if self._calls == 1:
            return list(self._core)
        return list(self._severity)


@pytest.fixture(autouse=True)
def _provenance_canonicalizers(monkeypatch):
    """Make _canonicalize_vendor a no-op for tests so vendor names pass through."""
    from atlas_brain.autonomous.tasks import _b2b_shared

    monkeypatch.setattr(_b2b_shared, "_canonicalize_vendor", lambda v: v)
    yield


@pytest.mark.asyncio
async def test_provenance_v4_strong_pricing_row_contributes():
    from atlas_brain.autonomous.tasks._b2b_shared import _fetch_pain_provenance

    pool = _ProvenancePool([_provenance_row(
        enrichment={
            "enrichment_schema_version": 4,
            "pain_category": "pricing",
            "pain_confidence": "strong",
            "urgency_score": 8.0,
        },
        rating=2.5, urgency=8.0,
    )])
    out = await _fetch_pain_provenance(pool, window_days=30)

    assert ("Shopify", "pricing") in out
    entry = out[("Shopify", "pricing")]
    assert entry["mention_count"] == 1
    assert entry["avg_urgency"] == 8.0
    assert entry["avg_rating"] == 2.5
    assert entry["source_distribution"] == {"g2": 1}
    assert "00000000-0000-0000-0000-000000000001" in entry["sample_review_ids"]


@pytest.mark.asyncio
async def test_provenance_v4_weak_specific_pain_contributes_at_default():
    """Default min_confidence='weak' permits weak signal."""
    from atlas_brain.autonomous.tasks._b2b_shared import _fetch_pain_provenance

    pool = _ProvenancePool([_provenance_row(
        enrichment={
            "enrichment_schema_version": 4,
            "pain_category": "ux",
            "pain_confidence": "weak",
            "urgency_score": 6.0,
        },
        rating=3.5, urgency=6.0,
    )])
    out = await _fetch_pain_provenance(pool, window_days=30)

    assert ("Shopify", "ux") in out
    assert out[("Shopify", "ux")]["mention_count"] == 1


@pytest.mark.asyncio
async def test_provenance_v4_confidence_none_row_dropped():
    """Rows with explicit pain_confidence='none' must NOT contribute."""
    from atlas_brain.autonomous.tasks._b2b_shared import _fetch_pain_provenance

    pool = _ProvenancePool([_provenance_row(
        enrichment={
            "enrichment_schema_version": 4,
            "pain_category": "overall_dissatisfaction",
            "pain_confidence": "none",
        },
    )])
    out = await _fetch_pain_provenance(pool, window_days=30)

    assert out == {}, "confidence=none row must not contribute to rollup"


@pytest.mark.asyncio
async def test_provenance_legacy_v3_row_resolves_via_recompute():
    """Legacy row missing pain_confidence resolves via the contract recompute."""
    from atlas_brain.autonomous.tasks._b2b_shared import _fetch_pain_provenance

    pool = _ProvenancePool([_provenance_row(
        enrichment={
            "pain_category": "pricing",
            "specific_complaints": ["far too expensive for value"],
            "urgency_score": 7.0,
            "churn_signals": {"intent_to_leave": True},
        },
        rating=2.0, urgency=7.0,
    )])
    out = await _fetch_pain_provenance(pool, window_days=30)

    # Resolved tier should be at least 'weak' for a legacy row with
    # intent + a complaint, so it survives the bucket gate
    assert ("Shopify", "pricing") in out


@pytest.mark.asyncio
async def test_provenance_aggregates_three_rows_same_bucket():
    """Three v4 strong rows for the same (vendor, pain) accumulate correctly."""
    from atlas_brain.autonomous.tasks._b2b_shared import _fetch_pain_provenance

    rows = [
        _provenance_row(
            review_id="00000000-0000-0000-0000-00000000000a",
            source="g2", rating=2.0, urgency=8.0,
        ),
        _provenance_row(
            review_id="00000000-0000-0000-0000-00000000000b",
            source="capterra", rating=3.0, urgency=7.0,
        ),
        _provenance_row(
            review_id="00000000-0000-0000-0000-00000000000c",
            source="g2", rating=2.5, urgency=9.0,
        ),
    ]
    pool = _ProvenancePool(rows)
    out = await _fetch_pain_provenance(pool, window_days=30)

    entry = out[("Shopify", "pricing")]
    assert entry["mention_count"] == 3
    assert entry["source_distribution"] == {"g2": 2, "capterra": 1}
    assert len(entry["sample_review_ids"]) == 3
    # avg_urgency = (8+7+9)/3 = 8.0
    assert entry["avg_urgency"] == 8.0
    # avg_rating = (2+3+2.5)/3 = 2.5
    assert entry["avg_rating"] == 2.5


@pytest.mark.asyncio
async def test_provenance_null_rating_does_not_skew_average():
    """Rows with null rating should be excluded from avg_rating denominator."""
    from atlas_brain.autonomous.tasks._b2b_shared import _fetch_pain_provenance

    rows = [
        _provenance_row(rating=4.0, urgency=8.0,
                        review_id="00000000-0000-0000-0000-00000000000a"),
        _provenance_row(rating=None, urgency=8.0,
                        review_id="00000000-0000-0000-0000-00000000000b"),
    ]
    pool = _ProvenancePool(rows)
    out = await _fetch_pain_provenance(pool, window_days=30)

    entry = out[("Shopify", "pricing")]
    assert entry["mention_count"] == 2
    # avg_rating = 4.0 / 1 row that had a rating, NOT 4.0/2
    assert entry["avg_rating"] == 4.0
    # both contributed urgency
    assert entry["avg_urgency"] == 8.0


@pytest.mark.asyncio
async def test_provenance_severity_augments_only_kept_keys():
    """Severity rows must ONLY augment (vendor, pain) keys that survived
    the gate. Severity for a gated-out key is silently dropped."""
    from atlas_brain.autonomous.tasks._b2b_shared import _fetch_pain_provenance

    core = [_provenance_row(
        enrichment={
            "enrichment_schema_version": 4,
            "pain_category": "pricing",
            "pain_confidence": "strong",
        },
    )]
    severity = [
        # Augments the kept (Shopify, pricing) key
        {"vendor_name": "Shopify", "pain_category": "pricing",
         "severity": "primary", "cnt": 5},
        # No corresponding core row -- must be dropped
        {"vendor_name": "Shopify", "pain_category": "support",
         "severity": "primary", "cnt": 99},
    ]
    pool = _ProvenancePool(core, severity)
    out = await _fetch_pain_provenance(pool, window_days=30)

    assert ("Shopify", "pricing") in out
    assert out[("Shopify", "pricing")]["primary_count"] == 5
    assert ("Shopify", "support") not in out


@pytest.mark.asyncio
async def test_provenance_gated_pain_may_differ_from_raw_pain_category():
    """If a row's raw pain_category is 'overall_dissatisfaction' but
    confidence='weak', the bucket gate keeps it under that same category
    (allow_generic=True default). Verify the bucket key matches gated value."""
    from atlas_brain.autonomous.tasks._b2b_shared import _fetch_pain_provenance

    pool = _ProvenancePool([_provenance_row(
        enrichment={
            "enrichment_schema_version": 4,
            "pain_category": "overall_dissatisfaction",
            "pain_confidence": "weak",
            "urgency_score": 6.0,
        },
    )])
    out = await _fetch_pain_provenance(pool, window_days=30)

    assert ("Shopify", "overall_dissatisfaction") in out
