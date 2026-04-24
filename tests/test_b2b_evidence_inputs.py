from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from atlas_brain.api import b2b_evidence as mod
from atlas_brain.autonomous.tasks._b2b_witnesses import derive_evidence_spans


@pytest.mark.asyncio
async def test_list_witnesses_normalizes_query_defaults_on_direct_call(monkeypatch):
    captured = {}

    async def _latest_snapshot(pool, vendor_name, window_days, target_date):
        captured.update(vendor_name=vendor_name, window_days=window_days, target_date=target_date)
        return None

    pool = SimpleNamespace(is_initialized=True)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "resolve_vendor_name", AsyncMock(return_value=None))
    monkeypatch.setattr(mod, "_latest_witness_snapshot_date", _latest_snapshot)

    result = await mod.list_witnesses("Salesforce", user=SimpleNamespace(account_id="not-a-uuid"))

    assert captured["vendor_name"] == "Salesforce"
    assert captured["window_days"] == mod._default_analysis_window_days()
    assert result["analysis_window_days"] == mod._default_analysis_window_days()
    assert result["limit"] == mod.DEFAULT_WITNESS_LIMIT
    assert result["offset"] == 0
    assert result["as_of_date"] is None


@pytest.mark.asyncio
async def test_get_witness_normalizes_query_default_window_days(monkeypatch):
    captured = {}

    async def _latest_snapshot(pool, vendor_name, window_days, target_date):
        captured["window_days"] = window_days
        return date(2026, 4, 1)

    pool = SimpleNamespace(is_initialized=True, fetchrow=AsyncMock(return_value=None))
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "resolve_vendor_name", AsyncMock(return_value=None))
    monkeypatch.setattr(mod, "_latest_witness_snapshot_date", _latest_snapshot)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.get_witness("w1", "Salesforce", user=SimpleNamespace(account_id=str(uuid4())))

    assert exc.value.status_code == 404
    assert captured["window_days"] == mod._default_analysis_window_days()
    assert pool.fetchrow.await_args.args[3] == mod._default_analysis_window_days()
    assert pool.fetchrow.await_args.args[4] == date(2026, 4, 1)


@pytest.mark.asyncio
async def test_get_vault_normalizes_query_default_window_days(monkeypatch):
    captured = {}

    async def _read_record(pool, vendor_name, *, as_of, analysis_window_days):
        captured["analysis_window_days"] = analysis_window_days
        return None

    monkeypatch.setattr(mod, "get_db_pool", lambda: SimpleNamespace(is_initialized=True))
    monkeypatch.setattr(mod, "resolve_vendor_name", AsyncMock(return_value=None))
    monkeypatch.setattr(mod, "_read_vendor_intelligence_record", _read_record)

    result = await mod.get_vault("Salesforce", user=SimpleNamespace(account_id=str(uuid4())))

    assert captured["analysis_window_days"] == mod._default_analysis_window_days()
    assert result == {
        "vendor_name": "Salesforce",
        "vault": None,
        "message": "No evidence vault found for this vendor",
    }


def _lazy_grounding_witness_row(**overrides):
    base = {
        "witness_id": "w-lazy-1",
        "vendor_name": "Salesforce",
        "as_of_date": date(2026, 4, 1),
        "analysis_window_days": 30,
        "schema_version": "witness_packet_v1",
        "review_id": "rev-1",
        "excerpt_text": "it was too expensive",
        "source_span_id": None,
        "review_text": "We loved it but it was too expensive for us.",
        "summary": "",
        "pros": "",
        "cons": "",
        "rating": 3,
        "review_source": "g2",
        "source_url": None,
        "reviewer_name": None,
        "reviewer_company": None,
        "reviewer_title": None,
        "enrichment_status": "enriched",
        "enrichment": {},
        "grounding_status": "pending",
        "grounding_checked_at": None,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_get_witness_lazy_grounding_fallback_rewrites_pending_row(monkeypatch):
    """Pending witness in the detail endpoint triggers a fresh grounding
    check, persists the result, and returns the new status in the response."""
    witness_row = _lazy_grounding_witness_row()
    update_calls = []

    async def _execute(query, *args):
        update_calls.append((query, args))

    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(return_value=witness_row),
        execute=_execute,
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "resolve_vendor_name", AsyncMock(return_value=None))
    monkeypatch.setattr(
        mod, "_latest_witness_snapshot_date",
        AsyncMock(return_value=date(2026, 4, 1)),
    )

    result = await mod.get_witness(
        "w-lazy-1", "Salesforce",
        user=SimpleNamespace(account_id=str(uuid4())),
    )
    witness = result["witness"]

    # Response reflects the fresh classification
    assert witness["grounding_status"] == "grounded"
    assert witness["grounding_checked_at"] is not None

    # UPDATE was issued with grounding_status = 'pending' in the WHERE clause
    # (safe under concurrent lazy calls)
    assert len(update_calls) == 1
    query, args = update_calls[0]
    assert "UPDATE b2b_vendor_witnesses" in query
    assert "AND grounding_status = 'pending'" in query
    assert args[0] == "grounded"


@pytest.mark.asyncio
async def test_get_witness_lazy_grounding_marks_not_grounded_when_phrase_absent(monkeypatch):
    witness_row = _lazy_grounding_witness_row(
        excerpt_text="phrase that never appears in the source",
        review_text="Totally different review content.",
    )
    update_calls = []

    async def _execute(query, *args):
        update_calls.append((query, args))

    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(return_value=witness_row),
        execute=_execute,
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "resolve_vendor_name", AsyncMock(return_value=None))
    monkeypatch.setattr(
        mod, "_latest_witness_snapshot_date",
        AsyncMock(return_value=date(2026, 4, 1)),
    )

    result = await mod.get_witness(
        "w-lazy-1", "Salesforce",
        user=SimpleNamespace(account_id=str(uuid4())),
    )
    witness = result["witness"]

    assert witness["grounding_status"] == "not_grounded"
    assert len(update_calls) == 1
    assert update_calls[0][1][0] == "not_grounded"


@pytest.mark.asyncio
async def test_get_witness_skips_lazy_path_for_already_graded_rows(monkeypatch):
    """Rows with grounding_status != 'pending' should NOT trigger a lazy
    UPDATE. Post-backfill this is the steady-state path -- no extra DB
    writes on the hot read path."""
    witness_row = _lazy_grounding_witness_row(
        grounding_status="grounded",
        grounding_checked_at=datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc),
    )
    update_calls = []

    async def _execute(query, *args):
        update_calls.append((query, args))

    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(return_value=witness_row),
        execute=_execute,
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "resolve_vendor_name", AsyncMock(return_value=None))
    monkeypatch.setattr(
        mod, "_latest_witness_snapshot_date",
        AsyncMock(return_value=date(2026, 4, 1)),
    )

    result = await mod.get_witness(
        "w-lazy-1", "Salesforce",
        user=SimpleNamespace(account_id=str(uuid4())),
    )
    witness = result["witness"]

    assert witness["grounding_status"] == "grounded"
    assert update_calls == [], "lazy UPDATE must not fire on non-pending rows"


@pytest.mark.asyncio
async def test_get_witness_quote_grade_true_when_grounded(monkeypatch):
    """Step 9: grounded witness exposes quote_grade=true and keeps its
    highlight bounds so the UI can render the verbatim quote."""
    # Non-empty enrichment so the highlight-computation block runs and
    # sets highlight_start/end via the substring-match path.
    witness_row = _lazy_grounding_witness_row(
        grounding_status="grounded",
        grounding_checked_at=datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc),
        excerpt_text="too expensive",
        review_text="We loved it but it was too expensive for us.",
        enrichment={"specific_complaints": ["too expensive"]},
    )

    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(return_value=witness_row),
        execute=AsyncMock(),
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "resolve_vendor_name", AsyncMock(return_value=None))
    monkeypatch.setattr(
        mod, "_latest_witness_snapshot_date",
        AsyncMock(return_value=date(2026, 4, 1)),
    )

    result = await mod.get_witness(
        "w-lazy-1", "Salesforce",
        user=SimpleNamespace(account_id=str(uuid4())),
    )
    witness = result["witness"]

    assert witness["quote_grade"] is True
    assert witness["grounding_status"] == "grounded"
    # highlight bounds were computed from the substring-match path and
    # MUST survive (not be suppressed) since the witness is quote-grade.
    assert "highlight_start" in witness
    assert "highlight_end" in witness


@pytest.mark.asyncio
async def test_get_witness_quote_grade_false_when_not_grounded(monkeypatch):
    """Step 9: not-grounded witness exposes quote_grade=false and the API
    strips highlight bounds so the UI cannot render a quote even if a
    stale client tries to."""
    witness_row = _lazy_grounding_witness_row(
        grounding_status="not_grounded",
        grounding_checked_at=datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc),
        # excerpt_text DOES substring-match review_text and the highlight
        # path WILL set bounds. The gate must still strip them because
        # grounding rejected this witness.
        excerpt_text="too expensive",
        review_text="It was too expensive for what we got.",
        enrichment={"specific_complaints": ["too expensive"]},
    )

    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(return_value=witness_row),
        execute=AsyncMock(),
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "resolve_vendor_name", AsyncMock(return_value=None))
    monkeypatch.setattr(
        mod, "_latest_witness_snapshot_date",
        AsyncMock(return_value=date(2026, 4, 1)),
    )

    result = await mod.get_witness(
        "w-lazy-1", "Salesforce",
        user=SimpleNamespace(account_id=str(uuid4())),
    )
    witness = result["witness"]

    assert witness["quote_grade"] is False
    assert witness["grounding_status"] == "not_grounded"
    # API strips highlight bounds so the UI can't render the quote
    assert "highlight_start" not in witness
    assert "highlight_end" not in witness


@pytest.mark.asyncio
async def test_get_witness_lazy_grounding_returns_fresh_status_even_when_update_fails(monkeypatch):
    """If the persist UPDATE raises (transient DB issue), the caller still
    gets the fresh classification in the response."""
    witness_row = _lazy_grounding_witness_row()

    async def _execute(query, *args):
        raise RuntimeError("simulated transient DB failure")

    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(return_value=witness_row),
        execute=_execute,
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "resolve_vendor_name", AsyncMock(return_value=None))
    monkeypatch.setattr(
        mod, "_latest_witness_snapshot_date",
        AsyncMock(return_value=date(2026, 4, 1)),
    )

    result = await mod.get_witness(
        "w-lazy-1", "Salesforce",
        user=SimpleNamespace(account_id=str(uuid4())),
    )
    witness = result["witness"]

    # Response still has the fresh value even though persist failed
    assert witness["grounding_status"] == "grounded"


@pytest.mark.asyncio
async def test_get_witness_prefers_current_derived_spans_for_highlights(monkeypatch):
    review_row = {
        "id": "rev-1",
        "review_text": "Tried going to the site again and talked to a robot who could not help me.",
        "summary": "",
        "pros": "",
        "cons": "",
        "reviewer_company": None,
        "reviewer_title": None,
    }
    enrichment = {
        "pain_category": "pricing",
        "specific_complaints": ["talked to a robot who could not help me"],
        "pricing_phrases": [],
        "recommendation_language": [],
        "positive_aspects": [],
        "competitors_mentioned": [],
        "event_mentions": [],
        "timeline": {},
        "budget_signals": {},
        "reviewer_context": {},
        "churn_signals": {},
        "evidence_spans": [
            {
                "span_id": "stale-span",
                "text": "talked to a robot who could not help me",
                "pain_category": "pricing",
                "signal_type": "complaint",
                "start_char": 0,
                "end_char": 12,
                "flags": [],
            }
        ],
    }
    derived_spans = derive_evidence_spans(enrichment, review_row)
    witness_row = {
        "witness_id": "w1",
        "review_id": "rev-1",
        "excerpt_text": "talked to a robot who could not help me",
        "source_span_id": derived_spans[0]["span_id"],
        "review_text": review_row["review_text"],
        "summary": review_row["summary"],
        "pros": review_row["pros"],
        "cons": review_row["cons"],
        "rating": 1,
        "review_source": "trustpilot",
        "source_url": None,
        "reviewer_name": None,
        "reviewer_company": None,
        "reviewer_title": None,
        "enrichment_status": "enriched",
        "enrichment": enrichment,
    }

    async def _latest_snapshot(_pool, _vendor_name, _window_days, _target_date):
        return date(2026, 4, 1)

    pool = SimpleNamespace(is_initialized=True, fetchrow=AsyncMock(return_value=witness_row))
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "resolve_vendor_name", AsyncMock(return_value=None))
    monkeypatch.setattr(mod, "_latest_witness_snapshot_date", _latest_snapshot)

    result = await mod.get_witness("w1", "Salesforce", user=SimpleNamespace(account_id=str(uuid4())))
    witness = result["witness"]

    assert witness["evidence_span_source"] == "refreshed"
    assert witness["evidence_spans"][0]["pain_category"] == "support"
    assert witness["highlight_start"] >= 0
    assert witness["highlight_end"] > witness["highlight_start"]


@pytest.mark.asyncio
async def test_list_annotations_normalizes_query_defaults_on_direct_call(monkeypatch):
    account_id = str(uuid4())
    pool = SimpleNamespace(is_initialized=True, fetch=AsyncMock(return_value=[]))
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "resolve_vendor_name", AsyncMock(side_effect=AssertionError("resolve touched")))

    result = await mod.list_annotations(user=SimpleNamespace(account_id=account_id))

    assert pool.fetch.await_args.args[1:] == (UUID(account_id),)
    assert result == {"annotations": [], "count": 0}
