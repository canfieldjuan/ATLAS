from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "backfill_witness_phrase_metadata.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "backfill_witness_phrase_metadata",
    _SCRIPT_PATH,
)
assert _SPEC is not None
backfill = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(backfill)


class FakeConn:
    def __init__(self):
        self.calls = []

    async def fetch(self, query, *args):
        self.calls.append((query, args))
        return []


@pytest.mark.asyncio
async def test_fetch_null_batch_initial_page_uses_composite_order_without_offset():
    conn = FakeConn()

    rows = await backfill._fetch_null_batch(
        conn,
        batch_size=100,
        last_key=None,
    )

    assert rows == []
    query, args = conn.calls[0]
    assert args == (100,)
    assert "OFFSET" not in query.upper()
    assert "ORDER BY w.vendor_name, w.as_of_date, w.analysis_window_days" in query


@pytest.mark.asyncio
async def test_fetch_null_batch_after_last_key_uses_keyset_pagination():
    conn = FakeConn()
    last_key = ("Acme", date(2026, 4, 1), 90, "v2", "witness-1")

    await backfill._fetch_null_batch(
        conn,
        batch_size=50,
        last_key=last_key,
    )

    query, args = conn.calls[0]
    assert args == (50, *last_key)
    assert "OFFSET" not in query.upper()
    assert ") > ($2, $3, $4, $5, $6)" in query
    assert "w.vendor_name" in query
    assert "w.witness_id" in query


def test_row_key_matches_fetch_order_columns():
    row = {
        "vendor_name": "Acme",
        "as_of_date": date(2026, 4, 1),
        "analysis_window_days": 90,
        "schema_version": "v2",
        "witness_id": "witness-1",
    }

    assert backfill._row_key(row) == (
        "Acme",
        date(2026, 4, 1),
        90,
        "v2",
        "witness-1",
    )


def test_classify_row_prefers_stamped_pain_confidence():
    row = {
        "enrichment": {"pain_confidence": "weak"},
        "pain_category": "pricing",
    }

    assert backfill._classify_row(row) == "weak"


def test_classify_row_recomputes_when_stamp_missing(monkeypatch):
    calls = []

    def fake_compute(enrichment, pain_category):
        calls.append((enrichment, pain_category))
        return "strong"

    monkeypatch.setattr(backfill, "_compute_pain_confidence", fake_compute)
    row = {
        "enrichment": {"specific_complaints": ["too expensive"]},
        "pain_category": "pricing",
    }

    assert backfill._classify_row(row) == "strong"
    assert calls == [({"specific_complaints": ["too expensive"]}, "pricing")]


def test_classify_row_returns_none_without_parseable_enrichment():
    assert backfill._classify_row({"enrichment": None}) is None
    assert backfill._classify_row({"enrichment": "not-json"}) is None
