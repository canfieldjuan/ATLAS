from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_review_source_postgres.py"
SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_review_source_postgres",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class _Pool:
    def __init__(
        self,
        *,
        summary_rows,
        source_rows,
        opportunity_rows=None,
        saved_draft_rows=None,
        existing_relations=None,
    ):
        self.summary_rows = list(summary_rows)
        self.source_rows = list(source_rows)
        self.opportunity_rows = list(opportunity_rows or [])
        self.saved_draft_rows = list(saved_draft_rows or [])
        self.existing_relations = set(
            existing_relations or ("campaign_opportunities", "b2b_campaigns")
        )
        self.fetch_calls = []
        self.execute_calls = []
        self.fetchval_calls = []
        self.fetchval_results = ["campaign-1", "campaign-2"]
        self.closed = False

    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": str(query), "args": args})
        call_number = len(self.fetch_calls)
        if call_number == 1:
            return self.summary_rows
        if call_number == 2:
            return self.source_rows
        if call_number == 3:
            return self.opportunity_rows
        return self.saved_draft_rows

    async def execute(self, query, *args):
        self.execute_calls.append({"query": str(query), "args": args})
        return "EXECUTE"

    async def fetchval(self, query, *args):
        self.fetchval_calls.append({"query": str(query), "args": args})
        if "to_regclass" in str(query):
            return args[0] if args and args[0] in self.existing_relations else None
        return self.fetchval_results.pop(0)

    async def close(self):
        self.closed = True


async def _return_pool(pool):
    return pool


def _summary_row(*, quote_grade_rows=3):
    return {
        "source": "g2",
        "total_rows": 10,
        "canonical_rows": 10,
        "enriched_rows": 8,
        "export_candidate_rows": 5,
        "quote_grade_rows": quote_grade_rows,
    }


def _source_row():
    return {
        "id": "review-1",
        "source": "g2",
        "source_url": "https://example.test/review-1",
        "source_review_id": "review-1",
        "vendor_name": "Slack",
        "rating": "4.0",
        "rating_max": "5",
        "review_text": "Notifications are noisy enough to slow the team down.",
        "reviewer_title": "Ops Manager",
        "reviewed_at": "2026-04-01T00:00:00Z",
        "enrichment": {
            "urgency_score": 7,
            "pain_category": ["performance"],
            "phrase_metadata": [
                {
                    "text": "Notifications are noisy enough to slow the team down.",
                    "field": "specific_complaints",
                    "polarity": "negative",
                    "subject": "subject_vendor",
                    "verbatim": True,
                    "category_hint": "performance",
                }
            ],
        },
    }


def _opportunity_row():
    return {
        "target_id": "review-1",
        "company_name": "Acme Logistics",
        "vendor_name": "Slack",
        "contact_email": "ops@example.com",
        "contact_name": "Jordan Lee",
        "pain_points": ["performance"],
        "evidence": [
            {
                "source_id": "review-1",
                "source_type": "review",
                "text": "Notifications are noisy enough to slow the team down.",
            }
        ],
        "raw_payload": {
            "target_id": "review-1",
            "company_name": "Acme Logistics",
            "vendor_name": "Slack",
            "contact_email": "ops@example.com",
            "source_type": "review",
        },
    }


def _saved_draft_row(*, body=None):
    return {
        "id": "campaign-1",
        "subject": "Acme Logistics: performance",
        "body": body
        or (
            "<p>Teams evaluating Slack are reporting pain around performance.</p>"
        ),
        "target_mode": "vendor_retention",
        "channel": "email_cold",
        "metadata": {
            "target_id": "review-1",
            "source_opportunity": {
                "target_id": "review-1",
            },
        },
    }


def _args(**overrides):
    values = {
        "source": "g2",
        "vendor": "Slack",
        "limit": 1,
        "target_mode": "vendor_retention",
        "channels": "email_cold",
        "min_drafts": None,
        "min_quote_grade_rows": 1,
        "min_review_text_chars": 80,
        "phrase_limit": 5,
        "polarities": ",".join(smoke.DEFAULT_POLARITIES),
        "phrase_fields": ",".join(smoke.DEFAULT_PHRASE_FIELDS),
        "summary_sources": ",".join(smoke.DEFAULT_SUMMARY_SOURCES),
        "allow_missing_source_url": False,
        "allow_ingestion_warnings": False,
        "default_field": [
            "company_name=Acme Logistics",
            "contact_email=ops@example.com",
            "contact_name=Jordan Lee",
        ],
        "account_id": "acct-smoke",
        "user_id": None,
        "opportunity_table": "campaign_opportunities",
        "keep_existing_opportunities": False,
        "forbidden_phrase": list(smoke.DEFAULT_FORBIDDEN_PHRASES),
        "output_source_rows": None,
        "output_result": None,
        "json": False,
        "database_url": "postgres://example",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.asyncio
async def test_review_source_postgres_smoke_imports_and_persists(monkeypatch, tmp_path):
    pool = _Pool(
        summary_rows=[_summary_row()],
        source_rows=[_source_row()],
        opportunity_rows=[_opportunity_row()],
        saved_draft_rows=[_saved_draft_row()],
    )
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    code, payload = await smoke.run_review_source_postgres_smoke(
        _args(),
        source_rows_path=tmp_path / "g2_sources.jsonl",
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["source_rows"] == 1
    assert payload["import"]["inserted"] == 1
    assert payload["import"]["replace_existing"] is True
    assert payload["drafts"]["generated"] == 1
    assert payload["saved_drafts"][0]["body"].startswith(
        "<p>Teams evaluating Slack are reporting pain around"
    )
    assert payload["saved_drafts"][0]["target_id"] == "review-1"
    assert pool.closed is True
    assert "DELETE FROM \"campaign_opportunities\"" in pool.execute_calls[0]["query"]
    assert "INSERT INTO \"campaign_opportunities\"" in pool.execute_calls[1]["query"]
    assert pool.execute_calls[0]["args"] == (
        "acct-smoke",
        "vendor_retention",
        ["review-1"],
    )
    assert any("INSERT INTO b2b_campaigns" in call["query"] for call in pool.fetchval_calls)
    assert "FROM b2b_campaigns" in pool.fetch_calls[-1]["query"]
    assert pool.fetch_calls[2]["args"] == ("vendor_retention", "acct-smoke", "review-1", 1)


@pytest.mark.asyncio
async def test_review_source_postgres_smoke_fails_before_import_when_schema_missing(
    monkeypatch,
    tmp_path,
):
    pool = _Pool(
        summary_rows=[_summary_row()],
        source_rows=[_source_row()],
        existing_relations={"b2b_campaigns"},
    )
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    code, payload = await smoke.run_review_source_postgres_smoke(
        _args(),
        source_rows_path=tmp_path / "g2_sources.jsonl",
    )

    assert code == 1
    assert any("required Content Ops table(s) missing" in error for error in payload["errors"])
    assert any("campaign_opportunities" in error for error in payload["errors"])
    assert any(
        "run_extracted_content_pipeline_migrations.py" in error
        for error in payload["errors"]
    )
    assert pool.execute_calls == []
    assert all("INSERT INTO b2b_campaigns" not in call["query"] for call in pool.fetchval_calls)


@pytest.mark.asyncio
async def test_review_source_postgres_smoke_can_keep_existing_opportunities(
    monkeypatch,
    tmp_path,
):
    pool = _Pool(
        summary_rows=[_summary_row()],
        source_rows=[_source_row()],
        opportunity_rows=[_opportunity_row()],
        saved_draft_rows=[_saved_draft_row()],
    )
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    code, payload = await smoke.run_review_source_postgres_smoke(
        _args(keep_existing_opportunities=True),
        source_rows_path=tmp_path / "g2_sources.jsonl",
    )

    assert code == 0
    assert payload["import"]["replace_existing"] is False
    assert all("DELETE FROM" not in call["query"] for call in pool.execute_calls)


@pytest.mark.asyncio
async def test_review_source_postgres_smoke_skips_export_error_when_quote_gate_fails(
    monkeypatch,
    tmp_path,
):
    pool = _Pool(
        summary_rows=[_summary_row(quote_grade_rows=0)],
        source_rows=[_source_row()],
    )
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    code, payload = await smoke.run_review_source_postgres_smoke(
        _args(),
        source_rows_path=tmp_path / "g2_sources.jsonl",
    )

    assert code == 1
    assert payload["source_rows"] == 0
    assert any("fewer quote-grade rows" in error for error in payload["errors"])
    assert all("expected 1 exported" not in error for error in payload["errors"])
    assert pool.execute_calls == []
    assert pool.fetchval_calls == []


@pytest.mark.asyncio
async def test_review_source_postgres_smoke_fails_on_forbidden_persisted_body(
    monkeypatch,
    tmp_path,
):
    pool = _Pool(
        summary_rows=[_summary_row()],
        source_rows=[_source_row()],
        opportunity_rows=[_opportunity_row()],
        saved_draft_rows=[_saved_draft_row(body="<p>Acme appears to be weighing Slack.</p>")],
    )
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    code, payload = await smoke.run_review_source_postgres_smoke(
        _args(),
        source_rows_path=tmp_path / "g2_sources.jsonl",
    )

    assert code == 1
    assert payload["ok"] is False
    assert any("forbidden phrase" in error for error in payload["errors"])


@pytest.mark.asyncio
async def test_review_source_postgres_smoke_fails_on_generation_errors_and_skips(
    monkeypatch,
    tmp_path,
):
    pool = _Pool(
        summary_rows=[_summary_row()],
        source_rows=[_source_row()],
        saved_draft_rows=[_saved_draft_row()],
    )
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    async def generate_with_error(**_kwargs):
        return {
            "requested": 1,
            "generated": 1,
            "skipped": 1,
            "reasoning_contexts_used": 0,
            "saved_ids": ["campaign-1"],
            "errors": [{"target_id": "review-1", "reason": "bad"}],
        }

    monkeypatch.setattr(smoke, "generate_imported_target_drafts", generate_with_error)

    code, payload = await smoke.run_review_source_postgres_smoke(
        _args(min_drafts=1),
        source_rows_path=tmp_path / "g2_sources.jsonl",
    )

    assert code == 1
    assert any("generation reported 1 error" in error for error in payload["errors"])
    assert any("generation skipped 1 draft" in error for error in payload["errors"])


@pytest.mark.asyncio
async def test_review_source_postgres_smoke_fails_on_wrong_persisted_target(
    monkeypatch,
    tmp_path,
):
    row = _saved_draft_row()
    row["metadata"] = {"target_id": "other-review"}
    pool = _Pool(
        summary_rows=[_summary_row()],
        source_rows=[_source_row()],
        opportunity_rows=[_opportunity_row()],
        saved_draft_rows=[row],
    )
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    code, payload = await smoke.run_review_source_postgres_smoke(
        _args(),
        source_rows_path=tmp_path / "g2_sources.jsonl",
    )

    assert code == 1
    assert any("persisted draft target_id was not imported" in error for error in payload["errors"])


@pytest.mark.asyncio
async def test_review_source_postgres_smoke_fails_on_missing_persisted_target(
    monkeypatch,
    tmp_path,
):
    row = _saved_draft_row()
    row["metadata"] = {}
    pool = _Pool(
        summary_rows=[_summary_row()],
        source_rows=[_source_row()],
        opportunity_rows=[_opportunity_row()],
        saved_draft_rows=[row],
    )
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    code, payload = await smoke.run_review_source_postgres_smoke(
        _args(),
        source_rows_path=tmp_path / "g2_sources.jsonl",
    )

    assert code == 1
    assert any("persisted draft missing target_id metadata" in error for error in payload["errors"])
