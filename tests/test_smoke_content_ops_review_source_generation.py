from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_review_source_generation.py"
SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_review_source_generation",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class _Pool:
    def __init__(self, *, summary_rows, review_rows):
        self.summary_rows = list(summary_rows)
        self.review_rows = list(review_rows)
        self.fetch_calls = 0
        self.closed = False

    async def fetch(self, query, *args):
        del query
        del args
        self.fetch_calls += 1
        if self.fetch_calls == 1:
            return self.summary_rows
        return self.review_rows

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


def _review_row(*, source_review_id="g2-review-1"):
    return {
        "id": "review-uuid-1",
        "source": "g2",
        "source_url": f"https://example.test/{source_review_id}",
        "source_review_id": source_review_id,
        "vendor_name": "Slack",
        "rating": "4.0",
        "rating_max": "5",
        "review_text": "Slack notifications are noisy enough to slow the team down.",
        "reviewer_title": "Ops Manager",
        "reviewed_at": "2026-04-01T00:00:00Z",
        "enrichment": {
            "urgency_score": 7,
            "pain_category": ["performance"],
            "phrase_metadata": [
                {
                    "text": "notifications are noisy enough to slow the team down",
                    "field": "specific_complaints",
                    "polarity": "negative",
                    "subject": "subject_vendor",
                    "verbatim": True,
                    "category_hint": "performance",
                }
            ],
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
        "min_review_text_chars": 1,
        "phrase_limit": 5,
        "polarities": "negative,mixed",
        "phrase_fields": "specific_complaints,pricing_phrases,feature_gaps,recommendation_language",
        "summary_sources": "g2,capterra,trustradius,trustpilot",
        "allow_missing_source_url": False,
        "allow_ingestion_warnings": False,
        "default_field": [
            "company_name=Acme Logistics",
            "contact_email=ops@example.com",
            "contact_name=Jordan Lee",
        ],
        "forbidden_phrase": list(smoke.DEFAULT_FORBIDDEN_PHRASES),
        "output_source_rows": None,
        "output_drafts": None,
        "json": False,
        "database_url": "postgres://example",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.asyncio
async def test_review_source_smoke_exports_inspects_and_generates(monkeypatch, tmp_path: Path):
    pool = _Pool(summary_rows=[_summary_row()], review_rows=[_review_row()])
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    code, payload = await smoke.run_review_source_generation_smoke(
        _args(),
        source_rows_path=tmp_path / "g2_sources.jsonl",
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["source_summary"]["quote_grade_rows"] == 3
    assert payload["source_rows"] == 1
    assert payload["ingestion"]["ok"] is True
    assert payload["drafts"]["result"]["generated"] == 1
    body = payload["drafts"]["drafts"][0]["body"]
    assert "Teams evaluating Slack are reporting pain around" in body
    assert "appears to be weighing" not in body
    assert pool.closed is True


@pytest.mark.asyncio
async def test_review_source_smoke_fails_on_unresolved_ingestion_warnings(monkeypatch, tmp_path: Path):
    pool = _Pool(summary_rows=[_summary_row()], review_rows=[_review_row()])
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    code, payload = await smoke.run_review_source_generation_smoke(
        _args(default_field=[]),
        source_rows_path=tmp_path / "g2_sources.jsonl",
    )

    assert code == 1
    assert payload["ok"] is False
    assert any("ingestion inspection produced warnings" in error for error in payload["errors"])
    assert payload["drafts"] is None


@pytest.mark.asyncio
async def test_review_source_smoke_fails_when_source_has_no_quote_grade_rows(monkeypatch, tmp_path: Path):
    pool = _Pool(summary_rows=[_summary_row(quote_grade_rows=0)], review_rows=[_review_row()])
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    code, payload = await smoke.run_review_source_generation_smoke(
        _args(),
        source_rows_path=tmp_path / "g2_sources.jsonl",
    )

    assert code == 1
    assert payload["source_rows"] == 0
    assert any("fewer quote-grade rows" in error for error in payload["errors"])


def test_print_payload_keeps_json_stdout_clean(capsys):
    smoke._print_payload({"ok": False, "errors": ["bad"]}, as_json=True)

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"ok": False, "errors": ["bad"]}
    assert captured.err == ""
