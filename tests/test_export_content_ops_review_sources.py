from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/export_content_ops_review_sources.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "export_content_ops_review_sources",
        CLI,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()


class _Pool:
    def __init__(self, rows) -> None:
        self.rows = list(rows)
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, query, *args):
        self.fetch_calls.append((str(query), args))
        return self.rows


def _enrichment(*, polarity: str = "negative", verbatim: bool = True, subject: str = "subject_vendor"):
    return {
        "pain_category": "overall_dissatisfaction",
        "pain_categories": [
            {"category": "pricing", "severity": "primary"},
            {"category": "overall_dissatisfaction", "severity": "secondary"},
        ],
        "urgency_score": 7,
        "phrase_metadata": [
            {
                "text": "The renewal cost became hard to justify.",
                "field": "pricing_phrases",
                "subject": subject,
                "polarity": polarity,
                "verbatim": verbatim,
                "category_hint": "pricing",
            },
            {
                "text": "The onboarding team was excellent.",
                "field": "positive_aspects",
                "subject": "subject_vendor",
                "polarity": "positive",
                "verbatim": True,
                "category_hint": "support",
            },
        ],
    }


def _row(**overrides):
    row = {
        "id": "review-1",
        "source": "g2",
        "source_url": "https://www.g2.com/reviews/review-1",
        "source_review_id": "g2-review-1",
        "vendor_name": "HubSpot",
        "product_name": "HubSpot",
        "product_category": "CRM",
        "rating": 3.5,
        "rating_max": 5,
        "summary": "Renewal pricing",
        "review_text": "Great onboarding, but the renewal cost became hard to justify.",
        "reviewer_title": "VP Revenue Operations",
        "reviewer_company": "",
        "reviewer_industry": "Logistics",
        "reviewed_at": "2026-05-01T00:00:00+00:00",
        "imported_at": "2026-05-02T00:00:00+00:00",
        "enrichment": _enrichment(),
    }
    row.update(overrides)
    return row


def test_quote_grade_review_phrases_filters_to_subject_vendor_negative_verbatim() -> None:
    phrases = mod.quote_grade_review_phrases({
        "phrase_metadata": [
            {
                "text": "Keep",
                "field": "specific_complaints",
                "subject": "subject_vendor",
                "polarity": "negative",
                "verbatim": True,
            },
            {
                "text": "Drop competitor",
                "field": "specific_complaints",
                "subject": "competitor",
                "polarity": "negative",
                "verbatim": True,
            },
            {
                "text": "Drop nonverbatim",
                "field": "specific_complaints",
                "subject": "subject_vendor",
                "polarity": "negative",
                "verbatim": False,
            },
            {
                "text": "Drop positive",
                "field": "positive_aspects",
                "subject": "subject_vendor",
                "polarity": "positive",
                "verbatim": True,
            },
        ]
    })

    assert phrases == [{
        "text": "Keep",
        "field": "specific_complaints",
        "polarity": "negative",
        "category_hint": "",
    }]


def test_review_row_to_source_row_uses_phrase_lane_as_text_and_preserves_full_review() -> None:
    enrichment = _enrichment()
    enrichment["pain_category"] = "pricing"
    out = mod.review_row_to_source_row(_row(enrichment=enrichment))

    assert out["id"] == "g2-review-1"
    assert out["source_id"] == "g2-review-1"
    assert out["review_id"] == "review-1"
    assert out["source"] == "g2"
    assert out["source_type"] == "review"
    assert out["vendor_name"] == "HubSpot"
    assert out["text"] == "The renewal cost became hard to justify."
    assert out["review_text"] == "Great onboarding, but the renewal cost became hard to justify."
    assert out["quote_grade_phrases"] == ["The renewal cost became hard to justify."]
    assert out["pain_points"] == ["pricing"]
    assert out["contact_title"] == "VP Revenue Operations"


def test_review_row_to_source_row_unescapes_exported_titles() -> None:
    out = mod.review_row_to_source_row(_row(reviewer_title="Co-Founder &amp; CEO"))

    assert out["contact_title"] == "Co-Founder & CEO"


def test_review_row_to_source_row_drops_rows_without_quote_grade_phrases() -> None:
    assert mod.review_row_to_source_row(_row(enrichment=_enrichment(verbatim=False))) == {}
    assert mod.review_row_to_source_row(_row(enrichment=_enrichment(polarity="positive"))) == {}
    assert mod.review_row_to_source_row(_row(enrichment=_enrichment(subject="competitor"))) == {}


def test_review_row_to_source_row_accepts_json_string_enrichment() -> None:
    out = mod.review_row_to_source_row(_row(enrichment=json.dumps(_enrichment())))

    assert out["quote_grade_phrases"] == ["The renewal cost became hard to justify."]


def test_build_review_source_query_filters_canonical_enriched_g2_rows() -> None:
    query, args = mod.build_review_source_query(
        source="g2",
        vendor_name="HubSpot",
        min_review_text_chars=120,
        require_review_url=True,
    )

    assert "LOWER(r.source) = LOWER($1)" in query
    assert "r.duplicate_of_review_id IS NULL" in query
    assert "r.enrichment_status = 'enriched'" in query
    assert "r.enrichment IS NOT NULL" in query
    assert "length(r.review_text) >= $2" in query
    assert "LOWER(r.vendor_name) = LOWER($3)" in query
    assert "NULLIF(BTRIM(r.source_url), '') IS NOT NULL" in query
    assert "LIMIT $4" in query
    assert "OFFSET $5" in query
    assert args == ["g2", 120, "HubSpot", 0, 0]


def test_build_review_source_summary_query_counts_quote_grade_rows() -> None:
    query, args = mod.build_review_source_summary_query(
        sources=("g2", "trustpilot"),
        min_review_text_chars=120,
        allowed_polarities=("negative", "mixed"),
        allowed_fields=("pricing_phrases",),
        require_review_url=True,
    )

    assert "lower(r.source) = ANY($1::text[])" in query
    assert "count(*) AS total_rows" in query
    assert "COALESCE(NULLIF(BTRIM(r.source_review_id), ''), r.id::text)" in query
    assert "count(DISTINCT" in query
    assert "AS export_candidate_rows" in query
    assert "AS quote_grade_rows" in query
    assert "jsonb_array_elements" in query
    assert "jsonb_typeof(r.enrichment->'phrase_metadata') = 'array'" in query
    assert "lower(BTRIM(pm->>'subject')) = 'subject_vendor'" in query
    assert "pm->'verbatim' = 'true'::jsonb" in query
    assert "lower(BTRIM(pm->>'polarity')) = ANY($3::text[])" in query
    assert "BTRIM(pm->>'field') = ANY($4::text[])" in query
    assert "NULLIF(BTRIM(r.source_url), '') IS NOT NULL" in query
    assert args == [
        ["g2", "trustpilot"],
        120,
        ["negative", "mixed"],
        ["pricing_phrases"],
    ]


@pytest.mark.asyncio
async def test_fetch_review_source_rows_dedupes_and_filters_after_fetch() -> None:
    pool = _Pool([
        _row(id="review-1", source_review_id="same-id"),
        _row(id="review-2", source_review_id="same-id"),
        _row(id="review-3", source_review_id="other-id", enrichment=_enrichment(verbatim=False)),
        _row(id="review-4", source_review_id="keep-id"),
    ])

    rows = await mod.fetch_review_source_rows(pool, source="g2", limit=2)

    assert [row["source_id"] for row in rows] == ["same-id", "keep-id"]
    query, args = pool.fetch_calls[0]
    assert "FROM b2b_reviews r" in query
    assert args[0] == "g2"
    assert args[-2:] == (6, 0)


@pytest.mark.asyncio
async def test_fetch_review_source_summary_returns_requested_sources_with_zero_fill() -> None:
    pool = _Pool([
        {
            "source": "g2",
            "total_rows": 10,
            "canonical_rows": 9,
            "enriched_rows": 8,
            "export_candidate_rows": 7,
            "quote_grade_rows": 6,
        }
    ])

    rows = await mod.fetch_review_source_summary(
        pool,
        sources=("g2", "trustpilot"),
        min_review_text_chars=100,
    )

    assert rows == [
        {
            "source": "g2",
            "total_rows": 10,
            "canonical_rows": 9,
            "enriched_rows": 8,
            "export_candidate_rows": 7,
            "quote_grade_rows": 6,
        },
        {
            "source": "trustpilot",
            "total_rows": 0,
            "canonical_rows": 0,
            "enriched_rows": 0,
            "export_candidate_rows": 0,
            "quote_grade_rows": 0,
        },
    ]
    query, args = pool.fetch_calls[0]
    assert "FROM b2b_reviews r" in query
    assert args[0] == ["g2", "trustpilot"]
    assert args[1] == 100


def test_render_jsonl_outputs_one_json_object_per_line() -> None:
    text = mod.render_jsonl([
        {"source_id": "a", "text": "one"},
        {"source_id": "b", "text": "two"},
    ])

    lines = text.splitlines()
    assert [json.loads(line)["source_id"] for line in lines] == ["a", "b"]


def test_parse_args_defaults_to_database_url_from_helper(monkeypatch) -> None:
    monkeypatch.setattr(mod, "_default_database_url", lambda: "postgres://example/db")

    args = mod._parse_args([])

    assert args.database_url == "postgres://example/db"


def test_parse_args_accepts_source_summary_sources(monkeypatch) -> None:
    monkeypatch.setattr(mod, "_default_database_url", lambda: "postgres://example/db")

    args = mod._parse_args(["--source-summary", "--summary-sources", "g2,trustpilot"])

    assert args.source_summary is True
    assert args.summary_sources == "g2,trustpilot"
