#!/usr/bin/env python3
"""Export canonical Atlas review rows as Content Ops source-row JSONL."""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - host dependency
    load_dotenv = None


DEFAULT_SOURCE = "g2"
DEFAULT_POLARITIES = ("negative", "mixed")
DEFAULT_PHRASE_FIELDS = (
    "specific_complaints",
    "pricing_phrases",
    "feature_gaps",
    "recommendation_language",
)


def _clean_text(value: Any) -> str:
    return html.unescape(str(value or "").strip())


def _safe_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _text_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        values = (value,)
    out: list[str] = []
    for item in values:
        text = _clean_text(item)
        if text and text not in out:
            out.append(text)
    return out


def _unique_texts(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = _clean_text(value)
        if text and text not in out:
            out.append(text)
    return out


def _pain_categories(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, Mapping):
        return _text_list(value.get("category") or value.get("name") or value.get("label"))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        out: list[str] = []
        for item in value:
            for text in _pain_categories(item):
                if text and text not in out:
                    out.append(text)
        return out
    return _text_list(value)


def _phrase_metadata(enrichment: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = enrichment.get("phrase_metadata")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def quote_grade_review_phrases(
    enrichment: Mapping[str, Any],
    *,
    allowed_polarities: Sequence[str] = DEFAULT_POLARITIES,
    allowed_fields: Sequence[str] = DEFAULT_PHRASE_FIELDS,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return coherent subject-vendor, verbatim negative/mixed phrases."""

    polarities = {p.strip().lower() for p in allowed_polarities if p.strip()}
    fields = {f.strip() for f in allowed_fields if f.strip()}
    phrases: list[dict[str, Any]] = []
    for entry in _phrase_metadata(enrichment):
        text = _clean_text(entry.get("text"))
        if not text:
            continue
        if entry.get("verbatim") is not True:
            continue
        if _clean_text(entry.get("subject")).lower() != "subject_vendor":
            continue
        polarity = _clean_text(entry.get("polarity")).lower()
        if polarities and polarity not in polarities:
            continue
        field = _clean_text(entry.get("field"))
        if fields and field not in fields:
            continue
        phrases.append({
            "text": text,
            "field": field,
            "polarity": polarity,
            "category_hint": _clean_text(entry.get("category_hint")),
        })
        if limit is not None and len(phrases) >= limit:
            break
    return phrases


def review_row_to_source_row(
    row: Mapping[str, Any],
    *,
    phrase_limit: int = 5,
    allowed_polarities: Sequence[str] = DEFAULT_POLARITIES,
    allowed_fields: Sequence[str] = DEFAULT_PHRASE_FIELDS,
    max_full_review_chars: int = 3000,
) -> dict[str, Any]:
    """Convert one canonical review row into a Content Ops source row."""

    enrichment = _safe_json_object(row.get("enrichment"))
    phrases = quote_grade_review_phrases(
        enrichment,
        allowed_polarities=allowed_polarities,
        allowed_fields=allowed_fields,
        limit=phrase_limit,
    )
    if not phrases:
        return {}

    review_id = _clean_text(row.get("id"))
    source = _clean_text(row.get("source")) or DEFAULT_SOURCE
    source_review_id = _clean_text(row.get("source_review_id"))
    source_id = source_review_id or review_id
    phrase_texts = [phrase["text"] for phrase in phrases]
    pain_points = _unique_texts(
        text
        for text in _pain_categories(enrichment.get("pain_category"))
        + _pain_categories(enrichment.get("pain_categories"))
        if text != "overall_dissatisfaction"
    )
    if not pain_points:
        pain_points = [
            phrase["category_hint"]
            for phrase in phrases
            if phrase.get("category_hint")
        ]
    out: dict[str, Any] = {
        "id": source_id,
        "source_id": source_id,
        "review_id": review_id,
        "source_type": "review",
        "source": source,
        "vendor_name": _clean_text(row.get("vendor_name")),
        "review_text": _clean_text(row.get("review_text"))[:max_full_review_chars],
        "text": "\n".join(phrase_texts),
        "quote_grade_phrases": phrase_texts,
        "quote_phrase_fields": [phrase["field"] for phrase in phrases if phrase.get("field")],
        "quote_polarities": sorted({phrase["polarity"] for phrase in phrases if phrase.get("polarity")}),
    }
    optional_keys = {
        "source_url": row.get("source_url"),
        "source_review_id": source_review_id,
        "source_title": row.get("summary"),
        "rating": row.get("rating"),
        "rating_max": row.get("rating_max"),
        "reviewed_at": row.get("reviewed_at"),
        "reviewer_title": row.get("reviewer_title"),
        "contact_title": row.get("reviewer_title"),
        "reviewer_company": row.get("reviewer_company"),
        "company_name": row.get("reviewer_company"),
        "reviewer_industry": row.get("reviewer_industry"),
        "urgency_score": enrichment.get("urgency_score"),
        "pain_points": pain_points,
    }
    for key, value in optional_keys.items():
        if value not in (None, "", [], {}):
            out[key] = _clean_text(value) if isinstance(value, str) else value
    return out


def build_review_source_query(
    *,
    source: str,
    vendor_name: str | None,
    min_review_text_chars: int,
    require_review_url: bool,
) -> tuple[str, list[Any]]:
    """Build the read-only Atlas review query and parameter list."""

    where = [
        "LOWER(r.source) = LOWER($1)",
        "r.duplicate_of_review_id IS NULL",
        "r.enrichment_status = 'enriched'",
        "r.enrichment IS NOT NULL",
        "NULLIF(BTRIM(r.review_text), '') IS NOT NULL",
        "length(r.review_text) >= $2",
    ]
    args: list[Any] = [source, min_review_text_chars]
    if vendor_name:
        args.append(vendor_name)
        where.append(f"LOWER(r.vendor_name) = LOWER(${len(args)})")
    if require_review_url:
        where.append("NULLIF(BTRIM(r.source_url), '') IS NOT NULL")
    args.extend([0, 0])
    limit_placeholder = f"${len(args) - 1}"
    offset_placeholder = f"${len(args)}"
    query = f"""
        SELECT r.id, r.source, r.source_url, r.source_review_id,
               r.vendor_name, r.product_name, r.product_category,
               r.rating, r.rating_max, r.summary, r.review_text,
               r.reviewer_title, r.reviewer_company, r.reviewer_industry,
               r.reviewed_at, r.imported_at, r.enrichment
        FROM b2b_reviews r
        WHERE {' AND '.join(where)}
        ORDER BY CASE
                   WHEN COALESCE(r.enrichment->>'urgency_score', '') ~ '^-?[0-9]+(\\.[0-9]+)?$'
                   THEN (r.enrichment->>'urgency_score')::numeric
                   ELSE 0
                 END DESC,
                 r.reviewed_at DESC NULLS LAST,
                 r.imported_at DESC NULLS LAST,
                 r.id ASC
        LIMIT {limit_placeholder}
        OFFSET {offset_placeholder}
    """
    return query, args


async def fetch_review_source_rows(
    pool: Any,
    *,
    source: str = DEFAULT_SOURCE,
    vendor_name: str | None = None,
    limit: int = 50,
    min_review_text_chars: int = 80,
    phrase_limit: int = 5,
    allowed_polarities: Sequence[str] = DEFAULT_POLARITIES,
    allowed_fields: Sequence[str] = DEFAULT_PHRASE_FIELDS,
    require_review_url: bool = True,
) -> list[dict[str, Any]]:
    """Fetch and convert canonical review rows from an asyncpg-shaped pool."""

    query, args = build_review_source_query(
        source=source,
        vendor_name=vendor_name,
        min_review_text_chars=min_review_text_chars,
        require_review_url=require_review_url,
    )
    fetch_limit_index = len(args) - 2
    fetch_offset_index = len(args) - 1
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    page_size = max(limit * 3, limit)
    max_scanned_rows = max(limit * 20, page_size)
    scanned_rows = 0
    while len(rows) < limit and scanned_rows < max_scanned_rows:
        args[fetch_limit_index] = page_size
        args[fetch_offset_index] = scanned_rows
        raw_rows = await pool.fetch(query, *args)
        if not raw_rows:
            break
        scanned_rows += len(raw_rows)
        for raw in raw_rows:
            row = dict(raw)
            source_row = review_row_to_source_row(
                row,
                phrase_limit=phrase_limit,
                allowed_polarities=allowed_polarities,
                allowed_fields=allowed_fields,
            )
            if not source_row:
                continue
            source_id = _clean_text(source_row.get("source_id"))
            if source_id in seen_ids:
                continue
            seen_ids.add(source_id)
            rows.append(source_row)
            if len(rows) >= limit:
                return rows
        if len(raw_rows) < page_size:
            break
    return rows


def render_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(json.dumps(dict(row), sort_keys=True, default=str) for row in rows)


def _parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value or "").split(",") if part.strip())


def _load_dotenv_files() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
        load_dotenv(ROOT / ".env.local", override=True)


def _default_database_url() -> str | None:
    raw = os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL")
    if raw:
        return raw
    try:
        from atlas_brain.storage.config import db_settings
    except Exception:
        return None
    return _clean_text(getattr(db_settings, "dsn", ""))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export canonical Atlas reviews as Content Ops source-row JSONL."
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--vendor", default=None, help="Optional vendor_name filter.")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--min-review-text-chars", type=int, default=80)
    parser.add_argument("--phrase-limit", type=int, default=5)
    parser.add_argument("--polarities", default=",".join(DEFAULT_POLARITIES))
    parser.add_argument("--phrase-fields", default=",".join(DEFAULT_PHRASE_FIELDS))
    parser.add_argument("--allow-missing-source-url", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--database-url",
        default=_default_database_url(),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    return parser.parse_args(argv)


async def _create_pool(database_url: str):
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError("asyncpg is required to export review sources") from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _main(argv: list[str] | None = None) -> int:
    _load_dotenv_files()
    args = _parse_args(argv)
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.min_review_text_chars < 1:
        raise SystemExit("--min-review-text-chars must be positive")
    if args.phrase_limit < 1:
        raise SystemExit("--phrase-limit must be positive")
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")

    pool = await _create_pool(args.database_url)
    try:
        rows = await fetch_review_source_rows(
            pool,
            source=args.source,
            vendor_name=args.vendor,
            limit=args.limit,
            min_review_text_chars=args.min_review_text_chars,
            phrase_limit=args.phrase_limit,
            allowed_polarities=_parse_csv_list(args.polarities),
            allowed_fields=_parse_csv_list(args.phrase_fields),
            require_review_url=not args.allow_missing_source_url,
        )
    finally:
        close = getattr(pool, "close", None)
        if close is not None:
            maybe_awaitable = close()
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable

    payload = render_jsonl(rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
        print(f"exported {len(rows)} source row(s) to {args.output}")
    else:
        if payload:
            print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
