#!/usr/bin/env python3
"""Sample a stratified corpus of B2B reviews for pain-classification benchmarking.

Produces a fixture skeleton at tests/fixtures/pain_classification_baseline.json.
Each entry enumerates the review's extracted phrases and leaves human-label
fields as null for manual annotation. Read-only against the database; no
pipeline code is modified.

Stratification buckets (5 total; default 15 reviews per bucket -> ~75 total):
  - structured_reviews  : g2, capterra, trustradius, gartner, peerspot, getapp
  - community_posts     : reddit, hackernews, quora, twitter
  - aggregators         : producthunt, trustpilot
  - technical           : github, stackoverflow
  - mixed_sentiment     : any source where enrichment has BOTH
                          positive_aspects non-empty AND
                          specific_complaints non-empty

After sampling, hand-label every phrase (subject/polarity/role/
grounded_in_source_text) and every review (expected_primary_pain, etc.).
Then run scripts/benchmark_pain_classifier.py to score current pipeline
behavior against the labeled ground truth.

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate
  python scripts/sample_pain_benchmark_corpus.py --per-bucket 15
  # outputs to tests/fixtures/pain_classification_baseline.json by default

  python scripts/sample_pain_benchmark_corpus.py --dry-run
  # prints summary without writing

  python scripts/sample_pain_benchmark_corpus.py --seed 42
  # reproducible sample

Safety: refuses to overwrite an existing fixture unless --force is passed,
since hand-labels are expensive to redo.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / ".env.local", override=True)

from atlas_brain.autonomous.tasks.b2b_enrichment import _MIN_REVIEW_TEXT_LENGTH
from atlas_brain.services.scraping.sources import ReviewSource
from atlas_brain.storage.database import close_database, get_db_pool, init_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sample_pain_benchmark_corpus")

# Source lists per bucket. Align with the benchmark strata in the plan doc.
_STRUCTURED_SOURCES: tuple[str, ...] = (
    ReviewSource.G2.value,
    ReviewSource.CAPTERRA.value,
    ReviewSource.TRUSTRADIUS.value,
    ReviewSource.GARTNER.value,
    ReviewSource.PEERSPOT.value,
    ReviewSource.GETAPP.value,
)
_COMMUNITY_SOURCES: tuple[str, ...] = (
    ReviewSource.REDDIT.value,
    ReviewSource.HACKERNEWS.value,
    ReviewSource.QUORA.value,
    ReviewSource.TWITTER.value,
)
_AGGREGATOR_SOURCES: tuple[str, ...] = (
    ReviewSource.PRODUCTHUNT.value,
    ReviewSource.TRUSTPILOT.value,
)
_TECHNICAL_SOURCES: tuple[str, ...] = (
    ReviewSource.GITHUB.value,
    ReviewSource.STACKOVERFLOW.value,
)

# Which enrichment array fields we enumerate for phrase-level labeling.
_PHRASE_SOURCE_FIELDS: tuple[str, ...] = (
    "specific_complaints",
    "pricing_phrases",
    "feature_gaps",
    "quotable_phrases",
    "recommendation_language",
    "positive_aspects",
)

_DEFAULT_OUTPUT = Path("tests/fixtures/pain_classification_baseline.json")


def _coerce_phrase_text(value: Any) -> str | None:
    """Accept both v1 string entries and any future dict entries with text field."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        text = str(value.get("text") or "").strip()
        return text or None
    return None


def _safe_json_column(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


async def _sample_by_sources(
    pool,
    *,
    sources: tuple[str, ...],
    per_bucket: int,
    seed: str,
    exclude_ids: set[str],
) -> list[dict[str, Any]]:
    """Pull `per_bucket` enriched reviews balanced across `sources`.

    Uses ROW_NUMBER() OVER (PARTITION BY source ORDER BY md5(...)) to take up
    to ceil(per_bucket/n_sources) rows from each source before capping at
    per_bucket. This prevents the md5 ordering from favoring whichever source
    has more rows, so every listed source contributes to the bucket when it
    has enough eligible reviews.

    Reproducible: same seed -> same rows. Cross-bucket dedup via exclude_ids.
    """
    n_sources = max(1, len(sources))
    per_source = (per_bucket + n_sources - 1) // n_sources  # ceil(per_bucket / n_sources)
    excluded = list(exclude_ids)
    conn = await pool.acquire()
    try:
        rows = await conn.fetch(
            """
            WITH ranked AS (
                SELECT r.id, r.vendor_name, r.source, r.rating, r.rating_max,
                       r.summary, r.review_text, r.enrichment,
                       ROW_NUMBER() OVER (
                           PARTITION BY r.source
                           ORDER BY md5(r.id::text || $5::text)
                       ) AS rn
                FROM b2b_reviews r
                WHERE r.source = ANY($1::text[])
                  AND r.enrichment_status = 'enriched'
                  AND r.enrichment IS NOT NULL
                  AND r.review_text IS NOT NULL
                  AND length(r.review_text) >= $4
                  AND NOT (r.id = ANY($3::uuid[]))
            )
            SELECT id, vendor_name, source, rating, rating_max,
                   summary, review_text, enrichment
            FROM ranked
            WHERE rn <= $6
            ORDER BY md5(id::text || $5::text)
            LIMIT $2
            """,
            list(sources),
            per_bucket,
            excluded,
            _MIN_REVIEW_TEXT_LENGTH,
            seed,
            per_source,
        )
    finally:
        await pool.release(conn)
    return [dict(r) for r in rows]


async def _sample_mixed_sentiment(
    pool,
    *,
    per_bucket: int,
    seed: str,
    exclude_ids: set[str],
) -> list[dict[str, Any]]:
    """Reviews where enrichment has BOTH positive_aspects and specific_complaints
    non-empty. Cross-cutting slice across all sources. Same md5-based
    deterministic ordering as _sample_by_sources."""
    excluded = list(exclude_ids)
    conn = await pool.acquire()
    try:
        rows = await conn.fetch(
            """
            SELECT r.id, r.vendor_name, r.source, r.rating, r.rating_max,
                   r.summary, r.review_text, r.enrichment
            FROM b2b_reviews r
            WHERE r.enrichment_status = 'enriched'
              AND r.enrichment IS NOT NULL
              AND r.review_text IS NOT NULL
              AND length(r.review_text) >= $3
              AND jsonb_typeof(r.enrichment->'positive_aspects') = 'array'
              AND jsonb_array_length(r.enrichment->'positive_aspects') > 0
              AND jsonb_typeof(r.enrichment->'specific_complaints') = 'array'
              AND jsonb_array_length(r.enrichment->'specific_complaints') > 0
              AND NOT (r.id = ANY($2::uuid[]))
            ORDER BY md5(r.id::text || $4::text)
            LIMIT $1
            """,
            per_bucket,
            excluded,
            _MIN_REVIEW_TEXT_LENGTH,
            seed,
        )
    finally:
        await pool.release(conn)
    return [dict(r) for r in rows]


def _build_fixture_entry(row: dict[str, Any], bucket: str) -> dict[str, Any]:
    """Produce one fixture entry with phrase enumeration + label placeholders."""
    enrichment = _safe_json_column(row.get("enrichment")) or {}
    review_text = str(row.get("review_text") or "")
    summary = str(row.get("summary") or "") or None

    extracted_phrases: list[dict[str, Any]] = []
    for field in _PHRASE_SOURCE_FIELDS:
        raw_list = enrichment.get(field)
        if not isinstance(raw_list, list):
            continue
        for idx, value in enumerate(raw_list):
            text = _coerce_phrase_text(value)
            if not text:
                continue
            extracted_phrases.append({
                "source_field": field,
                "index": idx,
                "text": text,
                "human_labels": {
                    "subject": None,
                    "polarity": None,
                    "role": None,
                    "grounded_in_source_text": None,
                    "notes": "",
                },
            })

    stored_enrichment_snapshot = {
        "enrichment_schema_version": enrichment.get("enrichment_schema_version"),
        "pain_category": enrichment.get("pain_category"),
        "pain_categories": enrichment.get("pain_categories"),
        "would_recommend": enrichment.get("would_recommend"),
        "urgency_score": enrichment.get("urgency_score"),
        "churn_signals": enrichment.get("churn_signals"),
        "contract_context": enrichment.get("contract_context"),
        "reviewer_context": enrichment.get("reviewer_context"),
        "sentiment_trajectory": enrichment.get("sentiment_trajectory"),
    }

    rating_value = row.get("rating")
    rating_max_value = row.get("rating_max")

    return {
        "review_id": str(row["id"]),
        "bucket": bucket,
        "source": row.get("source"),
        "vendor_name": row.get("vendor_name"),
        "rating": float(rating_value) if rating_value is not None else None,
        "rating_max": float(rating_max_value) if rating_max_value is not None else None,
        "summary": summary,
        "review_text": review_text,
        "stored_enrichment": stored_enrichment_snapshot,
        "extracted_phrases": extracted_phrases,
        "review_level_labels": {
            "expected_primary_pain": None,
            "expected_secondary_pains": [],
            "pricing_is_driver": None,
            "has_subject_attribution_issue": None,
            "has_polarity_trap": None,
            "expected_verbatim_witness_count": None,
            "notes": "",
        },
    }


async def _sample_all_buckets(
    pool,
    *,
    per_bucket: int,
    seed: float,
) -> list[dict[str, Any]]:
    """Sample every bucket, dedup across buckets, return fixture entries."""
    entries: list[dict[str, Any]] = []
    picked_ids: set[str] = set()

    bucket_specs: list[tuple[str, tuple[str, ...] | None]] = [
        ("structured_reviews", _STRUCTURED_SOURCES),
        ("community_posts", _COMMUNITY_SOURCES),
        ("aggregators", _AGGREGATOR_SOURCES),
        ("technical", _TECHNICAL_SOURCES),
        ("mixed_sentiment", None),  # special-case via query predicate
    ]

    for bucket, sources in bucket_specs:
        if sources is None:
            rows = await _sample_mixed_sentiment(
                pool, per_bucket=per_bucket, seed=seed, exclude_ids=picked_ids
            )
        else:
            rows = await _sample_by_sources(
                pool,
                sources=sources,
                per_bucket=per_bucket,
                seed=seed,
                exclude_ids=picked_ids,
            )
        logger.info("bucket %s: fetched %d rows", bucket, len(rows))
        if len(rows) < per_bucket:
            logger.warning(
                "bucket %s returned %d rows, below target %d",
                bucket, len(rows), per_bucket,
            )
        for row in rows:
            entry = _build_fixture_entry(row, bucket)
            entries.append(entry)
            picked_ids.add(entry["review_id"])

    return entries


def _summarize_corpus(entries: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    phrase_totals: dict[str, int] = {}
    total_phrases = 0
    for e in entries:
        counts[e["bucket"]] = counts.get(e["bucket"], 0) + 1
        src = str(e.get("source") or "")
        source_counts[src] = source_counts.get(src, 0) + 1
        for p in e.get("extracted_phrases") or []:
            field = p.get("source_field") or ""
            phrase_totals[field] = phrase_totals.get(field, 0) + 1
            total_phrases += 1
    return {
        "total_reviews": len(entries),
        "by_bucket": counts,
        "by_source": source_counts,
        "total_phrases": total_phrases,
        "phrases_by_field": phrase_totals,
    }


def _write_fixture(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_description": (
            "Pain classification baseline corpus. Each entry has review-level "
            "context and enumerated extracted_phrases. Human annotator fills in "
            "extracted_phrases[].human_labels and review_level_labels. See "
            "docs/PHRASE_SEMANTIC_TAGGING_PLAN.md Phase 0 for label definitions."
        ),
        "entries": entries,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-bucket",
        type=int,
        default=15,
        help="Target reviews per bucket (default 15; plan minimum is 12).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Path to write the fixture (default {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="atlas-pain-baseline-2026",
        help="Deterministic sampling seed (mixed into md5 ordering). Any string.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample and print summary without writing the fixture.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the existing fixture (refuses by default to protect labels).",
    )
    return parser.parse_args()


async def _main_async(args: argparse.Namespace) -> int:
    output_path: Path = args.output
    if output_path.exists() and not args.dry_run and not args.force:
        logger.error(
            "Refusing to overwrite existing fixture at %s. "
            "Re-run with --force to overwrite (will discard any hand-labels).",
            output_path,
        )
        return 2

    await init_database()
    pool = get_db_pool()
    try:
        entries = await _sample_all_buckets(
            pool, per_bucket=args.per_bucket, seed=args.seed
        )
    finally:
        await close_database()

    summary = _summarize_corpus(entries)
    logger.info("sampling summary: %s", json.dumps(summary, indent=2))

    if args.dry_run:
        logger.info("--dry-run: not writing fixture")
        return 0

    _write_fixture(output_path, entries)
    logger.info("wrote %d entries to %s", len(entries), output_path)
    return 0


def main() -> int:
    args = _parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
