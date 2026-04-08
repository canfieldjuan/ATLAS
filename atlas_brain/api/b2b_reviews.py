"""
REST API for B2B review import.

Accepts JSON array of reviews from any source and inserts into b2b_reviews
table with dedup via SHA-256 keys.
"""

import hashlib
import json
import logging
import time
import uuid as _uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import settings
from ..autonomous.visibility import record_dedup
from ..services.b2b.reviewer_identity import sanitize_reviewer_title
from ..services.b2b.review_dedup import (
    choose_cross_source_duplicate_survivor,
    load_cross_source_review_candidates,
    make_cross_source_identity_key,
    make_review_text_payload,
    normalize_review_date_key,
    normalize_reviewer_stem_key,
)
from ..services.company_normalization import normalize_company_name
from ..services.scraping.sources import ReviewSource
from ..services.vendor_registry import resolve_vendor_name
from ..storage.database import get_db_pool
from ..autonomous.tasks.b2b_scrape_intake import (
    _load_existing_review_identity_sets,
    _make_review_content_hash,
    _make_review_identity_key,
)

_VALID_SOURCES = {s.value for s in ReviewSource}

logger = logging.getLogger("atlas.api.b2b_reviews")

router = APIRouter(prefix="/b2b/reviews", tags=["b2b-reviews"])


class B2BReviewInput(BaseModel):
    source: str = Field(description="Review source: g2, capterra, trustradius, reddit, manual")
    vendor_name: str = Field(max_length=500, description="Vendor/company name")
    review_text: str = Field(max_length=10000, description="Main review body text")

    source_url: Optional[str] = None
    source_review_id: Optional[str] = None
    product_name: Optional[str] = None
    product_category: Optional[str] = None
    rating: Optional[float] = None
    rating_max: int = 5
    summary: Optional[str] = Field(default=None, max_length=500)
    pros: Optional[str] = Field(default=None, max_length=5000)
    cons: Optional[str] = Field(default=None, max_length=5000)
    reviewer_name: Optional[str] = None
    reviewer_title: Optional[str] = None
    reviewer_company: Optional[str] = None
    company_size_raw: Optional[str] = None
    reviewer_industry: Optional[str] = None
    reviewed_at: Optional[str] = None
    metadata: Optional[dict] = Field(default_factory=dict)


_INSERT_SQL = """
INSERT INTO b2b_reviews (
    dedup_key, source, source_url, source_review_id,
    vendor_name, product_name, product_category,
    rating, rating_max, summary, review_text, pros, cons,
    reviewer_name, reviewer_title, reviewer_company, reviewer_company_norm,
    company_size_raw, reviewer_industry, reviewed_at,
    import_batch_id, raw_metadata, parser_version,
    cross_source_content_hash, cross_source_identity_key,
    duplicate_of_review_id, duplicate_reason, deduped_at,
    enrichment_status, id
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23,
    $24, $25, $26::uuid, $27, $28, $29, $30::uuid
)
ON CONFLICT (dedup_key) DO NOTHING
"""


def _make_dedup_key(source: str, vendor_name: str, source_review_id: str | None,
                    reviewer_name: str | None, reviewed_at: str | None) -> str:
    if source_review_id:
        raw = f"{source}:{vendor_name}:{source_review_id}"
    else:
        raw = f"{source}:{vendor_name}:{reviewer_name or ''}:{reviewed_at or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()


@router.post("/import")
async def import_b2b_reviews(reviews: list[B2BReviewInput]) -> dict:
    """Import B2B reviews from any source. Accepts JSON array."""
    if not reviews:
        raise HTTPException(status_code=400, detail="Empty review list")
    if len(reviews) > 500:
        raise HTTPException(status_code=400, detail="Max 500 reviews per request")

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    # Validate all sources up front
    bad_sources = {r.source for r in reviews if r.source not in _VALID_SOURCES}
    if bad_sources:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown source(s): {', '.join(sorted(bad_sources))}. "
                   f"Valid: {', '.join(sorted(_VALID_SOURCES))}",
        )

    batch_id = f"api_{int(time.time())}"
    cfg = settings.b2b_scrape
    rows = []
    dedup_audit_events: list[tuple[str, str, str, dict]] = []
    seen_hashes: set[str] = set()
    seen_identities: set[str] = set()
    seen_content_hashes_by_source: dict[str, set[str]] = {}
    existing_cache: dict[tuple[str, str], tuple[set[str], set[str]]] = {}
    batch_canonical_by_content_hash: dict[str, dict] = {}
    batch_canonical_by_identity_key: dict[str, dict] = {}
    for r in reviews:
        reviewed_at_ts = None
        if r.reviewed_at:
            try:
                reviewed_at_ts = datetime.fromisoformat(r.reviewed_at.replace("Z", "+00:00"))
            except ValueError:
                pass

        # Resolve to canonical vendor name BEFORE dedup key so both use canonical form
        canonical_vendor = await resolve_vendor_name(r.vendor_name)

        dedup_key = _make_dedup_key(
            r.source, canonical_vendor, r.source_review_id,
            r.reviewer_name, r.reviewed_at,
        )
        identity_key = _make_review_identity_key(
            r.source,
            canonical_vendor,
            r.source_review_id,
            r.reviewer_name,
            reviewed_at_ts or r.reviewed_at,
        )
        content_hash = _make_review_content_hash(r.review_text, r.pros, r.cons)
        review_payload = make_review_text_payload(r.summary, r.review_text, r.pros, r.cons)
        cross_source_identity_key = make_cross_source_identity_key(
            canonical_vendor,
            r.reviewer_name,
            reviewed_at_ts or r.reviewed_at,
            r.rating,
        )
        cache_key = (canonical_vendor, r.source)
        if cache_key not in existing_cache:
            try:
                existing_cache[cache_key] = await _load_existing_review_identity_sets(
                    pool,
                    canonical_vendor,
                    r.source,
                )
            except Exception:
                existing_cache[cache_key] = (set(), set())
        known_hashes, known_identities = existing_cache[cache_key]
        seen_content_hashes = seen_content_hashes_by_source.setdefault(r.source, set())
        if (
            dedup_key in seen_hashes
            or identity_key in seen_identities
            or (content_hash is not None and content_hash in seen_content_hashes)
            or dedup_key in known_hashes
            or identity_key in known_identities
        ):
            continue
        seen_hashes.add(dedup_key)
        seen_identities.add(identity_key)
        if content_hash is not None:
            seen_content_hashes.add(content_hash)

        review_uuid = _uuid.uuid4()
        duplicate_of_review_id: _uuid.UUID | None = None
        duplicate_reason: str | None = None
        deduped_at: datetime | None = None
        enrichment_status = "pending"
        duplicate_detail: dict | None = None

        if cfg.cross_source_dedup_enabled:
            candidate_rows: list[dict] = []
            if content_hash is not None and content_hash in batch_canonical_by_content_hash:
                candidate_rows.append(batch_canonical_by_content_hash[content_hash])
            if (
                cross_source_identity_key is not None
                and cross_source_identity_key in batch_canonical_by_identity_key
            ):
                candidate = batch_canonical_by_identity_key[cross_source_identity_key]
                if candidate not in candidate_rows:
                    candidate_rows.append(candidate)
            if content_hash is not None or cross_source_identity_key is not None:
                candidate_rows.extend(
                    await load_cross_source_review_candidates(
                        pool,
                        vendor_name=canonical_vendor,
                        content_hash=content_hash,
                        identity_key=cross_source_identity_key,
                        max_candidates=cfg.cross_source_dedup_max_candidates,
                        reviewer_stem=normalize_reviewer_stem_key(
                            r.reviewer_name,
                            stem_length=cfg.cross_source_dedup_reviewer_stem_length,
                        ),
                        reviewed_date=normalize_review_date_key(reviewed_at_ts or r.reviewed_at),
                        rating=r.rating,
                        reviewer_stem_length=cfg.cross_source_dedup_reviewer_stem_length,
                        review_date_tolerance_days=cfg.cross_source_dedup_review_date_tolerance_days,
                        rating_tolerance=cfg.cross_source_dedup_rating_tolerance,
                    )
                )
            survivor, duplicate_reason, duplicate_detail = choose_cross_source_duplicate_survivor(
                candidates=candidate_rows,
                incoming_source=r.source,
                incoming_content_hash=content_hash,
                incoming_identity_key=cross_source_identity_key,
                incoming_payload=review_payload,
                similarity_threshold=cfg.cross_source_dedup_similarity_threshold,
                incoming_reviewer_name=r.reviewer_name,
                incoming_reviewed_at=reviewed_at_ts or r.reviewed_at,
                incoming_rating=r.rating,
                loose_similarity_threshold=cfg.cross_source_dedup_loose_similarity_threshold,
                reviewer_stem_length=cfg.cross_source_dedup_reviewer_stem_length,
                review_date_tolerance_days=cfg.cross_source_dedup_review_date_tolerance_days,
                rating_tolerance=cfg.cross_source_dedup_rating_tolerance,
            )
            if survivor is not None and survivor.get("id"):
                duplicate_of_review_id = (
                    survivor["id"] if isinstance(survivor["id"], _uuid.UUID)
                    else _uuid.UUID(str(survivor["id"]))
                )
                duplicate_reason = duplicate_reason or "cross_source_duplicate"
                deduped_at = datetime.now(timezone.utc)
                enrichment_status = "duplicate"

        reviewer_company_norm = normalize_company_name(r.reviewer_company or "") or None
        metadata = dict(r.metadata or {})
        if content_hash is not None:
            metadata["review_content_hash"] = content_hash
        if duplicate_of_review_id is not None:
            metadata["duplicate_of_review_id"] = str(duplicate_of_review_id)
            metadata["duplicate_reason"] = duplicate_reason
            if duplicate_detail:
                metadata["duplicate_detail"] = duplicate_detail

        rows.append((
            dedup_key,
            r.source,
            r.source_url,
            r.source_review_id,
            canonical_vendor,
            r.product_name,
            r.product_category,
            r.rating,
            r.rating_max,
            r.summary,
            r.review_text,
            r.pros,
            r.cons,
            r.reviewer_name,
            sanitize_reviewer_title(r.reviewer_title),
            r.reviewer_company,
            reviewer_company_norm,
            r.company_size_raw,
            r.reviewer_industry,
            reviewed_at_ts,
            batch_id,
            json.dumps(metadata),
            None,  # parser_version: N/A for API imports
            content_hash,
            cross_source_identity_key,
            duplicate_of_review_id,
            duplicate_reason,
            deduped_at,
            enrichment_status,
            review_uuid,
        ))
        if duplicate_of_review_id is None:
            canonical_row = {
                "id": review_uuid,
                "source": r.source,
                "cross_source_content_hash": content_hash,
                "cross_source_identity_key": cross_source_identity_key,
                "summary": r.summary,
                "review_text": r.review_text,
                "pros": r.pros,
                "cons": r.cons,
                "enrichment_status": enrichment_status,
                "source_weight": None,
                "imported_at": datetime.now(timezone.utc).isoformat(),
            }
            if content_hash is not None:
                batch_canonical_by_content_hash[content_hash] = canonical_row
            if cross_source_identity_key is not None:
                batch_canonical_by_identity_key[cross_source_identity_key] = canonical_row
        else:
            dedup_audit_events.append((
                str(review_uuid),
                str(duplicate_of_review_id),
                duplicate_reason or "cross_source_duplicate",
                duplicate_detail or {},
            ))

    try:
        async with pool.transaction() as conn:
            await conn.executemany(_INSERT_SQL, rows)
    except Exception:
        logger.exception("Failed to import B2B reviews")
        raise HTTPException(status_code=500, detail="Database insert failed")

    # Count how many actually inserted (not duplicates)
    count_row = await pool.fetchrow(
        "SELECT count(*) as cnt FROM b2b_reviews WHERE import_batch_id = $1",
        batch_id,
    )
    imported = count_row["cnt"] if count_row else 0
    duplicates = max(len(reviews) - imported, 0) + len(dedup_audit_events)

    for entity_id, survivor_id, reason_code, detail in dedup_audit_events:
        await record_dedup(
            pool,
            stage="review_import",
            entity_type="review",
            entity_id=entity_id,
            survivor_entity_id=survivor_id,
            reason=f"Suppressed cross-source duplicate review ({reason_code})",
            detail=detail,
        )

    logger.info("B2B review import: %d imported, %d duplicates (batch %s)", imported, duplicates, batch_id)

    return {
        "imported": imported,
        "duplicates": duplicates,
        "total": len(reviews),
        "batch_id": batch_id,
    }
