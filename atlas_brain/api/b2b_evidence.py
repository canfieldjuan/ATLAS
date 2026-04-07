"""
Evidence Explorer API.

Exposes witness records, evidence vaults, and claim-to-review traces
for the Evidence Explorer product -- the trust layer embedded across
watchlists, reports, and opportunity queues.

Endpoints:
    GET  /witnesses          -- list witnesses for a vendor with filters
    GET  /witnesses/{id}     -- single witness with full review context
    GET  /vault              -- evidence vault (weakness/strength claims)
    GET  /trace              -- full claim-to-review reasoning trace
"""

import json
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..auth.dependencies import AuthUser, require_b2b_plan
from ..services.vendor_registry import resolve_vendor_name
from ..storage.database import get_db_pool

# -- Pagination and query defaults --------------------------------------------

DEFAULT_WITNESS_LIMIT = 50
MAX_WITNESS_LIMIT = 200
DEFAULT_ANALYSIS_WINDOW_DAYS = 90
MIN_ANALYSIS_WINDOW_DAYS = 7
MAX_ANALYSIS_WINDOW_DAYS = 365
TRACE_WITNESS_SAMPLE_SIZE = 20

router = APIRouter(
    prefix="/b2b/evidence",
    tags=["b2b-evidence"],
)


# -- Helpers ------------------------------------------------------------------

def _pool_or_503():
    pool = get_db_pool()
    if pool is None or not getattr(pool, "is_initialized", False):
        raise HTTPException(status_code=503, detail="Database not ready")
    return pool


def _parse_target_date(as_of_date: Optional[str]) -> date:
    if not as_of_date:
        return date.today()
    try:
        return date.fromisoformat(as_of_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid as_of_date; expected YYYY-MM-DD") from exc


def _safe_json(value) -> list | dict | None:
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def _row_to_dict(row) -> dict:
    """Convert an asyncpg Record to a JSON-safe dict."""
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
        elif isinstance(v, date):
            d[k] = v.isoformat()
        elif hasattr(v, "hex"):
            d[k] = str(v)
    return d


# -- 1. List witnesses --------------------------------------------------------

@router.get("/witnesses")
async def list_witnesses(
    vendor_name: str,
    pain_category: Optional[str] = None,
    source: Optional[str] = None,
    competitor: Optional[str] = None,
    witness_type: Optional[str] = None,
    min_salience: Optional[float] = None,
    limit: int = Query(default=DEFAULT_WITNESS_LIMIT, le=MAX_WITNESS_LIMIT),
    offset: int = Query(default=0, ge=0),
    user: AuthUser = Depends(require_b2b_plan("b2b_trial")),
):
    """List witness records for a vendor with optional filters."""
    pool = _pool_or_503()


    resolved = await resolve_vendor_name(vendor_name)
    if resolved:
        vendor_name = resolved

    conditions = ["vendor_name = $1"]
    params: list = [vendor_name]
    idx = 2

    if pain_category:
        conditions.append(f"pain_category = ${idx}")
        params.append(pain_category)
        idx += 1

    if source:
        conditions.append(f"source = ${idx}")
        params.append(source)
        idx += 1

    if competitor:
        conditions.append(f"competitor = ${idx}")
        params.append(competitor)
        idx += 1

    if witness_type:
        conditions.append(f"witness_type = ${idx}")
        params.append(witness_type)
        idx += 1

    if min_salience is not None:
        conditions.append(f"salience_score >= ${idx}")
        params.append(min_salience)
        idx += 1

    where = " AND ".join(conditions)

    # Count total
    count_row = await pool.fetchrow(
        f"SELECT COUNT(*) AS total FROM b2b_vendor_witnesses WHERE {where}",
        *params,
    )
    total = count_row["total"] if count_row else 0

    # Fetch page
    params.append(limit)
    params.append(offset)
    rows = await pool.fetch(
        f"""
        SELECT witness_id, review_id, witness_type, excerpt_text, source,
               reviewed_at, reviewer_company, reviewer_title,
               pain_category, competitor, salience_score, specificity_score,
               selection_reason, signal_tags, as_of_date
        FROM b2b_vendor_witnesses
        WHERE {where}
        ORDER BY salience_score DESC NULLS LAST, reviewed_at DESC NULLS LAST
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *params,
    )

    # Fetch filter facets (for the UI filter sidebar)
    facets_rows = await pool.fetch(
        """
        SELECT
            pain_category,
            source,
            witness_type,
            COUNT(*) AS cnt
        FROM b2b_vendor_witnesses
        WHERE vendor_name = $1
        GROUP BY pain_category, source, witness_type
        ORDER BY cnt DESC
        """,
        vendor_name,
    )

    pain_cats = sorted({r["pain_category"] for r in facets_rows if r["pain_category"]})
    sources = sorted({r["source"] for r in facets_rows if r["source"]})
    types = sorted({r["witness_type"] for r in facets_rows if r["witness_type"]})

    return {
        "vendor_name": vendor_name,
        "witnesses": [_row_to_dict(r) for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
        "facets": {
            "pain_categories": pain_cats,
            "sources": sources,
            "witness_types": types,
        },
    }


# -- 2. Single witness with review context ------------------------------------

@router.get("/witnesses/{witness_id}")
async def get_witness(
    witness_id: str,
    vendor_name: str,
    user: AuthUser = Depends(require_b2b_plan("b2b_trial")),
):
    """Get a single witness record with full review text and evidence spans."""
    pool = _pool_or_503()


    resolved = await resolve_vendor_name(vendor_name)
    if resolved:
        vendor_name = resolved

    witness = await pool.fetchrow(
        """
        SELECT w.*, r.review_text, r.summary, r.pros, r.cons, r.rating,
               r.source AS review_source, r.source_url, r.enrichment,
               r.reviewer_name, r.enrichment_status
        FROM b2b_vendor_witnesses w
        LEFT JOIN b2b_reviews r ON r.id = w.review_id
        WHERE w.vendor_name = $1 AND w.witness_id = $2
        ORDER BY w.as_of_date DESC
        LIMIT 1
        """,
        vendor_name, witness_id,
    )

    if not witness:
        raise HTTPException(status_code=404, detail="Witness not found")

    result = _row_to_dict(witness)

    # Extract evidence spans from enrichment that relate to this witness
    enrichment = _safe_json(witness["enrichment"])
    if enrichment and isinstance(enrichment, dict):
        spans = enrichment.get("evidence_spans", [])
        # Filter spans relevant to this witness's pain/competitor/signal
        relevant_spans = []
        for span in (spans if isinstance(spans, list) else []):
            if not isinstance(span, dict):
                continue
            # Match by excerpt overlap or pain category
            span_text = span.get("excerpt_text", span.get("raw_text", "")).lower()
            witness_text = (witness["excerpt_text"] or "").lower()
            if (
                span_text and witness_text and (
                    span_text in witness_text
                    or witness_text in span_text
                    or span.get("pain_category") == witness["pain_category"]
                )
            ):
                relevant_spans.append(span)
        result["evidence_spans"] = relevant_spans
        result["all_evidence_span_count"] = len(spans) if isinstance(spans, list) else 0

    return {"witness": result}


# -- 3. Evidence vault ---------------------------------------------------------

@router.get("/vault")
async def get_vault(
    vendor_name: str,
    as_of_date: Optional[str] = None,
    window_days: int = Query(default=DEFAULT_ANALYSIS_WINDOW_DAYS, ge=MIN_ANALYSIS_WINDOW_DAYS, le=MAX_ANALYSIS_WINDOW_DAYS),
    user: AuthUser = Depends(require_b2b_plan("b2b_trial")),
):
    """Get the evidence vault for a vendor -- weakness/strength claims with provenance."""
    pool = _pool_or_503()


    resolved = await resolve_vendor_name(vendor_name)
    if resolved:
        vendor_name = resolved

    target_date = _parse_target_date(as_of_date)

    vault_row = await pool.fetchrow(
        """
        SELECT vendor_name, as_of_date, analysis_window_days, schema_version,
               vault, created_at
        FROM b2b_evidence_vault
        WHERE vendor_name = $1
          AND analysis_window_days = $2
          AND as_of_date <= $3
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        vendor_name, window_days, target_date,
    )

    if not vault_row:
        return {
            "vendor_name": vendor_name,
            "vault": None,
            "message": "No evidence vault found for this vendor",
        }

    vault = _safe_json(vault_row["vault"]) or {}

    # Count witnesses backing this vault
    witness_count = await pool.fetchrow(
        """
        SELECT COUNT(*) AS total
        FROM b2b_vendor_witnesses
        WHERE vendor_name = $1
          AND as_of_date = $2
          AND analysis_window_days = $3
        """,
        vendor_name,
        vault_row["as_of_date"],
        vault_row["analysis_window_days"],
    )

    return {
        "vendor_name": vendor_name,
        "as_of_date": vault_row["as_of_date"].isoformat(),
        "analysis_window_days": vault_row["analysis_window_days"],
        "schema_version": vault_row["schema_version"],
        "created_at": vault_row["created_at"].isoformat() if vault_row["created_at"] else None,
        "weakness_evidence": vault.get("weakness_evidence", []),
        "strength_evidence": vault.get("strength_evidence", []),
        "company_signals": vault.get("company_signals", []),
        "metric_snapshot": vault.get("metric_snapshot", {}),
        "provenance": vault.get("provenance", {}),
        "witness_count": witness_count["total"] if witness_count else 0,
    }


# -- 4. Claim-to-review trace -------------------------------------------------

@router.get("/trace")
async def get_trace(
    vendor_name: str,
    as_of_date: Optional[str] = None,
    window_days: int = Query(default=DEFAULT_ANALYSIS_WINDOW_DAYS, ge=MIN_ANALYSIS_WINDOW_DAYS, le=MAX_ANALYSIS_WINDOW_DAYS),
    user: AuthUser = Depends(require_b2b_plan("b2b_trial")),
):
    """Get the full claim-to-review reasoning trace for a vendor.

    Returns the chain: synthesis -> evidence_vault -> witnesses -> reviews,
    showing how each claim in the synthesis is backed by evidence.
    """
    pool = _pool_or_503()


    resolved = await resolve_vendor_name(vendor_name)
    if resolved:
        vendor_name = resolved

    target_date = _parse_target_date(as_of_date)

    # Layer 1: Reasoning synthesis
    synthesis_row = await pool.fetchrow(
        """
        SELECT vendor_name, as_of_date, analysis_window_days, schema_version,
               evidence_hash, synthesis, tokens_used, llm_model, created_at
        FROM b2b_reasoning_synthesis
        WHERE vendor_name = $1
          AND analysis_window_days = $2
          AND as_of_date <= $3
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        vendor_name, window_days, target_date,
    )

    synthesis = None
    evidence_hash = None
    if synthesis_row:
        synthesis = {
            "as_of_date": synthesis_row["as_of_date"].isoformat(),
            "schema_version": synthesis_row["schema_version"],
            "evidence_hash": synthesis_row["evidence_hash"],
            "sections": _safe_json(synthesis_row["synthesis"]) or {},
            "llm_model": synthesis_row["llm_model"],
            "tokens_used": synthesis_row["tokens_used"],
        }
        evidence_hash = synthesis_row["evidence_hash"]

    # Layer 2: Reasoning packet (witness assembly)
    packet_row = await pool.fetchrow(
        """
        SELECT vendor_name, as_of_date, analysis_window_days, schema_version,
               evidence_hash, packet, created_at
        FROM b2b_vendor_reasoning_packets
        WHERE vendor_name = $1
          AND analysis_window_days = $2
          AND as_of_date <= $3
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        vendor_name, window_days, target_date,
    )

    packet = None
    if packet_row:
        packet_data = _safe_json(packet_row["packet"]) or {}
        packet = {
            "as_of_date": packet_row["as_of_date"].isoformat(),
            "evidence_hash": packet_row["evidence_hash"],
            "section_count": len(packet_data.get("section_packets", [])),
            "witness_pack_size": len(packet_data.get("witness_pack", [])),
        }

    # Layer 3: Witnesses (sample, not all)
    witness_rows = await pool.fetch(
        f"""
        SELECT witness_id, review_id, witness_type, excerpt_text, source,
               reviewer_company, reviewer_title, pain_category, competitor,
               salience_score, specificity_score, signal_tags, reviewed_at
        FROM b2b_vendor_witnesses
        WHERE vendor_name = $1
          AND analysis_window_days = $2
          AND as_of_date <= $3
        ORDER BY salience_score DESC NULLS LAST
        LIMIT {TRACE_WITNESS_SAMPLE_SIZE}
        """,
        vendor_name, window_days, target_date,
    )

    witnesses = [_row_to_dict(r) for r in witness_rows]

    # Layer 4: Source reviews (unique review_ids from witnesses)
    review_ids = list({r["review_id"] for r in witness_rows if r["review_id"]})
    reviews = []
    if review_ids:
        review_rows = await pool.fetch(
            """
            SELECT id, source, source_url, vendor_name, rating, summary,
                   LEFT(review_text, 500) AS review_excerpt,
                   reviewer_name, reviewer_title, reviewer_company,
                   reviewed_at, enrichment_status
            FROM b2b_reviews
            WHERE id = ANY($1::uuid[])
            ORDER BY reviewed_at DESC
            """,
            review_ids,
        )
        reviews = [_row_to_dict(r) for r in review_rows]

    # Layer 5: Evidence diff (most recent)
    diff_row = await pool.fetchrow(
        """
        SELECT computed_date, confirmed_count, contradicted_count,
               novel_count, missing_count, diff_ratio, decision,
               has_core_contradiction
        FROM reasoning_evidence_diffs
        WHERE vendor_name = $1
          AND computed_date <= $2
        ORDER BY computed_date DESC
        LIMIT 1
        """,
        vendor_name,
        target_date,
    )

    evidence_diff = _row_to_dict(diff_row) if diff_row else None

    return {
        "vendor_name": vendor_name,
        "trace": {
            "synthesis": synthesis,
            "reasoning_packet": packet,
            "witnesses": witnesses,
            "source_reviews": reviews,
            "evidence_diff": evidence_diff,
        },
        "stats": {
            "witness_count": len(witnesses),
            "unique_reviews": len(reviews),
            "has_synthesis": synthesis is not None,
            "has_packet": packet is not None,
            "has_diff": evidence_diff is not None,
        },
    }
