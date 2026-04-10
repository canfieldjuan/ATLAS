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
import uuid as _uuid
from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..auth.dependencies import AuthUser, require_b2b_plan
from ..config import settings
from ..services.vendor_registry import resolve_vendor_name
from ..storage.database import get_db_pool

# -- Pagination and query defaults --------------------------------------------

DEFAULT_WITNESS_LIMIT = 50
MAX_WITNESS_LIMIT = 100
def _default_analysis_window_days() -> int:
    return int(getattr(settings.b2b_churn, 'evidence_default_analysis_window_days', 30))
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
        elif isinstance(v, _uuid.UUID):
            d[k] = str(v)
    return d


async def _latest_witness_snapshot_date(pool, vendor_name: str, window_days: int, target_date: date) -> date | None:
    row = await pool.fetchrow(
        """
        SELECT MAX(as_of_date) AS as_of_date
        FROM b2b_vendor_witnesses
        WHERE vendor_name = $1
          AND analysis_window_days = $2
          AND as_of_date <= $3
        """,
        vendor_name,
        window_days,
        target_date,
    )
    return row["as_of_date"] if row and row["as_of_date"] else None


def _annotation_join_parts(account_id: _uuid.UUID | None, param_idx: int) -> tuple[str, str, str, str, list]:
    if account_id is None:
        return "", "", "", "", []
    return (
        f"LEFT JOIN b2b_evidence_annotations ea ON ea.witness_id = w.witness_id AND ea.account_id = ${param_idx}",
        ", ea.annotation_type",
        " AND (ea.annotation_type IS NULL OR ea.annotation_type <> 'suppress')",
        "CASE WHEN ea.annotation_type = 'pin' THEN 0 ELSE 1 END,",
        [account_id],
    )


# -- 1. List witnesses --------------------------------------------------------

@router.get("/witnesses")
async def list_witnesses(
    vendor_name: str,
    as_of_date: Optional[str] = None,
    window_days: int = Query(default=_default_analysis_window_days(), ge=MIN_ANALYSIS_WINDOW_DAYS, le=MAX_ANALYSIS_WINDOW_DAYS),
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
    target_date = _parse_target_date(as_of_date)

    resolved = await resolve_vendor_name(vendor_name)
    if resolved:
        vendor_name = resolved

    snapshot_date = await _latest_witness_snapshot_date(pool, vendor_name, window_days, target_date)
    if snapshot_date is None:
        return {
            "vendor_name": vendor_name,
            "as_of_date": None,
            "analysis_window_days": window_days,
            "witnesses": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "facets": {
                "pain_categories": [],
                "sources": [],
                "witness_types": [],
            },
        }

    conditions = ["w.vendor_name = $1", "w.analysis_window_days = $2", "w.as_of_date = $3"]
    params: list = [vendor_name, window_days, snapshot_date]
    idx = 4

    if pain_category:
        conditions.append(f"w.pain_category = ${idx}")
        params.append(pain_category)
        idx += 1

    if source:
        conditions.append(f"w.source = ${idx}")
        params.append(source)
        idx += 1

    if competitor:
        conditions.append(f"w.competitor = ${idx}")
        params.append(competitor)
        idx += 1

    if witness_type:
        conditions.append(f"w.witness_type = ${idx}")
        params.append(witness_type)
        idx += 1

    if min_salience is not None:
        conditions.append(f"w.salience_score >= ${idx}")
        params.append(min_salience)
        idx += 1

    where = " AND ".join(conditions)

    acct_uuid: _uuid.UUID | None = None
    try:
        acct_uuid = _uuid.UUID(str(user.account_id))
    except (ValueError, TypeError):
        acct_uuid = None

    annotation_join, annotation_select, annotation_exclude, annotation_order, annotation_params = (
        _annotation_join_parts(acct_uuid, idx)
    )
    if annotation_params:
        params.extend(annotation_params)
        idx += len(annotation_params)

    # Count total (excluding suppressed)
    count_row = await pool.fetchrow(
        f"SELECT COUNT(*) AS total FROM b2b_vendor_witnesses w {annotation_join} WHERE {where}{annotation_exclude}",
        *params,
    )
    total = count_row["total"] if count_row else 0

    # Fetch page
    params.append(limit)
    params.append(offset)
    rows = await pool.fetch(
        f"""
        SELECT w.witness_id, w.review_id, w.witness_type, w.excerpt_text, w.source,
               w.reviewed_at, w.reviewer_company, w.reviewer_title,
               w.pain_category, w.competitor, w.salience_score, w.specificity_score,
               w.selection_reason, w.signal_tags, w.as_of_date
               {annotation_select}
        FROM b2b_vendor_witnesses w
        {annotation_join}
        WHERE {where}{annotation_exclude}
        ORDER BY {annotation_order} w.salience_score DESC NULLS LAST, w.reviewed_at DESC NULLS LAST
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *params,
    )

    # Fetch filter facets (for the UI filter sidebar)
    facet_params: list = [vendor_name, window_days, snapshot_date]
    facet_join = ""
    facet_exclude = ""
    if acct_uuid is not None:
        facet_join = "LEFT JOIN b2b_evidence_annotations ea ON ea.witness_id = w.witness_id AND ea.account_id = $4"
        facet_exclude = " AND (ea.annotation_type IS NULL OR ea.annotation_type <> 'suppress')"
        facet_params.append(acct_uuid)
    facets_rows = await pool.fetch(
        f"""
        SELECT
            w.pain_category,
            w.source,
            w.witness_type,
            COUNT(*) AS cnt
        FROM b2b_vendor_witnesses w
        {facet_join}
        WHERE w.vendor_name = $1
          AND w.analysis_window_days = $2
          AND w.as_of_date = $3
          {facet_exclude}
        GROUP BY w.pain_category, w.source, w.witness_type
        ORDER BY cnt DESC
        """,
        *facet_params,
    )

    pain_cats = sorted({r["pain_category"] for r in facets_rows if r["pain_category"]})
    sources = sorted({r["source"] for r in facets_rows if r["source"]})
    types = sorted({r["witness_type"] for r in facets_rows if r["witness_type"]})

    return {
        "vendor_name": vendor_name,
        "as_of_date": snapshot_date.isoformat(),
        "analysis_window_days": window_days,
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
    as_of_date: Optional[str] = None,
    window_days: int = Query(default=_default_analysis_window_days(), ge=MIN_ANALYSIS_WINDOW_DAYS, le=MAX_ANALYSIS_WINDOW_DAYS),
    user: AuthUser = Depends(require_b2b_plan("b2b_trial")),
):
    """Get a single witness record with full review text and evidence spans."""
    pool = _pool_or_503()
    target_date = _parse_target_date(as_of_date)

    resolved = await resolve_vendor_name(vendor_name)
    if resolved:
        vendor_name = resolved

    snapshot_date = await _latest_witness_snapshot_date(pool, vendor_name, window_days, target_date)
    if snapshot_date is None:
        raise HTTPException(status_code=404, detail="Witness not found")

    witness = await pool.fetchrow(
        """
        SELECT w.*, r.review_text, r.summary, r.pros, r.cons, r.rating,
               r.source AS review_source, r.source_url, r.enrichment,
               r.reviewer_name, r.enrichment_status
        FROM b2b_vendor_witnesses w
        LEFT JOIN b2b_reviews r ON r.id = w.review_id
        WHERE w.vendor_name = $1 AND w.witness_id = $2
          AND w.analysis_window_days = $3
          AND w.as_of_date = $4
        LIMIT 1
        """,
        vendor_name, witness_id, window_days, snapshot_date,
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
    window_days: int = Query(default=_default_analysis_window_days(), ge=MIN_ANALYSIS_WINDOW_DAYS, le=MAX_ANALYSIS_WINDOW_DAYS),
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
    window_days: int = Query(default=_default_analysis_window_days(), ge=MIN_ANALYSIS_WINDOW_DAYS, le=MAX_ANALYSIS_WINDOW_DAYS),
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


# ---------------------------------------------------------------------------
# Evidence Annotations (pin / flag / suppress)
# ---------------------------------------------------------------------------

_VALID_ANNOTATION_TYPES = {"pin", "flag", "suppress"}


class SetAnnotationRequest(BaseModel):
    witness_id: str = Field(..., min_length=1, max_length=600)
    vendor_name: str = Field(..., min_length=1, max_length=200)
    annotation_type: str = Field(..., min_length=1, max_length=20)
    note_text: str | None = Field(default=None, max_length=2000)


class RemoveAnnotationRequest(BaseModel):
    witness_ids: list[str] = Field(..., min_length=1, max_length=100)


def _annotation_payload(row) -> dict:
    return {
        "id": str(row["id"]),
        "witness_id": row["witness_id"],
        "vendor_name": row["vendor_name"],
        "annotation_type": row["annotation_type"],
        "note_text": row["note_text"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
    }


@router.get("/annotations")
async def list_annotations(
    vendor_name: str | None = Query(None),
    annotation_type: str | None = Query(None),
    user: AuthUser = Depends(require_b2b_plan("b2b_trial")),
):
    """List evidence annotations for this account."""
    pool = _pool_or_503()
    acct = _uuid.UUID(user.account_id)

    conditions = ["account_id = $1"]
    params: list = [acct]
    idx = 2

    if vendor_name:
        resolved = await resolve_vendor_name(vendor_name)
        vendor_name = resolved or vendor_name
        conditions.append(f"vendor_name = ${idx}")
        params.append(vendor_name)
        idx += 1
    if annotation_type:
        if annotation_type not in _VALID_ANNOTATION_TYPES:
            raise HTTPException(
                status_code=422,
                detail=f"annotation_type must be one of: {', '.join(sorted(_VALID_ANNOTATION_TYPES))}",
            )
        conditions.append(f"annotation_type = ${idx}")
        params.append(annotation_type)
        idx += 1

    where = " AND ".join(conditions)
    rows = await pool.fetch(
        f"""
        SELECT id, witness_id, vendor_name, annotation_type, note_text,
               created_at, updated_at
        FROM b2b_evidence_annotations
        WHERE {where}
        ORDER BY updated_at DESC
        """,
        *params,
    )
    annotations = [_annotation_payload(r) for r in rows]
    return {"annotations": annotations, "count": len(annotations)}


@router.post("/annotations", status_code=200)
async def set_annotation(
    req: SetAnnotationRequest,
    user: AuthUser = Depends(require_b2b_plan("b2b_trial")),
):
    """Upsert an evidence annotation (pin, flag, or suppress a witness)."""
    pool = _pool_or_503()

    if req.annotation_type not in _VALID_ANNOTATION_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"annotation_type must be one of: {', '.join(sorted(_VALID_ANNOTATION_TYPES))}",
        )

    acct = _uuid.UUID(user.account_id)
    now = datetime.now(timezone.utc)
    resolved_vendor = await resolve_vendor_name(req.vendor_name.strip())
    vendor_name = resolved_vendor or req.vendor_name.strip()

    witness_exists = await pool.fetchval(
        """
        SELECT 1
        FROM b2b_vendor_witnesses
        WHERE witness_id = $1
          AND vendor_name = $2
        LIMIT 1
        """,
        req.witness_id.strip(),
        vendor_name,
    )
    if not witness_exists:
        raise HTTPException(status_code=404, detail="Witness not found for vendor")

    row = await pool.fetchrow(
        """
        INSERT INTO b2b_evidence_annotations
            (id, account_id, witness_id, vendor_name, annotation_type, note_text,
             created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $7)
        ON CONFLICT (account_id, witness_id) DO UPDATE SET
            annotation_type = EXCLUDED.annotation_type,
            note_text = EXCLUDED.note_text,
            updated_at = NOW()
        RETURNING id, witness_id, vendor_name, annotation_type, note_text,
                  created_at, updated_at
        """,
        _uuid.uuid4(),
        acct,
        req.witness_id.strip(),
        vendor_name,
        req.annotation_type,
        req.note_text.strip() if req.note_text else None,
        now,
    )
    return _annotation_payload(row)


@router.post("/annotations/remove", status_code=200)
async def remove_annotations(
    req: RemoveAnnotationRequest,
    user: AuthUser = Depends(require_b2b_plan("b2b_trial")),
):
    """Remove annotations (restore witnesses to default state)."""
    pool = _pool_or_503()
    acct = _uuid.UUID(user.account_id)

    ids = [w.strip() for w in req.witness_ids if w.strip()]
    if not ids:
        return {"removed": 0}

    result = await pool.execute(
        """
        DELETE FROM b2b_evidence_annotations
        WHERE account_id = $1
          AND witness_id = ANY($2::text[])
        """,
        acct,
        ids,
    )
    removed = int(result.split()[-1]) if result else 0
    return {"removed": removed}
