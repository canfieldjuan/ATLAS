"""Async repository for the b2b_evidence_claims shadow table.

Pairs with atlas_brain/services/b2b/evidence_claim.py (the pure-sync
validator). This module owns:

  - upsert_claim(): write a ClaimValidation result on the
    (artifact_type, artifact_id, witness_id, claim_type, target_entity,
    secondary_target) replay key. Idempotent.

  - select_best_claim(): read the highest-quality validated claims for a
    given (vendor, claim_type, target_entity, as_of_date,
    analysis_window_days) tuple. Ranks by salience + grounding +
    pain_confidence with a stable witness_id tie-break. Dedups by
    source_excerpt_fingerprint so a single phrase that validated for
    multiple claim types is not returned twice.

  - dedup_selections(): cross-claim-type dedup helper for consumers that
    assemble mixed lists from several select_best_claim() calls (e.g. a
    battle card that mixes pain_claim_about_vendor and
    support_failure_claim quotes).

See docs/progress/evidence_claim_contract_plan_2026-04-25.md for the
shadow table schema and migration 305_b2b_evidence_claims.sql.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Literal
from uuid import UUID

from .evidence_claim import (
    ClaimType,
    ClaimValidation,
    ClaimValidationStatus,
    source_excerpt_fingerprint,
)


ArtifactType = Literal["synthesis", "intelligence"]


@dataclass(frozen=True)
class PersistedClaim:
    """All fields needed to write one row to b2b_evidence_claims.

    Callers typically build this from a ClaimValidation result plus the
    surrounding artifact context (vendor, as_of_date, analysis window,
    witness anchor data). The repository does the json-encoding and
    fingerprint computation.

    excerpt_text is the input from which source_excerpt_fingerprint is
    derived at write time. Pass excerpt_text and source_review_id and
    upsert_claim() will compute the fingerprint via the canonical
    helper. Passing source_excerpt_fingerprint directly is supported
    only for replay scenarios where a precomputed value already exists.
    For status='valid' rows without enough inputs to compute or supply
    a fingerprint, upsert_claim() raises ValueError -- this is the
    contract guard against silently writing un-deduppable rows.
    """

    artifact_type: ArtifactType
    artifact_id: UUID
    vendor_name: str
    claim_type: ClaimType
    target_entity: str
    status: ClaimValidationStatus
    rejection_reason: str | None = None
    secondary_target: str | None = None
    synthesis_id: UUID | None = None
    intelligence_id: UUID | None = None
    as_of_date: date | None = None
    analysis_window_days: int | None = None
    claim_schema_version: str = "v1"
    witness_id: str | None = None
    witness_hash: str | None = None
    source_review_id: UUID | None = None
    source_span_id: str | None = None
    salience_score: float = 0.0
    grounding_status: str | None = None
    pain_confidence: str | None = None
    excerpt_text: str | None = None
    source_excerpt_fingerprint: str | None = None
    supporting_fields: tuple[str, ...] = field(default_factory=tuple)
    claim_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClaimSelection:
    """Read-side row shape returned by select_best_claim()."""

    id: UUID
    artifact_type: ArtifactType
    artifact_id: UUID
    vendor_name: str
    claim_type: ClaimType
    target_entity: str
    secondary_target: str | None
    witness_id: str | None
    source_review_id: UUID | None
    source_span_id: str | None
    salience_score: float
    grounding_status: str | None
    pain_confidence: str | None
    source_excerpt_fingerprint: str | None
    supporting_fields: tuple[str, ...]
    claim_payload: dict[str, Any]


_INSERT_SQL = """
INSERT INTO b2b_evidence_claims (
    artifact_type,
    artifact_id,
    synthesis_id,
    intelligence_id,
    vendor_name,
    as_of_date,
    analysis_window_days,
    claim_schema_version,
    claim_type,
    target_entity,
    secondary_target,
    witness_id,
    witness_hash,
    source_review_id,
    source_span_id,
    salience_score,
    grounding_status,
    pain_confidence,
    source_excerpt_fingerprint,
    status,
    rejection_reason,
    supporting_fields,
    claim_payload,
    validated_at,
    created_at
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
    $21, $22::jsonb, $23::jsonb, now(), now()
)
ON CONFLICT (
    artifact_type,
    artifact_id,
    COALESCE(witness_id, ''),
    claim_type,
    target_entity,
    COALESCE(secondary_target, '')
) DO UPDATE SET
    synthesis_id = EXCLUDED.synthesis_id,
    intelligence_id = EXCLUDED.intelligence_id,
    vendor_name = EXCLUDED.vendor_name,
    as_of_date = EXCLUDED.as_of_date,
    analysis_window_days = EXCLUDED.analysis_window_days,
    claim_schema_version = EXCLUDED.claim_schema_version,
    witness_hash = EXCLUDED.witness_hash,
    source_review_id = EXCLUDED.source_review_id,
    source_span_id = EXCLUDED.source_span_id,
    salience_score = EXCLUDED.salience_score,
    grounding_status = EXCLUDED.grounding_status,
    pain_confidence = EXCLUDED.pain_confidence,
    source_excerpt_fingerprint = EXCLUDED.source_excerpt_fingerprint,
    status = EXCLUDED.status,
    rejection_reason = EXCLUDED.rejection_reason,
    supporting_fields = EXCLUDED.supporting_fields,
    claim_payload = EXCLUDED.claim_payload,
    validated_at = now()
"""


_SELECT_BEST_SQL = """
WITH ranked AS (
    SELECT
        id,
        artifact_type,
        artifact_id,
        vendor_name,
        claim_type,
        target_entity,
        secondary_target,
        witness_id,
        source_review_id,
        source_span_id,
        salience_score,
        grounding_status,
        pain_confidence,
        grounding_rank,
        pain_confidence_rank,
        source_excerpt_fingerprint,
        supporting_fields,
        claim_payload,
        ROW_NUMBER() OVER (
            -- COALESCE so NULL-fingerprint rows partition by their own id
            -- (each one becomes rn=1) instead of being collapsed together.
            PARTITION BY COALESCE(source_excerpt_fingerprint, id::text)
            ORDER BY
                salience_score DESC,
                grounding_rank ASC,
                pain_confidence_rank ASC,
                witness_id ASC
        ) AS dedup_rn
    FROM b2b_evidence_claims
    WHERE status = 'valid'
      AND vendor_name = $1
      AND claim_type = $2
      AND target_entity = $3
      AND as_of_date = $4
      AND analysis_window_days = $5
      AND ($6::text IS NULL OR secondary_target = $6)
)
SELECT
    id,
    artifact_type,
    artifact_id,
    vendor_name,
    claim_type,
    target_entity,
    secondary_target,
    witness_id,
    source_review_id,
    source_span_id,
    salience_score,
    grounding_status,
    pain_confidence,
    source_excerpt_fingerprint,
    supporting_fields,
    claim_payload
FROM ranked
WHERE dedup_rn = 1
ORDER BY
    salience_score DESC,
    grounding_rank ASC,
    pain_confidence_rank ASC,
    witness_id ASC
LIMIT $7
"""


def _supporting_fields_to_jsonb(fields: tuple[str, ...] | list[str] | None) -> str:
    return json.dumps(list(fields or []), separators=(",", ":"))


def _claim_payload_to_jsonb(payload: dict[str, Any] | None) -> str:
    return json.dumps(payload or {}, separators=(",", ":"), sort_keys=True, default=str)


def _coerce_supporting_fields(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError):
            return ()
        return tuple(str(item) for item in decoded if item)
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw if item)
    return ()


def _coerce_claim_payload(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return dict(decoded) if isinstance(decoded, dict) else {}
    return {}


def _resolve_fingerprint(claim: PersistedClaim) -> str | None:
    """Compute source_excerpt_fingerprint at write time when the caller
    supplied excerpt_text + source_review_id. A precomputed
    source_excerpt_fingerprint on the claim wins for replay scenarios.

    For status='valid' rows the fingerprint is required: a missing
    fingerprint would silently break dedup in select_best_claim. The
    caller MUST pass either the fingerprint or excerpt_text +
    source_review_id; otherwise upsert_claim raises.
    """
    if claim.source_excerpt_fingerprint:
        return claim.source_excerpt_fingerprint
    if claim.excerpt_text and claim.source_review_id:
        return source_excerpt_fingerprint(
            source_review_id=claim.source_review_id,
            excerpt_text=claim.excerpt_text,
        )
    return None


async def upsert_claim(pool, claim: PersistedClaim) -> None:
    """Write or replay-update a single claim row on the
    (artifact_type, artifact_id, witness_id, claim_type, target_entity,
    secondary_target) replay key. Idempotent across reprocessing.

    The writer owns source_excerpt_fingerprint computation so production
    callers cannot accidentally write un-deduppable rows. status='valid'
    rows without enough inputs to derive a fingerprint raise ValueError.
    """
    if claim.artifact_type not in ("synthesis", "intelligence"):
        raise ValueError(f"invalid artifact_type: {claim.artifact_type!r}")

    fingerprint = _resolve_fingerprint(claim)
    if (
        str(claim.status) == ClaimValidationStatus.VALID.value
        and not fingerprint
    ):
        raise ValueError(
            "valid claims require source_excerpt_fingerprint or "
            "(excerpt_text + source_review_id) so select_best_claim "
            "dedup cannot silently break"
        )

    await pool.execute(
        _INSERT_SQL,
        claim.artifact_type,
        claim.artifact_id,
        claim.synthesis_id,
        claim.intelligence_id,
        claim.vendor_name,
        claim.as_of_date,
        claim.analysis_window_days,
        claim.claim_schema_version,
        str(claim.claim_type),
        claim.target_entity,
        claim.secondary_target,
        claim.witness_id,
        claim.witness_hash,
        claim.source_review_id,
        claim.source_span_id,
        float(claim.salience_score or 0.0),
        claim.grounding_status,
        claim.pain_confidence,
        fingerprint,
        str(claim.status),
        claim.rejection_reason,
        _supporting_fields_to_jsonb(claim.supporting_fields),
        _claim_payload_to_jsonb(claim.claim_payload),
    )


async def select_best_claim(
    pool,
    *,
    claim_type: ClaimType,
    target_entity: str,
    vendor_name: str,
    as_of_date: date,
    analysis_window_days: int,
    secondary_target: str | None = None,
    limit: int = 1,
) -> list[ClaimSelection]:
    """Return the highest-quality validated claims for the given query.

    Filters on the partial index (status='valid', vendor_name, claim_type,
    target_entity, as_of_date, analysis_window_days). Ranks by:

      1. salience_score DESC
      2. grounding_rank ASC  (0 = grounded, 1 = otherwise)
      3. pain_confidence_rank ASC  (0 = strong, 1 = weak, 2 = none/null)
      4. witness_id ASC for stable tie-break

    Dedups across multi-claim-per-witness rows by
    source_excerpt_fingerprint so a single phrase that validated for
    several claim types is not returned twice for the same call. Dedup
    is enforced in SQL via ROW_NUMBER() PARTITION BY fingerprint, so
    `limit` truly means "up to N unique claims" -- no app-side
    over-fetch heuristic that could under-fill when the leading rows
    share fingerprints. NULL-fingerprint rows partition by id and never
    collapse with each other.

    secondary_target is matched if non-NULL; passing None matches any
    secondary_target value (including NULL). Returns empty list when no
    valid claim exists; consumer decides fallback.
    """
    if limit <= 0:
        return []
    rows = await pool.fetch(
        _SELECT_BEST_SQL,
        vendor_name,
        str(claim_type),
        target_entity,
        as_of_date,
        analysis_window_days,
        secondary_target,
        limit,
    )
    return [_row_to_selection(row) for row in rows]


def _row_to_selection(row: Any) -> ClaimSelection:
    return ClaimSelection(
        id=row["id"],
        artifact_type=row["artifact_type"],
        artifact_id=row["artifact_id"],
        vendor_name=row["vendor_name"],
        claim_type=ClaimType(row["claim_type"]),
        target_entity=row["target_entity"],
        secondary_target=row["secondary_target"],
        witness_id=row["witness_id"],
        source_review_id=row["source_review_id"],
        source_span_id=row["source_span_id"],
        salience_score=float(row["salience_score"] or 0.0),
        grounding_status=row["grounding_status"],
        pain_confidence=row["pain_confidence"],
        source_excerpt_fingerprint=row["source_excerpt_fingerprint"],
        supporting_fields=_coerce_supporting_fields(row["supporting_fields"]),
        claim_payload=_coerce_claim_payload(row["claim_payload"]),
    )


def dedup_selections(selections: Iterable[ClaimSelection]) -> list[ClaimSelection]:
    """Cross-claim-type dedup helper. When a consumer assembles a mixed
    list from multiple select_best_claim() calls (e.g. battle card showing
    both pain_claim and support_failure quotes), a single phrase that
    legitimately validated for both should appear only once.

    Order is preserved; the first occurrence of each fingerprint wins.
    Selections without a fingerprint are passed through unchanged.
    """
    seen: set[str] = set()
    out: list[ClaimSelection] = []
    for sel in selections:
        fp = sel.source_excerpt_fingerprint
        if fp:
            if fp in seen:
                continue
            seen.add(fp)
        out.append(sel)
    return out


__all__ = [
    "ArtifactType",
    "PersistedClaim",
    "ClaimSelection",
    "upsert_claim",
    "select_best_claim",
    "dedup_selections",
]
