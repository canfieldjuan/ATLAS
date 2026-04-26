"""Synthesis-side builder that turns witness rows into b2b_evidence_claims.

Phase 9 step 5: shadow-mode capture. For every witness picked into a
synthesis run, enumerate the eligible claim_types, validate each via the
pure validate_claim() API, and upsert the result into the shadow table.
Consumers still read from the legacy witness picks -- this only writes
the audit/best-evidence rows so the canary hand-audit (rollout step 7)
has data to inspect before any consumer migrates.

Three components:

  - eligible_claim_types_for(witness): selective enumeration so the
    audit table doesn't bloat with pain_category_mismatch rejections
    for every typed pain claim against every witness.
  - synthesis_artifact_id(...): deterministic UUID5 derived from the
    synthesis composite key, so replays of the same synthesis row
    produce identical artifact_ids and the upsert idempotency holds.
  - write_evidence_claims_for_synthesis(pool, ...): the orchestrator
    that loops witnesses x eligible claim types, validates, and writes.

The builder is feature-flagged via
settings.b2b_churn.evidence_claim_shadow_enabled. The synthesis caller
checks the flag before invoking; this module is a pure no-op if not
called.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Iterable
from uuid import UUID, uuid5

from .evidence_claim import (
    ClaimType,
    ClaimValidation,
    ClaimValidationStatus,
    validate_claim,
)
from .evidence_claim_repository import (
    PersistedClaim,
    upsert_claim,
)


logger = logging.getLogger("atlas.b2b.evidence_claim_builder")


# Stable namespace for synthesis-tied artifact_ids. Deriving artifact_id
# from the synthesis composite key via UUID5 makes replays idempotent
# without adding a surrogate UUID column to b2b_reasoning_synthesis.
_SYNTHESIS_NAMESPACE = UUID("c4a3f6f7-2c8b-5d1d-9e44-9f9b1d0e6a01")


def synthesis_artifact_id(
    *,
    vendor_name: str,
    as_of_date: date,
    analysis_window_days: int,
    schema_version: str,
) -> UUID:
    """Return the deterministic artifact_id for a synthesis row. Same
    composite key always maps to the same UUID, so the unique-index
    replay key on b2b_evidence_claims composes idempotently with synthesis
    upserts."""
    name = (
        f"synthesis::{vendor_name}::{as_of_date.isoformat()}::"
        f"{int(analysis_window_days)}::{schema_version}"
    )
    return uuid5(_SYNTHESIS_NAMESPACE, name)


# Witness fields used when building PersistedClaim. Kept as constants so
# the builder is robust to upstream field renames.
_WITNESS_FIELD_REVIEW_ID = "review_id"
_WITNESS_FIELD_EXCERPT_TEXT = "excerpt_text"
_WITNESS_FIELD_VENDOR_NAME = "vendor_name"


def eligible_claim_types_for(witness: dict[str, Any]) -> list[ClaimType]:
    """Decide which claim_types the validator should be run against for a
    given witness. The selector is structural eligibility, not semantic
    validation: we attempt a claim_type only when the witness's tags
    could plausibly support it. The validator still rejects bad
    candidates -- this just stops emitting predictable rejection rows
    that bloat the audit without surfacing a real signal.

    Eligibility rules:

      - pain_claim_about_vendor: always (universal frame).
      - counterevidence_about_vendor: only when phrase_polarity ==
        'positive'. A counterevidence claim on a negative or unclear
        phrase is structurally impossible -- the validator would always
        reject with polarity_not_positive. Tagless witnesses (None
        polarity) still skip; the cannot_validate audit signal lives on
        the pain_claim row instead, where it is not duplicated.
      - typed pain claims: only when pain_category matches the type's
        required set (so a UX phrase doesn't get tested as a pricing
        claim and rejected with pain_category_mismatch).
      - displacement (both directions): only when a competitor field
        exists.
      - named_account_anchor: only when reviewer_company exists.
    """
    types: list[ClaimType] = [
        ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
    ]

    polarity = str(witness.get("phrase_polarity") or "").strip().lower()
    if polarity == "positive":
        types.append(ClaimType.COUNTEREVIDENCE_ABOUT_VENDOR)

    pain_category = (
        str(witness.get("pain_category") or "").strip().lower()
    )
    if pain_category == "pricing":
        types.append(ClaimType.PRICING_URGENCY_CLAIM)
    if pain_category == "features":
        types.append(ClaimType.FEATURE_GAP_CLAIM)
    if pain_category == "support":
        types.append(ClaimType.SUPPORT_FAILURE_CLAIM)
    if pain_category in {"timing", "renewal", "deadline"}:
        types.append(ClaimType.TIMING_PRESSURE_CLAIM)
    if pain_category in {"onboarding", "adoption"}:
        types.append(ClaimType.ADOPTION_OR_ONBOARDING_CLAIM)
    if pain_category in {"reliability", "uptime", "outages"}:
        types.append(ClaimType.RELIABILITY_CLAIM)
    if pain_category in {"integrations", "workflow"}:
        types.append(ClaimType.INTEGRATION_OR_WORKFLOW_CLAIM)

    competitor = str(witness.get("competitor") or "").strip()
    if competitor:
        types.append(ClaimType.DISPLACEMENT_PROOF_TO_COMPETITOR)
        types.append(ClaimType.DISPLACEMENT_PROOF_FROM_COMPETITOR)

    reviewer_company = str(witness.get("reviewer_company") or "").strip()
    if reviewer_company:
        types.append(ClaimType.NAMED_ACCOUNT_ANCHOR)

    return types


def _secondary_target_for(
    claim_type: ClaimType, witness: dict[str, Any]
) -> str | None:
    if claim_type in {
        ClaimType.DISPLACEMENT_PROOF_TO_COMPETITOR,
        ClaimType.DISPLACEMENT_PROOF_FROM_COMPETITOR,
        ClaimType.FEATURE_GAP_CLAIM,
    }:
        comp = witness.get("competitor")
        if comp:
            return str(comp).strip()
    return None


def _coerce_review_id(value: Any) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (ValueError, TypeError):
        return None


def _coerce_salience(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _build_persisted_claim(
    *,
    validation: ClaimValidation,
    witness: dict[str, Any],
    claim_type: ClaimType,
    secondary_target: str | None,
    artifact_id: UUID,
    artifact_type: str,
    vendor_name: str,
    as_of_date: date,
    analysis_window_days: int,
    claim_schema_version: str = "v1",
) -> PersistedClaim:
    """Convert a ClaimValidation into a writer-ready PersistedClaim.

    The Step-5 invariant: when validation.status == VALID, the witness
    must carry excerpt_text + review_id, because validate_claim()
    enforces source_provenance_unavailable as a final gate. So if we
    reach this function with a VALID result, the source provenance is
    guaranteed present and the writer's contract guard cannot fire."""
    review_id = _coerce_review_id(witness.get(_WITNESS_FIELD_REVIEW_ID))
    excerpt_text = witness.get(_WITNESS_FIELD_EXCERPT_TEXT)
    synthesis_id = artifact_id if artifact_type == "synthesis" else None
    intelligence_id = artifact_id if artifact_type == "intelligence" else None

    return PersistedClaim(
        artifact_type=artifact_type,  # type: ignore[arg-type]
        artifact_id=artifact_id,
        synthesis_id=synthesis_id,
        intelligence_id=intelligence_id,
        vendor_name=vendor_name,
        as_of_date=as_of_date,
        analysis_window_days=analysis_window_days,
        claim_schema_version=claim_schema_version,
        claim_type=claim_type,
        target_entity=validation.target_entity or vendor_name,
        secondary_target=secondary_target,
        status=validation.status,
        rejection_reason=validation.rejection_reason,
        witness_id=validation.source_witness_id,
        witness_hash=witness.get("witness_hash"),
        source_review_id=review_id,
        source_span_id=witness.get("source_span_id"),
        salience_score=_coerce_salience(witness.get("salience_score")),
        grounding_status=witness.get("grounding_status"),
        pain_confidence=witness.get("pain_confidence"),
        excerpt_text=str(excerpt_text) if excerpt_text else None,
        supporting_fields=tuple(validation.supporting_fields),
        claim_payload={
            "excerpt_text": excerpt_text,
            "pain_category": witness.get("pain_category"),
            "phrase_subject": witness.get("phrase_subject"),
            "phrase_polarity": witness.get("phrase_polarity"),
            "phrase_role": witness.get("phrase_role"),
            "phrase_verbatim": witness.get("phrase_verbatim"),
            "competitor": witness.get("competitor"),
            "reviewer_company": witness.get("reviewer_company"),
        },
    )


async def write_evidence_claims_for_synthesis(
    pool,
    *,
    vendor_name: str,
    as_of_date: date,
    analysis_window_days: int,
    schema_version: str,
    witnesses: Iterable[dict[str, Any]],
    source_reviews: dict[str, dict[str, Any]] | None = None,
    known_vendor_names: frozenset[str] | None = None,
) -> dict[str, int]:
    """Validate every (witness x eligible claim_type) and upsert the
    result into b2b_evidence_claims under a deterministic
    synthesis-tied artifact_id. Returns counts for audit logging:

        {
            "valid": int,         # claims that validated
            "invalid": int,       # claims rejected by per-claim gates
            "cannot_validate": int,  # claims short-circuited
            "skipped": int,       # witnesses without minimum metadata
            "errors": int,        # writer raises (should be zero in steady state)
        }

    Idempotent: running it twice for the same synthesis row produces the
    same row set in b2b_evidence_claims (validated_at advances, nothing
    else moves). Run it AFTER the synthesis row is persisted, since the
    artifact_id is derived from the synthesis composite key.

    source_reviews is keyed by review_id (string). When a witness's
    review_id is present in this map, the corresponding review dict is
    passed to validate_claim() as source_review so the antecedent regex
    has access to the surrounding text. Pass an empty dict if you don't
    have review bodies plumbed; the validator falls through to the
    per-phrase gates without source-window checks.

    known_vendor_names is the canonical alias set used by the
    competitor-named antecedent patterns. Missing it degrades trap
    detection to self-transition templates only.
    """
    artifact_id = synthesis_artifact_id(
        vendor_name=vendor_name,
        as_of_date=as_of_date,
        analysis_window_days=analysis_window_days,
        schema_version=schema_version,
    )
    counts = {
        "valid": 0,
        "invalid": 0,
        "cannot_validate": 0,
        "skipped": 0,
        "errors": 0,
        "purged_stale": 0,
    }
    source_reviews = source_reviews or {}

    # Per-artifact rewrite. The replay-key upsert preserves rows the
    # current run still emits, but it cannot remove rows for attempts
    # the eligibility selector no longer makes (e.g. tightening
    # counterevidence to polarity=positive only). Without this purge,
    # tightening eligibility leaves stale invalid rows that taint the
    # audit. Delete first, then re-emit; idempotent and free under the
    # partial uniqueness index.
    purged = await pool.execute(
        "DELETE FROM b2b_evidence_claims WHERE artifact_type = $1 AND artifact_id = $2",
        "synthesis",
        artifact_id,
    )
    # asyncpg returns 'DELETE N' as the status string; parse the count.
    try:
        counts["purged_stale"] = int(str(purged).split()[-1])
    except (ValueError, IndexError):
        counts["purged_stale"] = 0

    for witness in witnesses:
        target_entity = (
            str(witness.get(_WITNESS_FIELD_VENDOR_NAME) or vendor_name).strip()
        )
        if not target_entity or not witness.get("witness_id"):
            counts["skipped"] += 1
            continue

        review_id = witness.get(_WITNESS_FIELD_REVIEW_ID)
        review_id_key = str(review_id) if review_id else ""
        source_review = source_reviews.get(review_id_key) if review_id_key else None

        for claim_type in eligible_claim_types_for(witness):
            secondary_target = _secondary_target_for(claim_type, witness)
            try:
                validation = validate_claim(
                    claim_type=claim_type,
                    witness=witness,
                    target_entity=target_entity,
                    secondary_target=secondary_target,
                    source_review=source_review,
                    known_vendor_names=known_vendor_names,
                )
            except Exception:
                counts["errors"] += 1
                logger.warning(
                    "validate_claim raised for %s/%s; skipping",
                    target_entity, claim_type,
                    exc_info=True,
                )
                continue

            persisted = _build_persisted_claim(
                validation=validation,
                witness=witness,
                claim_type=claim_type,
                secondary_target=secondary_target,
                artifact_id=artifact_id,
                artifact_type="synthesis",
                vendor_name=vendor_name,
                as_of_date=as_of_date,
                analysis_window_days=analysis_window_days,
            )
            try:
                await upsert_claim(pool, persisted)
            except Exception:
                counts["errors"] += 1
                logger.warning(
                    "upsert_claim raised for %s/%s/%s; skipping",
                    vendor_name, target_entity, claim_type,
                    exc_info=True,
                )
                continue

            if validation.status == ClaimValidationStatus.VALID:
                counts["valid"] += 1
            elif validation.status == ClaimValidationStatus.INVALID:
                counts["invalid"] += 1
            else:
                counts["cannot_validate"] += 1

    return counts


async def load_known_vendor_names(pool) -> frozenset[str]:
    """Load canonical vendor names + aliases from b2b_vendors. Used by
    the competitor-named antecedent patterns. Empty frozenset on
    missing table or empty registry -- the validator degrades to
    self-transition patterns only."""
    try:
        rows = await pool.fetch(
            "SELECT canonical_name, aliases FROM b2b_vendors"
        )
    except Exception:
        logger.warning("b2b_vendors not available; known_vendor_names empty", exc_info=True)
        return frozenset()
    names: set[str] = set()
    for row in rows:
        canonical = (row["canonical_name"] or "").strip()
        if canonical:
            names.add(canonical)
        aliases = row["aliases"] or []
        if isinstance(aliases, str):
            try:
                import json as _json

                aliases = _json.loads(aliases)
            except (TypeError, ValueError):
                aliases = []
        if isinstance(aliases, (list, tuple)):
            for alias in aliases:
                if alias and str(alias).strip():
                    names.add(str(alias).strip())
    return frozenset(names)


__all__ = [
    "synthesis_artifact_id",
    "eligible_claim_types_for",
    "write_evidence_claims_for_synthesis",
    "load_known_vendor_names",
]
