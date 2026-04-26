"""Claim validation contract for B2B evidence.

Pure deterministic, sync, pool-free. The validator decides whether a single
witness can support a single claim_type for a single target_entity. It is
the gate between witness selection (which proves a phrase exists with
certain tags) and report rendering (which makes a *claim* using that
phrase).

The DB-bound counterpart -- best-evidence selection and the b2b_evidence_claims
shadow table writer -- lives in evidence_claim_repository.py. This module
must not import asyncpg or open a pool.

See docs/progress/evidence_claim_contract_plan_2026-04-25.md for the design
doc the API and acceptance fixtures pin against.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ClaimType(StrEnum):
    PAIN_CLAIM_ABOUT_VENDOR = "pain_claim_about_vendor"
    COUNTEREVIDENCE_ABOUT_VENDOR = "counterevidence_about_vendor"
    DISPLACEMENT_PROOF_TO_COMPETITOR = "displacement_proof_to_competitor"
    DISPLACEMENT_PROOF_FROM_COMPETITOR = "displacement_proof_from_competitor"
    NAMED_ACCOUNT_ANCHOR = "named_account_anchor"
    PRICING_URGENCY_CLAIM = "pricing_urgency_claim"
    FEATURE_GAP_CLAIM = "feature_gap_claim"
    SUPPORT_FAILURE_CLAIM = "support_failure_claim"
    TIMING_PRESSURE_CLAIM = "timing_pressure_claim"
    ADOPTION_OR_ONBOARDING_CLAIM = "adoption_or_onboarding_claim"
    RELIABILITY_CLAIM = "reliability_claim"
    INTEGRATION_OR_WORKFLOW_CLAIM = "integration_or_workflow_claim"


class ClaimValidationStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    CANNOT_VALIDATE = "cannot_validate"


@dataclass(frozen=True)
class ClaimValidation:
    claim_type: ClaimType
    status: ClaimValidationStatus
    rejection_reason: str | None
    supporting_fields: tuple[str, ...]
    target_entity: str
    source_witness_id: str | None


# Pain-like claim types share the pain-claim gates plus a category requirement.
_PAIN_LIKE_CLAIM_TYPES: frozenset[ClaimType] = frozenset({
    ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
    ClaimType.PRICING_URGENCY_CLAIM,
    ClaimType.FEATURE_GAP_CLAIM,
    ClaimType.SUPPORT_FAILURE_CLAIM,
    ClaimType.TIMING_PRESSURE_CLAIM,
    ClaimType.ADOPTION_OR_ONBOARDING_CLAIM,
    ClaimType.RELIABILITY_CLAIM,
    ClaimType.INTEGRATION_OR_WORKFLOW_CLAIM,
})

# Typed pain-like claims must also match a pain_category. Plain
# pain_claim_about_vendor accepts any non-empty pain_category.
_REQUIRED_PAIN_CATEGORIES: dict[ClaimType, frozenset[str]] = {
    ClaimType.PRICING_URGENCY_CLAIM: frozenset({"pricing"}),
    ClaimType.FEATURE_GAP_CLAIM: frozenset({"features"}),
    ClaimType.SUPPORT_FAILURE_CLAIM: frozenset({"support"}),
    ClaimType.TIMING_PRESSURE_CLAIM: frozenset({"timing", "renewal", "deadline"}),
    ClaimType.ADOPTION_OR_ONBOARDING_CLAIM: frozenset({"onboarding", "adoption"}),
    ClaimType.RELIABILITY_CLAIM: frozenset({"reliability", "uptime", "outages"}),
    ClaimType.INTEGRATION_OR_WORKFLOW_CLAIM: frozenset({"integrations", "workflow"}),
}

# Claim types that REQUIRE a secondary_target. Missing -> cannot_validate.
_REQUIRES_SECONDARY_TARGET: frozenset[ClaimType] = frozenset({
    ClaimType.DISPLACEMENT_PROOF_TO_COMPETITOR,
    ClaimType.DISPLACEMENT_PROOF_FROM_COMPETITOR,
})

# Claim types that ACCEPT a secondary_target without requiring one. Passing
# a competitor for these is the "vs named competitor" variant; omitting it
# is the bare claim. Anything outside this set with a non-None
# secondary_target is rejected as unexpected.
_ACCEPTS_SECONDARY_TARGET: frozenset[ClaimType] = (
    _REQUIRES_SECONDARY_TARGET | frozenset({ClaimType.FEATURE_GAP_CLAIM})
)

# Self-transition antecedent patterns. These do not require a named
# competitor token; the trap is that the negative trait belongs to the
# reviewer's prior setup or in-house alternative, not to the subject
# vendor. See plan section "Antecedent Pattern Set (v1)" patterns 1-2.
_SELF_TRANSITION_TEMPLATES: tuple[str, ...] = (
    r"\b(?:before|prior to|previously|originally|initially|formerly)\s+{vendor}[^.]{{0,80}}\bwe\s+(?:used|had|were on|were using|relied on|ran)\b",
    r"\b(?:before|prior to|previously|originally|initially|formerly)\s+{vendor}[^.]{{0,80}}\bour\s+(?:team|company|org|stack)\b",
)

# Competitor-named antecedent patterns. These require the caller to pass
# a known_vendor_names set so the validator can match a real competitor
# token (not a stray capitalized word). See plan patterns 3-6.
_COMPETITOR_NAMED_TEMPLATES: tuple[str, ...] = (
    r"\bunlike\s+{vendor}[^.]{{0,80}}{competitor}\b",
    r"\b(?:switched|moved|migrated|graduated|upgraded)\s+(?:from|away from)\s+{competitor}[^.]{{0,80}}\bto\s+{vendor}\b",
    r"\bwe\s+(?:used to use|used to be on|came from|moved off)\s+{competitor}\b",
    r"\b{competitor}\s+(?:was our|were our|used to be)\b",
)


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFKC", str(text)).strip()


def _norm_lower(value: Any) -> str:
    return _normalize(value).lower()


def _vendor_pattern(vendor_name: str) -> str:
    """Return a regex fragment that matches the vendor name allowing minor
    formatting variation (whitespace, .com suffix, case)."""
    cleaned = re.escape(vendor_name.strip())
    # Make a trailing ".com" optional so "Monday" matches "Monday.com".
    cleaned = re.sub(r"\\\.com$", r"(?:\\.com)?", cleaned)
    return cleaned


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def _extract_source_window(
    witness: dict[str, Any],
    source_review: dict[str, Any] | None,
) -> str:
    """Return the sentence containing the excerpt plus the preceding
    sentence, scoped to review_text. Empty string if review_text or
    excerpt_text are missing or no match is found."""
    if not source_review:
        return ""
    review_text = _normalize(source_review.get("review_text"))
    excerpt = _normalize(witness.get("excerpt_text"))
    if not review_text or not excerpt:
        return ""
    sentences = _split_sentences(review_text)
    if not sentences:
        return review_text
    excerpt_lc = excerpt.lower()
    for idx, sentence in enumerate(sentences):
        if excerpt_lc in sentence.lower():
            if idx == 0:
                return sentence
            return f"{sentences[idx - 1]} {sentence}"
    # Excerpt didn't land on a sentence boundary; fall back to the whole
    # review so the regex still has something to scan.
    return review_text


def _check_attribution(
    *,
    target_vendor: str,
    source_window: str,
    known_vendor_names: frozenset[str] | None,
) -> tuple[ClaimValidationStatus, str] | None:
    """Decide whether the source window proves the negative trait belongs to
    a different entity (antecedent_trap, INVALID), to nobody we can be sure
    of (ambiguous_attribution, CANNOT_VALIDATE), or is consistent with the
    target vendor (None, fall through to per-phrase gates).

    Three passes:

      1. Self-transition patterns (templates 1-2). Fire without needing a
         named competitor token; the trap is "before [VENDOR] we used /
         had / our team ..." attributing the trait to the reviewer's prior
         setup. Pure regex.

      2. Competitor-named patterns (templates 3-6). Fire only when
         known_vendor_names contains a token that appears in the window
         and is not the subject vendor. Pure regex over canonical aliases.

      3. Ambiguous fallback. If known_vendor_names is provided AND the
         window mentions the target plus at least one other known vendor
         AND no transition pattern fired, attribution is unclear and the
         validator must return cannot_validate. Per the plan: "Do NOT
         return valid in ambiguous cases -- the claim contract's whole
         point is to be safer than the per-phrase gates."

    Returns None when none of the three pass triggered, meaning the window
    is unambiguously about the target vendor (or contains no other vendor
    references at all).
    """
    if not source_window or not target_vendor:
        return None
    vendor_re = _vendor_pattern(target_vendor)
    flags = re.IGNORECASE | re.DOTALL

    for template in _SELF_TRANSITION_TEMPLATES:
        if re.search(template.format(vendor=vendor_re), source_window, flags):
            return (ClaimValidationStatus.INVALID, "antecedent_trap")

    if not known_vendor_names:
        return None

    target_lc = target_vendor.strip().lower()
    competitors_in_window: list[str] = []
    for name in known_vendor_names:
        if not name:
            continue
        if name.strip().lower() == target_lc:
            continue
        if re.search(rf"\b{_vendor_pattern(name)}\b", source_window, flags):
            competitors_in_window.append(name)

    for competitor in competitors_in_window:
        comp_re = _vendor_pattern(competitor)
        for template in _COMPETITOR_NAMED_TEMPLATES:
            pattern = template.format(vendor=vendor_re, competitor=comp_re)
            if re.search(pattern, source_window, flags):
                return (ClaimValidationStatus.INVALID, "antecedent_trap")

    target_in_window = bool(re.search(rf"\b{vendor_re}\b", source_window, flags))
    if target_in_window and competitors_in_window:
        return (ClaimValidationStatus.CANNOT_VALIDATE, "ambiguous_attribution")

    return None


def _check_required_phrase_field(
    witness: dict[str, Any],
    field_name: str,
    rejection_reason: str,
) -> str | None:
    """Return rejection_reason if the field is missing or blank; None otherwise.
    Used for cannot_validate gates on v3-backed and synthesized rows."""
    value = witness.get(field_name)
    if value is None or _normalize(value) == "":
        return rejection_reason
    return None


def _has_source_provenance(witness: dict[str, Any]) -> bool:
    """A witness has source provenance when it carries enough data for the
    repository to compute source_excerpt_fingerprint at write time:
    excerpt_text AND review_id. Without this, a 'valid' result cannot be
    persisted -- the repository's writer guard would raise. The validator
    catches it earlier as cannot_validate(source_provenance_unavailable)
    so report-safe claims and audit rows stay deterministic.
    """
    return bool(_normalize(witness.get("excerpt_text"))) and bool(
        witness.get("review_id")
    )


def _build_supporting_fields(witness: dict[str, Any], names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in names if witness.get(name) not in (None, ""))


def _resolve_target_entity(target_entity: str, source_witness_id: str | None) -> str:
    return _normalize(target_entity)


def source_excerpt_fingerprint(
    *,
    source_review_id: Any,
    excerpt_text: str | None,
) -> str | None:
    """Stable fingerprint for cross-claim-type dedup. Stored top-level on
    b2b_evidence_claims so consumers don't recompute it. Lowercased,
    whitespace-normalized excerpt joined to the canonicalized review id."""
    if not source_review_id or not excerpt_text:
        return None
    excerpt = re.sub(r"\s+", " ", _normalize(excerpt_text)).lower()
    if not excerpt:
        return None
    rid = _normalize(source_review_id)
    payload = f"{rid}::{excerpt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_claim(
    *,
    claim_type: ClaimType,
    witness: dict[str, Any],
    target_entity: str,
    secondary_target: str | None = None,
    source_review: dict[str, Any] | None = None,
    known_vendor_names: frozenset[str] | None = None,
) -> ClaimValidation:
    """Return a ClaimValidation for the given (witness, claim_type) pair.

    Pure deterministic. No DB. No I/O. Persistence to b2b_evidence_claims
    is handled by the repository module; this function only decides.

    `secondary_target` is required for dual-target claim types (displacement
    pair, named-competitor feature gap). For single-target claim types,
    pass None. An unexpected non-None secondary_target on a single-target
    claim is rejected as 'unexpected_secondary_target'.

    `source_review` is optional. When provided, the antecedent-trap regex
    runs against the sentence containing the excerpt plus the preceding
    sentence. v3-backed and synthesized witnesses cannot trigger this
    path because they lack phrase tags and short-circuit upstream.
    """
    target = _resolve_target_entity(target_entity, witness.get("witness_id"))
    witness_id = witness.get("witness_id")
    witness_id_str = str(witness_id) if witness_id is not None else None

    def _result(
        status: ClaimValidationStatus,
        reason: str | None,
        supporting: tuple[str, ...] = (),
    ) -> ClaimValidation:
        return ClaimValidation(
            claim_type=claim_type,
            status=status,
            rejection_reason=reason,
            supporting_fields=supporting,
            target_entity=target,
            source_witness_id=witness_id_str,
        )

    if not target:
        return _result(ClaimValidationStatus.INVALID, "target_entity_missing")

    # Reject a stray secondary_target on claim types that don't accept one.
    if claim_type not in _ACCEPTS_SECONDARY_TARGET and secondary_target:
        return _result(ClaimValidationStatus.INVALID, "unexpected_secondary_target")

    # Required-secondary-target gate runs FIRST so a missing competitor
    # surfaces as cannot_validate rather than getting masked by a
    # phrase-field check. feature_gap_claim is in _ACCEPTS_ but not
    # _REQUIRES_, so it falls through to the pain-like validator.
    if claim_type in _REQUIRES_SECONDARY_TARGET:
        if not secondary_target or not _normalize(secondary_target):
            return _result(
                ClaimValidationStatus.CANNOT_VALIDATE,
                "secondary_target_not_provided",
            )

    # Per-claim dispatch.
    if claim_type == ClaimType.NAMED_ACCOUNT_ANCHOR:
        return _validate_named_account_anchor(witness, target, witness_id_str, claim_type, _result)

    if claim_type == ClaimType.COUNTEREVIDENCE_ABOUT_VENDOR:
        return _validate_counterevidence(witness, target, witness_id_str, claim_type, _result)

    if claim_type in (
        ClaimType.DISPLACEMENT_PROOF_TO_COMPETITOR,
        ClaimType.DISPLACEMENT_PROOF_FROM_COMPETITOR,
    ):
        return _validate_displacement(
            witness=witness,
            claim_type=claim_type,
            target=target,
            secondary_target=secondary_target,
            source_review=source_review,
            known_vendor_names=known_vendor_names,
            result=_result,
        )

    if claim_type in _PAIN_LIKE_CLAIM_TYPES:
        return _validate_pain_like(
            witness=witness,
            claim_type=claim_type,
            target=target,
            source_review=source_review,
            known_vendor_names=known_vendor_names,
            result=_result,
        )

    return _result(ClaimValidationStatus.CANNOT_VALIDATE, "unsupported_claim_type")


def _validate_pain_like(
    *,
    witness: dict[str, Any],
    claim_type: ClaimType,
    target: str,
    source_review: dict[str, Any] | None,
    known_vendor_names: frozenset[str] | None,
    result,
) -> ClaimValidation:
    # cannot_validate gates: required phrase metadata missing.
    for field_name, reason in (
        ("phrase_subject", "phrase_subject_unavailable"),
        ("phrase_polarity", "phrase_polarity_unavailable"),
        ("phrase_role", "phrase_role_unavailable"),
        ("pain_confidence", "pain_confidence_unavailable"),
    ):
        miss = _check_required_phrase_field(witness, field_name, reason)
        if miss:
            return result(ClaimValidationStatus.CANNOT_VALIDATE, miss)

    grounding = _norm_lower(witness.get("grounding_status"))
    if grounding != "grounded":
        return result(ClaimValidationStatus.INVALID, "not_grounded")

    subject = _norm_lower(witness.get("phrase_subject"))
    if subject != "subject_vendor":
        return result(ClaimValidationStatus.INVALID, "subject_not_subject_vendor")

    polarity = _norm_lower(witness.get("phrase_polarity"))
    if polarity not in {"negative", "mixed"}:
        return result(ClaimValidationStatus.INVALID, "polarity_not_negative_or_mixed")

    role = _norm_lower(witness.get("phrase_role"))
    if role == "passing_mention":
        return result(ClaimValidationStatus.INVALID, "role_passing_mention")

    pain_confidence = _norm_lower(witness.get("pain_confidence"))
    if pain_confidence == "none" or pain_confidence == "":
        return result(ClaimValidationStatus.INVALID, "pain_confidence_none")
    if pain_confidence not in {"strong", "weak"}:
        return result(ClaimValidationStatus.INVALID, "pain_confidence_invalid")

    pain_category = _norm_lower(witness.get("pain_category"))
    if not pain_category:
        return result(ClaimValidationStatus.INVALID, "pain_category_missing")

    required_categories = _REQUIRED_PAIN_CATEGORIES.get(claim_type)
    if required_categories is not None and pain_category not in required_categories:
        return result(ClaimValidationStatus.INVALID, "pain_category_mismatch")

    # Source-window attribution check (antecedent trap or ambiguous).
    window = _extract_source_window(witness, source_review)
    attribution = _check_attribution(
        target_vendor=target,
        source_window=window,
        known_vendor_names=known_vendor_names,
    )
    if attribution is not None:
        return result(attribution[0], attribution[1])

    if not _has_source_provenance(witness):
        return result(
            ClaimValidationStatus.CANNOT_VALIDATE,
            "source_provenance_unavailable",
        )

    supporting = _build_supporting_fields(
        witness,
        (
            "phrase_subject",
            "phrase_polarity",
            "phrase_role",
            "pain_confidence",
            "pain_category",
            "grounding_status",
            "salience_score",
            "phrase_verbatim",
        ),
    )
    return result(ClaimValidationStatus.VALID, None, supporting)


def _validate_counterevidence(
    witness: dict[str, Any],
    target: str,
    witness_id_str: str | None,
    claim_type: ClaimType,
    result,
) -> ClaimValidation:
    for field_name, reason in (
        ("phrase_subject", "phrase_subject_unavailable"),
        ("phrase_polarity", "phrase_polarity_unavailable"),
        ("phrase_role", "phrase_role_unavailable"),
    ):
        miss = _check_required_phrase_field(witness, field_name, reason)
        if miss:
            return result(ClaimValidationStatus.CANNOT_VALIDATE, miss)

    grounding = _norm_lower(witness.get("grounding_status"))
    if grounding != "grounded":
        return result(ClaimValidationStatus.INVALID, "not_grounded")

    subject = _norm_lower(witness.get("phrase_subject"))
    if subject != "subject_vendor":
        return result(ClaimValidationStatus.INVALID, "subject_not_subject_vendor")

    polarity = _norm_lower(witness.get("phrase_polarity"))
    if polarity != "positive":
        return result(ClaimValidationStatus.INVALID, "polarity_not_positive")

    role = _norm_lower(witness.get("phrase_role"))
    if role == "passing_mention":
        return result(ClaimValidationStatus.INVALID, "role_passing_mention")

    if not _has_source_provenance(witness):
        return result(
            ClaimValidationStatus.CANNOT_VALIDATE,
            "source_provenance_unavailable",
        )

    supporting = _build_supporting_fields(
        witness,
        ("phrase_subject", "phrase_polarity", "phrase_role", "grounding_status", "salience_score"),
    )
    return result(ClaimValidationStatus.VALID, None, supporting)


def _validate_named_account_anchor(
    witness: dict[str, Any],
    target: str,
    witness_id_str: str | None,
    claim_type: ClaimType,
    result,
) -> ClaimValidation:
    # Named-account anchor cares about role + grounding + a resolvable
    # reviewer company, but NOT about phrase_subject (the anchor value
    # is the company name, not the polarity).
    miss = _check_required_phrase_field(witness, "phrase_role", "phrase_role_unavailable")
    if miss:
        return result(ClaimValidationStatus.CANNOT_VALIDATE, miss)

    grounding = _norm_lower(witness.get("grounding_status"))
    if grounding != "grounded":
        return result(ClaimValidationStatus.INVALID, "not_grounded")

    role = _norm_lower(witness.get("phrase_role"))
    if role == "passing_mention":
        return result(ClaimValidationStatus.INVALID, "role_passing_mention")

    company = _normalize(witness.get("reviewer_company"))
    if not company:
        return result(ClaimValidationStatus.CANNOT_VALIDATE, "reviewer_company_unavailable")

    if not _has_source_provenance(witness):
        return result(
            ClaimValidationStatus.CANNOT_VALIDATE,
            "source_provenance_unavailable",
        )

    supporting = _build_supporting_fields(
        witness, ("phrase_role", "grounding_status", "reviewer_company", "salience_score")
    )
    return result(ClaimValidationStatus.VALID, None, supporting)


def _validate_displacement(
    *,
    witness: dict[str, Any],
    claim_type: ClaimType,
    target: str,
    secondary_target: str | None,
    source_review: dict[str, Any] | None,
    known_vendor_names: frozenset[str] | None,
    result,
) -> ClaimValidation:
    # Displacement requires phrase metadata when present, but the v1
    # validator treats v3 / synthesized rows as cannot_validate -- it does
    # not yet have an LLM-grade attribution path for tagless evidence.
    miss = _check_required_phrase_field(
        witness, "phrase_subject", "phrase_subject_unavailable"
    )
    if miss:
        return result(ClaimValidationStatus.CANNOT_VALIDATE, miss)
    miss = _check_required_phrase_field(
        witness, "phrase_polarity", "phrase_polarity_unavailable"
    )
    if miss:
        return result(ClaimValidationStatus.CANNOT_VALIDATE, miss)

    grounding = _norm_lower(witness.get("grounding_status"))
    if grounding != "grounded":
        return result(ClaimValidationStatus.INVALID, "not_grounded")

    subject = _norm_lower(witness.get("phrase_subject"))
    polarity = _norm_lower(witness.get("phrase_polarity"))

    if claim_type == ClaimType.DISPLACEMENT_PROOF_TO_COMPETITOR:
        # Active-evaluation / active-switch evidence about the subject
        # vendor losing share to the named competitor.
        if subject not in {"subject_vendor", "alternative"}:
            return result(ClaimValidationStatus.INVALID, "subject_not_resolvable")
        if polarity not in {"negative", "mixed"}:
            return result(ClaimValidationStatus.INVALID, "polarity_not_negative_or_mixed")
    else:
        # FROM_COMPETITOR: subject vendor wins share from the competitor.
        if subject != "subject_vendor":
            return result(ClaimValidationStatus.INVALID, "subject_not_subject_vendor")
        if polarity not in {"positive", "mixed"}:
            return result(ClaimValidationStatus.INVALID, "polarity_not_positive_or_mixed")

    window = _extract_source_window(witness, source_review)
    attribution = _check_attribution(
        target_vendor=target,
        source_window=window,
        known_vendor_names=known_vendor_names,
    )
    if attribution is not None:
        return result(attribution[0], attribution[1])

    if not _has_source_provenance(witness):
        return result(
            ClaimValidationStatus.CANNOT_VALIDATE,
            "source_provenance_unavailable",
        )

    supporting = _build_supporting_fields(
        witness,
        ("phrase_subject", "phrase_polarity", "phrase_role", "competitor", "grounding_status"),
    )
    return result(ClaimValidationStatus.VALID, None, supporting)


__all__ = [
    "ClaimType",
    "ClaimValidationStatus",
    "ClaimValidation",
    "validate_claim",
    "source_excerpt_fingerprint",
]
