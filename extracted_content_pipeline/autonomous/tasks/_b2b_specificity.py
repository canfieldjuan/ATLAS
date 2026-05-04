"""Atlas-side wrapper for the witness specificity pack.

The deterministic specificity helpers
(``surface_specificity_context``, ``merge_specificity_contexts``,
``specificity_signal_terms``, ``evaluate_specificity_support``,
``specificity_audit_snapshot``, ``campaign_proof_terms_from_audit``)
were lifted to ``extracted_quality_gate.witness_pack`` in PR-B5b
and are re-exported here so existing imports keep working.

This module retains:

  * ``campaign_policy_audit_snapshot`` -- the atlas-side adapter
    that runs the specificity audit, resolves campaign proof-terms,
    and delegates the policy validators to
    ``extracted_quality_gate.campaign_pack.evaluate_campaign``
    (PR-B4b).
  * ``latest_specificity_audit`` / ``specificity_quality_summary``
    -- atlas-side helpers that read pre-computed audit dicts out of
    ``b2b_*`` row metadata. They are not deterministic validators,
    so they do not belong in the pack.
"""

import copy
from typing import Any

from extracted_quality_gate.witness_pack import (
    campaign_proof_terms_from_audit,
    evaluate_specificity_support,
    merge_specificity_contexts,
    specificity_audit_snapshot,
    specificity_signal_terms,
    surface_specificity_context,
)


__all__ = [
    "campaign_policy_audit_snapshot",
    "campaign_proof_terms_from_audit",
    "evaluate_specificity_support",
    "latest_specificity_audit",
    "merge_specificity_contexts",
    "specificity_audit_snapshot",
    "specificity_quality_summary",
    "specificity_signal_terms",
    "surface_specificity_context",
]


def _dedupe_strings(values: list[str]) -> list[str]:
    """Stable de-dup by lowercase marker."""
    resolved: list[str] = []
    seen: set[str] = set()
    for value in values:
        marker = str(value or "").strip().lower()
        if not marker or marker in seen:
            continue
        seen.add(marker)
        resolved.append(str(value).strip())
    return resolved


def _campaign_collection(payload: dict[str, Any], key: str) -> Any:
    """Read ``payload[key]`` with metadata-fallback semantics."""
    value = payload.get(key)
    if value not in (None, "", [], {}):
        return value
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def campaign_policy_audit_snapshot(
    *,
    subject: str,
    body: str,
    cta: str,
    campaign: dict[str, Any] | None = None,
    anchor_examples: dict[str, list[dict[str, Any]]] | None = None,
    witness_highlights: list[dict[str, Any]] | None = None,
    reference_ids: dict[str, Any] | None = None,
    campaign_proof_terms: list[str] | None = None,
    min_anchor_hits: int = 1,
    require_anchor_support: bool = True,
    require_timing_or_numeric_when_available: bool = False,
    proof_term_limit: int = 3,
) -> dict[str, Any]:
    """Atlas-side adapter for campaign quality validation.

    The deterministic policy validators (proof-term coverage,
    report-tier banned language, forbidden competitor / incumbent
    names in cold email, private-account leak) live in
    ``extracted_quality_gate.campaign_pack.evaluate_campaign``. This
    wrapper:

      * runs the (now pack-resident) specificity audit
      * resolves the campaign proof-terms (caller-provided override
        / metadata fallback / regen-from-audit)
      * builds a ``QualityInput`` with the audit's blocking issues
        and warnings as pass-through context
      * calls the pack
      * merges the pack's findings with the rest of the audit dict
        so existing callers see the legacy shape.
    """
    from extracted_quality_gate.campaign_pack import evaluate_campaign
    from extracted_quality_gate.types import QualityInput, QualityPolicy

    payload = campaign if isinstance(campaign, dict) else {}
    channel = str(payload.get("channel") or "").strip()

    audit = specificity_audit_snapshot(
        body,
        anchor_examples=anchor_examples,
        witness_highlights=witness_highlights,
        reference_ids=reference_ids,
        allow_company_names=False,
        min_anchor_hits=min_anchor_hits,
        require_anchor_support=require_anchor_support,
        require_timing_or_numeric_when_available=require_timing_or_numeric_when_available,
        include_competitor_terms=channel != "email_cold",
    )

    proof_terms = _dedupe_strings(
        [str(term or "").strip() for term in (campaign_proof_terms or []) if str(term or "").strip()]
    )
    if not proof_terms:
        proof_terms = _dedupe_strings(
            [
                str(term or "").strip()
                for term in (_campaign_collection(payload, "campaign_proof_terms") or [])
                if str(term or "").strip()
            ]
        )
    if not proof_terms:
        proof_terms = campaign_proof_terms_from_audit(
            audit,
            channel=channel,
            limit=proof_term_limit,
        )

    pack_input = QualityInput(
        artifact_type="campaign_email",
        artifact_id=None,
        content=body,
        context={
            "subject": subject,
            "body": body,
            "cta": cta,
            "campaign": payload,
            "required_proof_terms": tuple(proof_terms),
            "anchor_examples": anchor_examples or {},
            "witness_highlights": tuple(witness_highlights or ()),
            "specificity_blocking_issues": tuple(audit.get("blocking_issues") or ()),
            "specificity_warnings": tuple(audit.get("warnings") or ()),
        },
    )
    pack_policy = QualityPolicy(
        name="campaign_email",
        thresholds={"require_anchor_support": require_anchor_support},
    )
    pack_report = evaluate_campaign(pack_input, policy=pack_policy)

    deduped_blockers = list(pack_report.metadata.get("blocking_issues") or ())
    deduped_warnings = list(pack_report.metadata.get("warnings") or ())
    used_proof_terms = list(pack_report.metadata.get("used_proof_terms") or ())
    return {
        **audit,
        "status": "fail" if deduped_blockers else "pass",
        "blocking_issues": deduped_blockers,
        "warnings": deduped_warnings,
        "campaign_proof_terms": proof_terms,
        "required_proof_terms": proof_terms,
        "used_proof_terms": used_proof_terms,
        "unused_proof_terms": [term for term in proof_terms if term not in used_proof_terms],
        "primary_blocker": deduped_blockers[0] if deduped_blockers else None,
    }


def latest_specificity_audit(metadata: Any) -> dict[str, Any]:
    """Pull the most recent specificity audit out of row metadata.

    Used by the consumer-side reporting flow; not a deterministic
    validator, so it stays atlas-side.
    """
    if not isinstance(metadata, dict):
        return {}
    current = metadata.get("latest_specificity_audit")
    if isinstance(current, dict):
        return copy.deepcopy(current)
    generation = metadata.get("generation_audit")
    if not isinstance(generation, dict):
        return {}
    specificity = generation.get("specificity")
    if isinstance(specificity, dict):
        return copy.deepcopy(specificity)
    if generation:
        return {
            "status": generation.get("status"),
            "blocking_issues": [],
            "warnings": [],
            "matched_groups": [],
        }
    return {}


def specificity_quality_summary(metadata: Any) -> dict[str, Any]:
    """Compact summary fields derived from the latest audit."""
    audit = latest_specificity_audit(metadata)
    blocking_issues = list(audit.get("blocking_issues") or [])
    warnings = list(audit.get("warnings") or [])
    return {
        "quality_status": audit.get("status"),
        "blocker_count": len(blocking_issues),
        "warning_count": len(warnings),
        "latest_error_summary": blocking_issues[0] if blocking_issues else None,
    }
