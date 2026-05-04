"""Source-quality pack (PR-B5c).

Owned by ``extracted_quality_gate``. Lifts the deterministic
witness-render gate from ``atlas_brain.services.b2b.witness_render_gate``
and the coverage-ratio helpers from
``atlas_brain.services.b2b.source_impact`` into the pack module.

Three API styles ship in this module:

  * ``apply_witness_render_gate(row, *, policy)`` -- legacy entry point
    lifted verbatim from
    ``atlas_brain.services.b2b.witness_render_gate``. Mutates the
    ``row`` dict in place with ``evidence_posture``, ``confidence``,
    ``render_allowed``, ``report_allowed``, ``suppression_reason``,
    and ``quote_grade`` keys, then returns the row. Existing callers
    (``atlas_brain.api.b2b_evidence``) keep working unchanged via
    the atlas re-export.
  * ``compute_coverage_ratio`` / ``row_count`` /
    ``build_non_empty_text_check`` -- pure helpers lifted from
    ``source_impact.py``. Useful for any caller that builds
    field-coverage queries against a source-typed table without
    pulling in the full Atlas baseline machinery.
  * ``evaluate_source_quality(input, *, policy)`` -- pack contract.
    Reads a sequence of witness rows from ``input.context['witnesses']``,
    applies the render gate to each, and returns a ``QualityReport``
    whose findings enumerate the rows that did not reach ``USABLE``
    posture.

Render-gate policy thresholds (``min_supporting_count``,
``min_direct_evidence``, etc.) are exposed via
``QualityPolicy.thresholds`` so a customer can tighten or loosen the
gate without modifying the pack. The defaults match the legacy atlas
policy so behaviour is preserved when no policy is passed.

Out of scope (kept atlas-side):

  * ``build_source_impact_ledger`` / ``get_consumer_wiring_baseline``
    -- read multiple ``settings.b2b_*`` flags and reference Atlas
    consumer / pool names.
  * ``summarize_source_field_baseline`` -- async SQL against
    ``b2b_reviews`` schema with JSONB paths specific to Atlas.
  * The ``_SOURCE_IMPACT_PROFILES`` registry -- ~330 LOC of profile
    data referencing Atlas-internal pool names. Could be lifted as
    data in a follow-up but is not behaviour.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .product_claim import (
    ClaimGatePolicy,
    ConfidenceLabel,
    EvidencePosture,
    SuppressionReason,
    decide_render_gates,
)
from .types import (
    GateDecision,
    GateFinding,
    GateSeverity,
    QualityInput,
    QualityPolicy,
    QualityReport,
)


# ---- Policy defaults (legacy atlas values, parametric via policy) ----

_DEFAULT_MIN_SUPPORTING_COUNT = 1
_DEFAULT_MIN_DIRECT_EVIDENCE = 1
_DEFAULT_HIGH_CONFIDENCE_MIN_SUPPORTING = 1
_DEFAULT_HIGH_CONFIDENCE_MIN_WITNESSES = 1
_DEFAULT_MEDIUM_CONFIDENCE_MIN_SUPPORTING = 1
_DEFAULT_MEDIUM_CONFIDENCE_MIN_WITNESSES = 1
_DEFAULT_COVERAGE_PRECISION = 3


def _resolve_int_threshold(
    thresholds: Mapping[str, Any] | None,
    key: str,
    default: int,
) -> int:
    """Read an int threshold with type-safe fallback to the legacy default."""
    if thresholds is None:
        return default
    value = thresholds.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return default


def _build_render_policy(thresholds: Mapping[str, Any] | None) -> ClaimGatePolicy:
    """Build a ClaimGatePolicy from caller thresholds with legacy fallbacks."""
    return ClaimGatePolicy(
        min_supporting_count=_resolve_int_threshold(
            thresholds, "min_supporting_count", _DEFAULT_MIN_SUPPORTING_COUNT
        ),
        min_direct_evidence=_resolve_int_threshold(
            thresholds, "min_direct_evidence", _DEFAULT_MIN_DIRECT_EVIDENCE
        ),
        high_confidence_min_supporting=_resolve_int_threshold(
            thresholds,
            "high_confidence_min_supporting",
            _DEFAULT_HIGH_CONFIDENCE_MIN_SUPPORTING,
        ),
        high_confidence_min_witnesses=_resolve_int_threshold(
            thresholds,
            "high_confidence_min_witnesses",
            _DEFAULT_HIGH_CONFIDENCE_MIN_WITNESSES,
        ),
        medium_confidence_min_supporting=_resolve_int_threshold(
            thresholds,
            "medium_confidence_min_supporting",
            _DEFAULT_MEDIUM_CONFIDENCE_MIN_SUPPORTING,
        ),
        medium_confidence_min_witnesses=_resolve_int_threshold(
            thresholds,
            "medium_confidence_min_witnesses",
            _DEFAULT_MEDIUM_CONFIDENCE_MIN_WITNESSES,
        ),
    )


# ---- Pure coverage helpers (lifted from source_impact.py) ----


def compute_coverage_ratio(
    numerator: int | float | None,
    denominator: int | float | None,
    *,
    precision: int = _DEFAULT_COVERAGE_PRECISION,
) -> float | None:
    """Return a stable coverage ratio with configurable decimal precision.

    Returns ``None`` when either operand is ``None`` or the
    denominator is zero, mirroring the legacy
    ``_compute_coverage_ratio`` semantics. ``precision`` defaults to
    3 (same as the atlas helper) and is overridable.
    """
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round(float(numerator) / float(denominator), precision)


def row_count(
    row: Any,
    key: str,
    *,
    fallback_key: str | None = None,
) -> int:
    """Read an integer count from a query row, optionally with a fallback alias.

    Mirrors the legacy ``_row_count`` helper. ``row`` only needs the
    ``__contains__`` and ``__getitem__`` protocols, so it accepts
    asyncpg ``Record``, plain dict, or any duck-typed substitute.
    """
    if key in row:
        return int(row[key] or 0)
    if fallback_key and fallback_key in row:
        return int(row[fallback_key] or 0)
    return 0


def build_non_empty_text_check(expression: str) -> str:
    """Build a SQL fragment that evaluates true when ``expression`` is non-empty.

    Wraps the input in ``NULLIF(TRIM(COALESCE(<expr>, ''))) IS NOT
    NULL`` -- the same form the legacy
    ``_build_non_empty_text_check`` produces. Caller is responsible
    for the SQL injection surface (the expression is interpolated
    verbatim).
    """
    return f"""
        NULLIF(
            TRIM(
                COALESCE(
                    {expression},
                    ''
                )
            ),
            ''
        ) IS NOT NULL
    """


# ---- Witness-render gate (lifted from witness_render_gate.py) ----


def _witness_confidence(row: Mapping[str, Any]) -> ConfidenceLabel:
    """Map a row's pain_confidence string to a ConfidenceLabel."""
    value = str(row.get("pain_confidence") or "").strip().lower()
    if value == "strong":
        return ConfidenceLabel.HIGH
    if value == "weak":
        return ConfidenceLabel.MEDIUM
    return ConfidenceLabel.LOW


def _witness_gate_payload(
    *,
    evidence_posture: EvidencePosture,
    confidence: ConfidenceLabel,
    policy: ClaimGatePolicy,
    suppression_reason: SuppressionReason | None = None,
) -> dict[str, Any]:
    """Build the keys ``apply_witness_render_gate`` writes onto a row."""
    if suppression_reason is None:
        render_allowed, report_allowed, derived_reason = decide_render_gates(
            evidence_posture=evidence_posture,
            confidence=confidence,
            supporting_count=1,
            direct_evidence_count=1,
            contradiction_count=0,
            denominator=None,
            sample_size=None,
            policy=policy,
        )
        suppression_reason = derived_reason
    else:
        render_allowed = False
        report_allowed = False

    return {
        "evidence_posture": evidence_posture.value,
        "confidence": confidence.value,
        "render_allowed": render_allowed,
        "report_allowed": report_allowed,
        "suppression_reason": (
            suppression_reason.value if suppression_reason else None
        ),
    }


def apply_witness_render_gate(
    row: dict[str, Any],
    *,
    policy: ClaimGatePolicy | None = None,
) -> dict[str, Any]:
    """Attach ProductClaim-style render gates to a witness row.

    Mutates ``row`` in place with ``quote_grade``,
    ``evidence_posture``, ``confidence``, ``render_allowed``,
    ``report_allowed``, and ``suppression_reason`` keys, then returns
    it. Mirrors the legacy
    ``atlas_brain.services.b2b.witness_render_gate.apply_witness_render_gate``
    behaviour. The optional ``policy`` argument is additive (default
    matches the legacy atlas constants) so existing callers passing
    only ``row`` keep working.
    """
    if policy is None:
        policy = _build_render_policy(None)
    confidence = _witness_confidence(row)
    grounding_status = str(row.get("grounding_status") or "pending").strip()
    row["quote_grade"] = grounding_status == "grounded"

    if not row["quote_grade"]:
        row.update(
            _witness_gate_payload(
                evidence_posture=EvidencePosture.UNVERIFIED,
                confidence=confidence,
                policy=policy,
            )
        )
        return row

    required_tags = ("phrase_subject", "phrase_polarity", "phrase_role")
    if any(row.get(field) is None for field in required_tags):
        row.update(
            _witness_gate_payload(
                evidence_posture=EvidencePosture.UNVERIFIED,
                confidence=confidence,
                policy=policy,
                suppression_reason=SuppressionReason.UNVERIFIED_EVIDENCE,
            )
        )
        return row

    subject = str(row.get("phrase_subject") or "").strip().lower()
    if subject != "subject_vendor":
        row.update(
            _witness_gate_payload(
                evidence_posture=EvidencePosture.INSUFFICIENT,
                confidence=confidence,
                policy=policy,
                suppression_reason=SuppressionReason.SUBJECT_NOT_SUBJECT_VENDOR,
            )
        )
        return row

    role = str(row.get("phrase_role") or "").strip().lower()
    if role == "passing_mention":
        row.update(
            _witness_gate_payload(
                evidence_posture=EvidencePosture.WEAK,
                confidence=confidence,
                policy=policy,
                suppression_reason=SuppressionReason.PASSING_MENTION_ONLY,
            )
        )
        return row

    if role not in {"primary_driver", "supporting_context"}:
        row.update(
            _witness_gate_payload(
                evidence_posture=EvidencePosture.INSUFFICIENT,
                confidence=confidence,
                policy=policy,
                suppression_reason=SuppressionReason.ROLE_NOT_RENDERABLE,
            )
        )
        return row

    polarity = str(row.get("phrase_polarity") or "").strip().lower()
    witness_type = str(row.get("witness_type") or "").strip().lower()
    positive_allowed = witness_type in {"strength", "counterevidence"}
    if polarity not in {"negative", "mixed"} and not (
        polarity == "positive" and positive_allowed
    ):
        row.update(
            _witness_gate_payload(
                evidence_posture=EvidencePosture.INSUFFICIENT,
                confidence=confidence,
                policy=policy,
                suppression_reason=SuppressionReason.POLARITY_NOT_RENDERABLE,
            )
        )
        return row

    if str(row.get("pain_confidence") or "").strip().lower() == "none":
        row.update(
            _witness_gate_payload(
                evidence_posture=EvidencePosture.WEAK,
                confidence=confidence,
                policy=policy,
                suppression_reason=SuppressionReason.LOW_CONFIDENCE,
            )
        )
        return row

    row.update(
        _witness_gate_payload(
            evidence_posture=EvidencePosture.USABLE,
            confidence=confidence,
            policy=policy,
        )
    )
    return row


# ---- Pack contract ----


def evaluate_source_quality(
    input: QualityInput,
    *,
    policy: QualityPolicy | None = None,
) -> QualityReport:
    """Pack-contract entry point.

    Reads ``input.context['witnesses']``: a sequence of dicts (one
    per witness row). Applies ``apply_witness_render_gate`` to each
    (in place) and assembles a ``QualityReport`` with one finding
    per row that did not reach ``EvidencePosture.USABLE``.

    Decision rules:
      * any witness with ``suppression_reason`` -> per-witness
        ``WARNING`` finding (caller can re-classify upstream)
      * the report ``decision`` is:
          - ``BLOCK`` when zero rows are render-allowed AND there is
            at least one input row
          - ``WARN`` when at least one row is suppressed but at
            least one is render-allowed
          - ``PASS`` when every row is render-allowed (or the input
            list is empty)

    Reads ``policy.thresholds`` for the underlying
    ``ClaimGatePolicy`` parameters
    (``min_supporting_count``, ``min_direct_evidence``,
    ``high_confidence_min_supporting``, ``high_confidence_min_witnesses``,
    ``medium_confidence_min_supporting``, ``medium_confidence_min_witnesses``).
    Defaults match the legacy atlas policy.
    """
    context = dict(input.context or {})
    raw_rows: Sequence[Any] = context.get("witnesses") or ()
    thresholds = policy.thresholds if policy is not None else None
    render_policy = _build_render_policy(thresholds)

    findings: list[GateFinding] = []
    total = 0
    rendered = 0
    suppressed = 0
    rows_out: list[dict[str, Any]] = []

    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            continue
        # Defensive copy so the pack does not surprise callers by
        # mutating their input dicts.
        row: dict[str, Any] = dict(raw)
        apply_witness_render_gate(row, policy=render_policy)
        rows_out.append(row)
        total += 1
        if bool(row.get("render_allowed")):
            rendered += 1
            continue
        suppressed += 1
        suppression_reason = str(row.get("suppression_reason") or "")
        witness_id = str(row.get("witness_id") or "").strip()
        message = (
            f"witness_suppressed:{witness_id}:{suppression_reason}"
            if witness_id
            else f"witness_suppressed::{suppression_reason}"
        )
        findings.append(
            GateFinding(
                code="witness_suppressed",
                message=message,
                severity=GateSeverity.WARNING,
                metadata={
                    "witness_id": witness_id or None,
                    "suppression_reason": suppression_reason or None,
                    "evidence_posture": row.get("evidence_posture"),
                    "confidence": row.get("confidence"),
                },
            )
        )

    if total == 0:
        decision = GateDecision.PASS
    elif rendered == 0:
        decision = GateDecision.BLOCK
        findings.insert(
            0,
            GateFinding(
                code="no_renderable_witnesses",
                message=f"no_renderable_witnesses:{total}_witnesses_all_suppressed",
                severity=GateSeverity.BLOCKER,
                metadata={"total_witnesses": total, "suppressed": suppressed},
            ),
        )
    elif suppressed > 0:
        decision = GateDecision.WARN
    else:
        decision = GateDecision.PASS

    metadata: dict[str, Any] = {
        "total_witnesses": total,
        "rendered_witnesses": rendered,
        "suppressed_witnesses": suppressed,
        "rows": tuple(rows_out),
    }
    return QualityReport(
        passed=decision != GateDecision.BLOCK,
        decision=decision,
        findings=tuple(findings),
        metadata=metadata,
    )


__all__ = [
    "apply_witness_render_gate",
    "build_non_empty_text_check",
    "compute_coverage_ratio",
    "evaluate_source_quality",
    "row_count",
]
