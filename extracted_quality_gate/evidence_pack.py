"""Evidence-claim coverage gate (PR-B5a).

Owned by ``extracted_quality_gate``. The deterministic pieces (rank
floor mapping, ID dedup / coercion, coverage-ratio rollup, gate
decision) are pure. The async DB read against ``b2b_evidence_claims``
is the only I/O step; the pack accepts a pool-shaped object the same
way ``extracted_llm_infrastructure.services.cost.drift`` does so the
package stays library-agnostic (the asyncpg ``Pool`` is the production
binding; tests can pass any object exposing ``.fetch(sql, *args)``).

Two API styles ship in this module:

  * ``audit_witness_evidence_coverage`` -- the legacy entry point
    lifted verbatim from
    ``atlas_brain.services.b2b.evidence_gate``. Returns the same
    ``dict[str, Any]`` shape so the existing caller in
    ``atlas_brain/autonomous/tasks/b2b_campaign_generation.py:1232``
    continues to work via the atlas-side re-export.
  * ``evaluate_evidence_coverage(pool, input, *, policy)`` -- the
    pack contract. Returns a ``QualityReport`` so future product
    packs can compose against it.

The deterministic pain-confidence rank map is part of the contract
(``b2b_evidence_claims.pain_confidence_rank`` is a STORED generated
column with values 0/1/2 for strong/weak/none) -- not parametric.
The coverage-ratio thresholds, valid-status filter, and precision
ARE parametric via ``QualityPolicy.thresholds``.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence
from uuid import UUID

from .types import (
    GateDecision,
    GateFinding,
    GateSeverity,
    QualityInput,
    QualityPolicy,
    QualityReport,
)

logger = logging.getLogger("extracted_quality_gate.evidence_pack")


# ---- Schema-fixed constants (NOT parametric -- these are the contract) ----

# pain_confidence_rank is a STORED generated column on
# b2b_evidence_claims: 0=strong, 1=weak, 2=none/null. Filtering by
# rank lets us treat 'strong' and 'weak' as floors without string
# comparisons. The map below is the exact contract.
_PAIN_CONFIDENCE_RANK_FLOOR: dict[str, int] = {
    "strong": 0,
    "weak": 1,
    "none": 2,
}


# ---- Parametric defaults (overridable via QualityPolicy.thresholds) ----

_DEFAULT_MIN_PAIN_CONFIDENCE = "strong"
_DEFAULT_VALID_STATUS = "valid"
_DEFAULT_COVERAGE_PRECISION = 3
# Block when coverage drops below this ratio. Default 0.0 means
# "never block" -- matches the current shadow-mode posture in
# atlas. Production callers tighten via policy.thresholds.
_DEFAULT_COVERAGE_BLOCK_THRESHOLD = 0.0
# Warn when coverage drops below this ratio (but stays above the
# block threshold). Default 1.0 means any uncovered review yields
# a warning.
_DEFAULT_COVERAGE_WARN_THRESHOLD = 1.0


def _rank_floor(min_pain_confidence: str) -> int:
    """Map a pain_confidence label to the max rank that still passes the gate.

    Unknown / empty labels fall back to the strictest floor (strong).
    """
    return _PAIN_CONFIDENCE_RANK_FLOOR.get(
        (min_pain_confidence or "").strip().lower(),
        _PAIN_CONFIDENCE_RANK_FLOOR[_DEFAULT_MIN_PAIN_CONFIDENCE],
    )


def _coerce_review_ids(source_review_ids: Iterable[UUID | str]) -> list[UUID]:
    """Coerce + dedup the caller's IDs into a clean list of UUIDs.

    Drops None, blank strings, malformed UUIDs, and duplicates.
    """
    cleaned: list[UUID] = []
    seen: set[str] = set()
    for rid in source_review_ids:
        if rid is None:
            continue
        try:
            uid = rid if isinstance(rid, UUID) else UUID(str(rid))
        except (ValueError, TypeError):
            continue
        key = str(uid)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(uid)
    return cleaned


async def audit_witness_evidence_coverage(
    pool: Any,
    *,
    vendor_name: str,
    source_review_ids: Iterable[UUID | str],
    min_pain_confidence: str = _DEFAULT_MIN_PAIN_CONFIDENCE,
    valid_status: str = _DEFAULT_VALID_STATUS,
    coverage_precision: int = _DEFAULT_COVERAGE_PRECISION,
) -> dict[str, Any]:
    """Report which source_review_ids have a backing valid claim at the floor.

    Lifted verbatim from
    ``atlas_brain.services.b2b.evidence_gate.audit_witness_evidence_coverage``
    (PR-B5a). The original three keyword arguments
    (``vendor_name``, ``source_review_ids``, ``min_pain_confidence``)
    keep their semantics and defaults so the existing caller in
    ``atlas_brain/autonomous/tasks/b2b_campaign_generation.py``
    continues to work via the atlas re-export. The two new arguments
    (``valid_status``, ``coverage_precision``) are additive and
    default to the legacy values, so they are non-breaking.

    Returns a dict:

      - ``vendor_name`` -- echoed for audit-log convenience
      - ``min_pain_confidence`` -- the floor applied
      - ``total_review_ids`` -- count of distinct, non-empty review_ids
      - ``covered_review_ids`` -- list[str] with at least one valid
        b2b_evidence_claims row whose pain_confidence_rank <= floor
      - ``uncovered_review_ids`` -- list[str] that did not meet the
        floor (would be dropped under enforcement)
      - ``covered_count`` / ``uncovered_count``
      - ``coverage_ratio`` -- (0..1, rounded to ``coverage_precision``)

    Returns ``coverage_ratio == 1.0`` with empty lists when no
    review_ids are supplied.
    """
    cleaned = _coerce_review_ids(source_review_ids)
    total = len(cleaned)
    floor = _rank_floor(min_pain_confidence)
    vendor = (vendor_name or "").strip()

    if not vendor or total == 0:
        return {
            "vendor_name": vendor,
            "min_pain_confidence": min_pain_confidence,
            "total_review_ids": total,
            "covered_review_ids": [],
            "uncovered_review_ids": [str(rid) for rid in cleaned],
            "covered_count": 0,
            "uncovered_count": total,
            "coverage_ratio": 1.0 if total == 0 else 0.0,
        }

    rows = await pool.fetch(
        """
        SELECT DISTINCT source_review_id::text AS review_id
        FROM b2b_evidence_claims
        WHERE status = $1
          AND vendor_name = $2
          AND source_review_id = ANY($3::uuid[])
          AND pain_confidence_rank <= $4
        """,
        valid_status,
        vendor,
        cleaned,
        floor,
    )

    covered = {row["review_id"] for row in rows}
    covered_list = sorted(rid for rid in covered if rid)
    uncovered_list = sorted(str(rid) for rid in cleaned if str(rid) not in covered)

    coverage_ratio = (len(covered_list) / total) if total else 0.0

    return {
        "vendor_name": vendor,
        "min_pain_confidence": min_pain_confidence,
        "total_review_ids": total,
        "covered_review_ids": covered_list,
        "uncovered_review_ids": uncovered_list,
        "covered_count": len(covered_list),
        "uncovered_count": len(uncovered_list),
        "coverage_ratio": round(coverage_ratio, coverage_precision),
    }


# ---- Pack contract ----


def _resolve_threshold(
    thresholds: Mapping[str, Any] | None,
    key: str,
    default: Any,
) -> Any:
    """Read a threshold value with type-safe fallback to default."""
    if thresholds is None:
        return default
    value = thresholds.get(key)
    if value is None:
        return default
    return value


def _build_quality_report(
    audit: dict[str, Any],
    *,
    block_threshold: float,
    warn_threshold: float,
) -> QualityReport:
    """Translate an audit dict into a typed QualityReport.

    Decision rules (pure):
      * coverage_ratio < block_threshold -> BLOCK with one
        ``low_evidence_coverage`` finding plus per-uncovered findings
      * coverage_ratio < warn_threshold -> WARN with per-uncovered findings
      * otherwise -> PASS with no findings

    Per-uncovered review_ids are surfaced as separate findings so an
    operator UI can render them individually; the summary metadata
    keeps the legacy dict shape so dashboards keep working.
    """
    findings: list[GateFinding] = []
    coverage_ratio = float(audit.get("coverage_ratio", 0.0))
    uncovered: Sequence[str] = tuple(audit.get("uncovered_review_ids") or ())

    if coverage_ratio < block_threshold:
        decision = GateDecision.BLOCK
        severity = GateSeverity.BLOCKER
        findings.append(
            GateFinding(
                code="low_evidence_coverage",
                message=(
                    f"low_evidence_coverage:{coverage_ratio} below "
                    f"block_threshold:{block_threshold}"
                ),
                severity=GateSeverity.BLOCKER,
                metadata={
                    "coverage_ratio": coverage_ratio,
                    "block_threshold": block_threshold,
                },
            )
        )
    elif coverage_ratio < warn_threshold:
        decision = GateDecision.WARN
        severity = GateSeverity.WARNING
    else:
        decision = GateDecision.PASS
        severity = None

    if severity is not None:
        for review_id in uncovered:
            findings.append(
                GateFinding(
                    code="unbacked_review",
                    message=f"unbacked_review:{review_id}",
                    severity=severity,
                    metadata={"review_id": review_id},
                )
            )

    return QualityReport(
        passed=decision != GateDecision.BLOCK,
        decision=decision,
        findings=tuple(findings),
        metadata={
            "vendor_name": audit.get("vendor_name", ""),
            "min_pain_confidence": audit.get("min_pain_confidence", ""),
            "total_review_ids": int(audit.get("total_review_ids", 0)),
            "covered_count": int(audit.get("covered_count", 0)),
            "uncovered_count": int(audit.get("uncovered_count", 0)),
            "coverage_ratio": coverage_ratio,
            "covered_review_ids": tuple(audit.get("covered_review_ids") or ()),
            "uncovered_review_ids": tuple(uncovered),
            "block_threshold": block_threshold,
            "warn_threshold": warn_threshold,
        },
    )


async def evaluate_evidence_coverage(
    pool: Any,
    input: QualityInput,
    *,
    policy: QualityPolicy | None = None,
) -> QualityReport:
    """Pack-contract entry point.

    Reads ``input.context``:
      * ``vendor_name``: str -- the vendor whose claims gate this report
      * ``source_review_ids``: Iterable[UUID | str] -- review IDs the
        artifact references as witness anchors
      * ``min_pain_confidence``: str (optional) -- per-call override of
        the policy default; falls back to ``policy.thresholds["min_pain_confidence"]``

    Reads ``policy.thresholds`` (all optional):
      * ``min_pain_confidence`` (str, default ``"strong"``)
      * ``valid_status`` (str, default ``"valid"``)
      * ``coverage_precision`` (int, default 3)
      * ``coverage_block_threshold`` (float, default 0.0)
      * ``coverage_warn_threshold`` (float, default 1.0)

    Returns a ``QualityReport`` whose ``metadata`` carries the same
    fields the legacy ``audit_witness_evidence_coverage`` dict
    produced (vendor_name, total_review_ids, covered/uncovered lists
    + counts, coverage_ratio) plus the active block/warn thresholds.
    """
    context = dict(input.context or {})
    thresholds = policy.thresholds if policy is not None else None

    policy_min_pain = _resolve_threshold(
        thresholds, "min_pain_confidence", _DEFAULT_MIN_PAIN_CONFIDENCE
    )
    min_pain_confidence = str(context.get("min_pain_confidence") or policy_min_pain)
    valid_status = str(
        _resolve_threshold(thresholds, "valid_status", _DEFAULT_VALID_STATUS)
    )
    coverage_precision = int(
        _resolve_threshold(
            thresholds, "coverage_precision", _DEFAULT_COVERAGE_PRECISION
        )
    )
    block_threshold = float(
        _resolve_threshold(
            thresholds, "coverage_block_threshold", _DEFAULT_COVERAGE_BLOCK_THRESHOLD
        )
    )
    warn_threshold = float(
        _resolve_threshold(
            thresholds, "coverage_warn_threshold", _DEFAULT_COVERAGE_WARN_THRESHOLD
        )
    )

    audit = await audit_witness_evidence_coverage(
        pool,
        vendor_name=str(context.get("vendor_name") or ""),
        source_review_ids=tuple(context.get("source_review_ids") or ()),
        min_pain_confidence=min_pain_confidence,
        valid_status=valid_status,
        coverage_precision=coverage_precision,
    )
    return _build_quality_report(
        audit,
        block_threshold=block_threshold,
        warn_threshold=warn_threshold,
    )


__all__ = [
    "audit_witness_evidence_coverage",
    "evaluate_evidence_coverage",
]
