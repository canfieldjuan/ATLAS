"""Evidence-claim coverage gate for campaign generation (Gap 3, shadow mode).

Provides audit_witness_evidence_coverage(): for a given vendor and a set of
source_review_ids surfaced as witness anchors during campaign prompt
assembly, returns which review_ids are backed by a valid b2b_evidence_claims
row at the configured pain_confidence tier. Used in shadow mode to log the
delta to campaign_audit_log (event_type='claim_gate_shadow') so we can
measure how many campaigns *would* be gated under enforcement before
flipping the enforce flag.

The pain_confidence_rank column on b2b_evidence_claims is a STORED
generated rank: 0=strong, 1=weak, 2=none/null. Filtering by rank lets us
treat 'strong' and 'weak' as floors without string comparisons.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable
from uuid import UUID

logger = logging.getLogger("atlas.services.b2b.evidence_gate")


_PAIN_CONFIDENCE_RANK_FLOOR: dict[str, int] = {
    "strong": 0,
    "weak": 1,
    "none": 2,
}


def _rank_floor(min_pain_confidence: str) -> int:
    """Map a pain_confidence label to the max rank that still passes the gate."""
    return _PAIN_CONFIDENCE_RANK_FLOOR.get(
        (min_pain_confidence or "").strip().lower(),
        _PAIN_CONFIDENCE_RANK_FLOOR["strong"],
    )


async def audit_witness_evidence_coverage(
    pool,
    *,
    vendor_name: str,
    source_review_ids: Iterable[UUID | str],
    min_pain_confidence: str = "strong",
) -> dict[str, Any]:
    """Report which source_review_ids have a backing valid claim at the floor.

    Returns a dict:
      - total_review_ids: count of distinct, non-empty review_ids supplied
      - covered_review_ids: list[str] of review_ids with at least one valid
        b2b_evidence_claims row whose pain_confidence_rank <= floor
      - uncovered_review_ids: list[str] of supplied review_ids that did not
        meet the floor (would be dropped under enforcement)
      - covered_count, uncovered_count, coverage_ratio (0..1, rounded to 3 dp)
      - min_pain_confidence: the floor applied
      - vendor_name: echoed for audit-log convenience

    Returns coverage_ratio == 1.0 with empty lists when no review_ids supplied.
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
        WHERE status = 'valid'
          AND vendor_name = $1
          AND source_review_id = ANY($2::uuid[])
          AND pain_confidence_rank <= $3
        """,
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
        "coverage_ratio": round(coverage_ratio, 3),
    }


__all__ = ["audit_witness_evidence_coverage"]
