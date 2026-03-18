"""Selection logic for cross-vendor reasoning targets.

Decides which vendor pairs and categories deserve LLM budget based on
signal strength, data availability, and resource divergence.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("atlas.reasoning.cross_vendor_selection")


def _canonical_vendor(name: str) -> str:
    return str(name or "").strip().lower().replace(" ", "_")


def _lookup_vendor_evidence(
    evidence_lookup: dict[str, dict[str, Any]],
    vendor_name: str,
) -> dict[str, Any] | None:
    wanted = _canonical_vendor(vendor_name)
    if not wanted:
        return None
    for key, value in evidence_lookup.items():
        if _canonical_vendor(key) == wanted:
            return value
    return None


def _lookup_vendor_record(
    records: dict[str, dict[str, Any]] | None,
    vendor_name: str,
) -> dict[str, Any] | None:
    if not records:
        return None
    wanted = _canonical_vendor(vendor_name)
    if not wanted:
        return None
    for key, value in records.items():
        if _canonical_vendor(key) == wanted:
            return value
    return None


def _text_tokens(value: Any) -> set[str]:
    items: list[str] = []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = [str(v) for v in value if v]
    elif isinstance(value, dict):
        items = [str(k) for k, v in value.items() if k and v]
    tokens = {
        str(item).strip().lower()
        for item in items
        if str(item).strip()
    }
    return tokens


def _buyer_role_tokens(evidence: dict[str, Any]) -> set[str]:
    buyer_auth = evidence.get("buyer_authority") or {}
    if not isinstance(buyer_auth, dict):
        return set()
    return _text_tokens(buyer_auth.get("role_types") or {})


def _use_case_tokens(
    evidence: dict[str, Any],
    profile: dict[str, Any] | None,
) -> set[str]:
    evidence_tokens = _text_tokens(evidence.get("top_use_cases") or [])
    profile_tokens = _text_tokens((profile or {}).get("primary_use_cases") or [])
    return evidence_tokens | profile_tokens


def _competitor_tokens(evidence: dict[str, Any]) -> set[str]:
    competitors = evidence.get("competitors") or []
    names = []
    for entry in competitors:
        if isinstance(entry, dict):
            names.append(entry.get("name") or "")
        elif entry:
            names.append(str(entry))
    return _text_tokens(names)


def _segment_tokens(profile: dict[str, Any] | None) -> set[str]:
    sizes = _text_tokens((profile or {}).get("typical_company_size") or [])
    segments: set[str] = set()
    for size in sizes:
        if any(token in size for token in ("enterprise", "1000+", "5000+", "large")):
            segments.add("enterprise")
        if any(token in size for token in ("smb", "startup", "1-10", "1-50", "1-100", "small")):
            segments.add("smb")
        if any(token in size for token in ("mid", "51-200", "201-500", "201-1000")):
            segments.add("midmarket")
    return segments


def _context_overlap_score(
    vendor_a: str,
    vendor_b: str,
    evidence_a: dict[str, Any],
    evidence_b: dict[str, Any],
    *,
    profile_a: dict[str, Any] | None = None,
    profile_b: dict[str, Any] | None = None,
    displacement_linked: bool = False,
) -> float:
    score = 1.0 if displacement_linked else 0.0
    if (
        evidence_a.get("product_category")
        and evidence_a.get("product_category") == evidence_b.get("product_category")
    ):
        score += 1.0
    if _use_case_tokens(evidence_a, profile_a) & _use_case_tokens(evidence_b, profile_b):
        score += 1.0
    if _buyer_role_tokens(evidence_a) & _buyer_role_tokens(evidence_b):
        score += 1.0
    if _segment_tokens(profile_a) & _segment_tokens(profile_b):
        score += 1.0
    competitors_a = _competitor_tokens(evidence_a)
    competitors_b = _competitor_tokens(evidence_b)
    if _canonical_vendor(vendor_b) in {_canonical_vendor(v) for v in competitors_a}:
        score += 1.0
    elif _canonical_vendor(vendor_a) in {_canonical_vendor(v) for v in competitors_b}:
        score += 1.0
    elif competitors_a & competitors_b:
        score += 1.0
    return score


def _has_contextual_evidence(evidence: dict[str, Any]) -> bool:
    return any(
        (
            evidence.get("top_use_cases"),
            (evidence.get("buyer_authority") or {}).get("role_types")
            if isinstance(evidence.get("buyer_authority"), dict)
            else None,
            evidence.get("pain_categories"),
            evidence.get("competitors"),
        )
    )


async def select_battles(
    pool: Any,
    displacement_edges: list[dict[str, Any]],
    evidence_lookup: dict[str, dict[str, Any]],
    *,
    product_profiles: dict[str, dict[str, Any]] | None = None,
    max_battles: int = 5,
    min_context_score: float = 2.0,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Select top displacement pairs for pairwise battle reasoning.

    Criteria (ranked):
    1. Displacement pair has enough deterministic overlap to be comparable
    2. signal_strength == "strong" AND velocity_7d > 0 (accelerating strong flow)
    3. signal_strength == "strong" (established flow)
    4. Deduplicate: if A->B and B->A both exist, keep the one with higher mention_count

    Returns: [(vendor_a, vendor_b, edge_dict), ...]
    """
    strength_rank = {"strong": 3, "moderate": 2, "emerging": 1}
    candidates: list[tuple[float, str, str, dict[str, Any]]] = []

    for edge in displacement_edges:
        from_v = edge.get("from_vendor", "")
        to_v = edge.get("to_vendor", "")
        if not from_v or not to_v:
            continue

        evidence_a = _lookup_vendor_evidence(evidence_lookup, from_v)
        evidence_b = _lookup_vendor_evidence(evidence_lookup, to_v)
        if not evidence_a or not evidence_b:
            continue

        profile_a = _lookup_vendor_record(product_profiles, from_v)
        profile_b = _lookup_vendor_record(product_profiles, to_v)
        context_score = _context_overlap_score(
            from_v,
            to_v,
            evidence_a,
            evidence_b,
            profile_a=profile_a,
            profile_b=profile_b,
            displacement_linked=True,
        )
        if context_score < min_context_score:
            continue

        strength = edge.get("signal_strength", "emerging")
        velocity_7d = edge.get("velocity_7d") or 0
        mention_count = edge.get("mention_count", 0)

        # Composite score: strength rank * 100 + velocity bonus + mention count
        score = strength_rank.get(strength, 0) * 100
        if velocity_7d > 0:
            score += 50  # Accelerating bonus
        score += min(mention_count, 50)
        score += context_score * 10.0

        candidates.append((score, from_v, to_v, edge))

    # Sort by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate bidirectional pairs (keep higher-scored direction)
    seen_pairs: set[tuple[str, str]] = set()
    result: list[tuple[str, str, dict[str, Any]]] = []

    for _score, from_v, to_v, edge in candidates:
        pair = tuple(sorted([from_v.lower(), to_v.lower()]))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        result.append((from_v, to_v, edge))
        if len(result) >= max_battles:
            break

    logger.info(
        "Selected %d/%d displacement edges for battle reasoning",
        len(result), len(displacement_edges),
    )
    return result


def select_categories(
    ecosystem_evidence: dict[str, dict[str, Any]],
    reasoning_lookup: dict[str, dict[str, Any]],
    evidence_lookup: dict[str, dict[str, Any]],
    *,
    min_vendors: int = 3,
    min_context_vendors: int = 2,
    min_displacement_intensity: float = 1.0,
    max_categories: int = 3,
) -> list[tuple[str, dict[str, Any]]]:
    """Select categories for category council reasoning.

    Criteria:
    1. At least min_vendors with reasoning evidence in that category
    2. At least min_context_vendors with richer contextual evidence
    3. displacement_intensity above threshold (active switching)
    4. Ranked by (reasoned_vendor_count * displacement_intensity) descending
    """
    candidates: list[tuple[float, str, dict[str, Any]]] = []

    for cat, eco in ecosystem_evidence.items():
        displacement_intensity = eco.get("displacement_intensity") or 0.0
        if displacement_intensity < min_displacement_intensity:
            continue

        category_vendors = [
            vendor
            for vendor in reasoning_lookup
            if (_lookup_vendor_evidence(evidence_lookup, vendor) or {}).get("product_category") == cat
        ]
        reasoned_vendor_count = len(category_vendors)
        if reasoned_vendor_count < min_vendors:
            continue

        context_vendor_count = sum(
            1 for vendor in category_vendors
            if _has_contextual_evidence(_lookup_vendor_evidence(evidence_lookup, vendor) or {})
        )
        if context_vendor_count < min_context_vendors:
            continue

        score = reasoned_vendor_count * displacement_intensity
        candidates.append((score, cat, eco))

    candidates.sort(key=lambda x: x[0], reverse=True)

    result = [(cat, eco) for _score, cat, eco in candidates[:max_categories]]
    logger.info(
        "Selected %d/%d categories for council reasoning",
        len(result), len(ecosystem_evidence),
    )
    return result


async def select_asymmetry_pairs(
    vendor_scores: list[dict[str, Any]],
    evidence_lookup: dict[str, dict[str, Any]],
    product_profiles: dict[str, dict[str, Any]],
    *,
    max_pairs: int = 3,
    pressure_delta_max: float = 1.5,
    review_ratio_min: float = 3.0,
    segment_divergence_bonus: float = 5.0,
    min_divergence_score: float = 2.0,
    min_context_score: float = 2.0,
) -> list[tuple[str, str]]:
    """Select vendor pairs with similar pressure but different resources.

    Criteria:
    1. pressure_score (or avg_urgency) within the configured delta
    2. Pair has enough deterministic overlap to be comparable
    3. Resource divergence: one has configured review-share gap, or one is
       enterprise-tilted while the other is SMB-tilted
    4. Both have per-vendor evidence
    """
    # Build vendor pressure list from vendor_scores
    vendor_pressure: list[tuple[str, float, int]] = []
    for vs in vendor_scores:
        vname = vs.get("vendor_name", "")
        if not vname:
            continue
        # Check if this vendor has evidence
        if not _lookup_vendor_evidence(evidence_lookup, vname):
            continue
        pressure = vs.get("avg_urgency", 0.0) or 0.0
        total_reviews = vs.get("total_reviews", 0) or 0
        vendor_pressure.append((vname, float(pressure), total_reviews))

    # Find pairs with similar pressure but divergent resources
    candidates: list[tuple[float, str, str]] = []

    for i, (va, pa, ra) in enumerate(vendor_pressure):
        for j, (vb, pb, rb) in enumerate(vendor_pressure):
            if j <= i:
                continue

            # Pressure similarity: within 15 points (on 0-10 scale)
            if abs(pa - pb) > pressure_delta_max:
                continue

            evidence_a = _lookup_vendor_evidence(evidence_lookup, va)
            evidence_b = _lookup_vendor_evidence(evidence_lookup, vb)
            if not evidence_a or not evidence_b:
                continue
            profile_a = _lookup_vendor_record(product_profiles, va)
            profile_b = _lookup_vendor_record(product_profiles, vb)
            context_score = _context_overlap_score(
                va,
                vb,
                evidence_a,
                evidence_b,
                profile_a=profile_a,
                profile_b=profile_b,
            )
            if context_score < min_context_score:
                continue

            # Resource divergence checks
            divergence_score = 0.0

            # Review count ratio (proxy for installed base)
            if ra > 0 and rb > 0:
                ratio = max(ra, rb) / max(min(ra, rb), 1)
                if ratio >= review_ratio_min:
                    divergence_score += ratio

            # Company size tilt from product profiles
            if profile_a and profile_b:
                seg_a = _segment_tokens(profile_a)
                seg_b = _segment_tokens(profile_b)
                a_ent = "enterprise" in seg_a
                a_smb = "smb" in seg_a
                b_ent = "enterprise" in seg_b
                b_smb = "smb" in seg_b
                if (a_ent and b_smb) or (a_smb and b_ent):
                    divergence_score += segment_divergence_bonus

            if divergence_score < min_divergence_score:
                continue

            candidates.append((divergence_score + context_score, va, vb))

    candidates.sort(key=lambda x: x[0], reverse=True)

    result = [(va, vb) for _score, va, vb in candidates[:max_pairs]]
    logger.info(
        "Selected %d asymmetry pairs from %d vendor scores",
        len(result), len(vendor_scores),
    )
    return result
