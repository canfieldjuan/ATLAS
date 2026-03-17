"""Selection logic for cross-vendor reasoning targets.

Decides which vendor pairs and categories deserve LLM budget based on
signal strength, data availability, and resource divergence.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("atlas.reasoning.cross_vendor_selection")


async def select_battles(
    pool: Any,
    displacement_edges: list[dict[str, Any]],
    evidence_lookup: dict[str, dict[str, Any]],
    *,
    max_battles: int = 5,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Select top displacement pairs for pairwise battle reasoning.

    Criteria (ranked):
    1. signal_strength == "strong" AND velocity_7d > 0 (accelerating strong flow)
    2. signal_strength == "strong" (established flow)
    3. Both vendors have per-vendor reasoning results (evidence available)
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

        # Both vendors must have evidence
        from_key = from_v.lower().replace(" ", "_")
        to_key = to_v.lower().replace(" ", "_")
        has_from = any(k.lower().replace(" ", "_") == from_key for k in evidence_lookup)
        has_to = any(k.lower().replace(" ", "_") == to_key for k in evidence_lookup)
        if not has_from or not has_to:
            continue

        strength = edge.get("signal_strength", "emerging")
        velocity_7d = edge.get("velocity_7d") or 0
        mention_count = edge.get("mention_count", 0)

        # Composite score: strength rank * 100 + velocity bonus + mention count
        score = strength_rank.get(strength, 0) * 100
        if velocity_7d > 0:
            score += 50  # Accelerating bonus
        score += min(mention_count, 50)

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
    *,
    min_vendors: int = 3,
    min_displacement_intensity: float = 1.0,
    max_categories: int = 3,
) -> list[tuple[str, dict[str, Any]]]:
    """Select categories for category council reasoning.

    Criteria:
    1. At least min_vendors with reasoning results in that category
    2. displacement_intensity above threshold (active switching)
    3. Ranked by (vendor_count * displacement_intensity) descending
    """
    candidates: list[tuple[float, str, dict[str, Any]]] = []

    for cat, eco in ecosystem_evidence.items():
        displacement_intensity = eco.get("displacement_intensity") or 0.0
        if displacement_intensity < min_displacement_intensity:
            continue

        eco_vendor_count = eco.get("vendor_count", 0)
        if eco_vendor_count < min_vendors:
            continue

        score = eco_vendor_count * displacement_intensity
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
) -> list[tuple[str, str]]:
    """Select vendor pairs with similar pressure but different resources.

    Criteria:
    1. pressure_score (or avg_urgency) within 15% of each other
    2. Resource divergence: one has 3x+ review share (total_reviews), or one
       is enterprise-tilted while the other is SMB-tilted
    3. Both have per-vendor evidence
    """
    # Build vendor pressure list from vendor_scores
    vendor_pressure: list[tuple[str, float, int]] = []
    for vs in vendor_scores:
        vname = vs.get("vendor_name", "")
        if not vname:
            continue
        # Check if this vendor has evidence
        canon = vname.lower().replace(" ", "_")
        has_evidence = any(
            k.lower().replace(" ", "_") == canon for k in evidence_lookup
        )
        if not has_evidence:
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
            if abs(pa - pb) > 1.5:
                continue

            # Resource divergence checks
            divergence_score = 0.0

            # Review count ratio (proxy for installed base)
            if ra > 0 and rb > 0:
                ratio = max(ra, rb) / max(min(ra, rb), 1)
                if ratio >= 3.0:
                    divergence_score += ratio

            # Company size tilt from product profiles
            profile_a = product_profiles.get(va)
            profile_b = product_profiles.get(vb)
            if profile_a and profile_b:
                size_a = (profile_a.get("typical_company_size") or "").lower()
                size_b = (profile_b.get("typical_company_size") or "").lower()
                enterprise_keywords = {"enterprise", "large", "1000+"}
                smb_keywords = {"smb", "small", "startup", "1-50", "1-100"}
                a_ent = any(kw in size_a for kw in enterprise_keywords)
                a_smb = any(kw in size_a for kw in smb_keywords)
                b_ent = any(kw in size_b for kw in enterprise_keywords)
                b_smb = any(kw in size_b for kw in smb_keywords)
                if (a_ent and b_smb) or (a_smb and b_ent):
                    divergence_score += 5.0

            if divergence_score < 2.0:
                continue

            candidates.append((divergence_score, va, vb))

    candidates.sort(key=lambda x: x[0], reverse=True)

    result = [(va, vb) for _score, va, vb in candidates[:max_pairs]]
    logger.info(
        "Selected %d asymmetry pairs from %d vendor scores",
        len(result), len(vendor_scores),
    )
    return result
