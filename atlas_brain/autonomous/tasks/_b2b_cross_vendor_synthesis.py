"""Cross-vendor synthesis packet builders and persistence helpers.

Builds deterministic evidence packets for pairwise battles, category
councils, and resource asymmetry analyses.  Each packet is a plain dict
suitable for JSON serialization and LLM prompting.

The ``to_legacy_cross_vendor_conclusion`` converter mirrors synthesis
output into the legacy ``b2b_cross_vendor_conclusions`` shape so existing
consumers (battle cards, blogs, challenger briefs) continue working
during migration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import date
from typing import Any

logger = logging.getLogger("atlas.autonomous.tasks._b2b_cross_vendor_synthesis")


# ---------------------------------------------------------------------------
# Vendor name helpers
# ---------------------------------------------------------------------------

def _canon(name: str) -> str:
    return (name or "").strip().lower()


def _sorted_vendors(*names: str | None) -> list[str]:
    return sorted(set(
        s for n in names
        if isinstance(n, str) and (s := n.strip())
    ))


def _slug(value: str | None) -> str:
    text = (value or "").strip().lower()
    if not text:
        return "unknown"
    chars = []
    for ch in text:
        chars.append(ch if ch.isalnum() else "_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unknown"


def empty_cross_vendor_lookup() -> dict[str, dict]:
    """Return the normalized empty cross-vendor lookup shape."""
    return {"battles": {}, "councils": {}, "asymmetries": {}}


# ---------------------------------------------------------------------------
# Evidence hashing
# ---------------------------------------------------------------------------

def compute_cross_vendor_evidence_hash(packet: dict[str, Any]) -> str:
    """Deterministic SHA-256 prefix from packet content."""
    raw = json.dumps(packet, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Pool summary extraction
# ---------------------------------------------------------------------------

def _vendor_pool_summary(
    vendor_name: str,
    pool_layers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Extract a compact vendor summary from pool layers."""
    layers = pool_layers.get(vendor_name) or pool_layers.get(_canon(vendor_name)) or {}
    if not layers:
        # Try fuzzy match
        for k, v in pool_layers.items():
            if _canon(k) == _canon(vendor_name):
                layers = v
                break

    core = layers.get("core") or layers.get("churn_signal") or {}
    pain = layers.get("pain_distribution") or []
    budget = layers.get("budget_pressure") or {}
    segment = layers.get("segment") or layers.get("affected_roles") or []
    temporal = layers.get("temporal") or {}
    displacement = layers.get("displacement") or layers.get("competitive_flows") or []

    return {
        "vendor": vendor_name,
        "total_reviews": core.get("total_reviews") or core.get("review_count") or 0,
        "avg_urgency": core.get("avg_urgency_score") or core.get("avg_urgency") or 0,
        "churn_density": core.get("churn_signal_density") or 0,
        "price_complaint_rate": core.get("price_complaint_rate") or budget.get("price_complaint_rate") or 0,
        "price_increase_rate": budget.get("price_increase_rate") or 0,
        "avg_seat_count": budget.get("avg_seat_count") or 0,
        "recommend_ratio": core.get("recommend_ratio"),
        "nps_proxy": core.get("nps_proxy"),
        "pain_distribution": pain[:5] if isinstance(pain, list) else [],
        "top_competitors": (core.get("top_competitors") or [])[:5],
        "displacement_targets": displacement[:5] if isinstance(displacement, list) else [],
        "segment_summary": segment[:3] if isinstance(segment, list) else [],
        "sentiment_direction": (temporal.get("sentiment_trajectory") or {}).get("direction"),
    }


def _vendor_profile_summary(
    vendor_name: str,
    product_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Extract a compact profile summary."""
    profile = product_profiles.get(vendor_name) or {}
    if not profile:
        for k, v in product_profiles.items():
            if _canon(k) == _canon(vendor_name):
                profile = v
                break
    return {
        "category": profile.get("product_category") or "",
        "strengths": (profile.get("strengths") or [])[:5],
        "weaknesses": (profile.get("weaknesses") or [])[:5],
        "integrations": (profile.get("top_integrations") or [])[:5],
        "use_cases": (profile.get("primary_use_cases") or [])[:5],
        "typical_company_size": profile.get("typical_company_size"),
        "typical_industries": (profile.get("typical_industries") or [])[:5],
    }


def _copy_reference_ids(refs: dict[str, Any] | None) -> dict[str, list[str]]:
    refs = refs or {}
    metric_ids = [
        str(item).strip()
        for item in (refs.get("metric_ids") or [])
        if str(item).strip()
    ]
    witness_ids = [
        str(item).strip()
        for item in (refs.get("witness_ids") or [])
        if str(item).strip()
    ]
    return {
        "metric_ids": list(dict.fromkeys(metric_ids)),
        "witness_ids": list(dict.fromkeys(witness_ids)),
    }


def _merge_reference_ids(*refs_groups: dict[str, Any] | None) -> dict[str, list[str]]:
    metric_ids: list[str] = []
    witness_ids: list[str] = []
    for refs in refs_groups:
        copied = _copy_reference_ids(refs)
        metric_ids.extend(copied["metric_ids"])
        witness_ids.extend(copied["witness_ids"])
    return {
        "metric_ids": list(dict.fromkeys(metric_ids)),
        "witness_ids": list(dict.fromkeys(witness_ids)),
    }


async def _fetch_pairwise_reference_fallbacks(
    pool,
    *,
    as_of: date,
    analysis_window_days: int,
) -> dict[tuple[str, ...], dict[str, list[str]]]:
    """Build pairwise witness refs from persisted displacement-edge provenance."""
    fallbacks: dict[tuple[str, ...], dict[str, list[str]]] = {}
    rows = await pool.fetch(
        """
        SELECT from_vendor, to_vendor, sample_review_ids,
               computed_date, created_at
        FROM b2b_displacement_edges
        WHERE computed_date <= $1
          AND computed_date > $1::date - make_interval(days => $2)
        ORDER BY from_vendor, to_vendor, computed_date DESC, created_at DESC
        """,
        as_of,
        analysis_window_days,
    )
    for row in rows:
        key = tuple(_sorted_vendors(row.get("from_vendor"), row.get("to_vendor")))
        if len(key) < 2 or key in fallbacks:
            continue
        witness_ids = [
            str(review_id).strip()
            for review_id in (row.get("sample_review_ids") or [])
            if str(review_id).strip()
        ]
        if witness_ids:
            fallbacks[key] = {"witness_ids": list(dict.fromkeys(witness_ids))}
    return fallbacks


def _append_citation_entry(
    registry: list[dict[str, Any]],
    *,
    sid: str,
    label: str,
    refs: dict[str, Any] | None = None,
) -> None:
    if not sid or not label:
        return
    registry.append({
        "_sid": sid,
        "label": label,
        "reference_ids": _copy_reference_ids(refs),
    })


def _normalize_citation_text(value: str) -> str:
    text = (value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("\u2192", "->")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _best_registry_sid_for_citation(
    citation: str,
    registry_map: dict[str, dict[str, Any]],
) -> str | None:
    citation_norm = _normalize_citation_text(citation)
    if not citation_norm:
        return None
    if citation in registry_map:
        return citation

    best_sid: str | None = None
    best_score = 0.0
    citation_tokens = set(citation_norm.split())
    for sid, item in registry_map.items():
        label = str(item.get("label") or "")
        label_norm = _normalize_citation_text(label)
        if not label_norm:
            continue
        if citation_norm == label_norm:
            return sid
        label_tokens = set(label_norm.split())
        overlap = citation_tokens.intersection(label_tokens)
        if not overlap:
            continue
        score = len(overlap) / max(1, len(citation_tokens.union(label_tokens)))
        if citation_norm in label_norm or label_norm in citation_norm:
            score += 0.35
        if score > best_score:
            best_score = score
            best_sid = sid
    if best_score >= 0.45:
        return best_sid
    return None


# ---------------------------------------------------------------------------
# Packet builders
# ---------------------------------------------------------------------------

def build_pairwise_battle_packet(
    vendor_a: str,
    vendor_b: str,
    edge: dict[str, Any],
    pool_layers: dict[str, dict[str, Any]],
    product_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic evidence packet for a pairwise battle.

    The ``locked_direction`` field tells the LLM which vendor is gaining
    (winner) and which is losing (loser) based on displacement evidence.
    """
    from_vendor = edge.get("from_vendor") or vendor_a
    to_vendor = edge.get("to_vendor") or vendor_b

    return {
        "analysis_type": "pairwise_battle",
        "locked_direction": {
            "winner": to_vendor,
            "loser": from_vendor,
        },
        "displacement_edge": {
            "from_vendor": from_vendor,
            "to_vendor": to_vendor,
            "mention_count": edge.get("mention_count") or 0,
            "signal_strength": edge.get("signal_strength") or "emerging",
            "primary_driver": edge.get("primary_driver") or "",
            "evidence_breakdown": edge.get("evidence_breakdown") or {},
            "velocity_7d": edge.get("velocity_7d") or 0,
        },
        "vendor_a_pool": _vendor_pool_summary(vendor_a, pool_layers),
        "vendor_b_pool": _vendor_pool_summary(vendor_b, pool_layers),
        "vendor_a_profile": _vendor_profile_summary(vendor_a, product_profiles),
        "vendor_b_profile": _vendor_profile_summary(vendor_b, product_profiles),
    }


def build_category_council_packet(
    category: str,
    ecosystem_evidence: dict[str, Any],
    pool_layers: dict[str, dict[str, Any]],
    product_profiles: dict[str, dict[str, Any]],
    displacement_edges: list[dict[str, Any]] | None = None,
    *,
    vendor_summary_limit: int = 10,
    flow_limit: int = 15,
) -> dict[str, Any]:
    """Build a deterministic evidence packet for a category council."""
    # Find vendors in this category from profiles
    category_vendors: list[str] = []
    for vname, profile in product_profiles.items():
        if _canon(profile.get("product_category") or "") == _canon(category):
            category_vendors.append(vname)

    vendor_summaries = [
        _vendor_pool_summary(v, pool_layers)
        for v in sorted(category_vendors)[: max(1, int(vendor_summary_limit))]
    ]

    # Filter displacement edges to this category's vendors
    cat_vendor_set = {_canon(v) for v in category_vendors}
    cat_edges = []
    for edge in (displacement_edges or []):
        if (_canon(edge.get("from_vendor") or "") in cat_vendor_set
                or _canon(edge.get("to_vendor") or "") in cat_vendor_set):
            cat_edges.append({
                "from_vendor": edge.get("from_vendor"),
                "to_vendor": edge.get("to_vendor"),
                "mention_count": edge.get("mention_count") or 0,
                "primary_driver": edge.get("primary_driver") or "",
            })

    return {
        "analysis_type": "category_council",
        "category": category,
        "vendor_count": len(category_vendors),
        "ecosystem_evidence": {
            "hhi": ecosystem_evidence.get("hhi"),
            "market_structure": ecosystem_evidence.get("market_structure"),
            "displacement_intensity": ecosystem_evidence.get("displacement_intensity"),
            "dominant_archetype": ecosystem_evidence.get("dominant_archetype"),
            "archetype_distribution": ecosystem_evidence.get("archetype_distribution") or {},
        },
        "vendor_summaries": vendor_summaries,
        "displacement_flows": cat_edges[: max(0, int(flow_limit))],
    }


def build_resource_asymmetry_packet(
    vendor_a: str,
    vendor_b: str,
    pool_layers: dict[str, dict[str, Any]],
    product_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic evidence packet for resource asymmetry analysis."""
    summary_a = _vendor_pool_summary(vendor_a, pool_layers)
    summary_b = _vendor_pool_summary(vendor_b, pool_layers)

    # Determine favored/disadvantaged by review count (proxy for installed base)
    reviews_a = summary_a.get("total_reviews") or 0
    reviews_b = summary_b.get("total_reviews") or 0

    return {
        "analysis_type": "resource_asymmetry",
        "vendor_a": vendor_a,
        "vendor_b": vendor_b,
        "pressure_scores": {
            "vendor_a_urgency": summary_a.get("avg_urgency") or 0,
            "vendor_b_urgency": summary_b.get("avg_urgency") or 0,
        },
        "resource_indicators": {
            "vendor_a_reviews": reviews_a,
            "vendor_b_reviews": reviews_b,
            "vendor_a_seat_count": summary_a.get("avg_seat_count") or 0,
            "vendor_b_seat_count": summary_b.get("avg_seat_count") or 0,
            "vendor_a_recommend_ratio": summary_a.get("recommend_ratio"),
            "vendor_b_recommend_ratio": summary_b.get("recommend_ratio"),
        },
        "divergence_score": abs(reviews_a - reviews_b) / max(reviews_a, reviews_b, 1),
        "vendor_a_pool": summary_a,
        "vendor_b_pool": summary_b,
        "vendor_a_profile": _vendor_profile_summary(vendor_a, product_profiles),
        "vendor_b_profile": _vendor_profile_summary(vendor_b, product_profiles),
    }


def attach_cross_vendor_citation_registry(
    packet: dict[str, Any],
    *,
    analysis_type: str,
    vendors: list[str],
    category: str | None,
    vendor_reference_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attach packet-level citation ids that map to underlying vendor refs."""
    vendor_reference_lookup = vendor_reference_lookup or {}
    sorted_vendors = _sorted_vendors(*vendors)
    combined_refs = _merge_reference_ids(
        *(vendor_reference_lookup.get(vendor) for vendor in sorted_vendors)
    )
    registry: list[dict[str, Any]] = []

    if analysis_type == "pairwise_battle":
        winner = ((packet.get("locked_direction") or {}).get("winner") or "").strip()
        loser = ((packet.get("locked_direction") or {}).get("loser") or "").strip()
        edge = packet.get("displacement_edge") or {}
        edge_sid = (
            f"xv:pairwise:edge:{_slug(loser)}_to_{_slug(winner)}"
            if winner and loser else
            f"xv:pairwise:edge:{_slug(sorted_vendors[0] if sorted_vendors else '')}"
        )
        _append_citation_entry(
            registry,
            sid=edge_sid,
            label=(
                f"{loser}->{winner} displacement edge: mention_count={edge.get('mention_count') or 0}, "
                f"signal_strength={edge.get('signal_strength') or 'unknown'}, "
                f"primary_driver={edge.get('primary_driver') or 'unknown'}, "
                f"velocity_7d={edge.get('velocity_7d') or 0}"
            ),
            refs=combined_refs,
        )
    elif analysis_type == "category_council":
        ecosystem = packet.get("ecosystem_evidence") or {}
        _append_citation_entry(
            registry,
            sid=f"xv:category:{_slug(category)}:ecosystem",
            label=(
                f"{category} ecosystem: displacement_intensity={ecosystem.get('displacement_intensity')}, "
                f"market_structure={ecosystem.get('market_structure') or 'unknown'}, "
                f"dominant_archetype={ecosystem.get('dominant_archetype') or 'unknown'}"
            ),
            refs=combined_refs,
        )
    elif analysis_type == "resource_asymmetry":
        pressure = packet.get("pressure_scores") or {}
        resources = packet.get("resource_indicators") or {}
        _append_citation_entry(
            registry,
            sid=f"xv:asymmetry:{_slug(sorted_vendors[0] if sorted_vendors else '')}_{_slug(sorted_vendors[1] if len(sorted_vendors) > 1 else '')}:summary",
            label=(
                f"resource asymmetry summary: vendor_a_urgency={pressure.get('vendor_a_urgency') or 0}, "
                f"vendor_b_urgency={pressure.get('vendor_b_urgency') or 0}, "
                f"vendor_a_reviews={resources.get('vendor_a_reviews') or 0}, "
                f"vendor_b_reviews={resources.get('vendor_b_reviews') or 0}, "
                f"divergence_score={packet.get('divergence_score') or 0}"
            ),
            refs=combined_refs,
        )
        for field in ("pressure_scores", "resource_indicators"):
            item = packet.get(field) or {}
            if not isinstance(item, dict):
                continue
            label_bits = ", ".join(
                f"{key}={value}"
                for key, value in item.items()
                if value is not None and value != ""
            )
            _append_citation_entry(
                registry,
                sid=f"xv:asymmetry:{_slug(sorted_vendors[0] if sorted_vendors else '')}_{_slug(sorted_vendors[1] if len(sorted_vendors) > 1 else '')}:{field}",
                label=f"{field}: {label_bits}",
                refs=combined_refs,
            )

    for vendor_key in ("vendor_a_pool", "vendor_b_pool", "vendor_a_profile", "vendor_b_profile"):
        item = packet.get(vendor_key) or {}
        vendor = str(item.get("vendor") or "").strip()
        if not vendor:
            if vendor_key.startswith("vendor_a"):
                vendor = sorted_vendors[0] if sorted_vendors else ""
            elif len(sorted_vendors) > 1:
                vendor = sorted_vendors[1]
        refs = vendor_reference_lookup.get(vendor) or {}
        if vendor_key.endswith("_pool"):
            _append_citation_entry(
                registry,
                sid=f"xv:vendor:{_slug(vendor)}:pool",
                label=(
                    f"{vendor} pool summary: total_reviews={item.get('total_reviews') or 0}, "
                    f"avg_urgency={item.get('avg_urgency') or 0}, "
                    f"churn_density={item.get('churn_density') or 0}, "
                    f"price_complaint_rate={item.get('price_complaint_rate') or 0}"
                ),
                refs=refs,
            )
            for idx, target in enumerate(item.get("displacement_targets") or []):
                competitor = str(
                    target.get("to_vendor") or target.get("name") or ""
                ).strip()
                flow_summary = target.get("flow_summary") or {}
                edge_metrics = target.get("edge_metrics") or {}
                switch_reasons = target.get("switch_reasons") or []
                reason_text = ""
                if switch_reasons and isinstance(switch_reasons[0], dict):
                    reason_text = str(
                        switch_reasons[0].get("reason")
                        or switch_reasons[0].get("reason_category")
                        or switch_reasons[0].get("switch_reason")
                        or ""
                    ).strip()
                _append_citation_entry(
                    registry,
                    sid=f"xv:vendor:{_slug(vendor)}:flow:{idx}:{_slug(competitor)}",
                    label=(
                        f"{vendor} displacement target: {vendor}->{competitor}, "
                        f"total_flow_mentions={flow_summary.get('total_flow_mentions') or edge_metrics.get('mention_count') or 0}, "
                        f"explicit_switch_count={flow_summary.get('explicit_switch_count') or 0}, "
                        f"active_evaluation_count={flow_summary.get('active_evaluation_count') or 0}, "
                        f"switch_reason={reason_text or 'unknown'}"
                    ),
                    refs=refs,
                )
        else:
            _append_citation_entry(
                registry,
                sid=f"xv:vendor:{_slug(vendor)}:profile",
                label=(
                    f"{vendor} profile: category={item.get('category') or 'unknown'}, "
                    f"typical_company_size={item.get('typical_company_size') or 'unknown'}"
                ),
                refs=refs,
            )

    for idx, summary in enumerate(packet.get("vendor_summaries") or []):
        vendor = str(summary.get("vendor") or "").strip()
        refs = vendor_reference_lookup.get(vendor) or {}
        _append_citation_entry(
            registry,
            sid=f"xv:category:{_slug(category)}:vendor_summary:{idx}:{_slug(vendor)}",
            label=(
                f"{vendor} vendor summary: total_reviews={summary.get('total_reviews') or 0}, "
                f"avg_urgency={summary.get('avg_urgency') or 0}, "
                f"churn_density={summary.get('churn_density') or 0}, "
                f"price_complaint_rate={summary.get('price_complaint_rate') or 0}"
            ),
            refs=refs,
        )

    for idx, flow in enumerate(packet.get("displacement_flows") or []):
        from_vendor = str(flow.get("from_vendor") or "").strip()
        to_vendor = str(flow.get("to_vendor") or "").strip()
        refs = _merge_reference_ids(
            vendor_reference_lookup.get(from_vendor),
            vendor_reference_lookup.get(to_vendor),
        )
        _append_citation_entry(
            registry,
            sid=f"xv:flow:{idx}:{_slug(from_vendor)}_to_{_slug(to_vendor)}",
            label=(
                f"displacement_flows: {from_vendor}->{to_vendor}, "
                f"mention_count={flow.get('mention_count') or 0}, "
                f"primary_driver={flow.get('primary_driver') or 'unknown'}"
            ),
            refs=refs,
        )

    packet["citation_registry"] = registry
    return packet


def materialize_cross_vendor_reference_ids(
    synthesis: dict[str, Any],
    packet: dict[str, Any],
) -> dict[str, Any]:
    """Map cited packet ids to underlying metric/witness reference ids."""
    registry = packet.get("citation_registry") or []
    registry_map: dict[str, dict[str, Any]] = {}
    for item in registry:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("_sid") or "").strip()
        if sid:
            registry_map[sid] = item

    citations = synthesis.get("citations") or []
    if not isinstance(citations, list):
        citations = []
    normalized_citations = []
    metric_ids: list[str] = []
    witness_ids: list[str] = []
    for sid in citations:
        sid_text = str(sid).strip()
        resolved_sid = _best_registry_sid_for_citation(sid_text, registry_map)
        if not resolved_sid:
            continue
        normalized_citations.append(resolved_sid)
        refs = _copy_reference_ids(registry_map[resolved_sid].get("reference_ids"))
        metric_ids.extend(refs["metric_ids"])
        witness_ids.extend(refs["witness_ids"])

    synthesis["citations"] = list(dict.fromkeys(normalized_citations))
    synthesis["reference_ids"] = {
        "metric_ids": sorted(set(metric_ids)),
        "witness_ids": sorted(set(witness_ids)),
    }
    return synthesis


# ---------------------------------------------------------------------------
# Contract normalization
# ---------------------------------------------------------------------------

def normalize_cross_vendor_contract(
    raw: dict[str, Any],
    analysis_type: str,
) -> dict[str, Any]:
    """Ensure a parsed LLM response has the expected contract fields."""
    if analysis_type == "pairwise_battle":
        return {
            "winner": raw.get("winner") or "",
            "loser": raw.get("loser") or "",
            "conclusion": raw.get("conclusion") or "",
            "confidence": _clamp_confidence(raw.get("confidence")),
            "durability_assessment": raw.get("durability_assessment") or "uncertain",
            "key_insights": _ensure_insight_list(raw.get("key_insights")),
            "falsification_conditions": raw.get("falsification_conditions") or [],
            "citations": raw.get("citations") or [],
            "meta": raw.get("meta") or {"analysis_type": analysis_type, "schema_version": "synthesis_v1"},
        }
    elif analysis_type == "category_council":
        return {
            "market_regime": raw.get("market_regime") or "uncertain",
            "conclusion": raw.get("conclusion") or "",
            "winner": raw.get("winner"),
            "loser": raw.get("loser"),
            "confidence": _clamp_confidence(raw.get("confidence")),
            "durability_assessment": raw.get("durability_assessment") or "uncertain",
            "key_insights": _ensure_insight_list(raw.get("key_insights")),
            "citations": raw.get("citations") or [],
            "meta": raw.get("meta") or {"analysis_type": analysis_type, "schema_version": "synthesis_v1"},
        }
    elif analysis_type == "resource_asymmetry":
        return {
            "favored_vendor": raw.get("favored_vendor") or "",
            "disadvantaged_vendor": raw.get("disadvantaged_vendor") or "",
            "conclusion": raw.get("conclusion") or "",
            "pressure_delta": float(raw.get("pressure_delta") or 0),
            "confidence": _clamp_confidence(raw.get("confidence")),
            "key_insights": _ensure_insight_list(raw.get("key_insights")),
            "citations": raw.get("citations") or [],
            "meta": raw.get("meta") or {"analysis_type": analysis_type, "schema_version": "synthesis_v1"},
        }
    return raw


def _clamp_confidence(val: Any) -> float:
    try:
        return max(0.0, min(1.0, float(val)))
    except (TypeError, ValueError):
        return 0.0


def _ensure_insight_list(val: Any) -> list[dict[str, str]]:
    if not isinstance(val, list):
        return []
    result = []
    for item in val:
        if isinstance(item, dict):
            result.append({
                "insight": str(item.get("insight") or ""),
                "evidence": str(item.get("evidence") or ""),
            })
        elif isinstance(item, str):
            result.append({"insight": item, "evidence": ""})
    return result


# ---------------------------------------------------------------------------
# Legacy compatibility mirror
# ---------------------------------------------------------------------------

def to_legacy_cross_vendor_conclusion(
    synthesis: dict[str, Any],
    analysis_type: str,
    vendors: list[str],
    category: str | None = None,
    evidence_hash: str = "",
    tokens_used: int = 0,
) -> dict[str, Any]:
    """Convert a synthesis contract into a legacy b2b_cross_vendor_conclusions row.

    Returns a dict with keys matching the legacy table columns.
    """
    conclusion: dict[str, Any]
    confidence: float

    if analysis_type == "pairwise_battle":
        conclusion = {
            "winner": synthesis.get("winner") or "",
            "loser": synthesis.get("loser") or "",
            "conclusion": synthesis.get("conclusion") or "",
            "market_regime": synthesis.get("meta", {}).get("market_regime"),
            "durability_assessment": synthesis.get("durability_assessment") or "uncertain",
            "key_insights": synthesis.get("key_insights") or [],
        }
        confidence = _clamp_confidence(synthesis.get("confidence"))
    elif analysis_type == "category_council":
        conclusion = {
            "winner": synthesis.get("winner"),
            "loser": synthesis.get("loser"),
            "conclusion": synthesis.get("conclusion") or "",
            "market_regime": synthesis.get("market_regime") or "",
            "durability_assessment": synthesis.get("durability_assessment") or "uncertain",
            "key_insights": synthesis.get("key_insights") or [],
        }
        confidence = _clamp_confidence(synthesis.get("confidence"))
    elif analysis_type == "resource_asymmetry":
        conclusion = {
            "favored_vendor": synthesis.get("favored_vendor") or "",
            "disadvantaged_vendor": synthesis.get("disadvantaged_vendor") or "",
            "conclusion": synthesis.get("conclusion") or "",
            "resource_advantage": synthesis.get("favored_vendor") or "",
            "pressure_delta": synthesis.get("pressure_delta") or 0,
        }
        confidence = _clamp_confidence(synthesis.get("confidence"))
    else:
        conclusion = dict(synthesis)
        confidence = _clamp_confidence(synthesis.get("confidence"))

    return {
        "analysis_type": analysis_type,
        "vendors": sorted(vendors),
        "category": category,
        "conclusion": conclusion,
        "confidence": confidence,
        "evidence_hash": evidence_hash,
        "tokens_used": tokens_used,
        "cached": False,
    }


# ---------------------------------------------------------------------------
# Cross-vendor synthesis reader
# ---------------------------------------------------------------------------


async def load_cross_vendor_synthesis_lookup(
    pool,
    *,
    as_of: date | None = None,
    analysis_window_days: int = 90,
) -> dict[str, dict]:
    """Read cross-vendor synthesis from the canonical table.

    Returns the same shape as ``reconstruct_cross_vendor_lookup`` so
    consumers can swap transparently:

        {"battles": {...}, "councils": {...}, "asymmetries": {...}}

    Each value uses sorted vendor tuples (battles/asymmetries) or category
    names (councils) as keys.
    """
    if as_of is None:
        as_of = date.today()

    try:
        pairwise_reference_fallbacks = await _fetch_pairwise_reference_fallbacks(
            pool,
            as_of=as_of,
            analysis_window_days=analysis_window_days,
        )
    except Exception:
        pairwise_reference_fallbacks = {}
        logger.debug("Pairwise reference fallback load failed", exc_info=True)

    rows = await pool.fetch(
        """
        SELECT analysis_type, vendors, category, synthesis,
               as_of_date, created_at
        FROM b2b_cross_vendor_reasoning_synthesis
        WHERE as_of_date <= $1
          AND analysis_window_days = $2
        ORDER BY analysis_type, vendors, category,
                 as_of_date DESC, created_at DESC
        """,
        as_of,
        analysis_window_days,
    )

    battles: dict[tuple[str, ...], dict] = {}
    councils: dict[str, dict] = {}
    asymmetries: dict[tuple[str, ...], dict] = {}

    def _prefer_candidate(existing: dict[str, Any] | None, candidate: dict[str, Any]) -> bool:
        if existing is None:
            return True
        existing_refs = bool(existing.get("reference_ids"))
        candidate_refs = bool(candidate.get("reference_ids"))
        if not existing_refs and candidate_refs:
            return True
        if existing_refs and not candidate_refs:
            return False
        existing_date = existing.get("computed_date")
        candidate_date = candidate.get("computed_date")
        if existing_date != candidate_date:
            return candidate_date is not None and (
                existing_date is None or candidate_date > existing_date
            )
        existing_created = str(existing.get("created_at") or "")
        candidate_created = str(candidate.get("created_at") or "")
        return candidate_created > existing_created

    for r in rows:
        atype = r["analysis_type"]
        vendors = list(r["vendors"]) if r["vendors"] else []
        category = r["category"] or ""
        raw = r["synthesis"]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(raw, dict):
            continue

        # The synthesis column stores either:
        # 1. a normalized top-level contract with winner/loser/conclusion keys
        # 2. a wrapper object with a nested "conclusion" dict
        nested_conclusion = raw.get("conclusion")
        if isinstance(nested_conclusion, dict):
            conclusion = nested_conclusion
        else:
            conclusion = raw
        confidence = float(
            conclusion.get("confidence") or raw.get("confidence") or 0,
        )

        entry = {
            "conclusion": conclusion,
            "confidence": confidence,
            "vendors": vendors,
            "category": category,
            "computed_date": r["as_of_date"],
            "created_at": r["created_at"],
            "source": "synthesis",
        }
        reference_ids = raw.get("reference_ids")
        if isinstance(reference_ids, dict) and reference_ids:
            entry["reference_ids"] = _copy_reference_ids(reference_ids)
        elif atype == "pairwise_battle" and len(vendors) >= 2:
            fallback_refs = pairwise_reference_fallbacks.get(tuple(sorted(vendors)))
            if fallback_refs:
                entry["reference_ids"] = _copy_reference_ids(fallback_refs)

        if atype == "pairwise_battle" and len(vendors) >= 2:
            key = tuple(sorted(vendors))
            if _prefer_candidate(battles.get(key), entry):
                battles[key] = entry
        elif atype == "category_council" and category:
            if _prefer_candidate(councils.get(category), entry):
                councils[category] = entry
        elif atype == "resource_asymmetry" and len(vendors) >= 2:
            key = tuple(sorted(vendors))
            if _prefer_candidate(asymmetries.get(key), entry):
                asymmetries[key] = entry

    return {"battles": battles, "councils": councils, "asymmetries": asymmetries}


def merge_cross_vendor_lookups(
    *,
    primary: dict[str, dict] | None,
    fallback: dict[str, dict] | None,
) -> tuple[dict[str, dict], int]:
    """Merge two cross-vendor lookups with primary entries winning per key."""
    merged = empty_cross_vendor_lookup()
    overrides = 0
    primary = primary or empty_cross_vendor_lookup()
    fallback = fallback or empty_cross_vendor_lookup()

    for bucket in ("battles", "councils", "asymmetries"):
        merged_bucket = dict(fallback.get(bucket, {}))
        for key, value in (primary.get(bucket, {}) or {}).items():
            if key in merged_bucket:
                overrides += 1
            merged_bucket[key] = value
        merged[bucket] = merged_bucket

    return merged, overrides


async def load_best_cross_vendor_lookup(
    pool,
    *,
    as_of: date | None = None,
    analysis_window_days: int = 90,
) -> dict[str, dict]:
    """Load canonical cross-vendor synthesis first, with legacy rows filling gaps."""
    try:
        synthesis_lookup = await load_cross_vendor_synthesis_lookup(
            pool,
            as_of=as_of,
            analysis_window_days=analysis_window_days,
        )
    except Exception:
        logger.debug("Cross-vendor synthesis lookup failed; falling back to legacy", exc_info=True)
        synthesis_lookup = empty_cross_vendor_lookup()

    try:
        from .b2b_churn_intelligence import reconstruct_cross_vendor_lookup

        legacy_lookup = await reconstruct_cross_vendor_lookup(pool, as_of=as_of)
    except Exception:
        logger.debug("Legacy cross-vendor lookup failed", exc_info=True)
        legacy_lookup = empty_cross_vendor_lookup()

    merged, _ = merge_cross_vendor_lookups(
        primary=synthesis_lookup,
        fallback=legacy_lookup,
    )
    try:
        pairwise_reference_fallbacks = await _fetch_pairwise_reference_fallbacks(
            pool,
            as_of=as_of or date.today(),
            analysis_window_days=analysis_window_days,
        )
    except Exception:
        pairwise_reference_fallbacks = {}
        logger.debug("Merged pairwise reference fallback load failed", exc_info=True)
    if pairwise_reference_fallbacks:
        for key, entry in (merged.get("battles") or {}).items():
            if not isinstance(entry, dict):
                continue
            if isinstance(entry.get("reference_ids"), dict) and entry.get("reference_ids"):
                continue
            fallback_refs = pairwise_reference_fallbacks.get(tuple(sorted(key)))
            if fallback_refs:
                entry["reference_ids"] = _copy_reference_ids(fallback_refs)
    return merged


def build_cross_vendor_conclusions_for_vendor(
    vendor_name: str,
    *,
    category: str | None = None,
    xv_lookup: dict[str, dict] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Materialize vendor-facing cross-vendor conclusions from a merged lookup."""
    vendor = str(vendor_name or "").strip()
    if not vendor:
        return []
    category_name = str(category or "").strip()
    lookup = xv_lookup or empty_cross_vendor_lookup()
    results: list[dict[str, Any]] = []

    def _append_entry(
        *,
        analysis_type: str,
        entry: dict[str, Any],
        vendors: list[str],
        category_value: str | None = None,
    ) -> None:
        conclusion = entry.get("conclusion") or {}
        if not isinstance(conclusion, dict):
            return
        summary = (
            conclusion.get("conclusion")
            or conclusion.get("summary")
            or ""
        )
        if not summary:
            return
        item: dict[str, Any] = {
            "analysis_type": analysis_type,
            "vendors": list(vendors),
            "confidence": float(entry.get("confidence") or 0),
            "summary": summary,
            "source": entry.get("source") or "",
        }
        if category_value:
            item["category"] = category_value
        computed_date = entry.get("computed_date")
        if computed_date is not None and hasattr(computed_date, "isoformat"):
            item["computed_date"] = computed_date.isoformat()
        reference_ids = entry.get("reference_ids")
        if isinstance(reference_ids, dict) and reference_ids:
            item["reference_ids"] = _copy_reference_ids(reference_ids)
        results.append(item)

    vendor_lower = vendor.lower()
    for pair_key, entry in (lookup.get("battles") or {}).items():
        members = [str(value or "").strip() for value in pair_key]
        if any(member.lower() == vendor_lower for member in members):
            _append_entry(
                analysis_type="pairwise_battle",
                entry=entry,
                vendors=members,
            )

    for pair_key, entry in (lookup.get("asymmetries") or {}).items():
        members = [str(value or "").strip() for value in pair_key]
        if any(member.lower() == vendor_lower for member in members):
            _append_entry(
                analysis_type="resource_asymmetry",
                entry=entry,
                vendors=members,
            )

    if category_name:
        council = (lookup.get("councils") or {}).get(category_name)
        if isinstance(council, dict):
            _append_entry(
                analysis_type="category_council",
                entry=council,
                vendors=list(council.get("vendors") or []),
                category_value=category_name,
            )

    results.sort(
        key=lambda item: (
            0 if item.get("analysis_type") == "pairwise_battle" else 1,
            -float(item.get("confidence") or 0),
            str(item.get("computed_date") or ""),
        ),
    )
    if limit > 0:
        return results[:limit]
    return results
