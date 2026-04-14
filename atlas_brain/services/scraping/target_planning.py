"""Planning helpers for scrape target coverage and noise audits."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from ...config import settings
from .source_fit import SourceFit, classify_source_fit
from .sources import ALL_SOURCES, SEARCH_SOURCES, parse_source_allowlist

_GENERIC_CATEGORIES = frozenset({"", "unknown", "<null>", "b2b software"})

_VERIFIED_CORE_TARGETS: dict[tuple[str, str, str], dict[str, str]] = {
    (
        "gartner",
        "Microsoft Defender for Endpoint",
        "Cybersecurity",
    ): {
        "product_slug": "endpoint-protection-platforms/microsoft/product/microsoft-defender-for-endpoint",
        "product_name": "Microsoft Defender for Endpoint",
    },
    (
        "software_advice",
        "Jira",
        "Project Management",
    ): {
        "product_slug": "project-management/atlassian-jira-profile",
        "product_name": "Jira",
    },
    (
        "software_advice",
        "Trello",
        "Project Management",
    ): {
        "product_slug": "project-management/trello-profile",
        "product_name": "Trello",
    },
    (
        "software_advice",
        "Microsoft Defender for Endpoint",
        "Cybersecurity",
    ): {
        "product_slug": "security/microsoft-365-defender-profile",
        "product_name": "Microsoft Defender XDR",
    },
}

_UNSUPPORTED_VENDOR_SOURCE_TARGETS: set[tuple[str, str, str]] = {
    (
        "software_advice",
        "Amazon Web Services",
        "Cloud Infrastructure",
    ),
}

_INVENTORY_SOURCE_PRIORITY: dict[str, int] = {
    "b2b_product_profiles": 0,
    "b2b_churn_signals": 1,
    "tracked_vendors": 2,
    "vendor_targets": 3,
}

_HIGH_IDENTITY_STRUCTURED_SOURCES = frozenset({"trustradius"})
_CONTEXT_RICH_STRUCTURED_SOURCES = frozenset(
    {"g2", "gartner", "capterra", "peerspot", "software_advice"}
)
_SEARCH_SOURCE_VALUES = frozenset(member.value for member in SEARCH_SOURCES)


def _normalize_vendor(value: str | None) -> str:
    return str(value or "").strip()


def _normalize_category(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text or None


def _is_generic_category(value: str | None) -> bool:
    return str(value or "").strip().lower() in _GENERIC_CATEGORIES


def _slugify_vendor(vendor_name: str) -> str:
    parts = re.findall(r"[a-z0-9]+", vendor_name.lower())
    return "-".join(parts) or "vendor"


def _safe_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _source_yield_priority(source: str) -> int:
    normalized = str(source or "").strip().lower()
    if normalized in _HIGH_IDENTITY_STRUCTURED_SOURCES:
        return 0
    if normalized in _CONTEXT_RICH_STRUCTURED_SOURCES:
        return 1
    if normalized in _SEARCH_SOURCE_VALUES:
        return 3
    return 2


def _source_yield_tier(source: str) -> str:
    normalized = str(source or "").strip().lower()
    if normalized in _HIGH_IDENTITY_STRUCTURED_SOURCES:
        return "high_yield"
    if normalized in _CONTEXT_RICH_STRUCTURED_SOURCES:
        return "context_rich"
    if normalized in _SEARCH_SOURCE_VALUES:
        return "search"
    return "standard"


def _is_auto_seedable(source: str) -> bool:
    return source in {member.value for member in SEARCH_SOURCES}


def get_verified_core_target(
    source: str,
    vendor_name: str,
    product_category: str | None,
) -> dict[str, str] | None:
    """Return a curated, verified target spec when available."""
    key = (
        str(source or "").strip().lower(),
        _normalize_vendor(vendor_name),
        _normalize_category(product_category) or "Unknown",
    )
    spec = _VERIFIED_CORE_TARGETS.get(key)
    return dict(spec) if spec else None


def is_vendor_source_target_supported(
    source: str,
    vendor_name: str,
    product_category: str | None,
) -> bool:
    key = (
        str(source or "").strip().lower(),
        _normalize_vendor(vendor_name),
        _normalize_category(product_category) or "Unknown",
    )
    return key not in _UNSUPPORTED_VENDOR_SOURCE_TARGETS


def _inventory_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    category = _normalize_category(row.get("product_category"))
    reviews = int(row.get("total_reviews_analyzed") or row.get("total_reviews") or 0)
    confidence = float(row.get("confidence_score") or 0.0)
    last_computed = str(row.get("last_computed_at") or "")
    inventory_source = str(row.get("inventory_source") or "b2b_product_profiles").strip().lower()
    return (
        1 if _is_generic_category(category) else 0,
        _INVENTORY_SOURCE_PRIORITY.get(inventory_source, 99),
        -reviews,
        -confidence,
        last_computed,
        category or "",
    )


def collapse_inventory_rows(profile_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick one primary category row per vendor for scrape planning."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in profile_rows:
        vendor = _normalize_vendor(row.get("vendor_name"))
        if not vendor:
            continue
        grouped.setdefault(vendor, []).append(dict(row))

    collapsed: list[dict[str, Any]] = []
    for vendor, rows in grouped.items():
        best = sorted(rows, key=_inventory_sort_key)[0]
        collapsed.append(
            {
                "vendor_name": vendor,
                "product_category": _normalize_category(best.get("product_category")) or "Unknown",
                "total_reviews_analyzed": int(best.get("total_reviews_analyzed") or 0),
                "confidence_score": float(best.get("confidence_score") or 0.0),
                "last_computed_at": best.get("last_computed_at"),
                "inventory_source": str(best.get("inventory_source") or "b2b_product_profiles"),
            }
        )
    return sorted(collapsed, key=lambda row: (row["vendor_name"], row["product_category"]))


def build_scrape_coverage_plan(
    inventory_rows: list[dict[str, Any]],
    existing_targets: list[dict[str, Any]],
    *,
    allowed_sources: list[str] | None = None,
) -> dict[str, Any]:
    """Return missing core coverage and enabled poor-fit targets."""
    collapsed_inventory = collapse_inventory_rows(inventory_rows)
    allowed = {src.strip().lower() for src in (allowed_sources or []) if src.strip()}
    if not allowed:
        allowed = {member.value for member in ALL_SOURCES}
    deferred_inventory_sources = set(
        parse_source_allowlist(getattr(settings.b2b_scrape, "deferred_inventory_sources", ""))
    )

    enabled_by_vendor_source: set[tuple[str, str]] = set()
    disabled_by_vendor_source: dict[tuple[str, str], dict[str, Any]] = {}
    poor_fit_enabled_targets: list[dict[str, Any]] = []
    conditional_opportunities: list[dict[str, Any]] = []
    deferred_conditional_inventory: list[dict[str, Any]] = []

    for row in existing_targets:
        vendor = _normalize_vendor(row.get("vendor_name"))
        source = str(row.get("source") or "").strip().lower()
        category = _normalize_category(row.get("product_category"))
        if not vendor or not source:
            continue
        key = (vendor, source)
        if row.get("enabled"):
            enabled_by_vendor_source.add(key)
            decision = classify_source_fit(source, category)
            metadata = _safe_metadata(row.get("metadata"))
            if decision.fit == SourceFit.avoid.value:
                poor_fit_enabled_targets.append(
                    {
                        "id": str(row.get("id") or ""),
                        "source": source,
                        "source_tier": _source_yield_tier(source),
                        "vendor_name": vendor,
                        "product_category": category,
                        "product_slug": row.get("product_slug"),
                        "scrape_mode": row.get("scrape_mode"),
                        "priority": row.get("priority"),
                        "vertical": decision.vertical,
                        "source_fit": decision.fit,
                        "reason": decision.reason,
                        "source_fit_override": str(metadata.get("source_fit_override") or ""),
                    }
                )
        else:
            disabled_by_vendor_source.setdefault(key, dict(row))

    missing_core_targets: list[dict[str, Any]] = []
    for row in collapsed_inventory:
        vendor = row["vendor_name"]
        category = row["product_category"]
        for source in sorted(allowed, key=lambda item: (_source_yield_priority(item), item)):
            if not is_vendor_source_target_supported(source, vendor, category):
                continue
            decision = classify_source_fit(source, category)
            existing_disabled = disabled_by_vendor_source.get((vendor, source))
            auto_seedable = _is_auto_seedable(source)
            verified_target = get_verified_core_target(source, vendor, category)
            can_seed_now = bool(existing_disabled or verified_target or auto_seedable)
            if decision.fit == SourceFit.conditional.value:
                key = (vendor, source)
                if key not in enabled_by_vendor_source:
                    candidate = {
                        "vendor_name": vendor,
                        "product_category": category,
                        "inventory_source": row.get("inventory_source"),
                        "source": source,
                        "source_tier": _source_yield_tier(source),
                        "vertical": decision.vertical,
                        "source_fit": decision.fit,
                        "reason": decision.reason,
                        "total_reviews_analyzed": row["total_reviews_analyzed"],
                        "confidence_score": row["confidence_score"],
                        "auto_seedable": auto_seedable,
                        "requires_product_slug": not auto_seedable,
                        "suggested_product_slug": _slugify_vendor(vendor) if auto_seedable else None,
                        "can_probation_now": can_seed_now,
                        "verified_product_slug": (verified_target or {}).get("product_slug"),
                        "verified_product_name": (verified_target or {}).get("product_name"),
                        "existing_disabled_target_id": str(existing_disabled.get("id")) if existing_disabled else None,
                        "existing_disabled_product_slug": existing_disabled.get("product_slug") if existing_disabled else None,
                    }
                    if source in deferred_inventory_sources and not candidate["can_probation_now"]:
                        deferred_conditional_inventory.append(candidate)
                    else:
                        conditional_opportunities.append(candidate)
                continue
            if decision.fit != SourceFit.core.value:
                continue
            key = (vendor, source)
            if key in enabled_by_vendor_source:
                continue
            missing_core_targets.append(
                {
                    "vendor_name": vendor,
                    "product_category": category,
                    "inventory_source": row.get("inventory_source"),
                    "source": source,
                    "source_tier": _source_yield_tier(source),
                    "vertical": decision.vertical,
                    "source_fit": decision.fit,
                    "reason": decision.reason,
                    "total_reviews_analyzed": row["total_reviews_analyzed"],
                    "confidence_score": row["confidence_score"],
                    "auto_seedable": auto_seedable,
                    "requires_product_slug": not auto_seedable,
                    "suggested_product_slug": _slugify_vendor(vendor) if auto_seedable else None,
                    "can_seed_now": bool(existing_disabled or verified_target or auto_seedable),
                    "verified_product_slug": (verified_target or {}).get("product_slug"),
                    "verified_product_name": (verified_target or {}).get("product_name"),
                    "existing_disabled_target_id": str(existing_disabled.get("id")) if existing_disabled else None,
                    "existing_disabled_product_slug": existing_disabled.get("product_slug") if existing_disabled else None,
                }
            )

    missing_by_source = Counter(item["source"] for item in missing_core_targets)
    missing_by_tier = Counter(item["source_tier"] for item in missing_core_targets)
    missing_seedable_by_tier = Counter(
        item["source_tier"] for item in missing_core_targets if item["can_seed_now"]
    )
    poor_fit_by_source = Counter(item["source"] for item in poor_fit_enabled_targets)
    poor_fit_by_tier = Counter(item["source_tier"] for item in poor_fit_enabled_targets)
    conditional_by_source = Counter(item["source"] for item in conditional_opportunities)
    conditional_by_tier = Counter(item["source_tier"] for item in conditional_opportunities)
    conditional_seedable_by_tier = Counter(
        item["source_tier"] for item in conditional_opportunities if item["can_probation_now"]
    )
    deferred_conditional_by_source = Counter(item["source"] for item in deferred_conditional_inventory)
    deferred_conditional_by_tier = Counter(item["source_tier"] for item in deferred_conditional_inventory)
    inventory_breakdown = Counter(
        str(row.get("inventory_source") or "b2b_product_profiles")
        for row in collapsed_inventory
    )
    inventory_sources = sorted(inventory_breakdown)

    return {
        "inventory_source": inventory_sources[0] if len(inventory_sources) == 1 else "mixed",
        "inventory_source_breakdown": dict(sorted(inventory_breakdown.items())),
        "vendors_considered": len(collapsed_inventory),
        "enabled_targets_considered": len(enabled_by_vendor_source),
        "summary": {
            "missing_core_targets": len(missing_core_targets),
            "conditional_opportunities": len(conditional_opportunities),
            "deferred_conditional_inventory": len(deferred_conditional_inventory),
            "poor_fit_enabled_targets": len(poor_fit_enabled_targets),
            "missing_core_auto_seedable": sum(1 for item in missing_core_targets if item["auto_seedable"]),
            "missing_core_seedable_now": sum(1 for item in missing_core_targets if item["can_seed_now"]),
            "conditional_probation_seedable_now": sum(
                1 for item in conditional_opportunities if item["can_probation_now"]
            ),
            "missing_core_by_source": dict(sorted(missing_by_source.items())),
            "missing_core_by_tier": dict(sorted(missing_by_tier.items())),
            "missing_core_seedable_now_by_tier": dict(sorted(missing_seedable_by_tier.items())),
            "conditional_by_source": dict(sorted(conditional_by_source.items())),
            "conditional_by_tier": dict(sorted(conditional_by_tier.items())),
            "conditional_probation_seedable_now_by_tier": dict(sorted(conditional_seedable_by_tier.items())),
            "deferred_conditional_by_source": dict(sorted(deferred_conditional_by_source.items())),
            "deferred_conditional_by_tier": dict(sorted(deferred_conditional_by_tier.items())),
            "poor_fit_by_source": dict(sorted(poor_fit_by_source.items())),
            "poor_fit_by_tier": dict(sorted(poor_fit_by_tier.items())),
        },
        "missing_core_targets": sorted(
            missing_core_targets,
            key=lambda item: (
                -item["total_reviews_analyzed"],
                _source_yield_priority(item["source"]),
                item["vendor_name"],
                item["source"],
            ),
        ),
        "conditional_opportunities": sorted(
            conditional_opportunities,
            key=lambda item: (
                not item["can_probation_now"],
                _source_yield_priority(item["source"]),
                -item["total_reviews_analyzed"],
                item["vendor_name"],
                item["source"],
            ),
        ),
        "deferred_conditional_inventory": sorted(
            deferred_conditional_inventory,
            key=lambda item: (
                _source_yield_priority(item["source"]),
                -item["total_reviews_analyzed"],
                item["vendor_name"],
                item["source"],
            ),
        ),
        "poor_fit_enabled_targets": sorted(
            poor_fit_enabled_targets,
            key=lambda item: (item["vendor_name"], item["source"]),
        ),
    }
