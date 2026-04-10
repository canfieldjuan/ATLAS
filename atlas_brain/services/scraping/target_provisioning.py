"""Shared helpers for scrape coverage planning and target provisioning."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from ...config import settings
from ..vendor_registry import resolve_vendor_name
from .sources import SEARCH_SOURCES, filter_deprecated_sources, parse_source_allowlist
from .target_planning import (
    build_scrape_coverage_plan,
    collapse_inventory_rows,
    get_verified_core_target,
)
from .target_validation import validate_target_input

_SEARCH_SOURCE_VALUES = frozenset(member.value for member in SEARCH_SOURCES)


def _safe_target_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


async def fetch_coverage_inputs(pool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from ...autonomous.tasks._b2b_shared import read_vendor_scorecard_inventory_rows

    profile_rows = await pool.fetch(
        """
        SELECT vendor_name, product_category, total_reviews_analyzed,
               confidence_score, last_computed_at,
               'b2b_product_profiles' AS inventory_source
        FROM b2b_product_profiles
        """
    )
    signal_rows = await read_vendor_scorecard_inventory_rows(pool)
    existing_targets = await pool.fetch(
        """
        SELECT id, source, vendor_name, product_name, product_category, product_slug,
               enabled, scrape_mode, priority, max_pages, scrape_interval_hours, metadata
        FROM b2b_scrape_targets
        """
    )
    targets = []
    for row in existing_targets:
        item = dict(row)
        item["metadata"] = _safe_target_metadata(item.get("metadata"))
        targets.append(item)
    inventory = [dict(row) for row in profile_rows]
    inventory.extend(dict(row) for row in signal_rows)
    return inventory, targets


def derive_seed_defaults(
    existing_targets: list[dict[str, Any]],
    source: str,
    product_category: str | None,
) -> dict[str, Any]:
    same_group = [
        row for row in existing_targets
        if str(row.get("source") or "").strip().lower() == source
        and str(row.get("product_category") or "").strip() == str(product_category or "").strip()
    ]

    def _pick(name: str, default: Any) -> Any:
        counts = Counter(row.get(name) for row in same_group if row.get(name) is not None)
        return counts.most_common(1)[0][0] if counts else default

    return {
        "priority": int(_pick("priority", 0)),
        "max_pages": int(_pick("max_pages", 5)),
        "scrape_interval_hours": int(_pick("scrape_interval_hours", 168)),
        "scrape_mode": str(_pick("scrape_mode", "incremental")),
    }


async def apply_missing_core_targets(
    pool,
    existing_targets: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    for item in candidates:
        if item.get("existing_disabled_target_id"):
            action = {
                "action": "enable_existing",
                "target_id": item["existing_disabled_target_id"],
                "vendor_name": item["vendor_name"],
                "source": item["source"],
                "product_slug": item.get("existing_disabled_product_slug"),
            }
            if not dry_run:
                await pool.execute(
                    "UPDATE b2b_scrape_targets SET enabled = true, updated_at = NOW() WHERE id = $1",
                    item["existing_disabled_target_id"],
                )
            applied.append(action)
            continue

        product_slug = item.get("verified_product_slug") or item.get("suggested_product_slug")
        if not product_slug:
            continue
        source, product_slug = validate_target_input(item["source"], product_slug)
        defaults = derive_seed_defaults(existing_targets, source, item.get("product_category"))
        product_name = item.get("verified_product_name") or item["vendor_name"]
        metadata: dict[str, Any] = {}
        if item.get("source_fit_probation"):
            metadata = {
                "source_fit_probation": True,
                "source_fit_override": "conditional_onboarding_signal",
                "onboarding_lane": item.get("onboarding_lane") or "signal",
            }
        action = {
            "action": "insert_target",
            "vendor_name": item["vendor_name"],
            "source": source,
            "product_slug": product_slug,
            "product_name": product_name,
            "product_category": item.get("product_category"),
            **defaults,
        }
        if metadata:
            action["metadata"] = metadata
        if not dry_run:
            vendor_name = await resolve_vendor_name(item["vendor_name"])
            row = await pool.fetchrow(
                """
                INSERT INTO b2b_scrape_targets
                     (source, vendor_name, product_name, product_slug, product_category,
                      max_pages, enabled, priority, scrape_interval_hours, scrape_mode, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, true, $7, $8, $9, $10::jsonb)
                RETURNING id
                """,
                source,
                vendor_name,
                product_name,
                product_slug,
                item.get("product_category"),
                defaults["max_pages"],
                defaults["priority"],
                defaults["scrape_interval_hours"],
                defaults["scrape_mode"],
                json.dumps(metadata),
            )
            action["target_id"] = str(row["id"]) if row else None
        applied.append(action)
    return applied


def _is_signal_lane_source(source: str | None) -> bool:
    return (source or "").strip().lower() in _SEARCH_SOURCE_VALUES


def _split_onboarding_candidates(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    core: list[dict[str, Any]] = []
    signal_lane: list[dict[str, Any]] = []
    for item in items:
        if _is_signal_lane_source(item.get("source")):
            signal_lane.append(item)
        else:
            core.append(item)
    return core, signal_lane


async def provision_missing_core_targets_for_vendors(
    pool,
    vendor_names: list[str],
    *,
    dry_run: bool = False,
    limit: int = 200,
) -> dict[str, Any]:
    requested_vendors = {
        str(value or "").strip().lower()
        for value in vendor_names
        if str(value or "").strip()
    }
    if not requested_vendors:
        return {
            "status": "noop",
            "requested": 0,
            "applied": 0,
            "matched_vendors": [],
            "unmatched_vendors": [],
            "actions": [],
        }

    inventory_rows, existing_targets = await fetch_coverage_inputs(pool)
    allowed_sources = filter_deprecated_sources(
        parse_source_allowlist(settings.b2b_scrape.source_allowlist),
        settings.b2b_scrape.deprecated_sources,
    )
    plan = build_scrape_coverage_plan(
        inventory_rows,
        existing_targets,
        allowed_sources=allowed_sources,
    )
    collapsed = collapse_inventory_rows(inventory_rows)
    matched_vendors = sorted(
        row["vendor_name"]
        for row in collapsed
        if str(row.get("vendor_name") or "").strip().lower() in requested_vendors
    )
    matched_lookup = {value.lower() for value in matched_vendors}
    candidates = [
        item for item in plan["missing_core_targets"]
        if item.get("can_seed_now")
        and str(item.get("vendor_name") or "").strip().lower() in requested_vendors
    ][:limit]
    actions = await apply_missing_core_targets(
        pool,
        existing_targets,
        candidates,
        dry_run=dry_run,
    )
    if dry_run:
        status = "dry_run"
    elif actions:
        status = "applied"
    else:
        status = "noop"
    return {
        "status": status,
        "requested": len(candidates),
        "applied": len(actions),
        "matched_vendors": matched_vendors,
        "unmatched_vendors": sorted(
            value for value in requested_vendors if value not in matched_lookup
        ),
        "inventory_source": plan.get("inventory_source"),
        "inventory_source_breakdown": plan.get("inventory_source_breakdown", {}),
        "actions": actions,
    }


def _normalize_source_slug_overrides(value: dict[str, str] | None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_source, raw_slug in (value or {}).items():
        source = str(raw_source or "").strip().lower()
        slug = str(raw_slug or "").strip()
        if not source or not slug:
            continue
        normalized[source] = slug
    return normalized


def _bootstrap_inventory_row(vendor_name: str, product_category: str) -> dict[str, Any]:
    return {
        "vendor_name": vendor_name,
        "product_category": product_category,
        "total_reviews_analyzed": 0,
        "confidence_score": 0.0,
        "last_computed_at": None,
        "inventory_source": "manual_bootstrap",
    }


async def provision_vendor_onboarding_targets(
    pool,
    vendor_name: str,
    *,
    product_category: str | None = None,
    source_slug_overrides: dict[str, str] | None = None,
    dry_run: bool = False,
    limit: int = 200,
) -> dict[str, Any]:
    canonical_vendor = await resolve_vendor_name(vendor_name)
    inventory_rows, existing_targets = await fetch_coverage_inputs(pool)
    collapsed = collapse_inventory_rows(inventory_rows)
    known_vendors = {
        str(row.get("vendor_name") or "").strip().lower()
        for row in collapsed
    }
    vendor_known = canonical_vendor.lower() in known_vendors
    bootstrap_category = str(product_category or "").strip()
    bootstrap_used = not vendor_known and bool(bootstrap_category)
    if bootstrap_used:
        inventory_rows = list(inventory_rows)
        inventory_rows.append(_bootstrap_inventory_row(canonical_vendor, bootstrap_category))

    allowed_sources = filter_deprecated_sources(
        parse_source_allowlist(settings.b2b_scrape.source_allowlist),
        settings.b2b_scrape.deprecated_sources,
    )
    plan = build_scrape_coverage_plan(
        inventory_rows,
        existing_targets,
        allowed_sources=allowed_sources,
    )
    slug_overrides = _normalize_source_slug_overrides(source_slug_overrides)
    candidates: list[dict[str, Any]] = []
    for item in plan["missing_core_targets"]:
        if str(item.get("vendor_name") or "").strip().lower() != canonical_vendor.lower():
            continue
        candidate = dict(item)
        override_slug = slug_overrides.get(str(candidate.get("source") or "").strip().lower())
        if override_slug:
            source, validated_slug = validate_target_input(candidate["source"], override_slug)
            candidate["source"] = source
            verified_target = get_verified_core_target(
                source, canonical_vendor, candidate.get("product_category")
            )
            candidate["verified_product_slug"] = validated_slug
            candidate["verified_product_name"] = (
                (verified_target or {}).get("product_name")
                or candidate.get("verified_product_name")
                or canonical_vendor
            )
            candidate["can_seed_now"] = True
        elif candidate.get("source") in {member.value for member in SEARCH_SOURCES}:
            candidate["can_seed_now"] = True
        if candidate.get("can_seed_now"):
            candidates.append(candidate)
        if len(candidates) >= limit:
            break

    has_verified_or_slug_seed = any(
        not _is_signal_lane_source(item.get("source"))
        for item in candidates
    )
    if not has_verified_or_slug_seed:
        for item in plan["conditional_opportunities"]:
            if str(item.get("vendor_name") or "").strip().lower() != canonical_vendor.lower():
                continue
            if not item.get("can_probation_now"):
                continue
            if not _is_signal_lane_source(item.get("source")):
                continue
            candidate = dict(item)
            candidate["source_fit_probation"] = True
            candidate["onboarding_lane"] = "signal"
            candidates.append(candidate)
            if len(candidates) >= limit:
                break

    actions = await apply_missing_core_targets(
        pool,
        existing_targets,
        candidates,
        dry_run=dry_run,
    )
    core_candidates, signal_candidates = _split_onboarding_candidates(candidates)
    core_actions, signal_actions = _split_onboarding_candidates(actions)
    vendor_inventory_source = "manual_bootstrap" if bootstrap_used else next(
        (
            str(row.get("inventory_source") or "unknown")
            for row in collapse_inventory_rows(inventory_rows)
            if str(row.get("vendor_name") or "").strip().lower() == canonical_vendor.lower()
        ),
        "unknown",
    )
    status = "dry_run" if dry_run else ("applied" if actions else "noop")
    return {
        "status": status,
        "requested": len(candidates),
        "applied": len(actions),
        "matched_vendors": [canonical_vendor] if vendor_known or bootstrap_used else [],
        "unmatched_vendors": [] if vendor_known or bootstrap_used else [canonical_vendor.lower()],
        "inventory_source": vendor_inventory_source,
        "inventory_source_breakdown": plan.get("inventory_source_breakdown", {}),
        "bootstrap_used": bootstrap_used,
        "requested_core_targets": len(core_candidates),
        "requested_signal_targets": len(signal_candidates),
        "applied_core_targets": len(core_actions),
        "applied_signal_targets": len(signal_actions),
        "actions": actions,
    }
