#!/usr/bin/env python3
"""Plan or apply targeted scrape-target upgrades for parser-version drift.

This is intentionally a scrape-target campaign tool, not a global enrichment
requeue. Parser fixes live on the review row, so old rows need rescrape or
historical cleanup before re-enrichment has any value.

Usage:
  python scripts/plan_parser_upgrade_rescrape_targets.py
  python scripts/plan_parser_upgrade_rescrape_targets.py --sources g2,gartner --limit-targets 10
  python scripts/plan_parser_upgrade_rescrape_targets.py --sources software_advice --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID
from urllib import error as urllib_error
from urllib import request as urllib_request

import asyncpg

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)

from atlas_brain.config import settings
from atlas_brain.services.scraping.parsers import get_all_parsers
from atlas_brain.storage.config import db_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("plan_parser_upgrade_rescrape_targets")


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().lower() for item in value.split(",") if item and item.strip()]


def _coerce_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return dict(parsed)
    return {}


def _source_rank_weight(source: str) -> int:
    normalized = str(source or "").strip().lower()
    if normalized == "trustradius":
        return 5
    if normalized == "g2":
        return 4
    if normalized in {"gartner", "capterra"}:
        return 3
    if normalized in {"software_advice", "peerspot", "sourceforge"}:
        return 2
    if normalized == "trustpilot":
        return 1
    return 0


def _recent_scrape_penalty(last_scraped_at: Any, *, cooldown_hours: int = 12) -> int:
    if not last_scraped_at:
        return 0
    if isinstance(last_scraped_at, str):
        try:
            parsed = datetime.fromisoformat(last_scraped_at.replace("Z", "+00:00"))
        except ValueError:
            return 0
    else:
        parsed = last_scraped_at
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(0, int(cooldown_hours)))
    return -1 if parsed >= cutoff else 0


def _is_in_recent_cooldown(last_scraped_at: Any, *, cooldown_hours: int = 12) -> bool:
    if not last_scraped_at:
        return False
    if isinstance(last_scraped_at, str):
        try:
            parsed = datetime.fromisoformat(last_scraped_at.replace("Z", "+00:00"))
        except ValueError:
            return False
    else:
        parsed = last_scraped_at
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(0, int(cooldown_hours)))
    return parsed >= cutoff


def _target_priority_score(item: dict[str, Any]) -> tuple[int, int, int, str, str]:
    return (
        _recent_scrape_penalty(item.get("last_scraped_at")),
        _source_rank_weight(str(item.get("source") or "")),
        int(item.get("outdated_reviews") or 0),
        int(item.get("missing_parser_version_reviews") or 0),
        str(item.get("vendor_name") or ""),
        str(item.get("target_id") or ""),
    )


def _parser_backlog_reviews(item: dict[str, Any]) -> int:
    return int(item.get("outdated_reviews") or 0) + int(item.get("missing_parser_version_reviews") or 0)


def _aggregate_source_backlog_reviews(items: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for item in items:
        source = str(item.get("source") or "").strip().lower()
        if not source:
            continue
        totals[source] = totals.get(source, 0) + _parser_backlog_reviews(item)
    return totals


def _is_recent_exhaustive_zero_insert_run(
    item: dict[str, Any],
    *,
    cooldown_hours: int,
) -> bool:
    status = str(item.get("last_scrape_status") or "").strip().lower()
    reviews_inserted = int(item.get("last_scrape_reviews_inserted") or 0)
    pages_scraped = int(item.get("last_scrape_pages_scraped") or 0)
    reviews_found = int(item.get("last_scrape_reviews_found") or 0)
    if status not in {"success", "partial"}:
        return False
    if reviews_inserted != 0:
        return False
    if not (reviews_found <= 0 or pages_scraped >= 3):
        return False
    return _is_in_recent_cooldown(item.get("last_scraped_at"), cooldown_hours=cooldown_hours)


async def _load_recent_stalled_partial_target_ids(
    conn: asyncpg.Connection,
    *,
    candidates: list[dict[str, Any]],
    cooldown_hours: int,
) -> set[str]:
    if not candidates:
        return set()
    current_versions = {
        str(item.get("target_id") or ""): str(item.get("current_parser_version") or "").strip()
        for item in candidates
        if str(item.get("target_id") or "").strip()
    }
    target_ids = [uuid.UUID(target_id) for target_id in current_versions]
    rows = await conn.fetch(
        """
        WITH ranked AS (
            SELECT
                target_id,
                started_at,
                status,
                stop_reason,
                reviews_inserted,
                pages_scraped,
                parser_version,
                ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY started_at DESC) AS rn
            FROM b2b_scrape_log
            WHERE target_id = ANY($1::uuid[])
              AND COALESCE(status, '') NOT LIKE 'skipped%'
        )
        SELECT
            target_id,
            started_at,
            status,
            stop_reason,
            reviews_inserted,
            pages_scraped,
            parser_version,
            rn
        FROM ranked
        WHERE rn <= 2
        ORDER BY target_id, rn
        """,
        target_ids,
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        target_id = str(row["target_id"])
        grouped.setdefault(target_id, []).append(dict(row))

    stalled: set[str] = set()
    for target_id, target_rows in grouped.items():
        if len(target_rows) < 2:
            continue
        current_version = current_versions.get(target_id, "")
        if not current_version:
            continue
        latest = target_rows[0]
        if not _is_in_recent_cooldown(latest.get("started_at"), cooldown_hours=cooldown_hours):
            continue
        if str(latest.get("stop_reason") or "").strip().lower() != "pages_exhausted":
            continue
        if int(latest.get("pages_scraped") or 0) > 1:
            continue
        if any(str(row.get("parser_version") or "").strip() != current_version for row in target_rows):
            continue
        if any(str(row.get("status") or "").strip().lower() not in {"success", "partial"} for row in target_rows):
            continue
        if any(int(row.get("reviews_inserted") or 0) != 0 for row in target_rows):
            continue
        stalled.add(target_id)
    return stalled


def _has_stable_pagination_signal(
    item: dict[str, Any],
    *,
    min_pages_scraped: int,
    baseline_run_max_pages: int | None,
) -> bool:
    pages_scraped = int(item.get("last_scrape_pages_scraped") or 0)
    reviews_found = int(item.get("last_scrape_reviews_found") or 0)
    stop_reason = str(item.get("last_scrape_stop_reason") or "").strip().lower()
    status = str(item.get("last_scrape_status") or "").strip().lower()
    required_pages = max(1, int(min_pages_scraped or 0))
    if baseline_run_max_pages:
        required_pages = max(required_pages, int(baseline_run_max_pages))
    return (
        status in {"success", "partial"}
        and pages_scraped >= required_pages
        and reviews_found > 0
        and stop_reason in {"page_cap", "pages_exhausted"}
    )


def _effective_run_overrides(
    item: dict[str, Any],
    args: argparse.Namespace,
    *,
    deep_targets_used: int,
) -> tuple[int | None, str | None, bool]:
    default_max_pages = int(args.run_max_pages or 0) or None
    default_scrape_mode = str(args.run_scrape_mode or "").strip() or None

    deep_sources = set(_parse_csv(getattr(args, "deep_sources", "")))
    deep_min_backlog = max(0, int(getattr(args, "deep_min_parser_backlog_reviews", 0) or 0))
    deep_run_max_pages = int(getattr(args, "deep_run_max_pages", 0) or 0)
    deep_min_stable_pages_scraped = max(1, int(getattr(args, "deep_min_stable_pages_scraped", 1) or 1))
    deep_max_targets = max(0, int(getattr(args, "deep_max_targets_per_batch", 0) or 0))

    qualifies_for_deep = (
        deep_targets_used < deep_max_targets
        and deep_run_max_pages > 0
        and str(item.get("source") or "").strip().lower() in deep_sources
        and _parser_backlog_reviews(item) >= deep_min_backlog
        and _has_stable_pagination_signal(
            item,
            min_pages_scraped=deep_min_stable_pages_scraped,
            baseline_run_max_pages=default_max_pages,
        )
    )
    if not qualifies_for_deep:
        return default_max_pages, default_scrape_mode, False

    effective_mode = default_scrape_mode or "exhaustive"
    effective_max_pages = max(int(default_max_pages or 0), deep_run_max_pages) or None
    return effective_max_pages, effective_mode, True


def _campaign_metadata(
    metadata: Any,
    *,
    current_version: str,
    outdated_reviews: int,
    missing_reviews: int,
    requested_at: datetime,
) -> dict[str, Any]:
    updated = _coerce_json_dict(metadata)
    updated["parser_upgrade_rescrape"] = {
        "requested": True,
        "requested_at": requested_at.astimezone(timezone.utc).isoformat(),
        "current_parser_version": current_version,
        "outdated_reviews": int(outdated_reviews),
        "missing_parser_version_reviews": int(missing_reviews),
        "reason": "parser_version_drift",
    }
    return updated


def _is_runnable_target(item: dict[str, Any], *, include_blocked: bool) -> bool:
    if not bool(item.get("enabled")):
        return False
    status = str(item.get("last_scrape_status") or "").strip().lower()
    if status == "blocked" and not include_blocked:
        return False
    return True


def _is_blocked_target(item: dict[str, Any]) -> bool:
    return str(item.get("last_scrape_status") or "").strip().lower() == "blocked"


def _mark_blocked_parser_upgrade_deferred(
    metadata: Any,
    *,
    deferred_at: datetime,
    reason: str,
) -> dict[str, Any]:
    updated = _coerce_json_dict(metadata)
    parser_upgrade = _coerce_json_dict(updated.get("parser_upgrade_rescrape"))
    parser_upgrade["blocked_deferred_at"] = deferred_at.astimezone(timezone.utc).isoformat()
    parser_upgrade["blocked_deferred_reason"] = reason
    updated["parser_upgrade_rescrape"] = parser_upgrade
    return updated


def _clear_blocked_parser_upgrade_deferred(metadata: Any) -> dict[str, Any]:
    updated = _coerce_json_dict(metadata)
    parser_upgrade = _coerce_json_dict(updated.get("parser_upgrade_rescrape"))
    parser_upgrade.pop("blocked_deferred_at", None)
    parser_upgrade.pop("blocked_deferred_reason", None)
    if parser_upgrade:
        updated["parser_upgrade_rescrape"] = parser_upgrade
    else:
        updated.pop("parser_upgrade_rescrape", None)
    return updated


def _is_blocked_parser_upgrade_deferred(
    metadata: Any,
    *,
    cooldown_hours: int,
) -> bool:
    parser_upgrade = _coerce_json_dict(_coerce_json_dict(metadata).get("parser_upgrade_rescrape"))
    deferred_at = parser_upgrade.get("blocked_deferred_at")
    if not deferred_at:
        return False
    return _is_in_recent_cooldown(deferred_at, cooldown_hours=cooldown_hours)


def _mark_noop_parser_upgrade_deferred(
    metadata: Any,
    *,
    deferred_at: datetime,
    parser_version: str,
    reason: str,
    reviews_found: int,
    pages_scraped: int,
) -> dict[str, Any]:
    updated = _coerce_json_dict(metadata)
    parser_upgrade = _coerce_json_dict(updated.get("parser_upgrade_rescrape"))
    parser_upgrade["noop_deferred_at"] = deferred_at.astimezone(timezone.utc).isoformat()
    parser_upgrade["noop_deferred_parser_version"] = str(parser_version or "").strip()
    parser_upgrade["noop_deferred_reason"] = reason
    parser_upgrade["noop_deferred_reviews_found"] = int(reviews_found)
    parser_upgrade["noop_deferred_pages_scraped"] = int(pages_scraped)
    updated["parser_upgrade_rescrape"] = parser_upgrade
    return updated


def _clear_noop_parser_upgrade_deferred(metadata: Any) -> dict[str, Any]:
    updated = _coerce_json_dict(metadata)
    parser_upgrade = _coerce_json_dict(updated.get("parser_upgrade_rescrape"))
    parser_upgrade.pop("noop_deferred_at", None)
    parser_upgrade.pop("noop_deferred_parser_version", None)
    parser_upgrade.pop("noop_deferred_reason", None)
    parser_upgrade.pop("noop_deferred_reviews_found", None)
    parser_upgrade.pop("noop_deferred_pages_scraped", None)
    if parser_upgrade:
        updated["parser_upgrade_rescrape"] = parser_upgrade
    else:
        updated.pop("parser_upgrade_rescrape", None)
    return updated


def _is_noop_parser_upgrade_deferred(
    metadata: Any,
    *,
    current_parser_version: str,
    cooldown_hours: int,
) -> bool:
    parser_upgrade = _coerce_json_dict(_coerce_json_dict(metadata).get("parser_upgrade_rescrape"))
    deferred_at = parser_upgrade.get("noop_deferred_at")
    deferred_version = str(parser_upgrade.get("noop_deferred_parser_version") or "").strip()
    if not deferred_at:
        return False
    if deferred_version and deferred_version != str(current_parser_version or "").strip():
        return False
    return _is_in_recent_cooldown(deferred_at, cooldown_hours=cooldown_hours)


def _extract_scrape_result_payload(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    if "reviews_inserted" in result or "reviews_found" in result:
        return result
    nested = result.get("result")
    if isinstance(nested, dict):
        payload = _extract_scrape_result_payload(nested)
        if payload is not None:
            return payload
    direct_result = result.get("direct_result")
    if isinstance(direct_result, dict):
        payload = _extract_scrape_result_payload(direct_result)
        if payload is not None:
            return payload
    api_result = result.get("api_result")
    if isinstance(api_result, dict):
        payload = _extract_scrape_result_payload(api_result)
        if payload is not None:
            return payload
    return None


def _classify_noop_maintenance_result(result: dict[str, Any] | None) -> tuple[str | None, dict[str, Any] | None]:
    payload = _extract_scrape_result_payload(result)
    if not payload:
        return None, None
    status = str(payload.get("status") or "").strip().lower()
    if status not in {"success", "partial"}:
        return None, payload
    reviews_found = int(payload.get("reviews_found") or 0)
    reviews_inserted = int(payload.get("reviews_inserted") or 0)
    duplicate_or_existing = int(payload.get("duplicate_or_existing") or 0)
    duplicate_existing = int(payload.get("duplicate_existing") or 0)
    duplicate_same_batch = int(payload.get("duplicate_same_batch") or 0)
    duplicate_db_conflict = int(payload.get("duplicate_db_conflict") or 0)
    if reviews_inserted != 0:
        return None, payload
    if reviews_found <= 0:
        return "empty_success_zero_insert", payload
    duplicate_total = duplicate_existing + duplicate_same_batch + duplicate_db_conflict
    if duplicate_or_existing >= reviews_found or duplicate_total >= reviews_found:
        return "duplicate_only_zero_insert", payload
    return None, payload


def _trigger_target_run(base_url: str, target_id: str, *, timeout_s: int = 330) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/v1/b2b/scrape/targets/{target_id}/run"
    req = urllib_request.Request(url, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=timeout_s) as resp:
            payload = resp.read().decode("utf-8")
            return json.loads(payload) if payload else {"status_code": resp.status}
    except urllib_error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(payload) if payload else {}
        except json.JSONDecodeError:
            parsed = {"detail": payload or exc.reason}
        return {
            "status": "failed",
            "http_status": exc.code,
            "error": parsed,
        }
    except Exception as exc:  # pragma: no cover - defensive runtime path
        return {
            "status": "failed",
            "error": {"detail": str(exc)},
        }


async def _trigger_target_run_direct(target_id: str) -> dict[str, Any]:
    from atlas_brain.api.b2b_scrape import trigger_scrape
    from atlas_brain.storage.database import init_database

    try:
        await init_database()
        result = await trigger_scrape(UUID(str(target_id)))
        return {
            "status": "completed",
            "mode": "direct",
            "result": result,
        }
    except Exception as exc:  # pragma: no cover - defensive runtime path
        detail = getattr(exc, "detail", None)
        return {
            "status": "failed",
            "mode": "direct",
            "error": {"detail": detail or str(exc)},
        }


async def _trigger_target_run_direct_bounded(
    target_id: str,
    *,
    max_pages: int | None,
    scrape_mode: str | None,
) -> dict[str, Any]:
    import asyncpg

    from atlas_brain.api.b2b_scrape import trigger_scrape
    from atlas_brain.storage.config import db_settings
    from atlas_brain.storage.database import init_database

    await init_database()
    conn = await asyncpg.connect(db_settings.dsn)
    original = await conn.fetchrow(
        """
        SELECT max_pages, scrape_mode, metadata
        FROM b2b_scrape_targets
        WHERE id = $1::uuid
        """,
        UUID(str(target_id)),
    )
    if not original:
        await conn.close()
        return {
            "status": "failed",
            "mode": "direct_bounded",
            "error": {"detail": "Target not found"},
        }
    original_max_pages = int(original["max_pages"] or 0)
    original_scrape_mode = str(original["scrape_mode"] or "")
    original_metadata = _coerce_json_dict(original.get("metadata"))
    try:
        if max_pages is not None or scrape_mode:
            updated_metadata = dict(original_metadata)
            effective_scrape_mode = str(scrape_mode or original_scrape_mode or "").strip().lower()
            if max_pages is not None and effective_scrape_mode == "exhaustive":
                updated_metadata["exhaustive_max_pages"] = int(max_pages)
            await conn.execute(
                """
                UPDATE b2b_scrape_targets
                SET max_pages = COALESCE($2::int, max_pages),
                    scrape_mode = COALESCE($3::text, scrape_mode),
                    metadata = $4::jsonb,
                    updated_at = NOW()
                WHERE id = $1::uuid
                """,
                UUID(str(target_id)),
                int(max_pages) if max_pages is not None else None,
                scrape_mode,
                json.dumps(updated_metadata, default=str),
            )
        result = await trigger_scrape(UUID(str(target_id)))
        return {
            "status": "completed",
            "mode": "direct_bounded",
            "result": result,
        }
    except Exception as exc:  # pragma: no cover - defensive runtime path
        detail = getattr(exc, "detail", None)
        return {
            "status": "failed",
            "mode": "direct_bounded",
            "error": {"detail": detail or str(exc)},
        }
    finally:
        await conn.execute(
            """
            UPDATE b2b_scrape_targets
            SET max_pages = $2::int,
                scrape_mode = $3::text,
                metadata = $4::jsonb,
                updated_at = NOW()
            WHERE id = $1::uuid
            """,
            UUID(str(target_id)),
            original_max_pages,
            original_scrape_mode,
            json.dumps(original_metadata, default=str),
        )
        await conn.close()


async def _sync_blocked_maintenance_metadata(conn: asyncpg.Connection, target_id: str) -> None:
    row = await conn.fetchrow(
        """
        SELECT last_scrape_status, metadata
        FROM b2b_scrape_targets
        WHERE id = $1::uuid
        """,
        UUID(str(target_id)),
    )
    if not row:
        return
    status = str(row.get("last_scrape_status") or "").strip().lower()
    metadata = row.get("metadata")
    if status == "blocked":
        updated_metadata = _mark_blocked_parser_upgrade_deferred(
            metadata,
            deferred_at=datetime.now(timezone.utc),
            reason="last_scrape_status_blocked",
        )
    else:
        updated_metadata = _clear_blocked_parser_upgrade_deferred(metadata)
    await conn.execute(
        """
        UPDATE b2b_scrape_targets
        SET metadata = $2::jsonb,
            updated_at = NOW()
        WHERE id = $1::uuid
        """,
        UUID(str(target_id)),
        json.dumps(updated_metadata, default=str),
    )


async def _sync_noop_maintenance_metadata(
    conn: asyncpg.Connection,
    target_id: str,
    *,
    current_parser_version: str,
    result: dict[str, Any],
) -> None:
    row = await conn.fetchrow(
        """
        SELECT metadata
        FROM b2b_scrape_targets
        WHERE id = $1::uuid
        """,
        UUID(str(target_id)),
    )
    if not row:
        return
    metadata = row.get("metadata")
    noop_reason, payload = _classify_noop_maintenance_result(result)
    payload_status = str((payload or {}).get("status") or "").strip().lower()
    payload_inserted = int((payload or {}).get("reviews_inserted") or 0)
    if noop_reason and payload is not None:
        updated_metadata = _mark_noop_parser_upgrade_deferred(
            metadata,
            deferred_at=datetime.now(timezone.utc),
            parser_version=current_parser_version,
            reason=noop_reason,
            reviews_found=int(payload.get("reviews_found") or 0),
            pages_scraped=int(payload.get("pages_scraped") or 0),
        )
    elif payload_status in {"success", "partial"} and payload_inserted > 0:
        updated_metadata = _clear_noop_parser_upgrade_deferred(metadata)
    else:
        return
    if updated_metadata == _coerce_json_dict(metadata):
        return
    await conn.execute(
        """
        UPDATE b2b_scrape_targets
        SET metadata = $2::jsonb,
            updated_at = NOW()
        WHERE id = $1::uuid
        """,
        UUID(str(target_id)),
        json.dumps(updated_metadata, default=str),
    )


def _should_fallback_to_direct(result: dict[str, Any]) -> bool:
    if str(result.get("status") or "").strip().lower() != "failed":
        return False
    http_status = result.get("http_status")
    error = result.get("error") or {}
    detail = str(error.get("detail") or "").strip().lower()
    if "connection refused" in detail:
        return True
    if "currently disabled by b2b scrape source governance" in detail:
        return True
    return http_status in {502, 503, 504}


async def _collect_candidates(
    conn: asyncpg.Connection,
    *,
    sources: list[str],
) -> list[dict[str, Any]]:
    parser_versions = {
        str(source).strip().lower(): getattr(parser, "version", None)
        for source, parser in get_all_parsers().items()
        if getattr(parser, "version", None)
    }
    deferred_sources = set(
        _parse_csv(getattr(settings.b2b_scrape, "parser_upgrade_deferred_sources", ""))
    )
    requested_sources = {
        source for source in set(sources or parser_versions.keys())
        if source not in deferred_sources
    }
    rows: list[dict[str, Any]] = []

    for source in sorted(requested_sources):
        current_version = parser_versions.get(source)
        if not current_version:
            continue
        query_rows = await conn.fetch(
            """
            SELECT t.id AS target_id,
                   t.source,
                   t.vendor_name,
                   t.product_name,
                   t.product_slug,
                   t.product_category,
                   t.enabled,
                   t.priority,
                   t.max_pages,
                   t.scrape_mode,
                   t.last_scraped_at,
                   t.last_scrape_status,
                   t.last_scrape_runtime_mode,
                   t.last_scrape_reviews AS last_scrape_reviews_inserted,
                   t.last_scrape_stop_reason,
                   t.last_scrape_pages_scraped,
                   t.last_scrape_reviews_found,
                   t.metadata,
                   COUNT(*) FILTER (
                       WHERE r.parser_version IS NOT NULL
                         AND r.parser_version <> $2
                         AND r.enrichment_status IN ('enriched', 'no_signal', 'quarantined')
                   ) AS outdated_reviews,
                   COUNT(*) FILTER (
                       WHERE r.parser_version IS NULL
                         AND r.enrichment_status IN ('enriched', 'no_signal', 'quarantined')
                   ) AS missing_parser_version_reviews,
                   COUNT(*) FILTER (
                       WHERE r.enrichment_status IN ('enriched', 'no_signal', 'quarantined')
                   ) AS processed_reviews
            FROM b2b_scrape_targets t
            JOIN b2b_reviews r
              ON r.source = t.source
             AND r.duplicate_of_review_id IS NULL
             AND r.raw_metadata ? 'scrape_target_id'
             AND (r.raw_metadata->>'scrape_target_id') = t.id::text
            WHERE t.source = $1
              AND t.enabled = true
            GROUP BY t.id
            HAVING COUNT(*) FILTER (
                       WHERE r.parser_version IS NOT NULL
                         AND r.parser_version <> $2
                         AND r.enrichment_status IN ('enriched', 'no_signal', 'quarantined')
                   ) > 0
                OR COUNT(*) FILTER (
                       WHERE r.parser_version IS NULL
                         AND r.enrichment_status IN ('enriched', 'no_signal', 'quarantined')
                   ) > 0
            """,
            source,
            current_version,
        )
        for row in query_rows:
            item = dict(row)
            if not bool(item.get("enabled")):
                continue
            item["current_parser_version"] = current_version
            rows.append(item)

    rows.sort(key=_target_priority_score, reverse=True)
    return rows


async def _apply_campaign(
    conn: asyncpg.Connection,
    *,
    targets: list[dict[str, Any]],
    priority_floor: int,
    max_pages_floor: int,
    force_exhaustive: bool,
) -> int:
    applied = 0
    requested_at = datetime.now(timezone.utc)
    for item in targets:
        metadata = _campaign_metadata(
            item.get("metadata"),
            current_version=str(item.get("current_parser_version") or ""),
            outdated_reviews=int(item.get("outdated_reviews") or 0),
            missing_reviews=int(item.get("missing_parser_version_reviews") or 0),
            requested_at=requested_at,
        )
        status = await conn.execute(
            """
            UPDATE b2b_scrape_targets
            SET priority = GREATEST(priority, $2::int),
                max_pages = GREATEST(max_pages, $3::int),
                scrape_mode = CASE
                    WHEN $4::boolean THEN 'exhaustive'
                    ELSE scrape_mode
                END,
                metadata = $5::jsonb,
                updated_at = NOW()
            WHERE id = $1::uuid
            """,
            item["target_id"],
            priority_floor,
            max_pages_floor,
            force_exhaustive,
            json.dumps(metadata, default=str),
        )
        applied += int(str(status).split()[-1])
    return applied


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="", help="Comma-separated sources to scope, for example g2,gartner")
    parser.add_argument("--limit-targets", type=int, default=25, help="Max ranked targets to emit or apply")
    parser.add_argument("--priority-floor", type=int, default=90, help="Minimum target priority when applying")
    parser.add_argument("--max-pages-floor", type=int, default=10, help="Minimum max_pages when applying")
    parser.add_argument(
        "--no-force-exhaustive",
        action="store_true",
        help="Do not switch selected targets to exhaustive mode during apply",
    )
    parser.add_argument("--apply", action="store_true", help="Persist campaign settings to matching scrape targets")
    parser.add_argument("--run-now", action="store_true", help="Trigger selected targets immediately through the live scrape API")
    parser.add_argument("--run-limit", type=int, default=0, help="Max selected targets to trigger immediately; default uses all selected")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Atlas API base URL for --run-now")
    parser.add_argument(
        "--recent-cooldown-hours",
        type=int,
        default=12,
        help="Skip targets scraped within this recent cooldown window",
    )
    parser.add_argument(
        "--min-target-parser-backlog-reviews",
        type=int,
        default=0,
        help="Minimum missing+outdated canonical reviews required for an individual target to be eligible",
    )
    parser.add_argument(
        "--min-source-parser-backlog-reviews",
        type=int,
        default=0,
        help="Minimum aggregate missing+outdated canonical reviews required for a source to be eligible",
    )
    parser.add_argument(
        "--run-now-mode",
        choices=("auto", "api", "direct"),
        default="auto",
        help="Execution mode for --run-now. 'auto' tries the API then falls back to in-process execution on local failures.",
    )
    parser.add_argument("--run-max-pages", type=int, default=0, help="Temporary max_pages override for direct --run-now execution only")
    parser.add_argument(
        "--run-scrape-mode",
        choices=("incremental", "exhaustive"),
        default="",
        help="Temporary scrape_mode override for direct --run-now execution only",
    )
    parser.add_argument("--include-blocked", action="store_true", help="Allow --run-now to include targets whose last scrape status is blocked")
    parser.add_argument(
        "--deep-sources",
        default="",
        help="Comma-separated sources eligible for selective deeper direct maintenance runs",
    )
    parser.add_argument(
        "--deep-min-parser-backlog-reviews",
        type=int,
        default=0,
        help="Minimum missing+outdated canonical reviews before a target qualifies for a deeper maintenance run",
    )
    parser.add_argument(
        "--deep-run-max-pages",
        type=int,
        default=0,
        help="Temporary max_pages override for qualifying deep maintenance targets",
    )
    parser.add_argument(
        "--deep-min-stable-pages-scraped",
        type=int,
        default=1,
        help="Require the most recent target scrape to have reached at least this many pages cleanly before deep maintenance is allowed",
    )
    parser.add_argument(
        "--deep-max-targets-per-batch",
        type=int,
        default=0,
        help="Maximum number of selectively deep maintenance targets per batch",
    )
    parser.add_argument(
        "--drain",
        action="store_true",
        help="Repeat selection and optional --run-now execution until the active queue is empty or the batch limit is reached",
    )
    parser.add_argument(
        "--drain-max-batches",
        type=int,
        default=25,
        help="Maximum batches to run when --drain is enabled",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser


def _render_result(result: dict[str, Any], *, emit_json: bool) -> None:
    if emit_json:
        print(json.dumps(result, indent=2, default=str, sort_keys=True))
        return
    if bool(result.get("drain")):
        logger.info(
            "drain sources=%s batches=%d remaining=%d deferred_noop=%d deferred_recent_zero_insert=%d deferred_stalled_partial=%d deferred_blocked=%d filtered_low_backlog_targets=%d filtered_low_backlog_sources=%d run_started=%d",
            ",".join(result.get("sources") or []),
            int(result.get("batches_run") or 0),
            int(result.get("requested_targets") or 0),
            int(result.get("deferred_noop_targets") or 0),
            int(result.get("deferred_recent_zero_insert_targets") or 0),
            int(result.get("deferred_stalled_partial_targets") or 0),
            int(result.get("deferred_blocked_targets") or 0),
            int(result.get("filtered_low_backlog_targets") or 0),
            int(result.get("filtered_low_backlog_sources") or 0),
            int(result.get("run_started") or 0),
        )
        for batch in result.get("batches") or []:
            logger.info(
                "batch=%s requested=%s run_started=%s deferred_noop=%s deferred_recent_zero_insert=%s deferred_stalled_partial=%s deferred_blocked=%s filtered_low_backlog_targets=%s filtered_low_backlog_sources=%s",
                batch.get("batch"),
                batch.get("requested_targets"),
                batch.get("run_started"),
                batch.get("deferred_noop_targets"),
                batch.get("deferred_recent_zero_insert_targets"),
                batch.get("deferred_stalled_partial_targets"),
                batch.get("deferred_blocked_targets"),
                batch.get("filtered_low_backlog_targets"),
                batch.get("filtered_low_backlog_sources"),
            )
        return
    logger.info(
        "sources=%s requested=%d emitted=%d applied=%d",
        ",".join(result["sources"]),
        result["requested_targets"],
        len(result["targets"]),
        int(result.get("applied") or 0),
    )
    for item in result["targets"]:
        logger.info(
            "target=%s source=%s vendor=%s outdated=%s missing=%s current=%s enabled=%s priority=%s mode=%s",
            item.get("target_id"),
            item.get("source"),
            item.get("vendor_name"),
            item.get("outdated_reviews"),
            item.get("missing_parser_version_reviews"),
            item.get("current_parser_version"),
            item.get("enabled"),
            item.get("priority"),
            item.get("scrape_mode"),
        )


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    sources = _parse_csv(args.sources)
    conn = await asyncpg.connect(db_settings.dsn)
    try:
        candidates = await _collect_candidates(conn, sources=sources)
        recent_cooldown_hours = max(0, int(args.recent_cooldown_hours))
        blocked_cooldown_hours = max(
            1,
            int(getattr(settings.b2b_scrape, "parser_upgrade_blocked_target_cooldown_hours", 168)),
        )
        noop_cooldown_hours = max(
            1,
            int(getattr(settings.b2b_scrape, "parser_upgrade_noop_target_cooldown_hours", 168)),
        )
        min_target_backlog_reviews = max(0, int(args.min_target_parser_backlog_reviews or 0))
        min_source_backlog_reviews = max(0, int(args.min_source_parser_backlog_reviews or 0))
        candidates = [
            item
            for item in candidates
            if not _is_in_recent_cooldown(
                item.get("last_scraped_at"),
                cooldown_hours=recent_cooldown_hours,
            )
        ]
        deferred_noop_targets = [
            item
            for item in candidates
            if _is_noop_parser_upgrade_deferred(
                item.get("metadata"),
                current_parser_version=str(item.get("current_parser_version") or ""),
                cooldown_hours=noop_cooldown_hours,
            )
        ]
        candidates = [
            item
            for item in candidates
            if not _is_noop_parser_upgrade_deferred(
                item.get("metadata"),
                current_parser_version=str(item.get("current_parser_version") or ""),
                cooldown_hours=noop_cooldown_hours,
            )
        ]
        deferred_recent_zero_insert_targets = [
            item
            for item in candidates
            if _is_recent_exhaustive_zero_insert_run(
                item,
                cooldown_hours=noop_cooldown_hours,
            )
        ]
        candidates = [
            item
            for item in candidates
            if not _is_recent_exhaustive_zero_insert_run(
                item,
                cooldown_hours=noop_cooldown_hours,
            )
        ]
        stalled_partial_target_ids = await _load_recent_stalled_partial_target_ids(
            conn,
            candidates=candidates,
            cooldown_hours=noop_cooldown_hours,
        )
        deferred_stalled_partial_targets = [
            item
            for item in candidates
            if str(item.get("target_id") or "") in stalled_partial_target_ids
        ]
        candidates = [
            item
            for item in candidates
            if str(item.get("target_id") or "") not in stalled_partial_target_ids
        ]
        filtered_low_backlog_targets = [
            item
            for item in candidates
            if min_target_backlog_reviews > 0
            and _parser_backlog_reviews(item) < min_target_backlog_reviews
        ]
        if min_target_backlog_reviews > 0:
            candidates = [
                item
                for item in candidates
                if _parser_backlog_reviews(item) >= min_target_backlog_reviews
            ]
        deferred_blocked_targets = [
            item
            for item in candidates
            if _is_blocked_target(item)
            and (
                _is_blocked_parser_upgrade_deferred(
                    item.get("metadata"),
                    cooldown_hours=blocked_cooldown_hours,
                )
                or not bool(args.include_blocked)
            )
        ]
        if not bool(args.include_blocked):
            candidates = [item for item in candidates if not _is_blocked_target(item)]
        elif deferred_blocked_targets:
            candidates = [
                item
                for item in candidates
                if not (
                    _is_blocked_target(item)
                    and _is_blocked_parser_upgrade_deferred(
                        item.get("metadata"),
                        cooldown_hours=blocked_cooldown_hours,
                    )
                )
            ]
        source_backlog_totals = _aggregate_source_backlog_reviews(candidates)
        low_backlog_sources = {
            source
            for source, total_backlog in source_backlog_totals.items()
            if min_source_backlog_reviews > 0 and total_backlog < min_source_backlog_reviews
        }
        filtered_low_backlog_sources = sorted(low_backlog_sources)
        if low_backlog_sources:
            candidates = [
                item
                for item in candidates
                if str(item.get("source") or "").strip().lower() not in low_backlog_sources
            ]
        selected = candidates[: max(0, int(args.limit_targets))]
        applied = 0
        if args.apply and selected:
            applied = await _apply_campaign(
                conn,
                targets=selected,
                priority_floor=int(args.priority_floor),
                max_pages_floor=int(args.max_pages_floor),
                force_exhaustive=not bool(args.no_force_exhaustive),
            )
        selected_targets = [
            {
                "target_id": str(item.get("target_id") or ""),
                "source": item.get("source"),
                "vendor_name": item.get("vendor_name"),
                "product_slug": item.get("product_slug"),
                "enabled": bool(item.get("enabled")),
                "priority": int(item.get("priority") or 0),
                "max_pages": int(item.get("max_pages") or 0),
                "scrape_mode": item.get("scrape_mode"),
                "last_scraped_at": str(item.get("last_scraped_at") or "") or None,
                "last_scrape_status": item.get("last_scrape_status"),
                "current_parser_version": item.get("current_parser_version"),
                "outdated_reviews": int(item.get("outdated_reviews") or 0),
                "missing_parser_version_reviews": int(item.get("missing_parser_version_reviews") or 0),
                "processed_reviews": int(item.get("processed_reviews") or 0),
            }
            for item in selected
        ]
        run_results: list[dict[str, Any]] = []
        if args.run_now and selected_targets:
            runnable_targets = [
                item
                for item in selected_targets
                if _is_runnable_target(item, include_blocked=bool(args.include_blocked))
            ]
            run_limit = int(args.run_limit) if int(args.run_limit) > 0 else len(runnable_targets)
            deep_targets_used = 0
            for item in runnable_targets[:run_limit]:
                run_mode = str(args.run_now_mode or "auto").strip().lower()
                effective_run_max_pages, effective_run_scrape_mode, deep_applied = _effective_run_overrides(
                    item,
                    args,
                    deep_targets_used=deep_targets_used,
                )
                if deep_applied:
                    deep_targets_used += 1
                if run_mode == "direct":
                    if effective_run_max_pages is not None or effective_run_scrape_mode:
                        result = await _trigger_target_run_direct_bounded(
                            str(item["target_id"]),
                            max_pages=effective_run_max_pages,
                            scrape_mode=effective_run_scrape_mode,
                        )
                    else:
                        result = await _trigger_target_run_direct(str(item["target_id"]))
                else:
                    result = await asyncio.to_thread(
                        _trigger_target_run,
                        str(args.base_url),
                        str(item["target_id"]),
                    )
                    if run_mode == "auto" and _should_fallback_to_direct(result):
                        if effective_run_max_pages is not None or effective_run_scrape_mode:
                            direct_result = await _trigger_target_run_direct_bounded(
                                str(item["target_id"]),
                                max_pages=effective_run_max_pages,
                                scrape_mode=effective_run_scrape_mode,
                            )
                        else:
                            direct_result = await _trigger_target_run_direct(str(item["target_id"]))
                        result = {
                            "status": direct_result.get("status"),
                            "mode": "auto_fallback_direct",
                            "api_result": result,
                            "direct_result": direct_result,
                        }
                run_results.append(
                    {
                        "target_id": item["target_id"],
                        "source": item["source"],
                        "vendor_name": item["vendor_name"],
                        "run_max_pages_applied": effective_run_max_pages,
                        "run_scrape_mode_applied": effective_run_scrape_mode,
                        "deep_maintenance_applied": deep_applied,
                        "result": result,
                    }
                )
                await _sync_noop_maintenance_metadata(
                    conn,
                    str(item["target_id"]),
                    current_parser_version=str(item.get("current_parser_version") or ""),
                    result=result,
                )
                await _sync_blocked_maintenance_metadata(conn, str(item["target_id"]))
        return {
            "sources": sources or sorted({str(item.get("source") or "") for item in candidates}),
            "requested_targets": len(candidates),
            "deferred_noop_targets": len(deferred_noop_targets),
            "deferred_recent_zero_insert_targets": len(deferred_recent_zero_insert_targets),
            "deferred_stalled_partial_targets": len(deferred_stalled_partial_targets),
            "deferred_blocked_targets": len(deferred_blocked_targets),
            "filtered_low_backlog_targets": len(filtered_low_backlog_targets),
            "filtered_low_backlog_sources": len(filtered_low_backlog_sources),
            "applied": applied,
            "run_started": len(run_results),
            "targets": selected_targets,
            "run_results": run_results,
        }
    finally:
        await conn.close()


async def _run_drain(args: argparse.Namespace) -> dict[str, Any]:
    batches: list[dict[str, Any]] = []
    max_batches = max(1, int(args.drain_max_batches or 1))

    for batch_number in range(1, max_batches + 1):
        batch_result = await _run(args)
        batches.append(
            {
                "batch": batch_number,
                "requested_targets": int(batch_result.get("requested_targets") or 0),
                "deferred_noop_targets": int(batch_result.get("deferred_noop_targets") or 0),
                "deferred_recent_zero_insert_targets": int(batch_result.get("deferred_recent_zero_insert_targets") or 0),
                "deferred_stalled_partial_targets": int(batch_result.get("deferred_stalled_partial_targets") or 0),
                "deferred_blocked_targets": int(batch_result.get("deferred_blocked_targets") or 0),
                "filtered_low_backlog_targets": int(batch_result.get("filtered_low_backlog_targets") or 0),
                "filtered_low_backlog_sources": int(batch_result.get("filtered_low_backlog_sources") or 0),
                "applied": int(batch_result.get("applied") or 0),
                "run_started": int(batch_result.get("run_started") or 0),
                "targets": list(batch_result.get("targets") or []),
                "run_results": list(batch_result.get("run_results") or []),
            }
        )
        if int(batch_result.get("requested_targets") or 0) <= 0:
            break
        if args.run_now and int(batch_result.get("run_started") or 0) <= 0:
            break

    final_state = await _run(
        argparse.Namespace(**{**vars(args), "run_now": False, "apply": False, "drain": False})
    )
    return {
        "drain": True,
        "sources": final_state.get("sources") or [],
        "batches_run": len(batches),
        "batches": batches,
        "final_state": final_state,
        "requested_targets": int(final_state.get("requested_targets") or 0),
        "deferred_noop_targets": int(final_state.get("deferred_noop_targets") or 0),
        "deferred_recent_zero_insert_targets": int(final_state.get("deferred_recent_zero_insert_targets") or 0),
        "deferred_stalled_partial_targets": int(final_state.get("deferred_stalled_partial_targets") or 0),
        "deferred_blocked_targets": int(final_state.get("deferred_blocked_targets") or 0),
        "filtered_low_backlog_targets": int(final_state.get("filtered_low_backlog_targets") or 0),
        "filtered_low_backlog_sources": int(final_state.get("filtered_low_backlog_sources") or 0),
        "applied": sum(int(batch.get("applied") or 0) for batch in batches),
        "run_started": sum(int(batch.get("run_started") or 0) for batch in batches),
        "targets": list(final_state.get("targets") or []),
        "run_results": [],
    }


def main() -> None:
    args = _build_parser().parse_args()
    runner = _run_drain if args.drain else _run
    result = asyncio.run(runner(args))
    _render_result(result, emit_json=args.json)


if __name__ == "__main__":
    main()
