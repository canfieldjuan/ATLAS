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
    requested_sources = set(sources or parser_versions.keys())
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
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser


def _render_result(result: dict[str, Any], *, emit_json: bool) -> None:
    if emit_json:
        print(json.dumps(result, indent=2, default=str, sort_keys=True))
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
        candidates = [
            item
            for item in candidates
            if not _is_in_recent_cooldown(
                item.get("last_scraped_at"),
                cooldown_hours=recent_cooldown_hours,
            )
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
            for item in runnable_targets[:run_limit]:
                run_mode = str(args.run_now_mode or "auto").strip().lower()
                if run_mode == "direct":
                    if int(args.run_max_pages or 0) > 0 or str(args.run_scrape_mode or "").strip():
                        result = await _trigger_target_run_direct_bounded(
                            str(item["target_id"]),
                            max_pages=int(args.run_max_pages or 0) or None,
                            scrape_mode=str(args.run_scrape_mode or "").strip() or None,
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
                        if int(args.run_max_pages or 0) > 0 or str(args.run_scrape_mode or "").strip():
                            direct_result = await _trigger_target_run_direct_bounded(
                                str(item["target_id"]),
                                max_pages=int(args.run_max_pages or 0) or None,
                                scrape_mode=str(args.run_scrape_mode or "").strip() or None,
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
                        "result": result,
                    }
                )
        return {
            "sources": sources or sorted({str(item.get("source") or "") for item in candidates}),
            "requested_targets": len(candidates),
            "applied": applied,
            "run_started": len(run_results),
            "targets": selected_targets,
            "run_results": run_results,
        }
    finally:
        await conn.close()


def main() -> None:
    args = _build_parser().parse_args()
    result = asyncio.run(_run(args))
    _render_result(result, emit_json=args.json)


if __name__ == "__main__":
    main()
