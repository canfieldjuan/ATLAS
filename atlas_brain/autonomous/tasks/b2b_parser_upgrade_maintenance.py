"""Scheduled parser-upgrade maintenance for healthy B2B scrape sources."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.b2b_parser_upgrade_maintenance")

_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import plan_parser_upgrade_rescrape_targets as parser_upgrade_runner  # noqa: E402


async def run(task: ScheduledTask) -> dict:
    """Drain parser-upgrade maintenance work for healthy structured sources."""
    cfg = settings.b2b_scrape
    if not cfg.parser_upgrade_maintenance_enabled:
        return {"_skip_synthesis": "Parser-upgrade maintenance disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    args = parser_upgrade_runner._build_parser().parse_args([])
    args.sources = cfg.parser_upgrade_maintenance_sources
    args.limit_targets = int(cfg.parser_upgrade_maintenance_limit_targets)
    args.run_now = True
    args.run_limit = 0
    args.run_now_mode = "direct"
    args.run_max_pages = int(cfg.parser_upgrade_maintenance_run_max_pages)
    args.run_scrape_mode = str(cfg.parser_upgrade_maintenance_run_scrape_mode or "").strip() or "exhaustive"
    args.deep_sources = str(getattr(cfg, "parser_upgrade_maintenance_deep_sources", "") or "")
    args.deep_min_parser_backlog_reviews = int(
        getattr(cfg, "parser_upgrade_maintenance_deep_min_parser_backlog_reviews", 20)
    )
    args.deep_run_max_pages = int(getattr(cfg, "parser_upgrade_maintenance_deep_run_max_pages", 0) or 0)
    args.deep_min_stable_pages_scraped = int(
        getattr(cfg, "parser_upgrade_maintenance_deep_min_stable_pages_scraped", 0) or 0
    )
    args.deep_max_targets_per_batch = int(
        getattr(cfg, "parser_upgrade_maintenance_deep_max_targets_per_batch", 0) or 0
    )
    args.recent_cooldown_hours = int(cfg.parser_upgrade_maintenance_recent_cooldown_hours)
    args.min_target_parser_backlog_reviews = int(
        getattr(cfg, "parser_upgrade_maintenance_min_target_parser_backlog_reviews", 0) or 0
    )
    args.min_source_parser_backlog_reviews = int(
        getattr(cfg, "parser_upgrade_maintenance_min_source_parser_backlog_reviews", 0) or 0
    )
    args.include_blocked = False
    args.apply = False
    args.base_url = "http://127.0.0.1:8000"
    args.drain = True
    args.drain_max_batches = int(cfg.parser_upgrade_maintenance_drain_max_batches)
    args.json = False

    result = await parser_upgrade_runner._run_drain(args)
    logger.info(
        "Parser-upgrade maintenance complete sources=%s batches=%d remaining=%d deferred_noop=%d deferred_blocked=%d filtered_low_backlog_targets=%d filtered_low_backlog_sources=%d run_started=%d",
        ",".join(result.get("sources") or []),
        int(result.get("batches_run") or 0),
        int(result.get("requested_targets") or 0),
        int(result.get("deferred_noop_targets") or 0),
        int(result.get("deferred_blocked_targets") or 0),
        int(result.get("filtered_low_backlog_targets") or 0),
        int(result.get("filtered_low_backlog_sources") or 0),
        int(result.get("run_started") or 0),
    )
    result["_skip_synthesis"] = True
    return result
