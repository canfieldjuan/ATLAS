"""Shared utilities for B2B Churn MCP domain modules."""

import json
import logging
import uuid as _uuid

logger = logging.getLogger("atlas.mcp.b2b_churn")

VALID_REPORT_TYPES = (
    "weekly_churn_feed",
    "vendor_scorecard",
    "displacement_report",
    "category_overview",
    "exploratory_overview",
    "vendor_comparison",
    "account_comparison",
    "account_deep_dive",
    "vendor_retention",
    "challenger_intel",
    "battle_card",
    "accounts_in_motion",
)

from atlas_brain.services.scraping.sources import ALL_SOURCES, ReviewSource  # noqa: E402

VALID_SOURCES = ALL_SOURCES

from atlas_brain.services.b2b.corrections import (  # noqa: E402
    suppress_predicate as _suppress_predicate,
    apply_field_overrides as _apply_field_overrides,
)


def get_pool():
    """Get the initialized DB pool."""
    from atlas_brain.storage.database import get_db_pool
    return get_db_pool()


def _is_uuid(value: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        _uuid.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def _safe_json(val):
    """Return val if it's already a list/dict, else try json.loads, else return val as-is."""
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass
    return val
