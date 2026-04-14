"""Validation helpers for B2B scrape target configuration.

These checks are intentionally structural rather than network-based. They catch
bad source names and obviously malformed product_slug values before targets are
persisted, while leaving site-specific availability checks to scrape logs.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from .sources import (
    SEARCH_SOURCES as _SEARCH_SOURCES,
    parse_source_allowlist,
    is_source_allowed,
)

_SLUG_RULES: dict[str, tuple[re.Pattern[str], str]] = {
    "g2": (re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*"), "asana"),
    "capterra": (re.compile(r"\d+/[A-Za-z0-9][A-Za-z0-9-]*"), "184581/Asana-PM"),
    "trustradius": (re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*"), "asana"),
    "gartner": (
        re.compile(
            r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*/[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*(?:/product/[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)?"
        ),
        "endpoint-protection-platforms/microsoft/product/microsoft-defender-for-endpoint",
    ),
    "peerspot": (re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*"), "asana"),
    "getapp": (
        re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*/a/[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*"),
        "project-management-planning-software/a/asana",
    ),
    "software_advice": (
        re.compile(
            r"(?:product/\d+-[A-Za-z0-9][A-Za-z0-9-]*|[A-Za-z0-9-]+/[A-Za-z0-9][A-Za-z0-9-]*-profile)"
        ),
        "project-management/asana-profile",
    ),
    "slashdot": (re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*"), "slack"),
    "producthunt": (re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*"), "asana"),
    "trustpilot": (re.compile(r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "asana.com"),
}


def validate_target_input(source: str, product_slug: str) -> tuple[str, str]:
    """Normalize and validate scrape target input.

    Returns ``(normalized_source, normalized_product_slug)``.
    Raises ``ValueError`` when the input is clearly malformed.
    """
    source_norm = (source or "").strip().lower()
    slug = (product_slug or "").strip()

    if not source_norm:
        raise ValueError("source is required")
    if not slug:
        raise ValueError("product_slug is required")

    if source_norm == "rss":
        parsed = urlparse(slug)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Invalid product_slug for rss. Expected a full http(s) feed URL.")
        return source_norm, slug

    if source_norm in _SEARCH_SOURCES:
        if any(ch in slug for ch in "\r\n\t"):
            raise ValueError(f"Invalid product_slug for {source_norm}. Control characters are not allowed.")
        return source_norm, slug

    rule = _SLUG_RULES.get(source_norm)
    if rule is None:
        return source_norm, slug

    pattern, example = rule
    if not pattern.fullmatch(slug):
        raise ValueError(
            f"Invalid product_slug for {source_norm}. Expected a format like '{example}'."
        )

    return source_norm, slug
