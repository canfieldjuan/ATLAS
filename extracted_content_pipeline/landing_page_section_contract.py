"""Canonical landing-page section metadata contract."""

from __future__ import annotations

import re
from typing import Any


LANDING_PAGE_SECTION_KINDS = (
    "problem",
    "solution",
    "how_it_works",
    "proof",
    "pricing",
    "faq",
    "objection",
    "conversion",
)

LANDING_PAGE_QUESTION_SECTION_KINDS = (
    "problem",
    "solution",
    "how_it_works",
    "faq",
    "objection",
)

LANDING_PAGE_PROBLEM_SECTION_KINDS = ("problem",)
LANDING_PAGE_SOLUTION_SECTION_KINDS = ("solution", "how_it_works")
LANDING_PAGE_OBJECTION_SECTION_KINDS = (
    "faq",
    "objection",
    "pricing",
    "proof",
    "how_it_works",
)


def normalize_landing_page_section_kind(value: Any) -> str:
    """Normalize a landing-page section kind for readiness scoring."""

    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


__all__ = [
    "LANDING_PAGE_OBJECTION_SECTION_KINDS",
    "LANDING_PAGE_PROBLEM_SECTION_KINDS",
    "LANDING_PAGE_QUESTION_SECTION_KINDS",
    "LANDING_PAGE_SECTION_KINDS",
    "LANDING_PAGE_SOLUTION_SECTION_KINDS",
    "normalize_landing_page_section_kind",
]
