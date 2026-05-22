from __future__ import annotations

import pytest

from extracted_quality_gate.landing_page_section_contract import (
    LANDING_PAGE_OBJECTION_SECTION_KINDS,
    LANDING_PAGE_PROBLEM_SECTION_KINDS,
    LANDING_PAGE_QUESTION_SECTION_KINDS,
    LANDING_PAGE_SECTION_KINDS,
    LANDING_PAGE_SOLUTION_SECTION_KINDS,
    normalize_landing_page_section_kind,
)


def test_landing_page_section_kind_groups_are_subsets_of_canonical_kinds() -> None:
    kinds = set(LANDING_PAGE_SECTION_KINDS)

    assert set(LANDING_PAGE_QUESTION_SECTION_KINDS).issubset(kinds)
    assert set(LANDING_PAGE_PROBLEM_SECTION_KINDS).issubset(kinds)
    assert set(LANDING_PAGE_SOLUTION_SECTION_KINDS).issubset(kinds)
    assert set(LANDING_PAGE_OBJECTION_SECTION_KINDS).issubset(kinds)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("How It Works", "how_it_works"),
        (" how-it-works ", "how_it_works"),
        ("FAQ", "faq"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_landing_page_section_kind(raw, expected) -> None:
    assert normalize_landing_page_section_kind(raw) == expected
