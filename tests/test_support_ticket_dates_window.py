"""S7: date-convention inference and the tolerant date window (C3 + M7).

Both audit defects are pinned in both error directions: day-first uploads
must not silently transpose (C3) and one blank date must not flip the
report onto the dateless x12 run-rate basis (M7) -- while sparse dates must
not fake a window either.
"""

from __future__ import annotations

from datetime import date

import pytest

from extracted_content_pipeline.support_ticket_dates import (
    DATE_CONVENTION_AMBIGUOUS,
    DATE_CONVENTION_DAY_FIRST,
    DATE_CONVENTION_MONTH_FIRST,
    DATE_CONVENTION_UNKNOWN,
    infer_support_ticket_date_convention,
    parse_support_ticket_source_date,
)
from extracted_content_pipeline.support_ticket_input_package import (
    build_support_ticket_input_package,
)


def _rows(dates: list[str]) -> list[dict[str, str]]:
    return [
        {
            "id": f"r{i}",
            "subject": "Cannot reset my password",
            "description": "I cannot reset my password from the login page.",
            "created_at": value,
        }
        for i, value in enumerate(dates)
    ]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["13/01/2026", "02/01/2026"], DATE_CONVENTION_DAY_FIRST),
        (["01/13/2026", "01/02/2026"], DATE_CONVENTION_MONTH_FIRST),
        (["13/01/2026", "01/13/2026"], DATE_CONVENTION_AMBIGUOUS),
        (["02/01/2026", "03/04/2026"], DATE_CONVENTION_UNKNOWN),
        (["2026-01-13", ""], DATE_CONVENTION_UNKNOWN),
    ],
)
def test_convention_inference(values: list[str], expected: str) -> None:
    assert infer_support_ticket_date_convention(values) == expected


def test_day_first_convention_parses_without_transposition() -> None:
    assert parse_support_ticket_source_date(
        "02-01-2026", convention=DATE_CONVENTION_DAY_FIRST
    ) == date(2026, 1, 2)
    assert parse_support_ticket_source_date(
        "13-01-2026", convention=DATE_CONVENTION_DAY_FIRST
    ) == date(2026, 1, 13)


def test_unknown_convention_keeps_the_us_default() -> None:
    assert parse_support_ticket_source_date("02-01-2026") == date(2026, 2, 1)
    assert parse_support_ticket_source_date("1/13/2026") == date(2026, 1, 13)


def test_ambiguous_convention_refuses_numeric_dates() -> None:
    assert parse_support_ticket_source_date(
        "02-01-2026", convention=DATE_CONVENTION_AMBIGUOUS
    ) is None
    # ISO stays parseable regardless of convention.
    assert parse_support_ticket_source_date(
        "2026-01-02", convention=DATE_CONVENTION_AMBIGUOUS
    ) == date(2026, 1, 2)


def test_day_first_upload_normalizes_created_at_end_to_end() -> None:
    package = build_support_ticket_input_package(
        _rows(["13/01/2026", "02/01/2026", "05/01/2026"])
    )
    import json

    payload = json.dumps(package.inputs, default=str)
    assert "2026-01-02" in payload  # 02/01 is Jan 2, not Feb 1
    assert "2026-02-01" not in payload
    assert package.inputs["has_dated_window"] is True


def test_contradictory_upload_warns_and_does_not_guess() -> None:
    package = build_support_ticket_input_package(
        _rows(["13/01/2026", "01/13/2026", "02/01/2026"])
    )
    codes = [w.get("code") for w in package.warnings]
    assert "support_ticket_date_convention_ambiguous" in codes
    assert package.inputs["has_dated_window"] is False


def test_one_blank_date_keeps_the_window() -> None:
    package = build_support_ticket_input_package(
        _rows(["2026-01-05"] * 19 + [""])
    )
    assert package.inputs["has_dated_window"] is True
    assert "Last" in str(package.inputs["source_period"])


def test_sparse_dates_do_not_fake_a_window() -> None:
    package = build_support_ticket_input_package(
        _rows(["2026-01-05"] * 8 + ["", ""])
    )
    assert package.inputs["has_dated_window"] is False


def test_coverage_threshold_edge() -> None:
    # 18/20 = exactly 90% keeps the window; 17/20 loses it.
    keeps = build_support_ticket_input_package(
        _rows(["2026-01-05"] * 18 + ["", ""])
    )
    assert keeps.inputs["has_dated_window"] is True
    loses = build_support_ticket_input_package(
        _rows(["2026-01-05"] * 17 + ["", "", ""])
    )
    assert loses.inputs["has_dated_window"] is False


def test_dateless_upload_still_uses_the_dateless_basis() -> None:
    package = build_support_ticket_input_package(
        _rows(["", "", ""])
    )
    assert package.inputs["has_dated_window"] is False
    assert "Last" not in str(package.inputs["source_period"])


def test_diagnostics_report_convention_and_counts() -> None:
    package = build_support_ticket_input_package(
        _rows(["13/01/2026", "02/01/2026", ""])
    )
    warning = [
        w for w in package.warnings
        if w.get("code") == "support_ticket_date_window_disabled"
    ]
    # 2/3 dated is below threshold: window disabled with counts, no content.
    assert package.inputs["has_dated_window"] is False
    assert warning and warning[0].get("missing_or_unparseable_date_count") == 1


# Round-1 review refinements: convention-consistent diagnostics, plausible
# inference evidence, and emitted convention diagnostics.


def test_ambiguous_upload_cannot_keep_the_window_via_the_us_default() -> None:
    package = build_support_ticket_input_package(
        _rows(["01/13/2026"] * 9 + ["13/01/2026"])
    )
    assert package.inputs["has_dated_window"] is False
    codes = [w.get("code") for w in package.warnings]
    assert "support_ticket_date_convention_ambiguous" in codes


def test_malformed_cells_are_not_inference_evidence() -> None:
    assert infer_support_ticket_date_convention(
        ["99/01/2026", "02/01/2026"]
    ) == DATE_CONVENTION_UNKNOWN
    assert infer_support_ticket_date_convention(
        ["00/13/2026"]
    ) == DATE_CONVENTION_UNKNOWN
    assert infer_support_ticket_date_convention(
        ["99/01/2026", "13/01/2026"]
    ) == DATE_CONVENTION_DAY_FIRST


def test_convention_is_emitted_in_diagnostics() -> None:
    package = build_support_ticket_input_package(
        _rows(["13/01/2026", "02/01/2026", "05/01/2026"])
    )
    assert package.metadata["support_ticket_date_convention"] == (
        DATE_CONVENTION_DAY_FIRST
    )
    disabled = build_support_ticket_input_package(
        _rows(["01/13/2026"] * 9 + ["13/01/2026"])
    )
    warning = next(
        w for w in disabled.warnings
        if w.get("code") == "support_ticket_date_window_disabled"
    )
    assert warning["date_convention"] == DATE_CONVENTION_AMBIGUOUS


# Round-2 refinement: unparseable created_at values never leak downstream.


def test_refused_dates_do_not_leak_to_source_material() -> None:
    import json

    package = build_support_ticket_input_package(
        _rows(["01/13/2026"] * 9 + ["13/01/2026"])
    )
    payload = json.dumps(package.inputs, default=str)
    # The invariant: created_at is canonical ISO or absent -- a value this
    # boundary refused to interpret must not be re-guessed downstream.
    assert "01/13/2026" not in payload
    assert "13/01/2026" not in payload
