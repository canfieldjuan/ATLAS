import pytest

from extracted_content_pipeline.control_surfaces import OUTPUT_CATALOG
from extracted_content_pipeline.landing_page_generation import LandingPageGenerationConfig
from extracted_content_pipeline.landing_page_repair_contract import (
    LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_DEFAULT,
    LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MAX,
    LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MIN,
    LANDING_PAGE_QUALITY_REPAIR_INPUT,
    landing_page_quality_repair_attempts_from_inputs,
    landing_page_quality_repair_input_contract,
    normalize_landing_page_quality_repair_attempts,
)


def test_landing_page_repair_defaults_share_contract_values() -> None:
    assert LandingPageGenerationConfig().quality_repair_attempts == (
        LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_DEFAULT
    )
    assert OUTPUT_CATALOG["landing_page"].default_quality_repair_attempts == (
        LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_DEFAULT
    )


def test_landing_page_repair_input_contract_matches_validator_bounds() -> None:
    assert landing_page_quality_repair_input_contract() == {
        "key": LANDING_PAGE_QUALITY_REPAIR_INPUT,
        "label": "Landing page quality repair attempts",
        "type": "integer",
        "min": LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MIN,
        "max": LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MAX,
        "default": LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_DEFAULT,
    }


def test_landing_page_repair_attempts_from_inputs_accepts_missing_default() -> None:
    assert landing_page_quality_repair_attempts_from_inputs({}, default=3) == 3
    assert landing_page_quality_repair_attempts_from_inputs(
        {LANDING_PAGE_QUALITY_REPAIR_INPUT: "2"},
        default=3,
    ) == 2


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        ("0", 0),
        (
            LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MAX,
            LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MAX,
        ),
        (
            str(LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MAX),
            LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MAX,
        ),
    ],
)
def test_normalize_landing_page_repair_attempts_accepts_valid_values(
    value,
    expected,
) -> None:
    assert normalize_landing_page_quality_repair_attempts(value) == expected


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (-1, "landing_page_quality_repair_attempts must be at least 0"),
        (11, "landing_page_quality_repair_attempts must be at most 10"),
        (True, "landing_page_quality_repair_attempts must be an integer"),
        (1.5, "landing_page_quality_repair_attempts must be an integer"),
        ("many", "landing_page_quality_repair_attempts must be an integer"),
    ],
)
def test_normalize_landing_page_repair_attempts_rejects_invalid_values(
    value,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_landing_page_quality_repair_attempts(value)
