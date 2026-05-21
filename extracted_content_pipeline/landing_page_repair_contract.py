"""Canonical landing-page quality-repair attempt contract."""

from __future__ import annotations

from typing import Any, Mapping

LANDING_PAGE_QUALITY_REPAIR_INPUT = "landing_page_quality_repair_attempts"
LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_DEFAULT = 1
LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MIN = 0
LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MAX = 10


def landing_page_quality_repair_input_contract() -> dict[str, Any]:
    """Return the wire contract for the repair-attempt input."""

    return {
        "key": LANDING_PAGE_QUALITY_REPAIR_INPUT,
        "label": "Landing page quality repair attempts",
        "type": "integer",
        "min": LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MIN,
        "max": LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MAX,
        "default": LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_DEFAULT,
    }


def landing_page_quality_repair_attempts_from_inputs(
    inputs: Mapping[str, Any],
    *,
    default: int | None = None,
) -> int | None:
    """Return the normalized repair-attempt override from request inputs."""

    raw = inputs.get(LANDING_PAGE_QUALITY_REPAIR_INPUT)
    if raw is None:
        return default
    return normalize_landing_page_quality_repair_attempts(raw)


def normalize_landing_page_quality_repair_attempts(value: Any) -> int:
    """Normalize and validate a landing-page quality-repair attempt count."""

    key = LANDING_PAGE_QUALITY_REPAIR_INPUT
    if isinstance(value, (bool, float)):
        raise ValueError(f"{key} must be an integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be an integer") from None
    if normalized < LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MIN:
        raise ValueError(
            f"{key} must be at least {LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MIN}; "
            f"got {normalized}"
        )
    if normalized > LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MAX:
        raise ValueError(
            f"{key} must be at most {LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MAX}; "
            f"got {normalized}"
        )
    return normalized


__all__ = [
    "LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_DEFAULT",
    "LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MAX",
    "LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MIN",
    "LANDING_PAGE_QUALITY_REPAIR_INPUT",
    "landing_page_quality_repair_attempts_from_inputs",
    "landing_page_quality_repair_input_contract",
    "normalize_landing_page_quality_repair_attempts",
]
