"""Deterministic output-variation angle helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VariantAngle:
    """One deterministic framing angle for an output variation."""

    id: str
    label: str
    instruction: str

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "label": self.label,
            "instruction": self.instruction,
        }


VARIANT_ANGLES: tuple[VariantAngle, ...] = (
    VariantAngle(
        id="pain_led",
        label="Pain-led",
        instruction=(
            "Pain-led: open with the customer's current friction, then show how "
            "the evidence supports the recommended next step."
        ),
    ),
    VariantAngle(
        id="outcome_led",
        label="Outcome-led",
        instruction=(
            "Outcome-led: lead with the business result the audience wants, then "
            "connect the supporting evidence to that result."
        ),
    ),
    VariantAngle(
        id="social_proof",
        label="Social-proof-led",
        instruction=(
            "Social-proof-led: frame the post around observed customer or market "
            "signals without adding unsupported testimonials."
        ),
    ),
    VariantAngle(
        id="objection_handling",
        label="Objection-handling",
        instruction=(
            "Objection-handling: anticipate the strongest buyer hesitation and "
            "answer it with only the supplied evidence."
        ),
    ),
    VariantAngle(
        id="urgency_led",
        label="Urgency-led",
        instruction=(
            "Urgency-led: emphasize why the audience should act now, but do not "
            "invent time-sensitive claims absent from the blueprint."
        ),
    ),
)


def normalize_variant_count(value: Any = None) -> int:
    """Return a positive variant count capped to the deterministic catalogue."""

    raw = 1 if value in (None, "") else int(value)
    if raw < 1:
        raise ValueError(f"variant_count must be at least 1; got {raw}")
    return min(raw, len(VARIANT_ANGLES))


def selected_variant_angles(count: int) -> tuple[VariantAngle, ...]:
    """Return the deterministic first-N variant angles."""

    return VARIANT_ANGLES[:normalize_variant_count(count)]


__all__ = [
    "VARIANT_ANGLES",
    "VariantAngle",
    "normalize_variant_count",
    "selected_variant_angles",
]
