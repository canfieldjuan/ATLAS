"""Sales-oriented wedge types for B2B reasoning synthesis.

A 'wedge' is the opening angle a sales rep uses to engage a churning
vendor's customers.  Wedges map 1:1 from the 8 base churn archetypes
produced by the stratified reasoner plus two compound patterns and
a stable fallback.

The wedge registry is the single source of truth for:
- Valid wedge enum values (prompt injection + post-LLM validation)
- Archetype-to-wedge mapping
- Sales motion guidance per wedge
- Required pool layers per wedge
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class Wedge(str, Enum):
    """Sales-oriented wedge types."""

    PRICE_SQUEEZE = "price_squeeze"
    FEATURE_PARITY = "feature_parity"
    SUPPORT_EROSION = "support_erosion"
    INTEGRATION_LOCK = "integration_lock"
    CATEGORY_SHIFT = "category_shift"
    ACQUISITION_HANGOVER = "acquisition_hangover"
    COMPLIANCE_EXPOSURE = "compliance_exposure"
    UX_REGRESSION = "ux_regression"
    SEGMENT_MISMATCH = "segment_mismatch"
    STABLE = "stable"


WEDGE_ENUM_VALUES: frozenset[str] = frozenset(w.value for w in Wedge)


@dataclass(frozen=True, slots=True)
class WedgeMeta:
    """Metadata for a wedge type."""

    wedge: Wedge
    label: str
    archetype_map: tuple[str, ...]
    sales_motion: str
    required_pools: tuple[str, ...] = field(default=("evidence_vault",))


_WEDGE_CATALOG: tuple[WedgeMeta, ...] = (
    WedgeMeta(
        wedge=Wedge.PRICE_SQUEEZE,
        label="Price Squeeze",
        archetype_map=("pricing_shock",),
        sales_motion="Lead with TCO comparison and hidden-cost analysis",
        required_pools=("evidence_vault", "segment", "accounts"),
    ),
    WedgeMeta(
        wedge=Wedge.FEATURE_PARITY,
        label="Feature Parity",
        archetype_map=("feature_gap",),
        sales_motion="Demo the missing capability live; anchor on workflow impact",
        required_pools=("evidence_vault", "displacement", "segment"),
    ),
    WedgeMeta(
        wedge=Wedge.SUPPORT_EROSION,
        label="Support Erosion",
        archetype_map=("support_collapse",),
        sales_motion="Offer white-glove onboarding and named CSM commitment",
        required_pools=("evidence_vault", "temporal", "accounts"),
    ),
    WedgeMeta(
        wedge=Wedge.INTEGRATION_LOCK,
        label="Integration Lock-in",
        archetype_map=("integration_break",),
        sales_motion="Show migration playbook with timeline and risk mitigation",
        required_pools=("evidence_vault", "displacement"),
    ),
    WedgeMeta(
        wedge=Wedge.CATEGORY_SHIFT,
        label="Category Shift",
        archetype_map=("category_disruption",),
        sales_motion="Position as the category leader; frame incumbent as legacy",
        required_pools=("evidence_vault", "category", "displacement"),
    ),
    WedgeMeta(
        wedge=Wedge.ACQUISITION_HANGOVER,
        label="Acquisition Hangover",
        archetype_map=("acquisition_decay", "leadership_redesign"),
        sales_motion="Exploit post-acquisition confusion and roadmap uncertainty",
        required_pools=("evidence_vault", "temporal", "accounts"),
    ),
    WedgeMeta(
        wedge=Wedge.COMPLIANCE_EXPOSURE,
        label="Compliance Exposure",
        archetype_map=("compliance_gap",),
        sales_motion="Lead with compliance matrix; urgency from regulatory deadline",
        required_pools=("evidence_vault", "temporal"),
    ),
    WedgeMeta(
        wedge=Wedge.UX_REGRESSION,
        label="UX Regression",
        archetype_map=("feature_gap",),
        sales_motion="Side-by-side UX walkthrough showing workflow friction",
        required_pools=("evidence_vault", "segment"),
    ),
    WedgeMeta(
        wedge=Wedge.SEGMENT_MISMATCH,
        label="Segment Mismatch",
        archetype_map=("mixed",),
        sales_motion="Target the underserved segment with tailored positioning",
        required_pools=("evidence_vault", "segment", "accounts"),
    ),
    WedgeMeta(
        wedge=Wedge.STABLE,
        label="Stable",
        archetype_map=("stable",),
        sales_motion="Long-game nurture; monitor for trigger events",
        required_pools=("evidence_vault",),
    ),
)

# Archetype -> Wedge lookup (first match wins)
_ARCHETYPE_TO_WEDGE: dict[str, Wedge] = {}
for _meta in _WEDGE_CATALOG:
    for _arch in _meta.archetype_map:
        _ARCHETYPE_TO_WEDGE.setdefault(_arch, _meta.wedge)

# Wedge -> WedgeMeta lookup
_WEDGE_TO_META: dict[Wedge, WedgeMeta] = {m.wedge: m for m in _WEDGE_CATALOG}


def wedge_from_archetype(archetype: str) -> Wedge:
    """Map a churn archetype to its sales wedge. Defaults to SEGMENT_MISMATCH."""
    return _ARCHETYPE_TO_WEDGE.get(archetype, Wedge.SEGMENT_MISMATCH)


def validate_wedge(value: str) -> Wedge | None:
    """Return the Wedge enum member if *value* is valid, else None."""
    try:
        return Wedge(value)
    except ValueError:
        return None


def get_wedge_meta(wedge: Wedge) -> WedgeMeta:
    """Return metadata for a wedge type."""
    return _WEDGE_TO_META[wedge]


def get_sales_motion(wedge: Wedge) -> str:
    """Return the recommended sales motion for a wedge."""
    return _WEDGE_TO_META[wedge].sales_motion


def get_required_pools(wedge: Wedge) -> Sequence[str]:
    """Return the pool layers required for a wedge."""
    return _WEDGE_TO_META[wedge].required_pools
