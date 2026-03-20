"""Read-only typed view over reasoning synthesis output.

Provides a clean contract for downstream consumers (battle cards,
challenger briefs, reports) instead of raw dict splatting.

Handles both v1 and v2 schemas transparently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...reasoning.wedge_registry import Wedge, validate_wedge


@dataclass(frozen=True, slots=True)
class TracedNumber:
    """A numeric value with its source provenance."""

    value: Any
    source_id: str

    @property
    def has_provenance(self) -> bool:
        return bool(self.source_id)


_CONFIDENCE_RANK = {
    "high": 4,
    "medium": 3,
    "low": 2,
    "insufficient": 1,
}

_REQUIRED_SECTIONS = (
    "causal_narrative", "segment_playbook", "timing_intelligence",
    "competitive_reframes", "migration_proof",
)


class SynthesisView:
    """Read-only view over a single vendor's reasoning synthesis.

    Wraps the raw dict and provides typed accessors, confidence checks,
    and source traceability.
    """

    __slots__ = ("vendor_name", "raw", "schema_version")

    def __init__(
        self,
        vendor_name: str,
        raw: dict[str, Any],
        schema_version: str = "v1",
    ) -> None:
        self.vendor_name = vendor_name
        self.raw = raw
        self.schema_version = schema_version

    @property
    def is_v2(self) -> bool:
        return self.schema_version.startswith("v2") or self.raw.get("schema_version") == "2.0"

    @property
    def primary_wedge(self) -> Wedge | None:
        cn = self.raw.get("causal_narrative")
        if not isinstance(cn, dict):
            return None
        return validate_wedge(cn.get("primary_wedge", ""))

    @property
    def wedge_label(self) -> str:
        w = self.primary_wedge
        if w is None:
            return ""
        return w.value.replace("_", " ").title()

    def section(self, name: str) -> dict[str, Any]:
        """Get a section dict, or empty dict if missing."""
        val = self.raw.get(name)
        return val if isinstance(val, dict) else {}

    def confidence(self, section: str) -> str:
        """Get the confidence level for a section."""
        sec = self.section(section)
        return sec.get("confidence", "insufficient")

    def should_suppress(self, section: str) -> bool:
        """True if section has insufficient confidence (should not display)."""
        return self.confidence(section) == "insufficient"

    def should_flag(self, section: str) -> bool:
        """True if section has low confidence (display with caveat)."""
        return self.confidence(section) == "low"

    def is_quotable(self, section: str) -> bool:
        """True if section is safe for outbound material.

        Requires confidence >= medium and no critical validation warnings.
        """
        conf = self.confidence(section)
        if _CONFIDENCE_RANK.get(conf, 0) < _CONFIDENCE_RANK["medium"]:
            return False
        # Check for critical warnings on this section
        warnings = self.raw.get("_validation_warnings") or []
        for w in warnings:
            if isinstance(w, dict) and w.get("path", "").startswith(section):
                if w.get("code") in ("hallucinated_source_id", "value_mismatch"):
                    return False
        return True

    def trace_number(self, section: str, field: str) -> TracedNumber:
        """Extract a {value, source_id} wrapper from a section field.

        Works with both v1 (bare values) and v2 (wrapped values).
        """
        sec = self.section(section)
        val = sec.get(field)

        if isinstance(val, dict) and "value" in val:
            # v2 wrapped value
            return TracedNumber(
                value=val.get("value"),
                source_id=val.get("source_id", ""),
            )
        # v1 bare value or missing
        return TracedNumber(value=val, source_id="")

    def citations(self, section: str) -> list[str]:
        """Get the citations array for a section."""
        sec = self.section(section)
        cites = sec.get("citations")
        return cites if isinstance(cites, list) else []

    def falsification_conditions(self) -> list[dict[str, Any]]:
        """Get structured falsification conditions from causal_narrative.

        Returns list of {condition, signal_source, monitorable} dicts.
        Falls back to bare strings wrapped in a dict for v1 compat.
        """
        cn = self.section("causal_narrative")
        raw = cn.get("what_would_weaken_thesis") or []
        if not isinstance(raw, list):
            return []
        result = []
        for item in raw:
            if isinstance(item, dict):
                result.append(item)
            elif isinstance(item, str):
                result.append({
                    "condition": item,
                    "signal_source": "",
                    "monitorable": False,
                })
        return result

    def data_gaps(self, section: str) -> list[str]:
        """Get data gaps for a section."""
        sec = self.section(section)
        gaps = sec.get("data_gaps")
        return gaps if isinstance(gaps, list) else []

    @property
    def meta(self) -> dict[str, Any]:
        m = self.raw.get("meta")
        if isinstance(m, dict):
            return m
        # v1 fallback: build from evidence_window
        ew = self.raw.get("evidence_window")
        if isinstance(ew, dict):
            return {
                "evidence_window_start": ew.get("earliest"),
                "evidence_window_end": ew.get("latest"),
            }
        return {}

    @property
    def validation_warnings(self) -> list[dict[str, Any]]:
        w = self.raw.get("_validation_warnings")
        return w if isinstance(w, list) else []


def load_synthesis_view(
    raw: dict[str, Any],
    vendor_name: str,
    schema_version: str = "",
) -> SynthesisView:
    """Construct a SynthesisView, handling v1 backward compatibility.

    If ``schema_version`` is not provided, infers from the raw dict.
    """
    if not schema_version:
        if raw.get("schema_version") == "2.0" or "meta" in raw:
            schema_version = "v2"
        else:
            schema_version = "v1"
    return SynthesisView(vendor_name=vendor_name, raw=raw, schema_version=schema_version)


def inject_synthesis_into_card(
    card_entry: dict[str, Any],
    view: SynthesisView,
) -> None:
    """Inject synthesis sections into a battle card entry.

    When synthesis is present, the synthesis wedge becomes the authoritative
    churn angle -- it overrides the archetype label so the card tells one
    consistent story.
    """
    for section in _REQUIRED_SECTIONS:
        if not view.should_suppress(section):
            card_entry[section] = view.section(section)

    # Always inject meta and wedge info
    card_entry["evidence_window"] = view.meta
    if view.primary_wedge:
        card_entry["synthesis_wedge"] = view.primary_wedge.value
        card_entry["synthesis_wedge_label"] = view.wedge_label
        # Override archetype with synthesis wedge so the card has one story.
        # The old archetype came from the stratified reasoner on different
        # data; the synthesis wedge is derived from the same pool evidence
        # the card displays.
        card_entry["archetype"] = view.primary_wedge.value
        card_entry["archetype_label"] = view.wedge_label

    # Flag low-confidence sections
    flagged = [s for s in _REQUIRED_SECTIONS if view.should_flag(s)]
    if flagged:
        card_entry["low_confidence_sections"] = flagged

    # Surface temporal depth warning in card header
    meta = view.meta
    ew_start = meta.get("evidence_window_start")
    ew_end = meta.get("evidence_window_end")
    if ew_start and ew_end:
        try:
            from datetime import date as _date
            start = _date.fromisoformat(str(ew_start)[:10])
            end = _date.fromisoformat(str(ew_end)[:10])
            window_days = (end - start).days
            card_entry["evidence_window_days"] = window_days
            from ._b2b_pool_compression import MIN_EVIDENCE_WINDOW_DAYS
            if window_days < MIN_EVIDENCE_WINDOW_DAYS:
                card_entry["evidence_depth_warning"] = (
                    f"Thin evidence window: {window_days} days. "
                    f"Confidence may improve with more data."
                )
        except (ValueError, TypeError):
            pass

    # Attach schema version for downstream branching
    card_entry["synthesis_schema_version"] = view.schema_version
