"""Read-only typed view over reasoning synthesis output.

Provides a clean contract for downstream consumers (battle cards,
challenger briefs, reports) instead of raw dict splatting.

Handles both v1 and v2 schemas transparently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
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

_EVIDENCE_SUMMARY_SECTIONS = (
    "executive_summary",
    "target_accounts",
    "displacement_rankings",
    "recommend_ratio",
) + _REQUIRED_SECTIONS

_SECTION_CONTRACT_PATHS = {
    "causal_narrative": ("vendor_core_reasoning", "causal_narrative"),
    "segment_playbook": ("vendor_core_reasoning", "segment_playbook"),
    "timing_intelligence": ("vendor_core_reasoning", "timing_intelligence"),
    "competitive_reframes": ("displacement_reasoning", "competitive_reframes"),
    "migration_proof": ("displacement_reasoning", "migration_proof"),
}

_CONSUMER_REQUIRED_CONTRACTS = {
    "battle_card": (
        "vendor_core_reasoning",
        "displacement_reasoning",
        "category_reasoning",
        "account_reasoning",
    ),
    "weekly_churn_feed": ("vendor_core_reasoning", "account_reasoning"),
    "vendor_scorecard": (
        "vendor_core_reasoning",
        "displacement_reasoning",
        "account_reasoning",
    ),
    "displacement_report": ("displacement_reasoning", "category_reasoning"),
    "accounts_in_motion": (
        "vendor_core_reasoning",
        "displacement_reasoning",
        "account_reasoning",
    ),
    "challenger_brief": (
        "vendor_core_reasoning",
        "displacement_reasoning",
        "category_reasoning",
        "account_reasoning",
    ),
}


def _repair_stale_timing_support(contracts: dict[str, Any]) -> dict[str, Any]:
    vendor_core = contracts.get("vendor_core_reasoning")
    if not isinstance(vendor_core, dict):
        return contracts
    segment = vendor_core.get("segment_playbook")
    timing = vendor_core.get("timing_intelligence")
    if not isinstance(segment, dict) and (not isinstance(timing, dict) or not timing):
        return contracts
    from ._b2b_reasoning_contracts import (
        _canonicalize_segment_playbook,
        _canonicalize_timing_intelligence,
        _merge_timing_signal_rows,
        _timing_signal_fallback_rows,
        _timing_signal_summary_rows,
    )

    repaired = dict(contracts)
    repaired_vendor_core = dict(vendor_core)
    changed = False

    if isinstance(segment, dict) and segment:
        repaired_segment = _canonicalize_segment_playbook(dict(segment))
        if repaired_segment != segment:
            repaired_vendor_core["segment_playbook"] = repaired_segment
            changed = True

    if isinstance(timing, dict) and timing:
        repaired_timing = _canonicalize_timing_intelligence(dict(timing))
        repaired_supporting = (
            dict(repaired_timing.get("supporting_evidence"))
            if isinstance(repaired_timing.get("supporting_evidence"), dict)
            else {}
        )
        timing_rows = list(repaired_supporting.get("top_timing_signals") or [])
        timing_rows = _merge_timing_signal_rows(
            timing_rows,
            _timing_signal_fallback_rows(repaired_timing),
        )
        timing_rows = _merge_timing_signal_rows(
            timing_rows,
            _timing_signal_summary_rows(repaired_supporting),
        )
        if timing_rows:
            repaired_supporting["top_timing_signals"] = timing_rows
            repaired_timing["supporting_evidence"] = repaired_supporting
        if repaired_timing != timing:
            repaired_vendor_core["timing_intelligence"] = repaired_timing
            changed = True

    if not changed:
        return contracts
    repaired["vendor_core_reasoning"] = repaired_vendor_core
    return repaired


class SynthesisView:
    """Read-only view over a single vendor's reasoning synthesis.

    Wraps the raw dict and provides typed accessors, confidence checks,
    and source traceability.
    """

    __slots__ = ("vendor_name", "raw", "schema_version", "as_of_date")

    def __init__(
        self,
        vendor_name: str,
        raw: dict[str, Any],
        schema_version: str = "v1",
        as_of_date: date | None = None,
    ) -> None:
        self.vendor_name = vendor_name
        self.raw = raw
        self.schema_version = schema_version
        self.as_of_date = as_of_date

    @property
    def is_v2(self) -> bool:
        return self.schema_version.startswith("v2") or str(self.raw.get("schema_version") or "").startswith("2.")

    @property
    def has_explicit_contracts(self) -> bool:
        return bool(self.reasoning_contracts)

    @property
    def as_of_date_iso(self) -> str:
        if self.as_of_date is None:
            return ""
        return self.as_of_date.isoformat()

    def is_stale(self, requested_as_of: date | None = None) -> bool:
        if self.as_of_date is None:
            return False
        if requested_as_of is None:
            return False
        return self.as_of_date < requested_as_of

    @property
    def primary_wedge(self) -> Wedge | None:
        cn = self.section("causal_narrative")
        if not cn:
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
        contracts = self.reasoning_contracts
        contract_path = _SECTION_CONTRACT_PATHS.get(name)
        if contract_path:
            contract_name, section_name = contract_path
            contract = contracts.get(contract_name)
            if isinstance(contract, dict):
                nested = contract.get(section_name)
                if isinstance(nested, dict):
                    return nested
            if self.has_explicit_contracts:
                return {}
        val = self.raw.get(name)
        if isinstance(val, dict):
            return val
        return {}

    def confidence(self, section: str) -> str:
        """Get the confidence level for a section."""
        sec = self.section(section)
        return sec.get("confidence", "insufficient")

    def should_suppress(self, section: str, vendor_evidence: dict[str, Any] | None = None) -> bool:
        """True if section should not display.

        Checks both synthesis confidence AND evidence-engine suppression rules
        when vendor_evidence is provided.
        """
        if self.confidence(section) == "insufficient":
            return True
        if vendor_evidence is not None:
            from ...reasoning.evidence_engine import get_evidence_engine
            result = get_evidence_engine().evaluate_suppression(section, vendor_evidence)
            if result.suppress:
                return True
        return False

    def should_flag(self, section: str, vendor_evidence: dict[str, Any] | None = None) -> bool:
        """True if section should display with a caveat/disclaimer.

        Checks both synthesis confidence AND evidence-engine degrade rules.
        """
        if self.confidence(section) == "low":
            return True
        if vendor_evidence is not None:
            from ...reasoning.evidence_engine import get_evidence_engine
            result = get_evidence_engine().evaluate_suppression(section, vendor_evidence)
            if result.degrade:
                return True
        return False

    def get_disclaimer(self, section: str, vendor_evidence: dict[str, Any] | None = None) -> str | None:
        """Get the disclaimer text for a degraded section, if any."""
        if vendor_evidence is not None:
            from ...reasoning.evidence_engine import get_evidence_engine
            result = get_evidence_engine().evaluate_suppression(section, vendor_evidence)
            if result.disclaimer:
                return result.disclaimer
        return None

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

    @property
    def reasoning_contracts(self) -> dict[str, Any]:
        rc = self.raw.get("reasoning_contracts")
        return rc if isinstance(rc, dict) else {}

    def contract(self, name: str) -> dict[str, Any]:
        rc = self.reasoning_contracts.get(name)
        return rc if isinstance(rc, dict) else {}

    def materialized_contracts(self) -> dict[str, Any]:
        """Return contract-first reasoning blocks with flat-section fallback.

        New synthesis rows persist explicit ``reasoning_contracts``. Older rows may
        only expose the flat battle-card-shaped sections. This view normalizes both
        cases so downstream consumers can treat contract blocks as canonical.
        """
        contracts: dict[str, Any] = {}
        allow_flat_fallback = not self.has_explicit_contracts

        vendor_core = self.contract("vendor_core_reasoning")
        if not vendor_core and allow_flat_fallback:
            vendor_sections = {
                section: self.section(section)
                for section in (
                    "causal_narrative",
                    "segment_playbook",
                    "timing_intelligence",
                )
                if self.section(section)
            }
            if vendor_sections:
                vendor_core = {
                    "schema_version": "v1",
                    **vendor_sections,
                }
        if vendor_core:
            contracts["vendor_core_reasoning"] = vendor_core

        displacement = self.contract("displacement_reasoning")
        if not displacement and allow_flat_fallback:
            displacement_sections = {
                section: self.section(section)
                for section in (
                    "competitive_reframes",
                    "migration_proof",
                )
                if self.section(section)
            }
            if displacement_sections:
                displacement = {
                    "schema_version": "v1",
                    **displacement_sections,
                }
        if displacement:
            contracts["displacement_reasoning"] = displacement

        category = self.contract("category_reasoning")
        if not category:
            raw_category = self.raw.get("category_reasoning")
            if isinstance(raw_category, dict) and raw_category:
                category = raw_category
        if category:
            contracts["category_reasoning"] = category

        account = self.contract("account_reasoning")
        if not account:
            raw_account = self.raw.get("account_reasoning")
            if isinstance(raw_account, dict) and raw_account:
                account = raw_account
        if account:
            contracts["account_reasoning"] = account

        contracts = _repair_stale_timing_support(contracts)

        if contracts:
            contracts["schema_version"] = (
                str(self.reasoning_contracts.get("schema_version") or "v1")
            )
        return contracts


def _coerce_iso_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None
    return None


def required_contracts_for_consumer(consumer: str) -> tuple[str, ...]:
    key = str(consumer or "").strip().lower()
    return _CONSUMER_REQUIRED_CONTRACTS.get(key, ())


def contract_gaps_for_consumer(view: SynthesisView | None, consumer: str) -> list[str]:
    required = required_contracts_for_consumer(consumer)
    if not required:
        return []
    if view is None:
        return list(required)
    materialized_contracts = getattr(view, "materialized_contracts", None)
    if callable(materialized_contracts):
        contracts = materialized_contracts() or {}
    else:
        contracts = {}
        contract_getter = getattr(view, "contract", None)
        if callable(contract_getter):
            for name in required:
                contract = contract_getter(name)
                if isinstance(contract, dict) and contract:
                    contracts[name] = contract
    gaps: list[str] = []
    for name in required:
        section = contracts.get(name)
        if not isinstance(section, dict) or not section:
            gaps.append(name)
    return gaps


def inject_synthesis_freshness(
    entry: dict[str, Any],
    view: SynthesisView,
    *,
    requested_as_of: date | None = None,
) -> None:
    as_of_date = getattr(view, "as_of_date", None)
    if as_of_date is not None:
        entry["data_as_of_date"] = as_of_date.isoformat()
    if requested_as_of is None:
        return
    is_stale = getattr(view, "is_stale", None)
    if callable(is_stale):
        entry["data_stale"] = bool(is_stale(requested_as_of))


def load_synthesis_view(
    raw: dict[str, Any],
    vendor_name: str,
    schema_version: str = "",
    as_of_date: date | str | None = None,
) -> SynthesisView:
    """Construct a SynthesisView, handling v1 backward compatibility.

    If ``schema_version`` is not provided, infers from the raw dict.
    """
    if not schema_version:
        if str(raw.get("schema_version") or "").startswith("2.") or "meta" in raw:
            schema_version = "v2"
        else:
            schema_version = "v1"
    return SynthesisView(
        vendor_name=vendor_name,
        raw=raw,
        schema_version=schema_version,
        as_of_date=_coerce_iso_date(as_of_date),
    )


def inject_synthesis_into_card(
    card_entry: dict[str, Any],
    view: SynthesisView,
    *,
    materialize_flat_sections: bool = False,
    requested_as_of: date | None = None,
    vendor_evidence: dict[str, Any] | None = None,
) -> None:
    """Inject synthesis sections into a battle card entry.

    When synthesis is present, the synthesis wedge becomes the authoritative
    churn angle -- it overrides the archetype label so the card tells one
    consistent story.

    If ``vendor_evidence`` is provided, evidence-engine suppression and
    conclusion gating rules are applied to sections and injected into the card.
    """
    if materialize_flat_sections:
        for section in _REQUIRED_SECTIONS:
            if not view.should_suppress(section, vendor_evidence):
                card_entry[section] = view.section(section)

    contracts = view.materialized_contracts()
    if contracts:
        card_entry["reasoning_contracts"] = contracts
        card_entry["reasoning_source"] = "b2b_reasoning_synthesis"
    inject_synthesis_freshness(
        card_entry,
        view,
        requested_as_of=requested_as_of,
    )
    contract_gaps = contract_gaps_for_consumer(view, "battle_card")
    if contract_gaps:
        card_entry["reasoning_contract_gaps"] = contract_gaps

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
    flagged = [s for s in _REQUIRED_SECTIONS if view.should_flag(s, vendor_evidence)]
    if flagged:
        card_entry["low_confidence_sections"] = flagged
    # Inject disclaimers for degraded sections
    disclaimers: dict[str, str] = {}
    for s in _REQUIRED_SECTIONS:
        disc = view.get_disclaimer(s, vendor_evidence)
        if disc:
            disclaimers[s] = disc
    if disclaimers:
        card_entry["section_disclaimers"] = disclaimers

    # Inject evidence-engine conclusion results when evidence is available
    if vendor_evidence is not None:
        card_entry["evidence_conclusions"] = evaluate_vendor_conclusions(vendor_evidence)

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
            card_entry["evidence_window_label"] = f"{window_days}-day evidence window"
            from ._b2b_pool_compression import MIN_EVIDENCE_WINDOW_DAYS
            if window_days < MIN_EVIDENCE_WINDOW_DAYS:
                card_entry["evidence_window_is_thin"] = True
                card_entry["evidence_depth_warning"] = (
                    f"Thin evidence window: {window_days} days. "
                    f"Confidence may improve with more data."
                )
            else:
                card_entry["evidence_window_is_thin"] = False
        except (ValueError, TypeError):
            pass

    # Attach schema version for downstream branching
    card_entry["synthesis_schema_version"] = view.schema_version


def evaluate_vendor_conclusions(
    vendor_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate evidence-engine conclusion gates for a vendor.

    Call this at report generation time with aggregated vendor-level evidence
    (total_reviews, pain_distribution, displacement_edge, indicator_counts, etc.).

    Returns a dict suitable for merging into the report payload:
    {
        "conclusions": {
            "pricing_crisis": {"met": True, "confidence": "high"},
            "losing_market_share": {"met": False, "fallback_label": "Emerging..."},
            ...
        },
        "suppressed_sections": ["target_accounts"],
        "degraded_sections": {"executive_summary": "Based on limited..."},
        "confidence_tier": "high",
    }
    """
    from ...reasoning.evidence_engine import get_evidence_engine

    engine = get_evidence_engine()

    # Evaluate all conclusion gates
    raw_conclusions = engine.evaluate_conclusions(vendor_evidence)
    conclusions: dict[str, dict[str, Any]] = {}
    for c in raw_conclusions:
        entry: dict[str, Any] = {"met": c.met, "confidence": c.confidence}
        if c.fallback_label:
            entry["fallback_label"] = c.fallback_label
        if c.fallback_action:
            entry["fallback_action"] = c.fallback_action
        conclusions[c.conclusion_id] = entry

    # Evaluate section suppressions
    suppressed: list[str] = []
    degraded: dict[str, str] = {}
    for section in _EVIDENCE_SUMMARY_SECTIONS:
        result = engine.evaluate_suppression(section, vendor_evidence)
        if result.suppress:
            suppressed.append(section)
        elif result.degrade and result.disclaimer:
            degraded[section] = result.disclaimer

    # Confidence tier
    total_reviews = int(vendor_evidence.get("total_reviews", 0))
    confidence_tier = engine.get_confidence_tier(total_reviews)

    return {
        "conclusions": conclusions,
        "suppressed_sections": suppressed,
        "degraded_sections": degraded,
        "confidence_tier": confidence_tier,
        "confidence_label": engine.get_confidence_label(total_reviews),
    }
