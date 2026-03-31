"""Read-only typed view over reasoning synthesis output.

Provides a clean contract for downstream consumers (battle cards,
challenger briefs, reports) instead of raw dict splatting.

Handles both v1 and v2 schemas transparently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

from ...config import settings
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
    "campaign": (
        "vendor_core_reasoning",
        "displacement_reasoning",
    ),
    "blog_reranker": (
        "vendor_core_reasoning",
        "category_reasoning",
    ),
    "vendor_briefing": (
        "vendor_core_reasoning",
        "displacement_reasoning",
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

    # --- Phase 3 accessors ------------------------------------------------

    @property
    def why_they_stay(self) -> dict[str, Any]:
        """Get the why_they_stay block from vendor_core_reasoning."""
        vc = self.contract("vendor_core_reasoning")
        wts = vc.get("why_they_stay")
        return wts if isinstance(wts, dict) else {}

    @property
    def confidence_posture(self) -> dict[str, Any]:
        """Get the confidence_posture block from vendor_core_reasoning."""
        vc = self.contract("vendor_core_reasoning")
        cp = vc.get("confidence_posture")
        return cp if isinstance(cp, dict) else {}

    @property
    def switch_triggers(self) -> list[dict[str, Any]]:
        """Get switch_triggers from displacement_reasoning."""
        dr = self.contract("displacement_reasoning")
        st = dr.get("switch_triggers")
        return st if isinstance(st, list) else []

    @property
    def evidence_governance(self) -> dict[str, Any]:
        """Get the evidence_governance contract block."""
        eg = self.contract("evidence_governance")
        if eg:
            return eg
        # Also check top-level raw for older schemas
        raw_eg = self.raw.get("evidence_governance")
        return raw_eg if isinstance(raw_eg, dict) else {}

    @property
    def metric_ledger(self) -> list[dict[str, Any]]:
        """Get metric_ledger from evidence_governance."""
        ml = self.evidence_governance.get("metric_ledger")
        return ml if isinstance(ml, list) else []

    @property
    def contradictions(self) -> list[dict[str, Any]]:
        """Get contradictions from evidence_governance."""
        c = self.evidence_governance.get("contradictions")
        return c if isinstance(c, list) else []

    @property
    def coverage_gaps(self) -> list[dict[str, Any]]:
        """Get coverage_gaps from evidence_governance."""
        cg = self.evidence_governance.get("coverage_gaps")
        return cg if isinstance(cg, list) else []

    @property
    def confidence_limits(self) -> list[str]:
        """Get confidence limits from confidence_posture."""
        return list(self.confidence_posture.get("limits") or [])

    @property
    def retention_strengths(self) -> list[dict[str, Any]]:
        """Get strengths from why_they_stay."""
        return list(self.why_they_stay.get("strengths") or [])

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

    @property
    def reference_ids(self) -> dict[str, list[str]]:
        raw = self.raw.get("reference_ids")
        if not isinstance(raw, dict):
            return {}
        resolved: dict[str, list[str]] = {}
        for key in ("metric_ids", "witness_ids"):
            values = raw.get(key)
            if isinstance(values, list):
                cleaned = [
                    str(value).strip()
                    for value in values
                    if str(value or "").strip()
                ]
                if cleaned:
                    resolved[key] = cleaned
        return resolved

    @property
    def packet_artifacts(self) -> dict[str, Any]:
        raw = self.raw.get("packet_artifacts")
        return raw if isinstance(raw, dict) else {}

    @property
    def witness_pack(self) -> list[dict[str, Any]]:
        raw = self.packet_artifacts.get("witness_pack")
        return [dict(item) for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []

    @property
    def section_packets(self) -> dict[str, Any]:
        raw = self.packet_artifacts.get("section_packets")
        return dict(raw) if isinstance(raw, dict) else {}

    def contract(self, name: str) -> dict[str, Any]:
        rc = self.reasoning_contracts.get(name)
        return rc if isinstance(rc, dict) else {}

    def witness(self, witness_id: str) -> dict[str, Any]:
        target = str(witness_id or "").strip()
        if not target:
            return {}
        for witness in self.witness_pack:
            sid = str(witness.get("witness_id") or witness.get("_sid") or "").strip()
            if sid == target:
                return dict(witness)
        return {}

    def anchor_examples(self) -> dict[str, list[dict[str, Any]]]:
        raw = self.section_packets.get("anchor_examples")
        if not isinstance(raw, dict):
            return {}
        resolved: dict[str, list[dict[str, Any]]] = {}
        for label, witness_ids in raw.items():
            if not isinstance(witness_ids, list):
                continue
            matches: list[dict[str, Any]] = []
            for witness_id in witness_ids:
                witness = self.witness(str(witness_id or ""))
                if witness:
                    matches.append(witness)
            if matches:
                resolved[str(label)] = matches
        return resolved

    def witness_highlights(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is None:
            limit = int(
                getattr(settings.b2b_churn, "reasoning_witness_highlight_limit", 4),
            )
        if limit <= 0:
            return []
        highlights: list[dict[str, Any]] = []
        seen: set[str] = set()

        def _add(witness: dict[str, Any]) -> None:
            witness_id = str(witness.get("witness_id") or witness.get("_sid") or "").strip()
            if not witness_id or witness_id in seen:
                return
            seen.add(witness_id)
            highlights.append(dict(witness))

        for rows in self.anchor_examples().values():
            for witness in rows:
                _add(witness)
                if len(highlights) >= limit:
                    return highlights

        ranked = sorted(
            self.witness_pack,
            key=lambda item: (
                -float(item.get("salience_score") or 0.0),
                str(item.get("witness_id") or item.get("_sid") or ""),
            ),
        )
        for witness in ranked:
            _add(witness)
            if len(highlights) >= limit:
                break
        return highlights

    def consumer_context(self, consumer: str) -> dict[str, Any]:
        contracts = self.materialized_contracts()
        context: dict[str, Any] = {
            "reasoning_contracts": contracts,
            "vendor_core_reasoning": dict(contracts.get("vendor_core_reasoning") or {}),
            "displacement_reasoning": dict(contracts.get("displacement_reasoning") or {}),
            "category_reasoning": dict(contracts.get("category_reasoning") or {}),
            "account_reasoning": dict(contracts.get("account_reasoning") or {}),
            "reasoning_contract_gaps": contract_gaps_for_consumer(self, consumer),
        }
        anchors = self.anchor_examples()
        if anchors:
            context["anchor_examples"] = anchors
        highlights = self.witness_highlights()
        if highlights:
            context["witness_highlights"] = highlights
        refs = self.reference_ids
        if refs:
            context["reference_ids"] = refs
        if self.packet_artifacts:
            context["packet_artifacts"] = self.packet_artifacts
        return context

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

        evidence_gov = self.contract("evidence_governance")
        if not evidence_gov:
            raw_eg = self.raw.get("evidence_governance")
            if isinstance(raw_eg, dict) and raw_eg:
                evidence_gov = raw_eg
        if evidence_gov:
            contracts["evidence_governance"] = evidence_gov

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
    context = view.consumer_context("battle_card")
    if materialize_flat_sections:
        for section in _REQUIRED_SECTIONS:
            if not view.should_suppress(section, vendor_evidence):
                card_entry[section] = view.section(section)

    contracts = context.get("reasoning_contracts") or {}
    if contracts:
        card_entry["reasoning_contracts"] = contracts
        card_entry["reasoning_source"] = "b2b_reasoning_synthesis"
    anchors = context.get("anchor_examples")
    if isinstance(anchors, dict) and anchors:
        card_entry["anchor_examples"] = anchors
    highlights = context.get("witness_highlights")
    if isinstance(highlights, list) and highlights:
        card_entry["witness_highlights"] = highlights
    reference_ids = context.get("reference_ids")
    if isinstance(reference_ids, dict) and reference_ids:
        card_entry["reference_ids"] = reference_ids
    inject_synthesis_freshness(
        card_entry,
        view,
        requested_as_of=requested_as_of,
    )
    contract_gaps = context.get("reasoning_contract_gaps") or []
    if contract_gaps:
        card_entry["reasoning_contract_gaps"] = contract_gaps

    # Always inject meta and wedge info
    card_entry["evidence_window"] = view.meta
    if view.primary_wedge:
        card_entry["synthesis_wedge"] = view.primary_wedge.value
        card_entry["synthesis_wedge_label"] = view.wedge_label
        # Override archetype with synthesis wedge so the card has one story.
        # Older archetype-style labels could come from a different producer;
        # the synthesis wedge is derived from the same pool evidence the card
        # displays.
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


# ---------------------------------------------------------------------------
# SynthesisView -> reasoning_lookup entry (for shared builders)
# ---------------------------------------------------------------------------

def synthesis_view_to_reasoning_entry(view: SynthesisView) -> dict[str, Any]:
    """Convert a SynthesisView into the dict shape that shared report builders expect.

    This produces the same ``{archetype, confidence, executive_summary,
    key_signals, ...}`` structure as ``reconstruct_reasoning_lookup()`` from
    ``b2b_churn_intelligence.py``, so it can be dropped into the same
    ``reasoning_lookup[vendor]`` slot.
    """
    cn = view.section("causal_narrative")
    wedge = view.primary_wedge

    # Prefer validated wedge, fall back to raw primary_wedge string
    archetype = wedge.value if wedge else cn.get("primary_wedge", "")

    confidence_str = view.confidence("causal_narrative")
    # Map confidence label -> numeric for backward compat
    confidence_num = _confidence_label_to_numeric(confidence_str)

    # Build summary from available fields -- v2.3 uses trigger/why_now/causal_chain
    executive_summary = cn.get("summary") or cn.get("executive_summary") or ""
    if not executive_summary:
        parts = [cn.get("trigger", ""), cn.get("why_now", "")]
        chain = cn.get("causal_chain", "")
        if chain:
            parts.append(chain)
        executive_summary = ". ".join(p for p in parts if p)

    key_signals = cn.get("key_signals") or []
    if not isinstance(key_signals, list):
        key_signals = []
    if not key_signals:
        for field in ("trigger", "who_most_affected", "why_now"):
            val = cn.get(field, "")
            if val:
                key_signals.append(val)

    falsification = [
        fc.get("condition", fc) if isinstance(fc, dict) else str(fc)
        for fc in view.falsification_conditions()
    ]

    uncertainty = cn.get("data_gaps") or []
    if not isinstance(uncertainty, list):
        uncertainty = []

    # risk_level from confidence
    risk_level = "high" if confidence_str in ("high", "medium") else "low"

    return {
        "archetype": archetype,
        "confidence": confidence_num,
        "mode": "synthesis" if view.schema_version != "legacy" else "synthesis_fallback",
        "risk_level": risk_level,
        "executive_summary": executive_summary,
        "key_signals": key_signals,
        "falsification_conditions": falsification,
        "uncertainty_sources": uncertainty,
    }


def build_reasoning_lookup_from_views(
    synthesis_views: dict[str, SynthesisView],
) -> dict[str, dict[str, Any]]:
    """Build a reasoning_lookup dict from synthesis views.

    Returns ``{vendor_name: reasoning_entry}`` in the same shape as
    ``reconstruct_reasoning_lookup()``.  Callers can merge this with
    legacy lookups for vendors not covered by synthesis.
    """
    lookup: dict[str, dict[str, Any]] = {}
    for vendor_name, view in synthesis_views.items():
        lookup[vendor_name] = synthesis_view_to_reasoning_entry(view)
    return lookup


# ---------------------------------------------------------------------------
# Legacy reasoning-to-contracts conversion
# ---------------------------------------------------------------------------

def legacy_reasoning_to_contracts(
    legacy: dict[str, Any],
    *,
    vendor_name: str = "",
) -> dict[str, Any]:
    """Convert a legacy b2b_churn_signals row into minimal reasoning contracts.

    The legacy schema stores ``archetype``, ``executive_summary``,
    ``key_signals``, ``falsification_conditions``, and
    ``uncertainty_sources`` as flat fields.  This normalizes them into the
    same contract shape that ``b2b_reasoning_synthesis`` produces, so
    downstream consumers can treat both sources identically.

    Returns a dict suitable for wrapping in a ``SynthesisView``.
    """
    archetype = legacy.get("archetype") or ""
    confidence_raw = legacy.get("archetype_confidence") or legacy.get("confidence")
    try:
        confidence_val = float(confidence_raw) if confidence_raw is not None else 0.0
    except (TypeError, ValueError):
        confidence_val = 0.0

    if confidence_val >= 0.7:
        confidence_label = "high"
    elif confidence_val >= 0.4:
        confidence_label = "medium"
    else:
        confidence_label = "low"

    executive_summary = (
        legacy.get("reasoning_executive_summary")
        or legacy.get("executive_summary")
        or ""
    )
    key_signals = legacy.get("reasoning_key_signals") or legacy.get("key_signals") or []
    if isinstance(key_signals, str):
        import json as _json
        try:
            key_signals = _json.loads(key_signals)
        except (ValueError, TypeError):
            key_signals = [key_signals]

    falsification = legacy.get("falsification_conditions") or []
    if isinstance(falsification, str):
        import json as _json
        try:
            falsification = _json.loads(falsification)
        except (ValueError, TypeError):
            falsification = [falsification]

    uncertainty = (
        legacy.get("reasoning_uncertainty_sources")
        or legacy.get("uncertainty_sources")
        or []
    )
    if isinstance(uncertainty, str):
        import json as _json
        try:
            uncertainty = _json.loads(uncertainty)
        except (ValueError, TypeError):
            uncertainty = [uncertainty]

    # Build minimal causal_narrative from legacy fields
    causal_narrative: dict[str, Any] = {
        "confidence": confidence_label,
        "data_gaps": list(uncertainty) if uncertainty else [],
    }
    if archetype:
        from ...reasoning.wedge_registry import wedge_from_archetype
        mapped_wedge = wedge_from_archetype(archetype)
        causal_narrative["primary_wedge"] = mapped_wedge.value
        causal_narrative["_legacy_archetype"] = archetype
    if executive_summary:
        causal_narrative["summary"] = executive_summary
    if key_signals:
        causal_narrative["key_signals"] = key_signals
    if falsification:
        causal_narrative["what_would_weaken_thesis"] = [
            {"condition": f, "signal_source": "", "monitorable": False}
            if isinstance(f, str) else f
            for f in falsification
        ]

    vendor_core: dict[str, Any] = {
        "schema_version": "legacy",
        "causal_narrative": causal_narrative,
    }

    contracts: dict[str, Any] = {
        "schema_version": "legacy",
        "vendor_core_reasoning": vendor_core,
    }

    return {
        "reasoning_contracts": contracts,
        "schema_version": "legacy",
        "_legacy_source": "b2b_churn_signals",
    }


# ---------------------------------------------------------------------------
# Shared DB loader: prefer synthesis, fall back to legacy
# ---------------------------------------------------------------------------

_logger = logging.getLogger("atlas.b2b.synthesis_reader")


def _confidence_label_to_numeric(confidence_label: str) -> float:
    return {
        "high": 0.85,
        "medium": 0.55,
        "low": 0.25,
        "insufficient": 0.0,
    }.get(str(confidence_label or "").strip().lower(), 0.0)


async def load_best_reasoning_view(
    pool,
    vendor_name: str,
    *,
    as_of: date | None = None,
    analysis_window_days: int = 30,
) -> SynthesisView | None:
    """Load the best available reasoning for a vendor.

    Preference order:
    1. Latest ``b2b_reasoning_synthesis`` row (contracts-first).
    2. Latest ``b2b_churn_signals`` row with an archetype, converted to
       minimal contracts via ``legacy_reasoning_to_contracts()``.
    3. ``None`` if neither source has data.

    This is the canonical entry point for any downstream consumer that
    needs vendor reasoning context.
    """
    # --- Try synthesis first -----------------------------------------------
    if as_of is not None:
        synth_row = await pool.fetchrow(
            """
            SELECT vendor_name, as_of_date, schema_version, synthesis
            FROM b2b_reasoning_synthesis
            WHERE LOWER(vendor_name) = LOWER($1)
              AND as_of_date <= $2
              AND analysis_window_days = $3
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 1
            """,
            vendor_name,
            as_of,
            analysis_window_days,
        )
    else:
        synth_row = await pool.fetchrow(
            """
            SELECT vendor_name, as_of_date, schema_version, synthesis
            FROM b2b_reasoning_synthesis
            WHERE LOWER(vendor_name) = LOWER($1)
              AND analysis_window_days = $2
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 1
            """,
            vendor_name,
            analysis_window_days,
        )

    if synth_row:
        import json as _json

        raw = synth_row["synthesis"]
        if isinstance(raw, str):
            raw = _json.loads(raw)
        if isinstance(raw, dict) and raw:
            return load_synthesis_view(
                raw,
                vendor_name=synth_row["vendor_name"],
                schema_version=str(synth_row.get("schema_version") or ""),
                as_of_date=synth_row["as_of_date"],
            )

    # --- Fall back to legacy churn signals ---------------------------------
    if as_of is not None:
        legacy_row = await pool.fetchrow(
            """
            SELECT archetype, archetype_confidence, falsification_conditions,
                   reasoning_executive_summary, reasoning_key_signals,
                   reasoning_uncertainty_sources, last_computed_at
            FROM b2b_churn_signals
            WHERE LOWER(vendor_name) = LOWER($1)
              AND archetype IS NOT NULL
              AND last_computed_at::date <= $2
            ORDER BY last_computed_at DESC
            LIMIT 1
            """,
            vendor_name,
            as_of,
        )
    else:
        legacy_row = await pool.fetchrow(
            """
            SELECT archetype, archetype_confidence, falsification_conditions,
                   reasoning_executive_summary, reasoning_key_signals,
                   reasoning_uncertainty_sources, last_computed_at
            FROM b2b_churn_signals
            WHERE LOWER(vendor_name) = LOWER($1)
              AND archetype IS NOT NULL
            ORDER BY last_computed_at DESC
            LIMIT 1
            """,
            vendor_name,
        )
    if not legacy_row or not legacy_row["archetype"]:
        return None

    legacy_dict = dict(legacy_row)
    raw = legacy_reasoning_to_contracts(legacy_dict, vendor_name=vendor_name)

    as_of_date = None
    lca = legacy_row.get("last_computed_at")
    if lca is not None:
        if hasattr(lca, "date"):
            as_of_date = lca.date()
        else:
            as_of_date = _coerce_iso_date(lca)

    return load_synthesis_view(
        raw,
        vendor_name=vendor_name,
        schema_version="legacy",
        as_of_date=as_of_date,
    )


async def load_best_reasoning_views(
    pool,
    vendor_names: list[str],
    *,
    as_of: date | None = None,
    analysis_window_days: int = 30,
) -> dict[str, SynthesisView]:
    """Batch-load best reasoning for multiple vendors.

    Returns a dict of vendor_name -> SynthesisView for each vendor that
    has reasoning available.  Uses two bulk queries instead of N+1.
    """
    views: dict[str, SynthesisView] = {}
    if not vendor_names:
        return views

    import json as _json

    lower_names = [v.lower() for v in vendor_names]

    # --- Bulk synthesis query ----------------------------------------------
    if as_of is not None:
        synth_rows = await pool.fetch(
            """
            SELECT DISTINCT ON (LOWER(vendor_name))
                   vendor_name, as_of_date, schema_version, synthesis
            FROM b2b_reasoning_synthesis
            WHERE LOWER(vendor_name) = ANY($1)
              AND as_of_date <= $2
              AND analysis_window_days = $3
            ORDER BY LOWER(vendor_name), as_of_date DESC, created_at DESC
            """,
            lower_names,
            as_of,
            analysis_window_days,
        )
    else:
        synth_rows = await pool.fetch(
            """
            SELECT DISTINCT ON (LOWER(vendor_name))
                   vendor_name, as_of_date, schema_version, synthesis
            FROM b2b_reasoning_synthesis
            WHERE LOWER(vendor_name) = ANY($1)
              AND analysis_window_days = $2
            ORDER BY LOWER(vendor_name), as_of_date DESC, created_at DESC
            """,
            lower_names,
            analysis_window_days,
        )

    covered: set[str] = set()
    for row in synth_rows:
        raw = row.get("synthesis")
        if raw is None:
            continue
        if isinstance(raw, str):
            raw = _json.loads(raw)
        if not isinstance(raw, dict) or not raw:
            continue
        vname = row.get("vendor_name")
        if not vname:
            continue
        views[vname] = load_synthesis_view(
            raw,
            vendor_name=vname,
            schema_version=str(row.get("schema_version") or ""),
            as_of_date=row["as_of_date"],
        )
        covered.add(vname.lower())

    # --- Bulk legacy fallback for remaining vendors ------------------------
    remaining = [v for v in vendor_names if v.lower() not in covered]
    if not remaining:
        return views

    remaining_lower = [v.lower() for v in remaining]
    if as_of is not None:
        legacy_rows = await pool.fetch(
            """
            SELECT DISTINCT ON (LOWER(vendor_name))
                   vendor_name, archetype, archetype_confidence,
                   falsification_conditions, reasoning_executive_summary,
                   reasoning_key_signals, reasoning_uncertainty_sources,
                   last_computed_at
            FROM b2b_churn_signals
            WHERE LOWER(vendor_name) = ANY($1)
              AND archetype IS NOT NULL
              AND last_computed_at::date <= $2
            ORDER BY LOWER(vendor_name), last_computed_at DESC
            """,
            remaining_lower,
            as_of,
        )
    else:
        legacy_rows = await pool.fetch(
            """
            SELECT DISTINCT ON (LOWER(vendor_name))
                   vendor_name, archetype, archetype_confidence,
                   falsification_conditions, reasoning_executive_summary,
                   reasoning_key_signals, reasoning_uncertainty_sources,
                   last_computed_at
            FROM b2b_churn_signals
            WHERE LOWER(vendor_name) = ANY($1)
              AND archetype IS NOT NULL
            ORDER BY LOWER(vendor_name), last_computed_at DESC
            """,
            remaining_lower,
        )
    for row in legacy_rows:
        if not row["archetype"]:
            continue
        vname = row["vendor_name"]
        legacy_dict = dict(row)
        raw = legacy_reasoning_to_contracts(legacy_dict, vendor_name=vname)

        as_of_date = None
        lca = row.get("last_computed_at")
        if lca is not None:
            if hasattr(lca, "date"):
                as_of_date = lca.date()
            else:
                as_of_date = _coerce_iso_date(lca)

        views[vname] = load_synthesis_view(
            raw,
            vendor_name=vname,
            schema_version="legacy",
            as_of_date=as_of_date,
        )

    return views


async def load_prior_reasoning_snapshots(
    pool,
    vendor_names: list[str],
    *,
    before_date: date | None = None,
    analysis_window_days: int = 30,
) -> dict[str, dict[str, Any]]:
    """Load the most recent prior reasoning snapshot before a cutoff date."""
    snapshots: dict[str, dict[str, Any]] = {}
    if not vendor_names:
        return snapshots

    import json as _json

    cutoff = before_date or date.today()
    requested_by_lower: dict[str, str] = {}
    lower_names: list[str] = []
    for vendor_name in vendor_names:
        raw_name = str(vendor_name or "").strip()
        if not raw_name:
            continue
        lowered = raw_name.lower()
        if lowered in requested_by_lower:
            continue
        requested_by_lower[lowered] = raw_name
        lower_names.append(lowered)
    if not lower_names:
        return snapshots

    synth_rows = await pool.fetch(
        """
        SELECT DISTINCT ON (LOWER(vendor_name))
               vendor_name, as_of_date, schema_version, synthesis
        FROM b2b_reasoning_synthesis
        WHERE LOWER(vendor_name) = ANY($1)
          AND as_of_date < $2
          AND analysis_window_days = $3
        ORDER BY LOWER(vendor_name), as_of_date DESC, created_at DESC
        """,
        lower_names,
        cutoff,
        analysis_window_days,
    )
    unresolved = set(lower_names)
    for row in synth_rows:
        lowered = str(row.get("vendor_name") or "").strip().lower()
        requested_name = requested_by_lower.get(lowered)
        if not requested_name:
            continue
        raw = row["synthesis"]
        if isinstance(raw, str):
            raw = _json.loads(raw)
        view = load_synthesis_view(
            raw,
            vendor_name=row["vendor_name"],
            schema_version=str(row.get("schema_version") or ""),
            as_of_date=row["as_of_date"],
        )
        snapshots[requested_name] = {
            "archetype": view.primary_wedge.value if view.primary_wedge else "",
            "confidence": _confidence_label_to_numeric(view.confidence("causal_narrative")),
            "snapshot_date": view.as_of_date_iso,
        }
        unresolved.discard(lowered)

    if unresolved:
        legacy_rows = await pool.fetch(
            """
            SELECT DISTINCT ON (LOWER(vendor_name))
                   vendor_name, archetype, archetype_confidence, last_computed_at
            FROM b2b_churn_signals
            WHERE LOWER(vendor_name) = ANY($1)
              AND last_computed_at::date < $2
              AND archetype IS NOT NULL
            ORDER BY LOWER(vendor_name), last_computed_at DESC
            """,
            list(unresolved),
            cutoff,
        )
        for row in legacy_rows:
            lowered = str(row.get("vendor_name") or "").strip().lower()
            requested_name = requested_by_lower.get(lowered)
            if not requested_name:
                continue
            last_computed_at = row.get("last_computed_at")
            if last_computed_at is not None and hasattr(last_computed_at, "date"):
                snapshot_date = last_computed_at.date().isoformat()
            else:
                snapshot_date = str(last_computed_at or "")
            snapshots[requested_name] = {
                "archetype": row.get("archetype") or "",
                "confidence": float(row["archetype_confidence"]) if row.get("archetype_confidence") is not None else None,
                "snapshot_date": snapshot_date,
            }

    return snapshots
