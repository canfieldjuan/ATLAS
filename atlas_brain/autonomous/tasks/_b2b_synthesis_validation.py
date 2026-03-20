"""Post-LLM, pre-persist validation for reasoning synthesis v2.

Runs between the LLM response parse and the DB persist.  Errors block
persistence (v1 fallback stays); warnings are attached to the row.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ...reasoning.wedge_registry import WEDGE_ENUM_VALUES, validate_wedge

logger = logging.getLogger("atlas.autonomous.tasks.b2b_synthesis_validation")

_VALID_CONFIDENCE = frozenset({"high", "medium", "low", "insufficient"})
_VALID_EVIDENCE_TYPE = frozenset({
    "explicit_switch", "active_evaluation", "insufficient_data",
})
_VALID_TRIGGER_TYPES = frozenset({
    "deadline", "spike", "announcement", "seasonal", "signal",
})
_VALID_SIGNAL_SOURCES = frozenset({
    "evidence_vault", "temporal", "displacement", "segment",
    "category", "accounts",
})
_REQUIRED_SECTIONS = (
    "causal_narrative", "segment_playbook", "timing_intelligence",
    "competitive_reframes", "migration_proof",
)
_UNCLASSIFIED_VAULT_PREFIXES = (
    "vault:weakness:unknown",
    "vault:strength:unknown",
)
_FIELD_SOURCE_RULES = {
    "migration_proof.switch_volume": {
        "displacement:aggregate:total_explicit_switches",
    },
    "migration_proof.active_evaluation_volume": {
        "displacement:aggregate:total_active_evaluations",
    },
    "migration_proof.displacement_mention_volume": {
        "vault:metric:displacement_mention_count",
        "category:aggregate:displacement_flow_count",
    },
    "timing_intelligence.active_eval_signals": {
        "accounts:summary:active_eval_signal_count",
        "temporal:signal:evaluation_deadline_signals",
    },
    "segment_playbook.priority_segments[*].estimated_reach": {
        "accounts:summary:total_accounts",
        "accounts:summary:high_intent_count",
    },
}


@dataclass(slots=True)
class ValidationError:
    """A single validation finding."""

    path: str          # e.g. "causal_narrative.primary_wedge"
    code: str          # e.g. "invalid_wedge"
    message: str
    severity: str      # "error" or "warning"


@dataclass(slots=True)
class ValidationResult:
    """Outcome of validating a synthesis dict against the source packet."""

    vendor_name: str
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        parts = [f"vendor={self.vendor_name}"]
        if self.errors:
            parts.append(f"errors={len(self.errors)}")
        if self.warnings:
            parts.append(f"warnings={len(self.warnings)}")
        if self.is_valid:
            parts.append("PASS")
        else:
            parts.append("FAIL")
        return " ".join(parts)


def _add(
    result: ValidationResult,
    path: str,
    code: str,
    message: str,
    severity: str = "error",
) -> None:
    finding = ValidationError(path=path, code=code, message=message, severity=severity)
    if severity == "error":
        result.errors.append(finding)
    else:
        result.warnings.append(finding)


def _normalized_path(path: str) -> str:
    return re.sub(r"\[\d+\]", "[*]", path)


def _is_unclassified_vault_source_id(source_id: str) -> bool:
    return any(source_id.startswith(prefix) for prefix in _UNCLASSIFIED_VAULT_PREFIXES)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_required_sections(
    synthesis: dict[str, Any], result: ValidationResult,
) -> None:
    for section in _REQUIRED_SECTIONS:
        if section not in synthesis or not isinstance(synthesis.get(section), dict):
            _add(result, section, "missing_section", f"Required section '{section}' is missing or not a dict")


def _check_wedge(
    synthesis: dict[str, Any], result: ValidationResult,
) -> None:
    cn = synthesis.get("causal_narrative")
    if not isinstance(cn, dict):
        return
    wedge_val = cn.get("primary_wedge")
    if wedge_val is None:
        _add(result, "causal_narrative.primary_wedge", "missing_wedge", "primary_wedge is required")
        return
    if validate_wedge(wedge_val) is None:
        _add(
            result,
            "causal_narrative.primary_wedge",
            "invalid_wedge",
            f"primary_wedge '{wedge_val}' is not in the wedge registry. "
            f"Valid values: {sorted(WEDGE_ENUM_VALUES)}",
        )


def _check_confidence_fields(
    synthesis: dict[str, Any], result: ValidationResult,
) -> None:
    for section in _REQUIRED_SECTIONS:
        sec = synthesis.get(section)
        if not isinstance(sec, dict):
            continue
        conf = sec.get("confidence")
        if conf is not None and conf not in _VALID_CONFIDENCE:
            _add(
                result,
                f"{section}.confidence",
                "invalid_confidence",
                f"confidence '{conf}' not in {sorted(_VALID_CONFIDENCE)}",
                severity="warning",
            )


def _check_data_gaps_on_insufficient(
    synthesis: dict[str, Any], result: ValidationResult,
) -> None:
    for section in _REQUIRED_SECTIONS:
        sec = synthesis.get(section)
        if not isinstance(sec, dict):
            continue
        if sec.get("confidence") == "insufficient":
            gaps = sec.get("data_gaps")
            if not gaps or not isinstance(gaps, list) or len(gaps) == 0:
                _add(
                    result,
                    f"{section}.data_gaps",
                    "missing_data_gaps",
                    f"Section '{section}' has confidence=insufficient but no data_gaps",
                    severity="warning",
                )


def _check_citations(
    synthesis: dict[str, Any],
    valid_source_ids: frozenset[str],
    result: ValidationResult,
) -> None:
    """Warn when sections lack citations arrays."""
    # Sections that should have top-level citations
    for section in ("causal_narrative", "timing_intelligence", "migration_proof"):
        sec = synthesis.get(section)
        if not isinstance(sec, dict):
            continue
        cites = sec.get("citations")
        if not cites or not isinstance(cites, list) or len(cites) == 0:
            _add(
                result,
                f"{section}.citations",
                "missing_citations",
                f"Section '{section}' has no citations array",
                severity="warning",
            )
        elif valid_source_ids:
            for sid in cites:
                if isinstance(sid, str) and sid and sid not in valid_source_ids:
                    _add(
                        result,
                        f"{section}.citations",
                        "hallucinated_source_id",
                        f"citation '{sid}' does not exist in the input packet",
                    )
                elif isinstance(sid, str) and _is_unclassified_vault_source_id(sid):
                    _add(
                        result,
                        f"{section}.citations",
                        "unclassified_source_id",
                        f"citation '{sid}' is not specific enough for synthesis output",
                    )

    # Per-segment citations
    sp = synthesis.get("segment_playbook")
    if isinstance(sp, dict):
        for i, seg in enumerate(sp.get("priority_segments") or []):
            if not isinstance(seg, dict):
                continue
            cites = seg.get("citations")
            if not cites or not isinstance(cites, list):
                _add(
                    result,
                    f"segment_playbook.priority_segments[{i}].citations",
                    "missing_citations",
                    "Segment has no citations array",
                    severity="warning",
                )

    # Per-reframe citations
    cr = synthesis.get("competitive_reframes")
    if isinstance(cr, dict):
        for i, rf in enumerate(cr.get("reframes") or []):
            if not isinstance(rf, dict):
                continue
            cites = rf.get("citations")
            if not cites or not isinstance(cites, list):
                _add(
                    result,
                    f"competitive_reframes.reframes[{i}].citations",
                    "missing_citations",
                    "Reframe has no citations array",
                    severity="warning",
                )


def _check_source_ids(
    synthesis: dict[str, Any],
    valid_source_ids: frozenset[str],
    result: ValidationResult,
) -> None:
    """Verify that every source_id in the synthesis exists in the packet."""
    if not valid_source_ids:
        return

    def _walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            sid = obj.get("source_id")
            if isinstance(sid, str) and sid:
                if sid not in valid_source_ids:
                    _add(
                        result, path, "hallucinated_source_id",
                        f"source_id '{sid}' does not exist in the input packet",
                    )
                elif _is_unclassified_vault_source_id(sid):
                    _add(
                        result, path, "unclassified_source_id",
                        f"source_id '{sid}' is not specific enough for synthesis output",
                    )
            for k, v in obj.items():
                _walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _walk(v, f"{path}[{i}]")

    for section in _REQUIRED_SECTIONS:
        sec = synthesis.get(section)
        if isinstance(sec, dict):
            _walk(sec, section)


def _check_value_wrappers(
    synthesis: dict[str, Any],
    aggregate_lookup: dict[str, Any],
    result: ValidationResult,
) -> None:
    """Verify {value, source_id} wrappers match pre-computed aggregates."""
    if not aggregate_lookup:
        return

    def _walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            if "value" in obj and "source_id" in obj:
                sid = obj.get("source_id", "")
                val = obj.get("value")
                if sid and sid in aggregate_lookup:
                    expected = aggregate_lookup[sid]
                    if val is not None and expected is not None:
                        if str(val) != str(expected):
                            _add(
                                result, path, "value_mismatch",
                                f"value {val!r} does not match aggregate "
                                f"{expected!r} for source_id '{sid}'",
                                severity="warning",
                            )
            for k, v in obj.items():
                _walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _walk(v, f"{path}[{i}]")

    for section in _REQUIRED_SECTIONS:
        sec = synthesis.get(section)
        if isinstance(sec, dict):
            _walk(sec, section)


def _check_migration_coherence(
    synthesis: dict[str, Any], result: ValidationResult,
) -> None:
    """Flag switching_is_real=true when switch_volume is 0."""
    mp = synthesis.get("migration_proof")
    if not isinstance(mp, dict):
        return
    switching = mp.get("switching_is_real")
    sv = mp.get("switch_volume")
    if isinstance(sv, dict):
        vol = sv.get("value")
    else:
        vol = sv
    # switching_is_real can be bool or string "true"/"false"
    is_real = switching is True or (isinstance(switching, str) and switching.lower() == "true")
    vol_zero = vol is not None and (vol == 0 or vol == "0")
    if is_real and vol_zero:
        et = mp.get("evidence_type", "")
        if et != "active_evaluation":
            _add(
                result,
                "migration_proof",
                "switching_real_but_zero_volume",
                "switching_is_real=true but switch_volume=0 and evidence_type "
                f"is '{et}', not 'active_evaluation'. This will confuse reps.",
                severity="warning",
            )


def _check_evidence_type(
    synthesis: dict[str, Any], result: ValidationResult,
) -> None:
    mp = synthesis.get("migration_proof")
    if not isinstance(mp, dict):
        return
    et = mp.get("evidence_type")
    if et is not None and et not in _VALID_EVIDENCE_TYPE:
        _add(
            result,
            "migration_proof.evidence_type",
            "invalid_evidence_type",
            f"evidence_type '{et}' not in {sorted(_VALID_EVIDENCE_TYPE)}",
            severity="warning",
        )


def _check_falsification_structure(
    synthesis: dict[str, Any], result: ValidationResult,
) -> None:
    """Warn if what_would_weaken_thesis items are bare strings instead of structured."""
    cn = synthesis.get("causal_narrative")
    if not isinstance(cn, dict):
        return
    thesis = cn.get("what_would_weaken_thesis")
    if not isinstance(thesis, list):
        return
    for i, item in enumerate(thesis):
        if isinstance(item, str):
            _add(
                result,
                f"causal_narrative.what_would_weaken_thesis[{i}]",
                "unstructured_falsification",
                "Falsification condition is a bare string; expected "
                "{condition, signal_source, monitorable}",
                severity="warning",
            )


def _check_trigger_types(
    synthesis: dict[str, Any], result: ValidationResult,
) -> None:
    """Warn if immediate_triggers use unknown type values."""
    ti = synthesis.get("timing_intelligence")
    if not isinstance(ti, dict):
        return
    triggers = ti.get("immediate_triggers")
    if not isinstance(triggers, list):
        return
    for i, t in enumerate(triggers):
        if not isinstance(t, dict):
            continue
        ttype = t.get("type")
        if ttype and ttype not in _VALID_TRIGGER_TYPES:
            _add(
                result,
                f"timing_intelligence.immediate_triggers[{i}].type",
                "invalid_trigger_type",
                f"trigger type '{ttype}' not in {sorted(_VALID_TRIGGER_TYPES)}",
                severity="warning",
            )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_synthesis(
    synthesis: dict[str, Any],
    packet: Any | None = None,
) -> ValidationResult:
    """Validate a parsed synthesis dict.

    Parameters
    ----------
    synthesis : dict
        The parsed LLM output.
    packet : CompressedPacket, optional
        The input packet (for source_id and value verification).

    Returns
    -------
    ValidationResult
        ``.is_valid`` is True if no errors (warnings are OK).
    """
    vendor_name = ""
    valid_source_ids: frozenset[str] = frozenset()
    aggregate_lookup: dict[str, Any] = {}

    if packet is not None:
        vendor_name = getattr(packet, "vendor_name", "")
        valid_source_ids = packet.source_ids()
        for agg in getattr(packet, "aggregates", []):
            aggregate_lookup[agg.source_id] = agg.value

    result = ValidationResult(vendor_name=vendor_name)

    _check_required_sections(synthesis, result)
    _check_wedge(synthesis, result)
    _check_confidence_fields(synthesis, result)
    _check_data_gaps_on_insufficient(synthesis, result)
    _check_citations(synthesis, valid_source_ids, result)
    _check_source_ids(synthesis, valid_source_ids, result)
    _check_value_wrappers(synthesis, aggregate_lookup, result)
    _check_evidence_type(synthesis, result)
    _check_migration_coherence(synthesis, result)
    _check_falsification_structure(synthesis, result)
    _check_trigger_types(synthesis, result)

    if result.errors:
        logger.warning("Synthesis validation FAILED for %s: %s", vendor_name, result.summary())
    elif result.warnings:
        logger.info("Synthesis validation passed with warnings for %s: %s", vendor_name, result.summary())

    return result
