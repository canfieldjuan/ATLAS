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
from ...services.company_normalization import normalize_company_name

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
_SECTION_CONTRACT_PATHS = {
    "causal_narrative": ("vendor_core_reasoning", "causal_narrative"),
    "segment_playbook": ("vendor_core_reasoning", "segment_playbook"),
    "timing_intelligence": ("vendor_core_reasoning", "timing_intelligence"),
    "competitive_reframes": ("displacement_reasoning", "competitive_reframes"),
    "migration_proof": ("displacement_reasoning", "migration_proof"),
    "account_reasoning": ("account_reasoning",),
    "category_reasoning": ("category_reasoning",),
}
_REQUIRED_SECTIONS = (
    "causal_narrative", "segment_playbook", "timing_intelligence",
    "competitive_reframes", "migration_proof", "account_reasoning",
    "category_reasoning",
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
        "accounts:summary:active_eval_signal_count",
        "segment:aggregate:active_eval_signal_count",
        "temporal:signal:evaluation_deadline_signals",
    },
    "migration_proof.displacement_mention_volume": {
        "vault:metric:displacement_mention_count",
        "category:aggregate:displacement_flow_count",
        "displacement:aggregate:total_flow_mentions",
    },
    "timing_intelligence.active_eval_signals": {
        "accounts:summary:active_eval_signal_count",
        "segment:aggregate:active_eval_signal_count",
        "temporal:signal:evaluation_deadline_signals",
        "displacement:aggregate:total_active_evaluations",
    },
    "segment_playbook.priority_segments[*].estimated_reach": {
        "accounts:summary:total_accounts",
        "accounts:summary:high_intent_count",
    },
    "account_reasoning.total_accounts": {
        "accounts:summary:total_accounts",
    },
    "account_reasoning.high_intent_count": {
        "accounts:summary:high_intent_count",
    },
    "account_reasoning.active_eval_count": {
        "accounts:summary:active_eval_signal_count",
    },
}

_FIELD_SOURCE_PREFIX_RULES = {
    "segment_playbook.priority_segments[*].estimated_reach": (
        "segment:reach:",
    ),
    # Weakness mention totals are acceptable broader-signal proxies when
    # displacement-specific counts are thin (prompt instructs the LLM to use
    # the broadest valid non-switch mention aggregate as a fallback).
    "migration_proof.displacement_mention_volume": (
        "displacement:flow:",
        "vault:weakness:",
        "vault:metric:",
        "category:",
    ),
}

_FIELDS_REQUIRING_AGGREGATE_SOURCE = {
    "competitive_reframes.reframes[*].proof_point",
}

_PLACEHOLDER_NAMED_EXAMPLE_PATTERNS = (
    re.compile(r"^unknown\b", re.I),
    re.compile(r"^anonymous\b", re.I),
    re.compile(r"^redacted\b", re.I),
    re.compile(r"\b(customer|prospect|company|account|buyer|reviewer|user|team|organization|organisation|business|department)\b", re.I),
    re.compile(r"\b(smb|mid[\s_-]?market|enterprise)\b", re.I),
)


def _looks_like_tool_label(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    generic_modifiers = ("custom", "homegrown", "home-grown", "in-house", "internal")
    generic_artifacts = (
        "integration", "utility", "workflow", "tool", "stack",
        "system", "automation", "bot",
    )
    return any(token in text for token in generic_modifiers) and any(
        token in text for token in generic_artifacts
    )


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


def _is_placeholder_named_example_company(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    normalized = normalize_company_name(text)
    if not normalized:
        return True
    if _looks_like_tool_label(text):
        return True
    return any(pattern.search(text) for pattern in _PLACEHOLDER_NAMED_EXAMPLE_PATTERNS)


def _build_source_alias_map(packet: Any | None) -> dict[str, str]:
    """Map common shorthand source labels back to canonical packet source IDs."""
    if packet is None:
        return {}

    alias_map: dict[str, str | None] = {}

    def _register(alias: str, canonical: str) -> None:
        key = str(alias or "").strip()
        if not key or key == canonical:
            return
        existing = alias_map.get(key)
        if existing is None:
            alias_map[key] = canonical
        elif existing != canonical:
            alias_map[key] = None

    for agg in getattr(packet, "aggregates", []):
        canonical = agg.source_id
        _register(agg.label, canonical)
        _register(canonical.rsplit(":", 1)[-1], canonical)
        if canonical == "category:aggregate:displacement_flow_count":
            _register("displacement:aggregate:total_flow_mentions", canonical)
        if canonical.endswith("_signal_count"):
            _register(canonical.replace("_signal_count", "_count"), canonical)
        if canonical.endswith(":mention_count_total"):
            _register(canonical.rsplit(":", 1)[0], canonical)
            prefix = canonical[: -len(":mention_count_total")]
            _register(f"{prefix}_mention_count_total", canonical)
        if canonical.endswith(":mention_count_recent"):
            prefix = canonical[: -len(":mention_count_recent")]
            _register(f"{prefix}_mention_count_recent", canonical)
        if canonical.startswith("segment:reach:"):
            _register(canonical.replace("segment:reach:", "segment:", 1), canonical)
    if any(
        getattr(agg, "source_id", "") == "accounts:summary:total_accounts"
        for agg in getattr(packet, "aggregates", []) or []
    ):
        _register("accounts", "accounts:summary:total_accounts")

    return {
        key: val for key, val in alias_map.items()
        if isinstance(val, str) and val
    }


def _section_dicts(synthesis: dict[str, Any], section: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current = synthesis.get(section)
    if isinstance(current, dict):
        blocks.append(current)
    contract_path = _SECTION_CONTRACT_PATHS.get(section)
    if contract_path:
        contract_name = contract_path[0]
        contract = _contract_block(synthesis, contract_name)
        nested: Any = contract
        for path_part in contract_path[1:]:
            if not isinstance(nested, dict):
                nested = None
                break
            nested = nested.get(path_part)
        if isinstance(nested, dict) and nested not in blocks:
            blocks.append(nested)
    return blocks


def _normalize_constrained_wrappers(
    synthesis: dict[str, Any],
    packet: Any | None,
) -> None:
    if packet is None:
        return
    alias_map = _build_source_alias_map(packet)
    aggregate_lookup = {
        getattr(agg, "source_id", ""): getattr(agg, "value", None)
        for agg in getattr(packet, "aggregates", []) or []
        if getattr(agg, "source_id", "")
    }
    has_segment_reach = any(
        getattr(agg, "source_id", "").startswith("segment:reach:")
        for agg in getattr(packet, "aggregates", []) or []
    )

    def _numeric_equal(left: Any, right: Any) -> bool:
        try:
            return float(left) == float(right)
        except (TypeError, ValueError):
            return left == right

    def _candidate_wrappers(field_path: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for sid in _FIELD_SOURCE_RULES.get(field_path, set()):
            if sid in aggregate_lookup:
                candidates.append({"value": aggregate_lookup[sid], "source_id": sid})
        for prefix in _FIELD_SOURCE_PREFIX_RULES.get(field_path, ()):
            for sid, value in aggregate_lookup.items():
                if sid.startswith(prefix):
                    candidates.append({"value": value, "source_id": sid})
        return candidates

    def _best_wrapper_for_field(
        wrapper: Any,
        field_path: str,
    ) -> dict[str, Any] | None:
        if not isinstance(wrapper, dict):
            return None
        sid = wrapper.get("source_id")
        alias = alias_map.get(sid) if isinstance(sid, str) else None
        if isinstance(alias, str) and alias and alias != sid:
            wrapper = {**wrapper, "source_id": alias}
        for candidate in _candidate_wrappers(field_path):
            if _numeric_equal(candidate.get("value"), wrapper.get("value")):
                return candidate
        if field_path in _FIELDS_REQUIRING_AGGREGATE_SOURCE:
            for sid, value in aggregate_lookup.items():
                if _numeric_equal(value, wrapper.get("value")):
                    return {"value": value, "source_id": sid}
        sid = wrapper.get("source_id")
        if isinstance(sid, str) and sid in aggregate_lookup:
            return {"value": aggregate_lookup[sid], "source_id": sid}
        return wrapper

    def _upgrade_wrapper_source(
        wrapper: Any,
        field_path: str,
    ) -> None:
        if not isinstance(wrapper, dict):
            return
        sid = wrapper.get("source_id")
        if not isinstance(sid, str) or not sid:
            return
        resolved_sid = sid
        alias = alias_map.get(sid)
        if not isinstance(alias, str) or not alias:
            alias = sid
        if not _field_source_allowed(field_path, sid):
            if (
                field_path not in _FIELDS_REQUIRING_AGGREGATE_SOURCE
                and not _field_source_allowed(field_path, alias)
            ):
                return
            wrapper["source_id"] = alias
            resolved_sid = alias
        if (
            (
                field_path in _FIELDS_REQUIRING_AGGREGATE_SOURCE
                or field_path in _FIELD_SOURCE_RULES
                or field_path in _FIELD_SOURCE_PREFIX_RULES
            )
            and resolved_sid in aggregate_lookup
        ):
            wrapper["value"] = aggregate_lookup[resolved_sid]

    explicit_switch = None
    for agg in getattr(packet, "aggregates", []) or []:
        if getattr(agg, "source_id", "") == "displacement:aggregate:total_explicit_switches":
            explicit_switch = {
                "value": getattr(agg, "value", None),
                "source_id": "displacement:aggregate:total_explicit_switches",
            }
            break
    if explicit_switch is None:
        return
    for migration_proof in _section_dicts(synthesis, "migration_proof"):
        if isinstance(migration_proof.get("switch_volume"), dict):
            migration_proof["switch_volume"] = dict(explicit_switch)
        for field_path, key in (
            ("migration_proof.active_evaluation_volume", "active_evaluation_volume"),
            ("migration_proof.displacement_mention_volume", "displacement_mention_volume"),
        ):
            normalized = _best_wrapper_for_field(migration_proof.get(key), field_path)
            if normalized is not None:
                migration_proof[key] = normalized

    for timing_intelligence in _section_dicts(synthesis, "timing_intelligence"):
        normalized = _best_wrapper_for_field(
            timing_intelligence.get("active_eval_signals"),
            "timing_intelligence.active_eval_signals",
        )
        if normalized is not None:
            timing_intelligence["active_eval_signals"] = normalized

    for account_reasoning in _section_dicts(synthesis, "account_reasoning"):
        for field_path, key in (
            ("account_reasoning.total_accounts", "total_accounts"),
            ("account_reasoning.high_intent_count", "high_intent_count"),
            ("account_reasoning.active_eval_count", "active_eval_count"),
        ):
            normalized = _best_wrapper_for_field(account_reasoning.get(key), field_path)
            if normalized is not None:
                account_reasoning[key] = normalized

    for segment_playbook in _section_dicts(synthesis, "segment_playbook"):
        if not has_segment_reach:
            segment_playbook["priority_segments"] = []
            continue
        normalized_segments: list[dict[str, Any]] = []
        for seg in segment_playbook.get("priority_segments") or []:
            if not isinstance(seg, dict):
                continue
            normalized = _best_wrapper_for_field(
                seg.get("estimated_reach"),
                "segment_playbook.priority_segments[*].estimated_reach",
            )
            if not isinstance(normalized, dict):
                continue
            sid = normalized.get("source_id")
            if not isinstance(sid, str) or not _field_source_allowed(
                "segment_playbook.priority_segments[*].estimated_reach",
                sid,
            ):
                continue
            reach_value = normalized.get("value")
            try:
                if float(reach_value) < 5:
                    continue
            except (TypeError, ValueError):
                continue
            seg["estimated_reach"] = normalized
            normalized_segments.append(seg)
        segment_playbook["priority_segments"] = normalized_segments

    for competitive_reframes in _section_dicts(synthesis, "competitive_reframes"):
        for reframe in competitive_reframes.get("reframes") or []:
            if not isinstance(reframe, dict):
                continue
            normalized = _best_wrapper_for_field(
                reframe.get("proof_point"),
                "competitive_reframes.reframes[*].proof_point",
            )
            if normalized is not None:
                reframe["proof_point"] = normalized


def normalize_synthesis_source_ids(
    synthesis: dict[str, Any],
    packet: Any | None = None,
) -> dict[str, Any]:
    """Normalize shorthand source labels to packet source IDs in-place."""
    if packet is None or not isinstance(synthesis, dict):
        return synthesis

    valid_source_ids = packet.source_ids()
    alias_map = _build_source_alias_map(packet)
    if alias_map:
        def _normalize_sid(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            sid = value.strip()
            if not sid or sid in valid_source_ids:
                return value
            return alias_map.get(sid, value)

        def _walk(obj: Any) -> None:
            if isinstance(obj, dict):
                sid = obj.get("source_id")
                if isinstance(sid, str):
                    obj["source_id"] = _normalize_sid(sid)
                cites = obj.get("citations")
                if isinstance(cites, list):
                    obj["citations"] = [_normalize_sid(sid) for sid in cites]
                for value in obj.values():
                    _walk(value)
            elif isinstance(obj, list):
                for value in obj:
                    _walk(value)

        _walk(synthesis)
    _normalize_constrained_wrappers(synthesis, packet)
    return synthesis


def _field_source_allowed(path: str, sid: str) -> bool:
    allowed_sources = _FIELD_SOURCE_RULES.get(path)
    if allowed_sources and sid in allowed_sources:
        return True
    allowed_prefixes = _FIELD_SOURCE_PREFIX_RULES.get(path) or ()
    return any(sid.startswith(prefix) for prefix in allowed_prefixes)


def _contract_block(synthesis: dict[str, Any], name: str) -> dict[str, Any]:
    contracts = synthesis.get("reasoning_contracts")
    if isinstance(contracts, dict):
        block = contracts.get(name)
        if isinstance(block, dict):
            return block
    return {}


def _normalized_synthesis_for_validation(synthesis: dict[str, Any]) -> dict[str, Any]:
    """Materialize required flat sections from contract blocks for validation.

    The persisted canonical shape is contracts-first, but validation rules are
    still expressed against the section-level names. This view lets validation
    operate on both legacy flat payloads and native contracts-first payloads.
    """
    normalized = dict(synthesis)
    contracts = synthesis.get("reasoning_contracts")
    has_explicit_contracts = isinstance(contracts, dict) and bool(contracts)
    if has_explicit_contracts:
        for section, contract_path in _SECTION_CONTRACT_PATHS.items():
            contract_name = contract_path[0] if contract_path else ""
            contract = _contract_block(synthesis, contract_name) if contract_name else {}
            if not contract:
                continue
            nested: Any = contract
            for path_part in contract_path[1:]:
                if not isinstance(nested, dict):
                    nested = None
                    break
                nested = nested.get(path_part)
            if isinstance(nested, dict):
                normalized.pop(section, None)
        if _contract_block(synthesis, "category_reasoning"):
            normalized.pop("category_reasoning", None)
    for section, contract_path in _SECTION_CONTRACT_PATHS.items():
        if isinstance(normalized.get(section), dict):
            continue
        if not contract_path:
            continue
        contract_name = contract_path[0]
        contract = _contract_block(synthesis, contract_name)
        if not contract:
            continue
        nested: Any = contract
        for path_part in contract_path[1:]:
            if not isinstance(nested, dict):
                nested = None
                break
            nested = nested.get(path_part)
        if isinstance(nested, dict):
            normalized[section] = nested
    # category_reasoning lives directly under reasoning_contracts (not nested)
    if not isinstance(normalized.get("category_reasoning"), dict):
        cr = _contract_block(synthesis, "category_reasoning")
        if cr:
            normalized["category_reasoning"] = cr
    # Do NOT auto-stub missing sections. Let _check_required_sections()
    # report them as real validation errors so synthesis can be retried
    # or rejected by the quality gate.
    return normalized


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
        if conf is None:
            _add(
                result,
                f"{section}.confidence",
                "missing_confidence",
                f"Required section '{section}' has no confidence field",
            )
        elif conf not in _VALID_CONFIDENCE:
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


def _collect_citation_entries(
    value: Any,
    path: str,
    sink: list[tuple[str, list[str]]],
) -> None:
    if isinstance(value, dict):
        citations = value.get("citations")
        if isinstance(citations, list):
            sink.append((
                f"{path}.citations",
                [
                    str(sid).strip()
                    for sid in citations
                    if isinstance(sid, str) and str(sid).strip()
                ],
            ))
        for key, item in value.items():
            _collect_citation_entries(item, f"{path}.{key}", sink)
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            _collect_citation_entries(item, f"{path}[{idx}]", sink)


def _check_packet_citations(
    synthesis: dict[str, Any],
    valid_source_ids: frozenset[str],
    result: ValidationResult,
) -> None:
    """Verify every citation id exists in the input packet."""
    if not valid_source_ids:
        return

    citation_entries: list[tuple[str, list[str]]] = []
    for section in _REQUIRED_SECTIONS:
        sec = synthesis.get(section)
        if isinstance(sec, dict):
            _collect_citation_entries(sec, section, citation_entries)

    for path, citations in citation_entries:
        if not citations:
            continue
        unclassified_ids = sorted({
            sid for sid in citations
            if _is_unclassified_vault_source_id(sid)
        })
        if unclassified_ids:
            _add(
                result,
                path,
                "unclassified_source_id",
                "citations are not specific enough for synthesis output: "
                + ", ".join(unclassified_ids),
            )
        unknown_ids = sorted({
            sid for sid in citations
            if sid not in valid_source_ids and not _is_unclassified_vault_source_id(sid)
        })
        if unknown_ids:
            _add(
                result,
                path,
                "unknown_packet_citation",
                "citations reference ids not present in the input packet: "
                + ", ".join(unknown_ids),
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


def _check_witness_pack_governance(
    packet: Any | None,
    result: ValidationResult,
    *,
    governance_blocking: bool = False,
) -> None:
    """Check witness pack quality. Blocking for schema >= 2.3."""
    if packet is None:
        return
    governance = {}
    if isinstance(getattr(packet, "section_packets", None), dict):
        governance = getattr(packet, "section_packets", {}).get("_witness_governance") or {}
    witness_pack = getattr(packet, "witness_pack", []) or []

    # Empty witness pack
    if not witness_pack:
        _add(
            result,
            "packet_artifacts.witness_pack",
            "empty_witness_pack",
            "synthesis has no witness evidence to anchor reasoning",
            severity="error" if governance_blocking else "warning",
        )
        return

    # Thin specific pool (only generic fallbacks used)
    thin_pool = bool(governance.get("thin_specific_witness_pool"))
    if not thin_pool:
        thin_pool = any(
            str(witness.get("generic_reason") or "").strip()
            for witness in witness_pack
            if isinstance(witness, dict)
        )
    if thin_pool:
        _add(
            result,
            "packet_artifacts.witness_pack",
            "thin_specific_witness_pool",
            "witness selection fell back to generic excerpts because the specific witness pool was too thin",
            severity="error" if governance_blocking else "warning",
        )


def _check_value_wrappers(
    synthesis: dict[str, Any],
    aggregate_lookup: dict[str, Any],
    result: ValidationResult,
) -> None:
    """Verify {value, source_id} wrappers match pre-computed aggregates."""
    if not aggregate_lookup:
        return

    def _numeric_equal(left: Any, right: Any) -> bool:
        try:
            return abs(float(left) - float(right)) < 1e-9
        except (TypeError, ValueError):
            return False

    def _walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            if "value" in obj and "source_id" in obj:
                sid = obj.get("source_id", "")
                val = obj.get("value")
                normalized_path = _normalized_path(path)
                allowed_sources = _FIELD_SOURCE_RULES.get(normalized_path)
                if (allowed_sources or _FIELD_SOURCE_PREFIX_RULES.get(normalized_path)) and not _field_source_allowed(normalized_path, sid):
                    _add(
                        result,
                        path,
                        "invalid_field_source",
                        f"field '{normalized_path}' cannot cite source_id '{sid}'",
                    )
                if normalized_path in _FIELDS_REQUIRING_AGGREGATE_SOURCE and sid not in aggregate_lookup:
                    _add(
                        result,
                        path,
                        "proof_point_requires_aggregate_source",
                        f"field '{normalized_path}' must cite a precomputed aggregate source_id, not '{sid}'",
                    )
                if sid and sid in aggregate_lookup:
                    expected = aggregate_lookup[sid]
                    if val is not None and expected is not None:
                        if not _numeric_equal(val, expected) and str(val) != str(expected):
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
    """Flag inconsistent migration-proof semantics."""
    mp = synthesis.get("migration_proof")
    if not isinstance(mp, dict):
        return
    switching = mp.get("switching_is_real")
    sv = mp.get("switch_volume")
    av = mp.get("active_evaluation_volume")
    if isinstance(sv, dict):
        vol = sv.get("value")
    else:
        vol = sv
    if isinstance(av, dict):
        eval_vol = av.get("value")
    else:
        eval_vol = av
    # switching_is_real can be bool or string "true"/"false"
    is_real = switching is True or (isinstance(switching, str) and switching.lower() == "true")
    vol_zero = vol is not None and (vol == 0 or vol == "0")
    et = mp.get("evidence_type", "")

    if is_real and vol_zero:
        if et != "active_evaluation":
            _add(
                result,
                "migration_proof",
                "switching_real_but_zero_volume",
                "switching_is_real=true but switch_volume=0 and evidence_type "
                f"is '{et}', not 'active_evaluation'. This will confuse reps.",
                severity="warning",
            )
    if is_real and vol_zero:
        _add(
            result,
            "migration_proof.switching_is_real",
            "switching_requires_confirmed_switches",
            "switching_is_real must be false when confirmed switch volume is zero",
        )
    confidence = mp.get("confidence")
    if confidence == "high" and et != "explicit_switch":
        _add(
            result,
            "migration_proof.confidence",
            "migration_confidence_too_high",
            "migration_proof confidence cannot be high without explicit switch evidence",
        )
    if et == "explicit_switch" and vol_zero:
        _add(
            result,
            "migration_proof.evidence_type",
            "explicit_switch_without_volume",
            "evidence_type is explicit_switch but confirmed switch volume is zero",
        )
    if et == "insufficient_data":
        eval_zero = eval_vol is None or eval_vol == 0 or eval_vol == "0"
        if not vol_zero or not eval_zero:
            _add(
                result,
                "migration_proof.evidence_type",
                "insufficient_data_with_signal_volume",
                "evidence_type is insufficient_data despite non-zero switching or evaluation volume",
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


def _check_named_examples(
    synthesis: dict[str, Any], result: ValidationResult,
) -> None:
    mp = synthesis.get("migration_proof")
    if not isinstance(mp, dict):
        return
    for idx, item in enumerate(mp.get("named_examples") or []):
        if not isinstance(item, dict):
            continue
        company = item.get("company")
        if _is_placeholder_named_example_company(company):
            _add(
                result,
                f"migration_proof.named_examples[{idx}].company",
                "placeholder_named_example",
                "named_examples should not use placeholder or segment-style company labels",
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


def _check_segment_constraints(
    synthesis: dict[str, Any], result: ValidationResult,
) -> None:
    """Reject seller-facing segments that are too thin or unsupported."""
    sp = synthesis.get("segment_playbook")
    if not isinstance(sp, dict):
        return
    for i, seg in enumerate(sp.get("priority_segments") or []):
        if not isinstance(seg, dict):
            continue
        reach = seg.get("estimated_reach")
        reach_value = None
        if isinstance(reach, dict):
            reach_value = reach.get("value")
        elif reach is not None:
            reach_value = reach
        reach_num = None
        if reach_value is not None:
            try:
                reach_num = float(reach_value)
            except (TypeError, ValueError):
                reach_num = None
        path = f"segment_playbook.priority_segments[{i}]"
        if reach_num is None:
            _add(
                result,
                f"{path}.estimated_reach",
                "missing_estimated_reach",
                "Priority segment is missing a usable estimated_reach value",
                severity="warning",
            )
        elif reach_num < 5:
            _add(
                result,
                f"{path}.estimated_reach",
                "segment_sample_too_small",
                "Priority segment estimated_reach must be at least 5 for seller-facing output",
            )

        segment_text = " ".join(
            str(seg.get(field) or "")
            for field in ("segment", "why_vulnerable", "best_opening_angle")
        )
        if re.search(r"(?<!\d)\d{1,3}%(?!\d)", segment_text) and (reach_num is None or reach_num < 5):
            _add(
                result,
                path,
                "unsupported_segment_percentage",
                "Percentage-based segment claim requires estimated_reach >= 5",
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


def _check_category_reasoning(
    synthesis: dict[str, Any], result: ValidationResult,
) -> None:
    """Warn when category_reasoning is missing or has empty regime/narrative."""
    cr = synthesis.get("category_reasoning")
    if not isinstance(cr, dict):
        _add(
            result,
            "category_reasoning",
            "missing_category_reasoning",
            "category_reasoning section is missing or not a dict",
            severity="warning",
        )
        return
    regime = str(cr.get("market_regime") or "").strip()
    narrative = str(cr.get("narrative") or "").strip()
    if not regime and not narrative:
        _add(
            result,
            "category_reasoning",
            "empty_category_reasoning",
            "category_reasoning has no market_regime or narrative",
            severity="warning",
        )


# ---------------------------------------------------------------------------
# Phase 3 governance validation
# ---------------------------------------------------------------------------

def _check_metric_ledger_numbers(
    synthesis: dict[str, Any],
    packet: Any | None,
    result: ValidationResult,
) -> None:
    """Reject numeric claims that are not present in the metric_ledger.

    Walks all sections looking for {value, source_id} wrappers and checks
    that their source_id exists in the packet's metric_ledger.
    """
    if packet is None:
        return
    ledger = getattr(packet, "metric_ledger", None) or []
    ledger_sids: set[str] = set()
    for entry in ledger:
        sid = entry.get("_sid")
        if sid:
            ledger_sids.add(sid)

    # All valid packet source IDs (for the "not in packet at all" error)
    all_packet_sids = set(packet.source_ids())

    def _walk_wrappers(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            if "value" in obj and "source_id" in obj:
                sid = obj.get("source_id", "")
                if not sid:
                    return
                val = obj.get("value")
                is_numeric = isinstance(val, (int, float))
                if sid not in all_packet_sids:
                    _add(
                        result,
                        path,
                        "unsupported_number" if is_numeric else "unsupported_source",
                        f"{'numeric' if is_numeric else 'text'} claim cites source_id '{sid}' not in packet",
                        severity="error" if is_numeric else "warning",
                    )
                elif is_numeric and sid not in ledger_sids:
                    _add(
                        result,
                        path,
                        "number_not_in_ledger",
                        f"numeric claim cites source_id '{sid}' present in packet "
                        f"but not surfaced in metric_ledger",
                        severity="warning",
                    )
            else:
                for key, val in obj.items():
                    _walk_wrappers(val, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _walk_wrappers(item, f"{path}[{i}]")

    for section in ("causal_narrative", "segment_playbook", "timing_intelligence",
                     "competitive_reframes", "migration_proof"):
        sec = synthesis.get(section)
        if isinstance(sec, dict):
            _walk_wrappers(sec, section)


def _check_contradiction_hedging(
    synthesis: dict[str, Any],
    packet: Any | None,
    result: ValidationResult,
    *,
    governance_blocking: bool = False,
) -> None:
    """Enforce contradiction-aware hedging when contradiction_rows exist.

    The prompt requires three things:
    1. Confidence capped at medium (error if high)
    2. Contradicting dimensions listed in data_gaps
    3. Contradiction-specific entries in confidence_posture.limits

    Severity is ``error`` when governance_blocking (schema >= 2.3), else ``warning``.
    """
    if packet is None:
        return
    contradictions = getattr(packet, "contradiction_rows", None) or []
    if not contradictions:
        return

    sev = "error" if governance_blocking else "warning"

    cn = synthesis.get("causal_narrative")
    if not isinstance(cn, dict):
        return

    # 1. Confidence cap
    confidence = cn.get("confidence", "")
    if confidence == "high":
        _add(
            result,
            "causal_narrative.confidence",
            "contradiction_confidence_cap",
            f"{len(contradictions)} contradiction(s) detected but confidence is 'high'; "
            "must be 'medium' or lower when contradictory evidence exists",
            severity=sev,
        )

    # 2. Contradicting dimensions must appear in data_gaps
    data_gaps = cn.get("data_gaps") or []
    gap_text = " ".join(str(g) for g in data_gaps).lower()
    contradiction_dims = {
        c.get("dimension", "").lower()
        for c in contradictions if isinstance(c, dict) and c.get("dimension")
    }
    missing_dims = [
        d for d in contradiction_dims
        if d and d not in gap_text
    ]
    if missing_dims:
        _add(
            result,
            "causal_narrative.data_gaps",
            "contradiction_dims_missing_from_data_gaps",
            f"contradictions on [{', '.join(sorted(missing_dims))}] not reflected in data_gaps",
            severity=sev,
        )

    # 3. confidence_posture.limits must reference contradictions
    cp = _get_confidence_posture(synthesis)
    cp_limits = cp.get("limits") or [] if isinstance(cp, dict) else []
    limits_text = " ".join(str(lim) for lim in cp_limits).lower()
    if contradiction_dims and "contradict" not in limits_text:
        missing_in_limits = [
            d for d in contradiction_dims
            if d and d not in limits_text
        ]
        if missing_in_limits:
            _add(
                result,
                "vendor_core_reasoning.confidence_posture.limits",
                "contradiction_dims_missing_from_limits",
                f"contradictions on [{', '.join(sorted(missing_in_limits))}] "
                f"not reflected in confidence_posture.limits",
                severity=sev,
            )


def _get_confidence_posture(synthesis: dict[str, Any]) -> dict[str, Any]:
    """Extract confidence_posture from contracts or flat vendor_core."""
    contracts = synthesis.get("reasoning_contracts")
    if isinstance(contracts, dict):
        vc = contracts.get("vendor_core_reasoning")
        if isinstance(vc, dict):
            cp = vc.get("confidence_posture")
            if isinstance(cp, dict):
                return cp
    vc_flat = synthesis.get("vendor_core_reasoning")
    if isinstance(vc_flat, dict):
        cp = vc_flat.get("confidence_posture")
        if isinstance(cp, dict):
            return cp
    return {}


def _check_why_they_stay(
    synthesis: dict[str, Any],
    packet: Any | None,
    result: ValidationResult,
    *,
    governance_blocking: bool = False,
) -> None:
    """Require why_they_stay when retention proof exists.

    Severity is ``error`` when governance_blocking (schema >= 2.3), else ``warning``.
    """
    if packet is None:
        return
    retention_proof = getattr(packet, "retention_proof", None) or []
    if not retention_proof:
        return

    # Check if why_they_stay exists in contracts
    contracts = synthesis.get("reasoning_contracts")
    if isinstance(contracts, dict):
        vc = contracts.get("vendor_core_reasoning")
        if isinstance(vc, dict) and vc.get("why_they_stay"):
            return

    # Also check flat vendor_core
    vc_flat = synthesis.get("vendor_core_reasoning")
    if isinstance(vc_flat, dict) and vc_flat.get("why_they_stay"):
        return

    _add(
        result,
        "vendor_core_reasoning.why_they_stay",
        "missing_why_they_stay",
        f"{len(retention_proof)} retention strength(s) in evidence but no why_they_stay in contracts",
        severity="error" if governance_blocking else "warning",
    )


def _check_confidence_posture_limits(
    synthesis: dict[str, Any],
    packet: Any | None,
    result: ValidationResult,
    *,
    governance_blocking: bool = False,
) -> None:
    """Require confidence_posture.limits when coverage gaps exist.

    Severity is ``error`` when governance_blocking (schema >= 2.3), else ``warning``.
    """
    if packet is None:
        return
    coverage_gaps = getattr(packet, "coverage_gaps", None) or []
    if not coverage_gaps:
        return

    # Check if confidence_posture.limits exists in contracts
    cp = _get_confidence_posture(synthesis)
    if isinstance(cp, dict) and cp.get("limits"):
        return

    _add(
        result,
        "vendor_core_reasoning.confidence_posture.limits",
        "missing_confidence_limits",
        f"{len(coverage_gaps)} coverage gap(s) detected but no confidence_posture.limits in contracts",
        severity="error" if governance_blocking else "warning",
    )


def _check_high_confidence_without_evidence(
    synthesis: dict[str, Any],
    packet: Any | None,
    result: ValidationResult,
    *,
    governance_blocking: bool = False,
) -> None:
    """Flag sections claiming high confidence when their supporting pool is thin.

    Blocking for schema >= 2.3 to prevent high-confidence claims built on
    zero or insufficient evidence from reaching downstream consumers.
    """
    if packet is None:
        return
    coverage_gaps = getattr(packet, "coverage_gaps", None) or []
    gap_areas = {
        str(g.get("area") or "").lower()
        for g in coverage_gaps
        if isinstance(g, dict)
    }

    # Map pool gap areas to the sections they feed.
    # Coverage gaps use area strings like "displacement", "accounts",
    # "temporal_depth", "strength_pricing", "weakness_support" etc.
    # We match by prefix to handle all variants.
    _POOL_PREFIX_TO_SECTIONS = {
        "displacement": ("migration_proof", "competitive_reframes"),
        "accounts": ("account_reasoning",),
        "segment": ("segment_playbook",),
        "temporal": ("timing_intelligence",),
        "category": ("category_reasoning",),
        # Thin segment samples use area like "strength_X" or "weakness_X"
        "strength": ("segment_playbook",),
        "weakness": ("segment_playbook",),
    }

    # Build effective set of affected sections from gap areas
    affected_sections: set[str] = set()
    for gap_area in gap_areas:
        for prefix, sections in _POOL_PREFIX_TO_SECTIONS.items():
            if gap_area == prefix or gap_area.startswith(f"{prefix}_"):
                affected_sections.update(sections)

    for section_name in affected_sections:
        sec = synthesis.get(section_name)
        if not isinstance(sec, dict):
            continue
        conf = str(sec.get("confidence") or "").strip().lower()
        if conf == "high":
            _add(
                result,
                f"{section_name}.confidence",
                "high_confidence_without_evidence",
                f"section '{section_name}' claims high confidence but its supporting pool has a coverage gap",
                severity="error" if governance_blocking else "warning",
            )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_synthesis(
    synthesis: dict[str, Any],
    packet: Any | None = None,
    *,
    schema_version: str = "",
    governance_blocking: bool | None = None,
) -> ValidationResult:
    """Validate a parsed synthesis dict.

    Parameters
    ----------
    synthesis : dict
        The parsed LLM output.
    packet : CompressedPacket, optional
        The input packet (for source_id and value verification).
    schema_version : str, optional
        Code-controlled schema version. Used for governance gate if
        governance_blocking is not explicitly set.
    governance_blocking : bool, optional
        Explicitly enable/disable binding governance. When None,
        derived from schema_version >= "2.3" or always True for
        current production versions.

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
    normalized = _normalized_synthesis_for_validation(synthesis)

    _check_required_sections(normalized, result)
    _check_wedge(normalized, result)
    _check_confidence_fields(normalized, result)
    _check_data_gaps_on_insufficient(normalized, result)
    _check_citations(normalized, result)
    _check_packet_citations(normalized, valid_source_ids, result)
    _check_source_ids(normalized, valid_source_ids, result)
    _check_value_wrappers(normalized, aggregate_lookup, result)
    _check_evidence_type(normalized, result)
    _check_migration_coherence(normalized, result)
    _check_named_examples(normalized, result)
    _check_falsification_structure(normalized, result)
    _check_segment_constraints(normalized, result)
    _check_trigger_types(normalized, result)
    _check_category_reasoning(normalized, result)

    # Phase 3 governance checks -- blocking when explicitly enabled or for
    # current production versions. Warnings-only for legacy/older rows.
    if governance_blocking is None:
        _schema_raw = schema_version or str(synthesis.get("schema_version") or "2.1")
        # Current production synthesis always gets binding governance
        _governance_blocking = _schema_raw in ("v2", "2.3") or _schema_raw >= "2.3"
    else:
        _governance_blocking = governance_blocking
    _check_metric_ledger_numbers(normalized, packet, result)
    _check_contradiction_hedging(normalized, packet, result, governance_blocking=_governance_blocking)
    _check_why_they_stay(normalized, packet, result, governance_blocking=_governance_blocking)
    _check_confidence_posture_limits(normalized, packet, result, governance_blocking=_governance_blocking)
    _check_high_confidence_without_evidence(normalized, packet, result, governance_blocking=_governance_blocking)
    _check_witness_pack_governance(packet, result, governance_blocking=_governance_blocking)

    if result.errors:
        logger.warning("Synthesis validation FAILED for %s: %s", vendor_name, result.summary())
    elif result.warnings:
        logger.info("Synthesis validation passed with warnings for %s: %s", vendor_name, result.summary())

    return result
