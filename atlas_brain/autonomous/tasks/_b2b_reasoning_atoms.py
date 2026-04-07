"""Deterministic reasoning atoms derived from persisted synthesis contracts.

This module adds a reusable atom layer on top of the existing Stage 5
section contracts without requiring an additional model call.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any


def _copy_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _copy_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _normalize_text(value).lower()
    return text in {"true", "1", "yes"}


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = _normalize_text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _iso_or_empty(value: Any) -> str:
    dt = _coerce_datetime(value)
    return dt.isoformat() if dt else ""


def _normalize_id_list(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = _normalize_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _collect_source_ids(value: Any, sink: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"source_id", "_sid"} and isinstance(item, str) and item.strip():
                sink.add(item.strip())
                continue
            if key == "citations" and isinstance(item, list):
                for citation in item:
                    text = _normalize_text(citation)
                    if text:
                        sink.add(text)
                continue
            _collect_source_ids(item, sink)
    elif isinstance(value, list):
        for item in value:
            _collect_source_ids(item, sink)


def _packet_metric_ids(packet: Any) -> set[str]:
    metric_ids = {
        _normalize_text(getattr(agg, "source_id", ""))
        for agg in getattr(packet, "aggregates", []) or []
        if _normalize_text(getattr(agg, "source_id", ""))
    }
    for entry in getattr(packet, "metric_ledger", []) or []:
        if not isinstance(entry, dict):
            continue
        sid = _normalize_text(entry.get("_sid"))
        if sid:
            metric_ids.add(sid)
    return metric_ids


def _packet_witness_lookup(packet: Any) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for witness in getattr(packet, "witness_pack", []) or []:
        if not isinstance(witness, dict):
            continue
        witness_id = _normalize_text(
            witness.get("witness_id") or witness.get("_sid"),
        )
        if witness_id:
            lookup[witness_id] = dict(witness)
    return lookup


def _lineage_for(value: Any, packet: Any) -> dict[str, Any]:
    source_ids: set[str] = set()
    _collect_source_ids(value, source_ids)
    witness_lookup = _packet_witness_lookup(packet)
    metric_ids = sorted(source_ids.intersection(_packet_metric_ids(packet)))
    witness_ids = sorted(source_ids.intersection(set(witness_lookup.keys())))
    last_supported_at = ""
    datetimes = [
        _coerce_datetime(witness_lookup[witness_id].get("reviewed_at"))
        for witness_id in witness_ids
    ]
    datetimes = [value for value in datetimes if value is not None]
    if datetimes:
        last_supported_at = max(datetimes).isoformat()
    return {
        "metric_ids": metric_ids,
        "witness_ids": witness_ids,
        "source_ids": sorted(source_ids),
        "evidence_count": len(source_ids),
        "last_supported_at": last_supported_at,
    }


def _time_bucket(reviewed_at: Any) -> str:
    dt = _coerce_datetime(reviewed_at)
    if dt is None:
        return "unknown"
    days_old = max((datetime.now(timezone.utc) - dt).days, 0)
    if days_old <= 30:
        return "last_30_days"
    if days_old <= 90:
        return "last_90_days"
    if days_old <= 180:
        return "last_180_days"
    return "older"


def _witness_mix(packet: Any) -> dict[str, int]:
    section_packets = dict(getattr(packet, "section_packets", {}) or {})
    anchor_examples = _copy_dict(section_packets.get("anchor_examples"))
    anchor_ids = {
        _normalize_text(item)
        for values in anchor_examples.values()
        if isinstance(values, list)
        for item in values
        if _normalize_text(item)
    }
    timing_ids = set(
        _normalize_id_list(
            _copy_dict(section_packets.get("timing_packet")).get("witness_ids"),
        ),
    )
    displacement_ids = set(
        _normalize_id_list(
            _copy_dict(section_packets.get("displacement_packet")).get("witness_ids"),
        ),
    )
    retention_ids = set(
        _normalize_id_list(
            _copy_dict(section_packets.get("retention_packet")).get("witness_ids"),
        ),
    )
    contradiction_rows = _copy_list(getattr(packet, "contradiction_rows", []) or [])
    return {
        "anchor_examples": len(anchor_ids),
        "timing": len(timing_ids),
        "displacement": len(displacement_ids),
        "retention": len(retention_ids),
        "contradictions": len(contradiction_rows),
    }


def build_scope_manifest(
    packet: Any,
    *,
    selection_strategy: str = "vendor_facet_packet_v1",
) -> dict[str, Any]:
    witness_pack = [
        dict(item) for item in (getattr(packet, "witness_pack", []) or [])
        if isinstance(item, dict)
    ]
    review_ids = {
        _normalize_text(item.get("review_id"))
        for item in witness_pack
        if _normalize_text(item.get("review_id"))
    }
    coverage_by_source = dict(
        sorted(
            Counter(
                _normalize_text(item.get("source")) or "unknown"
                for item in witness_pack
            ).items(),
        ),
    )
    coverage_by_time_bucket = dict(
        sorted(
            Counter(_time_bucket(item.get("reviewed_at")) for item in witness_pack).items(),
        ),
    )
    total_reviews = 0
    for agg in getattr(packet, "aggregates", []) or []:
        if getattr(agg, "label", "") == "total_reviews":
            total_reviews = _safe_int(getattr(agg, "value", 0), 0)
            break
    governance = _copy_dict(_copy_dict(getattr(packet, "section_packets", {})).get("_witness_governance"))
    coverage_gaps = [
        dict(item)
        for item in (getattr(packet, "coverage_gaps", []) or [])
        if isinstance(item, dict)
    ]
    reasons_dropped: list[str] = []
    filtered_generic = _safe_int(governance.get("filtered_generic_candidates"), 0)
    if filtered_generic > 0:
        reasons_dropped.append(f"filtered_generic_candidates:{filtered_generic}")
    if _normalize_bool(governance.get("thin_specific_witness_pool")):
        reasons_dropped.append("thin_specific_witness_pool")
    for gap in coverage_gaps:
        code = _normalize_text(gap.get("gap") or gap.get("code") or gap.get("label"))
        if code and code not in reasons_dropped:
            reasons_dropped.append(code)
    reviews_in_scope = len(review_ids)
    return {
        "selection_strategy": selection_strategy,
        "reviews_considered_total": total_reviews,
        "reviews_in_scope": reviews_in_scope,
        "witnesses_in_scope": len(witness_pack),
        "witness_mix": _witness_mix(packet),
        "coverage_by_source": coverage_by_source,
        "coverage_by_time_bucket": coverage_by_time_bucket,
        "dropped_evidence_count": max(0, total_reviews - reviews_in_scope),
        "reasons_dropped": reasons_dropped,
    }


def _theses_from_contracts(contracts: dict[str, Any], packet: Any) -> list[dict[str, Any]]:
    vendor_core = _copy_dict(contracts.get("vendor_core_reasoning"))
    displacement = _copy_dict(contracts.get("displacement_reasoning"))
    causal = _copy_dict(vendor_core.get("causal_narrative"))
    if not causal:
        return []
    theses: list[dict[str, Any]] = []
    primary = {
        "thesis_id": "primary_wedge",
        "wedge": _normalize_text(causal.get("primary_wedge")),
        "summary": _normalize_text(causal.get("causal_chain") or causal.get("summary") or causal.get("trigger")),
        "why_now": _normalize_text(causal.get("why_now") or causal.get("trigger")),
        "confidence": _normalize_text(causal.get("confidence")),
        "counter_witness_ids": _normalize_id_list(
            _copy_dict(
                _copy_dict(getattr(packet, "section_packets", {})).get("anchor_examples"),
            ).get("counterevidence"),
        ),
        "monitorable_conditions": _copy_list(causal.get("what_would_weaken_thesis")),
        "ui_priority": 1,
    }
    primary.update(_lineage_for(causal, packet))
    theses.append(primary)

    migration = _copy_dict(displacement.get("migration_proof"))
    if migration:
        summary_parts = [
            _normalize_text(
                _copy_dict(migration.get("top_destination")).get("value")
                or migration.get("top_destination"),
            ),
            _normalize_text(
                _copy_dict(migration.get("primary_switch_driver")).get("value")
                or migration.get("primary_switch_driver"),
            ),
        ]
        migration_thesis = {
            "thesis_id": "migration_pressure",
            "wedge": "switching_pressure",
            "summary": " ".join(part for part in summary_parts if part) or _normalize_text(
                migration.get("evaluation_vs_switching"),
            ),
            "why_now": _normalize_text(migration.get("evaluation_vs_switching")),
            "confidence": _normalize_text(migration.get("confidence")),
            "counter_witness_ids": primary["counter_witness_ids"],
            "monitorable_conditions": [],
            "ui_priority": 2,
        }
        migration_thesis.update(_lineage_for(migration, packet))
        theses.append(migration_thesis)

    reframes = _copy_list(_copy_dict(displacement.get("competitive_reframes")).get("reframes"))
    if reframes:
        top_reframe = _copy_dict(reframes[0])
        reframe_thesis = {
            "thesis_id": "competitive_reframe",
            "wedge": "competitive_reframe",
            "summary": _normalize_text(top_reframe.get("reframe")),
            "why_now": _normalize_text(top_reframe.get("why_buyers_believe_it")),
            "confidence": _normalize_text(
                _copy_dict(displacement.get("competitive_reframes")).get("confidence"),
            ) or "medium",
            "counter_witness_ids": primary["counter_witness_ids"],
            "monitorable_conditions": [],
            "ui_priority": 3,
        }
        reframe_thesis.update(_lineage_for(top_reframe, packet))
        theses.append(reframe_thesis)
    return theses


def _timing_windows_from_contracts(contracts: dict[str, Any], packet: Any) -> list[dict[str, Any]]:
    vendor_core = _copy_dict(contracts.get("vendor_core_reasoning"))
    timing = _copy_dict(vendor_core.get("timing_intelligence"))
    displacement = _copy_dict(contracts.get("displacement_reasoning"))
    windows: list[dict[str, Any]] = []
    best_window = _normalize_text(timing.get("best_timing_window"))
    if best_window:
        row = {
            "window_type": "evaluation",
            "start_or_anchor": best_window,
            "urgency": _normalize_text(timing.get("confidence")) or "medium",
            "recommended_action": _normalize_text(
                _copy_list(timing.get("immediate_triggers"))[:1][0].get("action")
                if _copy_list(timing.get("immediate_triggers"))
                and isinstance(_copy_list(timing.get("immediate_triggers"))[0], dict)
                else "",
            ),
            "supporting_witness_ids": _lineage_for(timing, packet)["witness_ids"],
        }
        row.update(_lineage_for(timing, packet))
        windows.append(row)
    for idx, trigger in enumerate(_copy_list(timing.get("immediate_triggers"))):
        if not isinstance(trigger, dict):
            continue
        row = {
            "window_type": _normalize_text(trigger.get("type")) or "signal",
            "start_or_anchor": _normalize_text(trigger.get("trigger")),
            "urgency": _normalize_text(trigger.get("urgency")),
            "recommended_action": _normalize_text(trigger.get("action")),
            "supporting_witness_ids": _lineage_for(trigger, packet)["witness_ids"],
        }
        row.update(_lineage_for(trigger, packet))
        row["window_id"] = f"trigger_{idx + 1}"
        windows.append(row)
    for idx, trigger in enumerate(_copy_list(displacement.get("switch_triggers"))):
        if not isinstance(trigger, dict):
            continue
        row = {
            "window_type": _normalize_text(trigger.get("type")) or "migration",
            "start_or_anchor": _normalize_text(
                trigger.get("description") or trigger.get("trigger") or trigger.get("window"),
            ),
            "urgency": _normalize_text(trigger.get("urgency")) or "medium",
            "recommended_action": _normalize_text(
                trigger.get("recommended_action") or trigger.get("action"),
            ),
            "supporting_witness_ids": _lineage_for(trigger, packet)["witness_ids"],
        }
        row.update(_lineage_for(trigger, packet))
        row["window_id"] = f"switch_trigger_{idx + 1}"
        windows.append(row)
    return windows


def _proof_points_from_contracts(contracts: dict[str, Any], packet: Any) -> list[dict[str, Any]]:
    proof_points: list[dict[str, Any]] = []
    displacement = _copy_dict(contracts.get("displacement_reasoning"))
    reframes = _copy_list(_copy_dict(displacement.get("competitive_reframes")).get("reframes"))
    for idx, reframe in enumerate(reframes):
        if not isinstance(reframe, dict):
            continue
        proof_point = _copy_dict(reframe.get("proof_point"))
        if not proof_point:
            continue
        row = {
            "label": _normalize_text(reframe.get("incumbent_claim")) or f"proof_point_{idx + 1}",
            "claim_type": _normalize_text(proof_point.get("field")),
            "value": proof_point.get("value"),
            "source_id": _normalize_text(proof_point.get("source_id")),
            "interpretation": _normalize_text(proof_point.get("interpretation")),
            "best_segment": _normalize_text(reframe.get("best_segment")),
            "confidence": _normalize_text(
                _copy_dict(displacement.get("competitive_reframes")).get("confidence"),
            ) or "medium",
        }
        row.update(_lineage_for(reframe, packet))
        proof_points.append(row)
    migration = _copy_dict(displacement.get("migration_proof"))
    switch_volume = _copy_dict(migration.get("switch_volume"))
    if switch_volume:
        row = {
            "label": "switch_volume",
            "claim_type": "migration_volume",
            "value": switch_volume.get("value"),
            "source_id": _normalize_text(switch_volume.get("source_id")),
            "interpretation": _normalize_text(migration.get("evaluation_vs_switching")),
            "best_segment": "",
            "confidence": _normalize_text(migration.get("confidence")),
        }
        row.update(_lineage_for(migration, packet))
        proof_points.append(row)
    return proof_points


def _account_signals_from_contracts(contracts: dict[str, Any], packet: Any) -> list[dict[str, Any]]:
    account_signals: list[dict[str, Any]] = []
    account_reasoning = _copy_dict(contracts.get("account_reasoning"))
    for idx, account in enumerate(_copy_list(account_reasoning.get("top_accounts"))):
        if not isinstance(account, dict):
            continue
        row = {
            "company": _normalize_text(
                account.get("name") or account.get("company") or account.get("account_name"),
            ),
            "role_type": _normalize_text(account.get("role_type")),
            "buying_stage": _normalize_text(account.get("buying_stage")),
            "urgency": account.get("urgency") or account.get("intent_score"),
            "competitor_context": _normalize_text(
                account.get("competitor_context") or account.get("competitor"),
            ),
            "contract_end": _normalize_text(account.get("contract_end")),
            "decision_timeline": _normalize_text(account.get("decision_timeline")),
            "primary_pain": _normalize_text(account.get("primary_pain")),
            "quote": _normalize_text(account.get("quote") or account.get("evidence")),
            "trust_tier": _normalize_text(account.get("trust_tier")) or "medium",
        }
        row.update(_lineage_for(account, packet))
        row["account_signal_id"] = f"account_{idx + 1}"
        account_signals.append(row)
    migration = _copy_dict(_copy_dict(contracts.get("displacement_reasoning")).get("migration_proof"))
    for idx, example in enumerate(_copy_list(migration.get("named_examples"))):
        if not isinstance(example, dict):
            continue
        row = {
            "company": _normalize_text(example.get("company")),
            "role_type": "",
            "buying_stage": "switching",
            "urgency": "",
            "competitor_context": _normalize_text(
                _copy_dict(migration.get("top_destination")).get("value")
                or migration.get("top_destination"),
            ),
            "contract_end": "",
            "decision_timeline": "",
            "primary_pain": _normalize_text(
                _copy_dict(migration.get("primary_switch_driver")).get("value")
                or migration.get("primary_switch_driver"),
            ),
            "quote": _normalize_text(example.get("evidence")),
            "trust_tier": "high" if _normalize_bool(example.get("quotable")) else "medium",
        }
        row.update(_lineage_for(example, packet))
        row["account_signal_id"] = f"migration_example_{idx + 1}"
        account_signals.append(row)
    return account_signals


def _counterevidence_from_contracts(contracts: dict[str, Any], packet: Any) -> list[dict[str, Any]]:
    evidence_governance = _copy_dict(contracts.get("evidence_governance"))
    counterevidence: list[dict[str, Any]] = []
    for idx, row in enumerate(_copy_list(evidence_governance.get("contradictions"))):
        if not isinstance(row, dict):
            continue
        item = {
            "counterevidence_id": f"counterevidence_{idx + 1}",
            "statement": _normalize_text(
                row.get("summary") or row.get("contradiction") or row.get("label") or row.get("signal"),
            ),
        }
        item.update(_lineage_for(row, packet))
        counterevidence.append(item)
    return counterevidence


def _coverage_limits_from_contracts(contracts: dict[str, Any], packet: Any) -> list[dict[str, Any]]:
    vendor_core = _copy_dict(contracts.get("vendor_core_reasoning"))
    posture = _copy_dict(vendor_core.get("confidence_posture"))
    governance = _copy_dict(contracts.get("evidence_governance"))
    limits: list[dict[str, Any]] = []
    for idx, limit in enumerate(_copy_list(posture.get("limits"))):
        text = _normalize_text(limit)
        if not text:
            continue
        limits.append({
            "coverage_limit_id": f"limit_{idx + 1}",
            "type": "confidence_limit",
            "label": text,
            **_lineage_for(posture, packet),
        })
    for idx, gap in enumerate(_copy_list(governance.get("coverage_gaps"))):
        if not isinstance(gap, dict):
            continue
        label = _normalize_text(gap.get("gap") or gap.get("code") or gap.get("label"))
        if not label:
            continue
        limits.append({
            "coverage_limit_id": f"gap_{idx + 1}",
            "type": "coverage_gap",
            "label": label,
            **_lineage_for(gap, packet),
        })
    return limits


def build_reasoning_atoms(contracts: dict[str, Any], packet: Any) -> dict[str, Any]:
    return {
        "schema_version": "v1",
        "theses": _theses_from_contracts(contracts, packet),
        "timing_windows": _timing_windows_from_contracts(contracts, packet),
        "proof_points": _proof_points_from_contracts(contracts, packet),
        "account_signals": _account_signals_from_contracts(contracts, packet),
        "counterevidence": _counterevidence_from_contracts(contracts, packet),
        "coverage_limits": _coverage_limits_from_contracts(contracts, packet),
    }


def build_reasoning_delta(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    *,
    current_as_of_date: Any = None,
    previous_as_of_date: Any = None,
) -> dict[str, Any]:
    current_contracts = _copy_dict(current.get("reasoning_contracts"))
    previous_contracts = _copy_dict((previous or {}).get("reasoning_contracts"))
    current_atoms = _copy_dict(current.get("reasoning_atoms"))
    previous_atoms = _copy_dict((previous or {}).get("reasoning_atoms"))

    current_causal = _copy_dict(
        _copy_dict(current_contracts.get("vendor_core_reasoning")).get("causal_narrative"),
    )
    previous_causal = _copy_dict(
        _copy_dict(previous_contracts.get("vendor_core_reasoning")).get("causal_narrative"),
    )
    current_migration = _copy_dict(
        _copy_dict(current_contracts.get("displacement_reasoning")).get("migration_proof"),
    )
    previous_migration = _copy_dict(
        _copy_dict(previous_contracts.get("displacement_reasoning")).get("migration_proof"),
    )

    current_theses = {
        _normalize_text(item.get("thesis_id") or item.get("summary")): item
        for item in _copy_list(current_atoms.get("theses"))
        if isinstance(item, dict)
    }
    previous_theses = {
        _normalize_text(item.get("thesis_id") or item.get("summary")): item
        for item in _copy_list(previous_atoms.get("theses"))
        if isinstance(item, dict)
    }
    current_windows = {
        _normalize_text(item.get("window_id") or item.get("start_or_anchor")): item
        for item in _copy_list(current_atoms.get("timing_windows"))
        if isinstance(item, dict)
    }
    previous_windows = {
        _normalize_text(item.get("window_id") or item.get("start_or_anchor")): item
        for item in _copy_list(previous_atoms.get("timing_windows"))
        if isinstance(item, dict)
    }
    current_accounts = {
        _normalize_text(item.get("company")) for item in _copy_list(current_atoms.get("account_signals"))
        if isinstance(item, dict) and _normalize_text(item.get("company"))
    }
    previous_accounts = {
        _normalize_text(item.get("company")) for item in _copy_list(previous_atoms.get("account_signals"))
        if isinstance(item, dict) and _normalize_text(item.get("company"))
    }
    current_limits = {
        _normalize_text(item.get("label")) for item in _copy_list(current_atoms.get("coverage_limits"))
        if isinstance(item, dict) and _normalize_text(item.get("label"))
    }
    previous_limits = {
        _normalize_text(item.get("label")) for item in _copy_list(previous_atoms.get("coverage_limits"))
        if isinstance(item, dict) and _normalize_text(item.get("label"))
    }

    current_counterevidence = len(_copy_list(current_atoms.get("counterevidence")))
    previous_counterevidence = len(_copy_list(previous_atoms.get("counterevidence")))
    current_destination = _normalize_text(
        _copy_dict(current_migration.get("top_destination")).get("value")
        or current_migration.get("top_destination"),
    )
    previous_destination = _normalize_text(
        _copy_dict(previous_migration.get("top_destination")).get("value")
        or previous_migration.get("top_destination"),
    )
    delta = {
        "schema_version": "v1",
        "previous_as_of_date": _normalize_text(previous_as_of_date),
        "current_as_of_date": _normalize_text(current_as_of_date),
        "wedge_changed": _normalize_text(current_causal.get("primary_wedge")) != _normalize_text(
            previous_causal.get("primary_wedge"),
        ),
        "confidence_changed": _normalize_text(current_causal.get("confidence")) != _normalize_text(
            previous_causal.get("confidence"),
        ),
        "theses_added": sorted(key for key in current_theses if key and key not in previous_theses),
        "theses_removed": sorted(key for key in previous_theses if key and key not in current_theses),
        "new_timing_windows": sorted(
            key for key in current_windows if key and key not in previous_windows
        ),
        "closed_timing_windows": sorted(
            key for key in previous_windows if key and key not in current_windows
        ),
        "new_account_signals": sorted(current_accounts - previous_accounts),
        "top_destination_changed": current_destination != previous_destination,
        "current_top_destination": current_destination,
        "previous_top_destination": previous_destination,
        "counterevidence_delta": current_counterevidence - previous_counterevidence,
        "coverage_improved": sorted(previous_limits - current_limits),
        "coverage_worsened": sorted(current_limits - previous_limits),
    }
    delta["changed"] = any(
        bool(delta[key]) for key in (
            "wedge_changed",
            "confidence_changed",
            "theses_added",
            "theses_removed",
            "new_timing_windows",
            "closed_timing_windows",
            "new_account_signals",
            "top_destination_changed",
            "counterevidence_delta",
            "coverage_improved",
            "coverage_worsened",
        )
    )
    return delta
