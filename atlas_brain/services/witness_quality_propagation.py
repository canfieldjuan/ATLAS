"""Helpers for decorating persisted witness JSON with quality fields."""

from __future__ import annotations

import copy
import json
from typing import Any

from atlas_brain.services.reasoning_delivery_audit import WITNESS_QUALITY_FIELDS


def decode_json_payload(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _field_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _witness_id(value: dict[str, Any]) -> str:
    for key in ("witness_id", "_sid", "source_id"):
        text = str(value.get(key) or "").strip()
        if text:
            return text
    return ""


def _is_quote_bearing_witness(value: dict[str, Any]) -> bool:
    if not _witness_id(value):
        return False
    return _field_present(value.get("excerpt_text")) or _field_present(value.get("quote"))


def collect_quote_witness_ids(value: Any) -> set[str]:
    payload = decode_json_payload(value)
    witness_ids: set[str] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if _is_quote_bearing_witness(node):
                witness_ids.add(_witness_id(node))
            for child in node.values():
                _walk(child)
        elif isinstance(node, list):
            for child in node:
                _walk(child)

    _walk(payload)
    return witness_ids


def normalize_witness_quality_row(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    witness_id = str(row.get("witness_id") or "").strip()
    quality: dict[str, Any] = {}
    for field in WITNESS_QUALITY_FIELDS:
        value = row.get(field)
        if _field_present(value):
            quality[field] = value
    return witness_id, quality


def decorate_witness_quality_fields(
    value: Any,
    quality_by_witness_id: dict[str, dict[str, Any]],
    *,
    overwrite: bool = False,
) -> tuple[Any, dict[str, int]]:
    """Return a payload copy with missing witness-quality fields filled.

    Only quote-bearing witness objects are decorated. ID-only provenance rows
    such as coverage gaps are intentionally ignored.
    """
    payload = decode_json_payload(value)
    stats = {
        "witness_objects_seen": 0,
        "witness_objects_matched": 0,
        "witness_objects_updated": 0,
        "fields_written": 0,
    }

    def _decorate(node: Any) -> Any:
        if isinstance(node, dict):
            out = {key: _decorate(child) for key, child in node.items()}
            if not _is_quote_bearing_witness(out):
                return out
            stats["witness_objects_seen"] += 1
            witness_id = _witness_id(out)
            quality = quality_by_witness_id.get(witness_id)
            if not quality:
                return out
            stats["witness_objects_matched"] += 1
            object_updated = False
            for field in WITNESS_QUALITY_FIELDS:
                if field not in quality:
                    continue
                if not overwrite and _field_present(out.get(field)):
                    continue
                out[field] = copy.deepcopy(quality[field])
                object_updated = True
                stats["fields_written"] += 1
            if object_updated:
                stats["witness_objects_updated"] += 1
            return out
        if isinstance(node, list):
            return [_decorate(child) for child in node]
        return node

    return _decorate(payload), stats
