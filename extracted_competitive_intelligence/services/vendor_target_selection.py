"""Vendor target selection helpers for Competitive Intelligence workflows."""

from __future__ import annotations

from typing import Any, Mapping


def _target_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("company_name") or "").strip().lower(),
        str(row.get("target_mode") or "").strip().lower(),
    )


def _target_priority(row: Mapping[str, Any]) -> tuple[int, int, str]:
    freshness = str(row.get("updated_at") or row.get("created_at") or "")
    return (
        1 if row.get("account_id") else 0,
        1 if row.get("contact_email") else 0,
        freshness,
    )


def dedupe_vendor_target_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep the strongest row per ``(company_name, target_mode)`` pair."""
    best_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        key = _target_key(row)
        existing = best_by_key.get(key)
        if existing is None or _target_priority(row) > _target_priority(existing):
            best_by_key[key] = row
    return sorted(
        best_by_key.values(),
        key=lambda row: (
            str(row.get("company_name") or "").strip().lower(),
            str(row.get("target_mode") or "").strip().lower(),
        ),
    )
