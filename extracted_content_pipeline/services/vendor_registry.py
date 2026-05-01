from __future__ import annotations


def resolve_vendor_name_cached(value: str | None) -> str:
    return str(value or "").strip()
