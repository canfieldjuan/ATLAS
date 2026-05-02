from __future__ import annotations


def resolve_vendor_name_cached(value: str | None) -> str:
    return str(value or "").strip()


async def resolve_vendor_name(value: str | None) -> str:
    """Async-compatible resolver used by copied Atlas campaign tasks."""
    return resolve_vendor_name_cached(value)
