from __future__ import annotations

from typing import Any


def build_business_trace_context(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}
