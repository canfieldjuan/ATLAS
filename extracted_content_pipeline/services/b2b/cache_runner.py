from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

if os.getenv("EXTRACTED_PIPELINE_STANDALONE", "0") == "1":
    @dataclass(frozen=True)
    class B2BExactStageRequest:
        stage_id: str
        provider: str = "standalone"
        model: str = "standalone"
        request_envelope: dict[str, Any] = field(default_factory=dict)

    def prepare_b2b_exact_stage_request(
        stage_id: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        llm: Any | None = None,
        messages: list[Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        request_envelope: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> B2BExactStageRequest:
        resolved_provider = provider or getattr(llm, "provider", None) or "standalone"
        resolved_model = model or getattr(llm, "model", None) or "standalone"
        envelope = dict(request_envelope or {})
        if messages is not None:
            envelope["messages"] = messages
        if max_tokens is not None:
            envelope["max_tokens"] = max_tokens
        if temperature is not None:
            envelope["temperature"] = temperature
        if metadata:
            envelope["metadata"] = metadata
        return B2BExactStageRequest(
            stage_id=stage_id,
            provider=str(resolved_provider),
            model=str(resolved_model),
            request_envelope=envelope,
        )

    def bind_b2b_exact_stage_request(
        stage_id: str,
        *,
        provider: str,
        model: str,
        request_envelope: dict[str, Any],
        **metadata: Any,
    ) -> B2BExactStageRequest:
        envelope = dict(request_envelope or {})
        if metadata:
            envelope["metadata"] = metadata
        return B2BExactStageRequest(
            stage_id=stage_id,
            provider=str(provider),
            model=str(model),
            request_envelope=envelope,
        )

    async def lookup_b2b_exact_stage_text(*args: Any, **kwargs: Any) -> None:
        return None

    async def store_b2b_exact_stage_text(*args: Any, **kwargs: Any) -> bool:
        return False
else:
    from atlas_brain.services.b2b.cache_runner import *
