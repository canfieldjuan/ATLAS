from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

if os.getenv("EXTRACTED_PIPELINE_STANDALONE", "0") == "1":
    @dataclass
    class AnthropicBatchItem:
        custom_id: str
        artifact_type: str
        artifact_id: str
        vendor_name: str | None = None
        messages: list[Any] = field(default_factory=list)
        max_tokens: int | None = None
        temperature: float | None = None
        trace_span_name: str | None = None
        trace_metadata: dict[str, Any] = field(default_factory=dict)
        request_metadata: dict[str, Any] = field(default_factory=dict)
        cached_response_text: str | None = None
        cached_usage: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class AnthropicBatchExecution:
        local_batch_id: str = "standalone"
        results_by_custom_id: dict[str, Any] = field(default_factory=dict)

    async def run_anthropic_message_batch(*args: Any, **kwargs: Any) -> AnthropicBatchExecution:
        return AnthropicBatchExecution()

    async def mark_batch_fallback_result(*args: Any, **kwargs: Any) -> None:
        return None
else:
    from extracted_llm_infrastructure.services.b2b.anthropic_batch import *
