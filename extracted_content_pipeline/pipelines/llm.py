from __future__ import annotations

import json
import os
from typing import Any

if os.getenv("EXTRACTED_PIPELINE_STANDALONE", "0") == "1":
    def get_pipeline_llm(*args: Any, **kwargs: Any):
        return None

    def clean_llm_output(text: str) -> str:
        return (text or "").strip()

    def parse_json_response(text: str) -> dict[str, Any] | list[Any] | None:
        cleaned = clean_llm_output(text)
        if not cleaned:
            return None
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    def trace_llm_call(*args: Any, **kwargs: Any) -> None:
        return None

    def call_llm_with_skill(*args: Any, **kwargs: Any):
        return None
else:
    from extracted_llm_infrastructure.pipelines.llm import *
