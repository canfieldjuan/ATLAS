from __future__ import annotations

import os
from typing import Any

if os.getenv("EXTRACTED_PIPELINE_STANDALONE", "0") == "1":
    def build_business_trace_context(**kwargs: Any) -> dict[str, Any]:
        return {k: v for k, v in kwargs.items() if v is not None}
else:
    from atlas_brain.services.tracing import *
