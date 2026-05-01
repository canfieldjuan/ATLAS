from __future__ import annotations

import os

if os.getenv("EXTRACTED_PIPELINE_STANDALONE", "0") == "1":
    def resolve_vendor_name_cached(value: str | None) -> str:
        return str(value or "").strip()
else:
    from atlas_brain.services.vendor_registry import *
