from __future__ import annotations

import os
import re

if os.getenv("EXTRACTED_PIPELINE_STANDALONE", "0") == "1":
    def normalize_company_name(value: str | None) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text
else:
    from atlas_brain.services.company_normalization import *
