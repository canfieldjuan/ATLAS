from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

if os.getenv("EXTRACTED_PIPELINE_STANDALONE", "0") == "1":
    class Wedge(str, Enum):
        PRICING = "pricing"
        SUPPORT = "support"
        RELIABILITY = "reliability"
        FEATURES = "features"
        INTEGRATIONS = "integrations"

    @dataclass
    class WedgeMeta:
        label: str

    def get_wedge_meta(wedge: Wedge) -> WedgeMeta:
        return WedgeMeta(label=str(wedge.value).replace("_", " ").title())

    def validate_wedge(value: str) -> str:
        norm = str(value or "").strip().lower()
        if norm in {w.value for w in Wedge}:
            return norm
        return Wedge.SUPPORT.value
else:
    from atlas_brain.reasoning.wedge_registry import *
