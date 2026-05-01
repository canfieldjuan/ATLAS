from __future__ import annotations

import os
from dataclasses import dataclass

if os.getenv("EXTRACTED_PIPELINE_STANDALONE", "0") == "1":
    @dataclass
    class Message:
        role: str
        content: str
else:
    from atlas_brain.services.protocols import *
