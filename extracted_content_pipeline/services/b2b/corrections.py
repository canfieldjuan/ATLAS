from __future__ import annotations

import os

if os.getenv("EXTRACTED_PIPELINE_STANDALONE", "0") == "1":
    def suppress_predicate(*args, **kwargs) -> bool:
        return False
else:
    from atlas_brain.services.b2b.corrections import *
