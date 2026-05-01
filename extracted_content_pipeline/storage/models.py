from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

if os.getenv("EXTRACTED_PIPELINE_STANDALONE", "0") == "1":
    @dataclass
    class ScheduledTask:
        id: UUID = field(default_factory=uuid4)
        metadata: dict[str, Any] = field(default_factory=dict)
else:
    from atlas_brain.storage.models import *
