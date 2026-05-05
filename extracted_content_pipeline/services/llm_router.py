"""LLM router bridge for extracted content pipeline.

Default mode delegates to the extracted LLM infrastructure product.
Standalone mode keeps task imports runnable without provisioning an LLM.
"""

from __future__ import annotations

import importlib as _importlib
import os as _os
from typing import Any


def _bridge(module_name: str) -> None:
    src = _importlib.import_module(module_name)
    globals_dict = globals()
    for name in dir(src):
        if not name.startswith("__"):
            globals_dict[name] = getattr(src, name)


if _os.environ.get("EXTRACTED_PIPELINE_STANDALONE") == "1":

    def get_llm(*args: Any, **kwargs: Any) -> None:
        return None

else:
    _bridge("extracted_llm_infrastructure.services.llm_router")


del _bridge, _importlib, _os
