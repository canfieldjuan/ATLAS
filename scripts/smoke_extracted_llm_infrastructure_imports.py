#!/usr/bin/env python3
"""Smoke-import every public LLM-infrastructure module from the scaffold.

The scaffold is a verbatim snapshot of atlas_brain sources, so an
ImportError here means the manifest pulled in a module whose imports
are not resolvable in the scaffold's package layout. Phase 1 accepts
that some modules import from atlas_brain directly (those go into
import_debt_allowlist.txt). Phase 2 will eliminate that debt.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULES = [
    "extracted_llm_infrastructure.services.b2b.anthropic_batch",
    "extracted_llm_infrastructure.services.b2b.cache_strategy",
    "extracted_llm_infrastructure.services.b2b.llm_exact_cache",
    "extracted_llm_infrastructure.pipelines.llm",
    "extracted_llm_infrastructure.reasoning.semantic_cache",
    "extracted_llm_infrastructure.services.llm_router",
    "extracted_llm_infrastructure.services.llm.anthropic",
    "extracted_llm_infrastructure.services.llm.openrouter",
    "extracted_llm_infrastructure.services.llm.ollama",
    "extracted_llm_infrastructure.services.llm.vllm",
    "extracted_llm_infrastructure.services.llm.groq",
    "extracted_llm_infrastructure.services.llm.together",
    "extracted_llm_infrastructure.services.llm.hybrid",
    "extracted_llm_infrastructure.services.llm.cloud",
    "extracted_llm_infrastructure.services.tracing",
]


def main() -> int:
    failed: list[str] = []
    for module in MODULES:
        try:
            importlib.import_module(module)
            print(f"OK {module}")
        except Exception as exc:
            print(f"FAIL {module}: {exc}")
            failed.append(module)

    if failed:
        print(f"Import smoke failed for {len(failed)} module(s)")
        return 1

    print("Import smoke passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
