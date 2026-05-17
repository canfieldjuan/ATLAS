#!/usr/bin/env python3
"""Smoke the extracted reasoning core as a standalone package."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "extracted_reasoning_core" / "manifest.json"
ENV_VAR = "EXTRACTED_REASONING_CORE_STANDALONE"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _FakeLLM:
    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        del messages, max_tokens, temperature, metadata
        return {
            "response": json.dumps({
                "summary": "Reasoning core standalone smoke passed.",
                "claims": [
                    {
                        "claim": "Standalone reasoning used local core ports.",
                        "confidence": "high",
                        "source_ids": ["smoke-1"],
                    }
                ],
                "confidence": 0.91,
            }),
            "usage": {"input_tokens": 4, "output_tokens": 6},
        }


def _module_name(target: str) -> str | None:
    path = Path(target)
    if path.suffix != ".py":
        return None
    if path.name == "__init__.py":
        return ".".join(path.parent.parts)
    return ".".join(path.with_suffix("").parts)


def _manifest_python_modules() -> list[str]:
    data = json.loads(MANIFEST.read_text())
    modules: list[str] = []
    for entry in data.get("owned", []):
        module = _module_name(str(entry.get("target") or ""))
        if module:
            modules.append(module)
    for entry in data.get("mappings", []):
        module = _module_name(str(entry.get("target") or ""))
        if module:
            modules.append(module)
    return sorted(set(modules))


async def _run_smoke() -> dict[str, Any]:
    # Marker for parity with other extracted product smokes. Reasoning core has
    # no Atlas fallback path to disable; decoupling is enforced by import gates.
    os.environ[ENV_VAR] = "1"

    imported: list[str] = []
    for module in _manifest_python_modules():
        importlib.import_module(module)
        imported.append(module)

    from extracted_reasoning_core.api import run_reasoning
    from extracted_reasoning_core.types import (
        EvidenceItem,
        ReasoningInput,
        ReasoningPack,
        ReasoningPorts,
    )

    result = await run_reasoning(
        ReasoningInput(
            entity_id="standalone-smoke",
            entity_type="test",
            goal="verify standalone reasoning core execution",
            evidence=(
                EvidenceItem(
                    source_type="smoke",
                    source_id="smoke-1",
                    text="Standalone host supplies ports.",
                ),
            ),
        ),
        pack=ReasoningPack(
            name="reasoning_synthesis",
            prompts={"reasoning_synthesis": "Return JSON."},
        ),
        ports=ReasoningPorts(llm=_FakeLLM()),
    )

    return {
        "ok": result.summary == "Reasoning core standalone smoke passed.",
        "imported_modules": len(imported),
        "summary": result.summary,
        "claims": len(result.claims),
        "tokens_used": result.state.get("tokens_used"),
    }


def main() -> int:
    result = asyncio.run(_run_smoke())
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
