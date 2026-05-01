#!/usr/bin/env python3
"""Smoke-check the competitive-intelligence standalone substrate."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["EXTRACTED_COMP_INTEL_STANDALONE"] = "1"
os.environ.setdefault("EXTRACTED_LLM_INFRA_STANDALONE", "1")

MODULES = [
    "extracted_competitive_intelligence.config",
    "extracted_competitive_intelligence.storage.database",
    "extracted_competitive_intelligence.auth.dependencies",
    "extracted_competitive_intelligence.services.protocols",
    "extracted_competitive_intelligence.services.campaign_sender",
    "extracted_competitive_intelligence.autonomous.tasks.campaign_suppression",
    "extracted_competitive_intelligence.pipelines.llm",
    "extracted_competitive_intelligence.templates.email.vendor_briefing",
    "extracted_competitive_intelligence.services.b2b.source_impact",
    "extracted_competitive_intelligence.services.b2b_competitive_sets",
]


def _assert_owner(module_name: str, attr_name: str, expected_prefix: str) -> None:
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    owner = getattr(value, "__module__", "")
    if not owner.startswith(expected_prefix):
        raise AssertionError(
            f"{module_name}.{attr_name} resolved to {owner}, expected {expected_prefix}"
        )


def main() -> int:
    failed: list[str] = []
    for module_name in MODULES:
        try:
            importlib.import_module(module_name)
            print(f"OK {module_name}", flush=True)
        except Exception as exc:
            print(f"FAIL {module_name}: {type(exc).__name__}: {exc}", flush=True)
            failed.append(module_name)

    checks = [
        (
            "extracted_competitive_intelligence.config",
            "CompIntelSettings",
            "extracted_competitive_intelligence._standalone.config",
        ),
        (
            "extracted_competitive_intelligence.storage.database",
            "DatabasePool",
            "extracted_llm_infrastructure._standalone.database",
        ),
        (
            "extracted_competitive_intelligence.auth.dependencies",
            "AuthUser",
            "extracted_competitive_intelligence._standalone.auth",
        ),
        (
            "extracted_competitive_intelligence.services.protocols",
            "Message",
            "extracted_llm_infrastructure._standalone.protocols",
        ),
    ]
    for module_name, attr_name, expected_prefix in checks:
        try:
            _assert_owner(module_name, attr_name, expected_prefix)
        except Exception as exc:
            print(f"FAIL {module_name}.{attr_name}: {exc}", flush=True)
            failed.append(f"{module_name}.{attr_name}")

    if failed:
        print(f"Standalone smoke failed for {len(failed)} check(s)")
        return 1

    print("Standalone smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

