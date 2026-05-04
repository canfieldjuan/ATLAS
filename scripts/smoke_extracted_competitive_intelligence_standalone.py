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
    "extracted_competitive_intelligence.services.vendor_registry",
    "extracted_competitive_intelligence.services.campaign_sender",
    "extracted_competitive_intelligence.services.scraping.capabilities",
    "extracted_competitive_intelligence.services.scraping.sources",
    "extracted_competitive_intelligence.autonomous.tasks.campaign_suppression",
    "extracted_competitive_intelligence.mcp.b2b.vendor_registry",
    "extracted_competitive_intelligence.mcp.b2b.write_ports",
    "extracted_competitive_intelligence.mcp.b2b.write_intelligence",
    "extracted_competitive_intelligence.pipelines.llm",
    "extracted_competitive_intelligence.templates.email.vendor_briefing",
    "extracted_competitive_intelligence.services.b2b.source_impact",
    "extracted_competitive_intelligence.services.b2b.challenger_dashboard_claims",
    "extracted_competitive_intelligence.services.b2b.competitive_set_ports",
    "extracted_competitive_intelligence.services.b2b_competitive_sets",
    "extracted_competitive_intelligence.autonomous.tasks._b2b_cross_vendor_synthesis",
    "extracted_competitive_intelligence.reasoning.ecosystem",
    "extracted_competitive_intelligence.reasoning.cross_vendor_selection",
    "extracted_competitive_intelligence.reasoning.single_pass_prompts.cross_vendor_battle",
    "extracted_competitive_intelligence.reasoning.single_pass_prompts.battle_card_reasoning",
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
        (
            "extracted_competitive_intelligence.services.scraping.sources",
            "ReviewSource",
            "extracted_competitive_intelligence.services.scraping.sources",
        ),
    ]
    for module_name, attr_name, expected_prefix in checks:
        try:
            _assert_owner(module_name, attr_name, expected_prefix)
        except Exception as exc:
            print(f"FAIL {module_name}.{attr_name}: {exc}", flush=True)
            failed.append(f"{module_name}.{attr_name}")

    for module_name in (
        "extracted_competitive_intelligence.services",
        "extracted_competitive_intelligence.services.b2b",
        "extracted_competitive_intelligence.templates.email",
        "extracted_competitive_intelligence.reasoning",
        "extracted_competitive_intelligence.autonomous",
        "extracted_competitive_intelligence.autonomous.tasks",
    ):
        module = importlib.import_module(module_name)
        try:
            getattr(module, "__atlas_fallback_probe__")
        except AttributeError:
            continue
        print(f"FAIL {module_name}: Atlas fallback did not fail closed", flush=True)
        failed.append(module_name)

    owned_files = (
        ROOT / "extracted_competitive_intelligence" / "services" / "vendor_registry.py",
        ROOT / "extracted_competitive_intelligence" / "mcp" / "b2b" / "vendor_registry.py",
        ROOT / "extracted_competitive_intelligence" / "mcp" / "b2b" / "displacement.py",
        ROOT / "extracted_competitive_intelligence" / "mcp" / "b2b" / "cross_vendor.py",
        ROOT / "extracted_competitive_intelligence" / "mcp" / "b2b" / "write_intelligence.py",
        ROOT / "extracted_competitive_intelligence" / "mcp" / "b2b" / "write_ports.py",
        ROOT / "extracted_competitive_intelligence" / "services" / "scraping" / "capabilities.py",
        ROOT / "extracted_competitive_intelligence" / "services" / "b2b" / "source_impact.py",
        ROOT / "extracted_competitive_intelligence" / "services" / "b2b" / "challenger_dashboard_claims.py",
        ROOT / "extracted_competitive_intelligence" / "services" / "b2b" / "competitive_set_ports.py",
        ROOT / "extracted_competitive_intelligence" / "services" / "b2b_competitive_sets.py",
        ROOT / "extracted_competitive_intelligence" / "autonomous" / "tasks" / "_b2b_cross_vendor_synthesis.py",
        ROOT / "extracted_competitive_intelligence" / "templates" / "email" / "vendor_briefing.py",
        ROOT / "extracted_competitive_intelligence" / "reasoning" / "ecosystem.py",
        ROOT / "extracted_competitive_intelligence" / "reasoning" / "cross_vendor_selection.py",
        ROOT / "extracted_competitive_intelligence" / "reasoning" / "single_pass_prompts" / "cross_vendor_battle.py",
        ROOT / "extracted_competitive_intelligence" / "reasoning" / "single_pass_prompts" / "battle_card_reasoning.py",
    )
    for module_path in owned_files:
        if "atlas_brain." in module_path.read_text():
            print(f"FAIL {module_path.relative_to(ROOT)}: still imports Atlas", flush=True)
            failed.append(str(module_path.relative_to(ROOT)))

    if failed:
        print(f"Standalone smoke failed for {len(failed)} check(s)")
        return 1

    print("Standalone smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
