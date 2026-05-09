#!/usr/bin/env python3
"""Standalone-mode smoke-check for extracted_competitive_intelligence.

Four phases:

1. **Import sweep** -- manifest-driven (mappings + owned) plus a small
   curated list of standalone substrate modules not yet on the
   manifest. Catches drift when a new mirror is added.
2. **Owner verification** -- specific module-attr-substrate triples
   that assert the standalone substrate is being used (not a fallback).
   Hardcoded by design: each entry encodes a substrate-routing
   decision.
3. **Atlas-fallback probes** -- namespaces that must fail closed
   under the toggle. Hardcoded by design.
4. **Owned-files atlas_brain scan** -- manifest-driven (owned only).
   Mappings are byte-synced from atlas_brain and may legitimately
   contain atlas_brain. text; scanning them is a category error.

Failure gate: any exception or check failure breaks the gate.
Tighter than the default-mode smokes (#427, #428) -- under the
standalone toggle, the substrate is meant to handle everything; any
import failure is a substrate gap.

See plans/PR-Audit-ManifestDrivenSmokes-2.md for rationale.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "extracted_competitive_intelligence"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["EXTRACTED_COMP_INTEL_STANDALONE"] = "1"
os.environ.setdefault("EXTRACTED_LLM_INFRA_STANDALONE", "1")


# Standalone substrate modules that are imported but not yet listed in
# the manifest. Adding them to manifest.owned is a separate slice (see
# Deferred section in plans/PR-Audit-ManifestDrivenSmokes-2.md).
_EXTRA_STANDALONE_SHIMS = (
    "extracted_competitive_intelligence.autonomous.tasks.campaign_suppression",
    "extracted_competitive_intelligence.pipelines.llm",
    "extracted_competitive_intelligence.services.b2b.pdf_renderer",
    "extracted_competitive_intelligence.services.campaign_sender",
    "extracted_competitive_intelligence.services.crm_provider",
    "extracted_competitive_intelligence.services.email_provider",
)


def _target_to_module(target: str) -> str:
    assert target.endswith(".py"), target
    base = target[: -len(".py")]
    if base.endswith("/__init__"):
        base = base[: -len("/__init__")]
    return base.replace("/", ".")


def _load_manifest() -> dict:
    return json.loads((ROOT / PACKAGE / "manifest.json").read_text(encoding="utf-8"))


def _load_modules(manifest: dict) -> list[str]:
    """Manifest entries (mappings + owned) plus _EXTRA_STANDALONE_SHIMS."""
    entries = manifest.get("mappings", []) + manifest.get("owned", [])
    manifest_modules = {
        _target_to_module(e["target"])
        for e in entries
        if e["target"].endswith(".py")
        and "/migrations/" not in e["target"]
        and not e["target"].endswith("/__init__.py")
    }
    return sorted(manifest_modules | set(_EXTRA_STANDALONE_SHIMS))


def _load_owned_files(manifest: dict) -> list[Path]:
    """Owned .py files from manifest -- the canonical set for the
    'still imports atlas_brain' scan."""
    return sorted(
        ROOT / e["target"]
        for e in manifest.get("owned", [])
        if e["target"].endswith(".py")
    )


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
    manifest = _load_manifest()

    # Phase 1 -- import sweep (manifest-driven + curated shims)
    modules = _load_modules(manifest)
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"OK {module_name}", flush=True)
        except Exception as exc:
            print(f"FAIL {module_name}: {type(exc).__name__}: {exc}", flush=True)
            failed.append(module_name)

    # Phase 2 -- owner verification (substrate routing assertions)
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
            "extracted_competitive_intelligence.services.llm_router",
            "get_llm",
            "extracted_llm_infrastructure.services.llm_router",
        ),
        (
            "extracted_competitive_intelligence.services.scraping.sources",
            "ReviewSource",
            "extracted_competitive_intelligence.services.scraping.sources",
        ),
        (
            "extracted_competitive_intelligence.services.b2b.llm_exact_cache",
            "build_skill_messages",
            "extracted_llm_infrastructure.services.b2b.llm_exact_cache",
        ),
        (
            "extracted_competitive_intelligence.services.b2b.anthropic_batch",
            "AnthropicBatchItem",
            "extracted_llm_infrastructure.services.b2b.anthropic_batch",
        ),
        (
            "extracted_competitive_intelligence.services.b2b.battle_card_ports",
            "get_battle_card_support_port",
            "extracted_competitive_intelligence.services.b2b.battle_card_ports",
        ),
        (
            "extracted_competitive_intelligence.services.b2b.vendor_briefing_ports",
            "get_vendor_briefing_intelligence_port",
            "extracted_competitive_intelligence.services.b2b.vendor_briefing_ports",
        ),
        (
            "extracted_competitive_intelligence.services.b2b.vendor_briefing_api_ports",
            "get_vendor_briefing_api_port",
            "extracted_competitive_intelligence.services.b2b.vendor_briefing_api_ports",
        ),
        (
            "extracted_competitive_intelligence.autonomous.tasks._b2b_batch_utils",
            "anthropic_batch_requested",
            "extracted_competitive_intelligence.autonomous.tasks._b2b_batch_utils",
        ),
    ]
    for module_name, attr_name, expected_prefix in checks:
        try:
            _assert_owner(module_name, attr_name, expected_prefix)
        except Exception as exc:
            print(f"FAIL {module_name}.{attr_name}: {exc}", flush=True)
            failed.append(f"{module_name}.{attr_name}")

    # Phase 3 -- fallback probes (substrate must fail closed)
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

    # Phase 4 -- owned-files atlas_brain scan (manifest-driven)
    for module_path in _load_owned_files(manifest):
        if not module_path.exists():
            print(f"FAIL {module_path.relative_to(ROOT)}: missing owned file", flush=True)
            failed.append(str(module_path.relative_to(ROOT)))
            continue
        if "atlas_brain." in module_path.read_text(encoding="utf-8"):
            print(f"FAIL {module_path.relative_to(ROOT)}: still imports Atlas", flush=True)
            failed.append(str(module_path.relative_to(ROOT)))

    if failed:
        print(f"Standalone smoke failed for {len(failed)} check(s)")
        return 1

    print("Standalone smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
