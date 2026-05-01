#!/usr/bin/env python3
"""Smoke-import every public module in the extracted_competitive_intelligence
scaffold.

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
    "extracted_competitive_intelligence.services.vendor_registry",
    "extracted_competitive_intelligence.mcp.b2b.vendor_registry",
    "extracted_competitive_intelligence.mcp.b2b.displacement",
    "extracted_competitive_intelligence.mcp.b2b.cross_vendor",
    "extracted_competitive_intelligence.mcp.b2b.write_intelligence",
    "extracted_competitive_intelligence.services.b2b.source_impact",
    "extracted_competitive_intelligence.autonomous.tasks.b2b_battle_cards",
    "extracted_competitive_intelligence.autonomous.tasks.b2b_vendor_briefing",
    "extracted_competitive_intelligence.autonomous.tasks._b2b_cross_vendor_synthesis",
    "extracted_competitive_intelligence.services.b2b_competitive_sets",
    "extracted_competitive_intelligence.reasoning.cross_vendor_selection",
    "extracted_competitive_intelligence.reasoning.single_pass_prompts.cross_vendor_battle",
    "extracted_competitive_intelligence.reasoning.single_pass_prompts.battle_card_reasoning",
    "extracted_competitive_intelligence.templates.email.vendor_briefing",
    "extracted_competitive_intelligence.api.b2b_vendor_briefing",
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
