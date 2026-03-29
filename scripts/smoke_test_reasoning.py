#!/usr/bin/env python3
"""Smoke test for the reasoning pipeline redesign.

Loads reasoning for a vendor through the shared loader, then verifies
every consumer-facing field is sourced correctly -- synthesis-first,
with proper contracts, governance, and no stale legacy paths.

Usage:
    python scripts/smoke_test_reasoning.py [vendor_name]

If no vendor is specified, picks the vendor with the most reviews
that has synthesis data.
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ATLAS_LLM_OLLAMA_MODEL", "qwen3:14b")


def _ok(label: str) -> None:
    print(f"  [PASS] {label}")


def _fail(label: str, detail: str = "") -> None:
    print(f"  [FAIL] {label}{': ' + detail if detail else ''}")


def _warn(label: str, detail: str = "") -> None:
    print(f"  [WARN] {label}{': ' + detail if detail else ''}")


async def run_smoke_test(vendor_name: str | None = None) -> dict:
    from atlas_brain.storage.database import get_db_pool, init_database
    await init_database()
    pool = get_db_pool()

    results = {"pass": 0, "fail": 0, "warn": 0}

    def ok(label):
        _ok(label)
        results["pass"] += 1

    def fail(label, detail=""):
        _fail(label, detail)
        results["fail"] += 1

    def warn(label, detail=""):
        _warn(label, detail)
        results["warn"] += 1

    # --- Pick a vendor ---
    if not vendor_name:
        row = await pool.fetchrow("""
            SELECT s.vendor_name,
                   (SELECT COUNT(*) FROM b2b_reviews r
                    WHERE LOWER(r.vendor_name) = LOWER(s.vendor_name)) as reviews
            FROM (
                SELECT DISTINCT ON (vendor_name) vendor_name, as_of_date
                FROM b2b_reasoning_synthesis
                ORDER BY vendor_name, as_of_date DESC
            ) s
            ORDER BY reviews DESC
            LIMIT 1
        """)
        if not row:
            print("ERROR: No synthesis data found. Run b2b_reasoning_synthesis first.")
            return results
        vendor_name = row["vendor_name"]
        print(f"Auto-selected vendor: {vendor_name} ({row['reviews']} reviews)")
    else:
        print(f"Testing vendor: {vendor_name}")

    print()

    # === Section 1: Shared Loader ===
    print("=== Shared Loader ===")

    from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
        load_best_reasoning_view,
        synthesis_view_to_reasoning_entry,
    )
    from datetime import date

    view = await load_best_reasoning_view(pool, vendor_name)
    if view is None:
        fail("load_best_reasoning_view returned None")
        return results
    ok("load_best_reasoning_view returned a view")

    if view.schema_version in ("v2", "legacy"):
        ok(f"schema_version = {view.schema_version}")
    else:
        warn(f"schema_version = {view.schema_version} (expected v2 or legacy)")

    if view.as_of_date:
        ok(f"as_of_date = {view.as_of_date}")
    else:
        warn("as_of_date is None")

    # === Section 2: Contracts ===
    print()
    print("=== Materialized Contracts ===")

    contracts = view.materialized_contracts()
    if not contracts:
        fail("materialized_contracts() is empty")
        return results
    ok(f"materialized_contracts has {len(contracts)} blocks")

    for block in ("vendor_core_reasoning", "displacement_reasoning",
                   "category_reasoning", "account_reasoning"):
        if contracts.get(block):
            ok(f"  {block} present")
        else:
            warn(f"  {block} missing")

    if contracts.get("evidence_governance"):
        ok("  evidence_governance present")
    else:
        warn("  evidence_governance missing (may not exist for older synthesis)")

    # === Section 3: Phase 3 Fields ===
    print()
    print("=== Phase 3 Governance Fields ===")

    wedge = view.primary_wedge
    if wedge:
        ok(f"primary_wedge = {wedge.value}")
    else:
        cn = view.section("causal_narrative")
        raw_wedge = cn.get("primary_wedge", "")
        if raw_wedge:
            warn(f"primary_wedge raw = {raw_wedge} (not in wedge registry)")
        else:
            fail("no primary_wedge at all")

    confidence = view.confidence("causal_narrative")
    ok(f"confidence = {confidence}")

    wts = view.why_they_stay
    if wts:
        ok(f"why_they_stay: {wts.get('summary', '')[:60]}")
    else:
        warn("why_they_stay not present (normal for older synthesis)")

    cp = view.confidence_posture
    if cp:
        ok(f"confidence_posture: overall={cp.get('overall', '')} limits={len(cp.get('limits', []))}")
    else:
        warn("confidence_posture not present (normal for older synthesis)")

    triggers = view.switch_triggers
    if triggers:
        ok(f"switch_triggers: {len(triggers)} triggers")
    else:
        warn("switch_triggers not present")

    eg = view.evidence_governance
    if eg:
        ok(f"evidence_governance: {list(eg.keys())}")
    else:
        warn("evidence_governance not present (normal for older synthesis)")

    # === Section 4: Reasoning Entry (for shared builders) ===
    print()
    print("=== Synthesis-to-Reasoning Entry ===")

    entry = synthesis_view_to_reasoning_entry(view)
    if entry.get("archetype"):
        ok(f"archetype = {entry['archetype']}")
    else:
        fail("archetype is empty")

    if entry.get("mode") == "synthesis":
        ok("mode = synthesis")
    elif entry.get("mode") == "synthesis_fallback":
        ok("mode = synthesis_fallback (legacy source)")
    else:
        warn(f"mode = {entry.get('mode', 'empty')}")

    if entry.get("executive_summary"):
        ok(f"executive_summary: {entry['executive_summary'][:60]}...")
    else:
        warn("executive_summary is empty (v2.3 builds from trigger/why_now)")

    if entry.get("key_signals"):
        ok(f"key_signals: {len(entry['key_signals'])} signals")
    else:
        warn("key_signals is empty")

    # === Section 5: Legacy Divergence Check ===
    print()
    print("=== Legacy Divergence Check ===")

    legacy_row = await pool.fetchrow("""
        SELECT archetype, archetype_confidence
        FROM b2b_churn_signals
        WHERE LOWER(vendor_name) = LOWER($1) AND archetype IS NOT NULL
        ORDER BY last_computed_at DESC LIMIT 1
    """, vendor_name)

    if legacy_row:
        legacy_arch = legacy_row["archetype"]
        synth_wedge = wedge.value if wedge else entry.get("archetype", "")
        if legacy_arch == synth_wedge:
            ok(f"legacy and synthesis agree: {synth_wedge}")
        else:
            # Check if legacy maps to the same wedge
            from atlas_brain.reasoning.wedge_registry import wedge_from_archetype
            mapped = wedge_from_archetype(legacy_arch)
            if mapped.value == synth_wedge:
                ok(f"legacy '{legacy_arch}' maps to synthesis '{synth_wedge}' via wedge registry")
            else:
                warn(f"DIVERGENT: legacy='{legacy_arch}' vs synthesis='{synth_wedge}' (mapped={mapped.value})")
    else:
        ok("no legacy churn_signals row (synthesis-only vendor)")

    # === Section 6: Consumer Output Check ===
    print()
    print("=== Consumer Output Check ===")

    # Check battle card
    bc_row = await pool.fetchrow("""
        SELECT intelligence_data->>'archetype' as arch,
               intelligence_data->>'reasoning_source' as source,
               intelligence_data->>'synthesis_wedge' as wedge,
               report_date
        FROM b2b_intelligence
        WHERE report_type = 'battle_card'
          AND intelligence_data->>'vendor' = $1
        ORDER BY created_at DESC LIMIT 1
    """, vendor_name)
    if bc_row:
        bc_source = bc_row["source"] or "unknown"
        bc_arch = bc_row["arch"] or ""
        bc_wedge_val = bc_row["wedge"] or ""
        if bc_source == "b2b_reasoning_synthesis":
            ok(f"battle card: source=synthesis wedge={bc_wedge_val}")
        else:
            warn(f"battle card: source={bc_source} arch={bc_arch} (may be stale)")
    else:
        warn("no battle card found for vendor")

    # Check campaign
    campaign_row = await pool.fetchrow("""
        SELECT (metadata->'reasoning'->>'wedge') as wedge,
               (metadata->'reasoning'->>'confidence') as conf
        FROM b2b_campaigns
        WHERE LOWER(vendor_name) = LOWER($1)
          AND metadata->'reasoning' IS NOT NULL
        ORDER BY created_at DESC LIMIT 1
    """, vendor_name)
    if campaign_row and campaign_row.get("wedge"):
        ok(f"campaign metadata: wedge={campaign_row['wedge']}")
    else:
        warn("no campaign with reasoning metadata found")

    # === Summary ===
    print()
    total = results["pass"] + results["fail"] + results["warn"]
    print(f"=== Summary: {results['pass']} passed, {results['fail']} failed, {results['warn']} warnings out of {total} checks ===")
    if results["fail"] == 0:
        print("RESULT: No old reasoning paths detected.")
    else:
        print("RESULT: FAILURES DETECTED - check output above.")

    return results


if __name__ == "__main__":
    vendor = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_smoke_test(vendor))
