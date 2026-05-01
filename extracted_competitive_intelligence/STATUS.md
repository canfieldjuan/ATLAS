# extracted_competitive_intelligence — STATUS

## Phase 1 — Scaffold creation ✅

| Step | Status |
|---|---|
| Manifest of source → scaffold mappings | ✅ done |
| Verbatim byte-snapshot of 15 Python files | ✅ done |
| Verbatim byte-snapshot of 9 migration SQL files | ✅ done |
| Package `__init__.py` files at every level | ✅ done |
| Sync + validate scripts (with `src.exists()` guards) | ✅ done |
| ASCII / smoke-import / import-debt checks | ✅ done |
| Driver script `run_extracted_competitive_intelligence_checks.sh` | ✅ done |
| GitHub Actions workflow (with `pip install -r requirements.txt` step) | ✅ done |
| README + this STATUS file | ✅ done |
| `import_debt_allowlist.txt` (empty by design — corrected resolver) | ✅ done |

## Phase 2 — Standalone toggle 🔲 (separate PR)

Goal: every scaffolded module is importable and runnable without `atlas_brain` on `sys.path`, gated by `EXTRACTED_COMP_INTEL_STANDALONE=1`.

| Task | Notes |
|---|---|
| Carve a slim `CompIntelSettings` Pydantic class out of `atlas_brain/config.py` | Mix-in fields from b2b_churn (vendor_briefing_*, cross_vendor_*, competitive_intelligence_*) |
| Local DB pool abstraction | Either share `extracted_llm_infrastructure/_standalone/database.py` from PR #40, or create a thin local wrapper |
| Email-send provider Protocol | Replace `atlas_brain.services.campaign_sender:get_campaign_sender()` with an injectable `EmailSender` Protocol so the scaffold does not require the Resend singleton |
| Suppression-callback Protocol | Replace `atlas_brain.autonomous.tasks.campaign_suppression:is_suppressed()` with an injectable `SuppressionPolicy` Protocol |
| Bridge stubs gate on `EXTRACTED_COMP_INTEL_STANDALONE=1` | Mirror the LLM-infra Phase 2 pattern from PR #40 |
| Standalone smoke script + CI | Add a second smoke that exercises the standalone path |

## Phase 3 — Decoupling 🔲 (later PRs)

| Task | Source file referenced |
|---|---|
| Rewire `b2b_battle_cards.py` LLM calls to consume `extracted_llm_infrastructure` directly | `autonomous/tasks/b2b_battle_cards.py:260` (`call_llm_with_skill`), `b2b_vendor_briefing.py:1201` (`get_llm`) |
| Replace `_b2b_shared.py` cross-imports with explicit `Protocol`-based interfaces | `vendor_briefing.py:40-47` reads from `_b2b_shared` for vendor intelligence records |
| Decouple from `atlas_brain.services.b2b.challenger_dashboard_claims` | `b2b_battle_cards.py:21` imports `aggregate_direct_displacement_claims_for_incumbent` |
| Generic `EvidenceClaimReader` Protocol | `services/b2b/evidence_claim_*.py` stays in atlas-core; scaffold consumes via Protocol |
| Open-source-grade README + LICENSE + pyproject.toml | scaffold root |
| Publishable PyPI package | scaffold root |

## Per-file extraction state

| Scaffold file | Phase 1 (snapshot) | Phase 2 (standalone-ready) | Phase 3 (decoupled) |
|---|---|---|---|
| `services/vendor_registry.py` | ✅ | 🔲 | 🔲 |
| `mcp/b2b/vendor_registry.py` | ✅ | 🔲 | 🔲 |
| `mcp/b2b/displacement.py` | ✅ | 🔲 | 🔲 |
| `mcp/b2b/cross_vendor.py` | ✅ | 🔲 | 🔲 |
| `mcp/b2b/write_intelligence.py` | ✅ | 🔲 | 🔲 |
| `services/b2b/source_impact.py` | ✅ | 🔲 (mostly pure data; should be easy) | 🔲 |
| `autonomous/tasks/b2b_battle_cards.py` | ✅ | 🔲 | 🔲 |
| `autonomous/tasks/b2b_vendor_briefing.py` | ✅ | 🔲 | 🔲 |
| `autonomous/tasks/_b2b_cross_vendor_synthesis.py` | ✅ | 🔲 | 🔲 |
| `services/b2b_competitive_sets.py` | ✅ | 🔲 | 🔲 |
| `reasoning/cross_vendor_selection.py` | ✅ | 🔲 | 🔲 |
| `reasoning/single_pass_prompts/cross_vendor_battle.py` | ✅ | ✅ (pure prompt string; no atlas imports) | n/a |
| `reasoning/single_pass_prompts/battle_card_reasoning.py` | ✅ | ✅ (pure prompt string; no atlas imports) | n/a |
| `templates/email/vendor_briefing.py` | ✅ | 🔲 | 🔲 |
| `api/b2b_vendor_briefing.py` | ✅ | 🔲 | 🔲 |
| `storage/migrations/095_b2b_vendor_registry.sql` | ✅ | n/a | n/a |
| `storage/migrations/099_displacement_edges_and_company_signals.sql` | ✅ | n/a | n/a |
| `storage/migrations/101_vendor_buyer_profiles.sql` | ✅ | n/a | n/a |
| `storage/migrations/147_displacement_velocity.sql` | ✅ | n/a | n/a |
| `storage/migrations/158_cross_vendor_conclusions.sql` | ✅ | n/a | n/a |
| `storage/migrations/245_cross_vendor_reasoning_synthesis.sql` | ✅ | n/a | n/a |
| `storage/migrations/261_b2b_competitive_sets.sql` | ✅ | n/a | n/a |
| `storage/migrations/262_b2b_competitive_set_runs.sql` | ✅ | n/a | n/a |
| `storage/migrations/263_b2b_competitive_set_run_constraints.sql` | ✅ | n/a | n/a |
