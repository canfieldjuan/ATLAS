# extracted_competitive_intelligence — STATUS

## Phase 1 — Scaffold creation ✅

| Step | Status |
|---|---|
| Manifest of source → scaffold mappings | ✅ done |
| Verbatim byte-snapshot of 6 mapped Python files | ✅ done |
| Verbatim byte-snapshot of 9 migration SQL files | ✅ done |
| Package `__init__.py` files at every level | ✅ done |
| Sync + validate scripts (with `src.exists()` guards) | ✅ done |
| ASCII / smoke-import / import-debt checks | ✅ done |
| Driver script `run_extracted_competitive_intelligence_checks.sh` | ✅ done |
| GitHub Actions workflow (with `pip install -r requirements.txt` step) | ✅ done |
| README + this STATUS file | ✅ done |
| `import_debt_allowlist.txt` (empty by design — corrected resolver) | ✅ done |

## Phase 2 — Standalone toggle 🔲

Goal: core substrate and selected product surfaces are importable without
`atlas_brain` on `sys.path`, gated by `EXTRACTED_COMP_INTEL_STANDALONE=1`.
Full task/runtime decoupling remains Phase 3.

| Task | Notes |
|---|---|
| Carve a slim `CompIntelSettings` Pydantic class out of `atlas_brain/config.py` | ✅ done for config fields used by current scaffold |
| Local DB pool abstraction | ✅ uses `extracted_llm_infrastructure.storage.database` in standalone mode |
| Email-send provider Protocol | ✅ `services.campaign_sender` routes to injectable standalone campaign sender |
| CRM provider Protocol | ✅ `services.crm_provider` routes to an injectable standalone CRM adapter |
| Checkout email provider Protocol | ✅ `services.email_provider` routes to an injectable standalone email adapter |
| Gated report PDF renderer Protocol | ✅ `services.b2b.pdf_renderer` routes to an injectable standalone PDF renderer |
| LLM exact-cache bridge | ✅ `services.b2b.llm_exact_cache` routes to extracted LLM infrastructure in standalone mode |
| Anthropic batch bridge | ✅ `services.b2b.anthropic_batch` routes to extracted LLM infrastructure in standalone mode |
| Anthropic batch helper boundary | ✅ `autonomous.tasks._b2b_batch_utils` is product-owned helper logic |
| Campaign LLM router bridge | ✅ `services.llm_router` routes vendor-briefing campaign LLM selection through extracted LLM infrastructure in standalone mode |
| Battle-card support port | ✅ `services.b2b.battle_card_ports` replaces direct `_b2b_shared.py`, churn-scope, execution-progress, and synthesis-reader imports for battle-card support |
| Vendor briefing intelligence port | ✅ `services.b2b.vendor_briefing_ports` replaces direct `_b2b_shared.py` imports for vendor briefing evidence readers |
| Suppression-callback Protocol | ✅ `autonomous.tasks.campaign_suppression` routes to injectable standalone suppression policy |
| Bridge stubs gate on `EXTRACTED_COMP_INTEL_STANDALONE=1` | ✅ config, DB, auth, campaign sender, suppression, protocols, LLM pipeline/router bridges, and service package fallback |
| Standalone smoke script + CI | ✅ `smoke_extracted_competitive_intelligence_standalone.py` runs in the local check driver |
| MCP package import boundary | ✅ extracted MCP server/shared helpers no longer import `atlas_brain.mcp.b2b` just to import tool modules |
| Source registry support module | ✅ `services.scraping.sources` is extracted-owned instead of an Atlas bridge |
| Package-level Atlas fallbacks | ✅ standalone mode fails closed for lazy package access in services, B2B services, templates, reasoning, autonomous, and autonomous tasks |
| Product-owned manifest entries | ✅ `manifest.json` supports `owned` modules that stay in ASCII/import checks but are skipped by byte-sync validation |
| MCP write tool boundary | ✅ `write_intelligence.py` owns simple DB writes and exposes explicit host ports for deep runtime builders/enrichers |
| Source impact support boundary | ✅ `source_impact.py` and its static scrape capability registry are product-owned |
| Cross-vendor selection boundary | ✅ `reasoning/cross_vendor_selection.py` is product-owned pure selection logic |
| Prompt contract boundary | ✅ single-pass battle prompts are product-owned LLM contracts |
| Ecosystem analyzer boundary | ✅ `reasoning/ecosystem.py` is a host adapter port, not an Atlas import |
| Challenger claim aggregation boundary | ✅ `services/b2b/challenger_dashboard_claims.py` is a fail-closed host adapter port |
| Competitive-set planner boundary | ✅ `services/b2b_competitive_sets.py` is product-owned and uses `competitive_set_ports.py` for reasoning/task support |
| Cross-vendor synthesis boundary | ✅ `_b2b_cross_vendor_synthesis.py` is product-owned packet/contract/reader logic with extracted semantic-cache hashing |

### Current audit snapshot

| Metric | Count |
|---|---:|
| Extracted files | 92 |
| Manifest mappings | 12 |
| Manifest Python snapshots | 3 |
| Manifest SQL snapshots | 9 |
| Product-owned modules | 20 |

Product-owned modules:

- `mcp/b2b/vendor_registry.py`
- `mcp/b2b/displacement.py`
- `mcp/b2b/cross_vendor.py`
- `mcp/b2b/write_intelligence.py`
- `mcp/b2b/write_ports.py`
- `services/vendor_registry.py`
- `services/scraping/capabilities.py`
- `services/b2b/source_impact.py`
- `services/b2b/challenger_dashboard_claims.py`
- `services/b2b/competitive_set_ports.py`
- `services/b2b/battle_card_ports.py`
- `services/b2b/vendor_briefing_ports.py`
- `services/b2b_competitive_sets.py`
- `autonomous/tasks/_b2b_batch_utils.py`
- `autonomous/tasks/_b2b_cross_vendor_synthesis.py`
- `templates/email/vendor_briefing.py`
- `reasoning/ecosystem.py`
- `reasoning/cross_vendor_selection.py`
- `reasoning/single_pass_prompts/cross_vendor_battle.py`
- `reasoning/single_pass_prompts/battle_card_reasoning.py`

## Phase 3 — Decoupling 🔲 (later PRs)

| Task | Source file referenced |
|---|---|
| Rewire remaining non-LLM battle-card/vendor-briefing host dependencies | LLM calls now route through `pipelines.llm` / `services.llm_router` into extracted LLM infrastructure in standalone mode. Remaining blockers are task/runtime host dependencies outside the LLM surface. |
| Replace remaining `_b2b_shared.py` cross-imports with explicit `Protocol`-based interfaces | Vendor briefing and battle cards now consume product-owned support ports; remaining direct consumers stay in other task surfaces |
| Provide host adapters for write-tool builders | `mcp/b2b/write_ports.py` defines ports for challenger brief and accounts-in-motion builders |
| Generic `EvidenceClaimReader` Protocol | `services/b2b/evidence_claim_*.py` stays in atlas-core; scaffold consumes via Protocol |
| Open-source-grade README + LICENSE + pyproject.toml | scaffold root |
| Publishable PyPI package | scaffold root |

## Per-file extraction state

| Scaffold file | Phase 1 (snapshot) | Phase 2 (standalone-ready) | Phase 3 (decoupled) |
|---|---|---|---|
| `services/vendor_registry.py` | ✅ | ✅ | 🔲 |
| `mcp/b2b/vendor_registry.py` | ✅ | ✅ | ✅ |
| `mcp/b2b/displacement.py` | ✅ | ✅ | ✅ |
| `mcp/b2b/cross_vendor.py` | ✅ | ✅ | ✅ |
| `mcp/b2b/write_intelligence.py` | ✅ | ✅ | 🔲 (deep builders require host adapters) |
| `mcp/b2b/_shared.py` | n/a | ✅ | 🔲 |
| `mcp/b2b/server.py` | n/a | ✅ | 🔲 |
| `services/b2b/source_impact.py` | ✅ | ✅ | ✅ |
| `services/b2b/challenger_dashboard_claims.py` | ✅ | ✅ | ✅ |
| `services/b2b/battle_card_ports.py` | n/a | ✅ | ✅ |
| `services/b2b/vendor_briefing_ports.py` | n/a | ✅ | ✅ |
| `services/scraping/sources.py` | n/a | ✅ | ✅ |
| `reasoning/ecosystem.py` | n/a | ✅ | ✅ |
| `autonomous/tasks/_b2b_batch_utils.py` | n/a | ✅ | ✅ |
| `autonomous/tasks/b2b_battle_cards.py` | ✅ | 🔲 (shared-helper, churn-scope, progress, and synthesis-reader imports routed through `battle_card_ports.py`; remaining runtime seams need follow-up ports) | 🔲 |
| `autonomous/tasks/b2b_vendor_briefing.py` | ✅ | 🔲 | 🔲 |
| `autonomous/tasks/_b2b_cross_vendor_synthesis.py` | ✅ | ✅ | ✅ |
| `services/b2b_competitive_sets.py` | ✅ | ✅ | ✅ |
| `reasoning/cross_vendor_selection.py` | ✅ | ✅ | ✅ |
| `reasoning/single_pass_prompts/cross_vendor_battle.py` | ✅ | ✅ | n/a (product-owned LLM contract) |
| `reasoning/single_pass_prompts/battle_card_reasoning.py` | ✅ | ✅ | n/a (product-owned LLM contract) |
| `templates/email/vendor_briefing.py` | ✅ | ✅ | ✅ |
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
