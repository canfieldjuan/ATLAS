# extracted_competitive_intelligence

Phase 1 scaffold for extracting the **Competitive / Vendor Intelligence Platform** from `atlas_brain/` into a standalone, sellable package.

## Product framing

The customer brings their own clean vendor / product / win-loss data; this platform produces:

- **Vendor displacement edges** — A → B competitive flows with strength, velocity, and confidence
- **Head-to-head battle cards** — discovery questions, objection handlers, talk tracks, recommended plays per vendor pair
- **Executive vendor briefings** — weekly churn intelligence emails with evidence-backed pressure scores
- **Fuzzy-matched vendor registry** — canonical vendor names + aliases, the foundation every other artifact references
- **Cross-vendor reasoning** — pairwise battles, category councils, resource asymmetry analyses
- **Win/loss prediction inputs** — displacement momentum, churn severity, pain concentration

Differentiator: every output is grounded in real switching signals and uses the same evidence-claim substrate that powers the email-campaign engine — no template-shuffling, no template "battle cards" written from feature lists.

## What's in scope (Phase 1)

| Path | Purpose |
|---|---|
| `services/vendor_registry.py` | Canonical vendor names + aliases + cache |
| `mcp/b2b/vendor_registry.py` | MCP tools for list/search/fuzzy-match |
| `mcp/b2b/displacement.py` | MCP tools for querying displacement edges |
| `mcp/b2b/cross_vendor.py` | MCP tools for cross-vendor conclusions |
| `mcp/b2b/write_intelligence.py` | Write-back MCP tools for persisting conclusions |
| `services/b2b/source_impact.py` | Source impact ledger (which sources feed which products) |
| `autonomous/tasks/b2b_battle_cards.py` | Deterministic battle card builder + LLM overlay (~5K LOC) |
| `autonomous/tasks/b2b_vendor_briefing.py` | Vendor churn briefing assembly + Resend send |
| `autonomous/tasks/_b2b_batch_utils.py` | Product-owned Anthropic batch helper logic |
| `autonomous/tasks/_b2b_cross_vendor_synthesis.py` | Cross-vendor packet builders |
| `services/b2b_competitive_sets.py` | Planner that scopes synthesis to a competitive set |
| `reasoning/cross_vendor_selection.py` | Selection logic for which vendor pairs deserve LLM budget |
| `reasoning/single_pass_prompts/cross_vendor_battle.py` | Single-pass cross-vendor battle prompt |
| `reasoning/single_pass_prompts/battle_card_reasoning.py` | Contracts-first battle card reasoning prompt |
| `templates/email/vendor_briefing.py` | HTML email template (Outlook-compatible) |
| `api/b2b_vendor_briefing.py` | REST endpoints for preview / send / approve / reject |

Plus 9 migrations: `095_b2b_vendor_registry.sql`, `099_displacement_edges_and_company_signals.sql`, `101_vendor_buyer_profiles.sql`, `147_displacement_velocity.sql`, `158_cross_vendor_conclusions.sql`, `245_cross_vendor_reasoning_synthesis.sql`, `261_b2b_competitive_sets.sql`, `262_b2b_competitive_set_runs.sql`, `263_b2b_competitive_set_run_constraints.sql`.

## What's out of scope (remaining Phase 3)

- Deep runtime decoupling for remaining battle-card LLM calls; standalone mode
  routes the LLM bridge, exact-cache message builder, and Anthropic batch
  support through `extracted_llm_infrastructure/`, but task-level LLM seams
  still need Phase 3 hardening.
- API endpoint extraction beyond the briefing endpoints (`/b2b/win-loss`, dashboard endpoints stay in atlas_brain)
- Full runtime exercise without `atlas_brain` on `sys.path`; this slice adds the standalone substrate and smoke coverage, but deep task modules still carry Atlas-owned domain dependencies.

## Cross-product dependencies (acknowledged)

| Dependency | Status | Notes |
|---|---|---|
| **LLM Infrastructure** | extracted package available | Standalone bridge delegates to `extracted_llm_infrastructure`; task-level LLM seams remain Phase 3 work |
| **Evidence claims** | atlas-core | `services/b2b/evidence_claim_*.py` is shared with churn intel — keep central |
| **Campaign suppression** | injectable in standalone mode | Atlas bridge remains default; standalone mode uses a configured `SuppressionPolicy` |
| **Campaign sender (Resend)** | injectable in standalone mode | Atlas bridge remains default; standalone mode uses a configured campaign sender |
| **`_b2b_shared.py`** | atlas-core | Circular-import risk; not extracted |
| **`challenger_dashboard_claims.py`** | injectable in standalone mode | Battle-card displacement gates use configured host claim readers |

## Standalone toggle

Set `EXTRACTED_COMP_INTEL_STANDALONE=1` to route core substrate imports away from Atlas:

- `config.py` uses `extracted_competitive_intelligence._standalone.config`
- `storage/database.py` uses `extracted_llm_infrastructure.storage.database`
- `auth/dependencies.py` uses fail-closed standalone auth hooks
- `services/campaign_sender.py` and `autonomous/tasks/campaign_suppression.py` use injectable product-owned ports
- `services/crm_provider.py` uses an injectable CRM provider port for standalone lead/contact writes
- `services/email_provider.py` uses an injectable email provider port for standalone checkout confirmations
- `services/b2b/pdf_renderer.py` uses an injectable PDF renderer port for standalone gated report delivery
- `services/b2b/llm_exact_cache.py` uses `extracted_llm_infrastructure` for standalone battle-card prompt envelopes
- `services/b2b/anthropic_batch.py` uses `extracted_llm_infrastructure` for standalone battle-card batch overlays
- `services/protocols.py` and `pipelines/llm.py` use `extracted_llm_infrastructure`
- `services/scraping/sources.py` owns the source enum and classification sets locally
- MCP shared/server modules are extracted-owned and importable without the optional `mcp` package installed
- `services/b2b/challenger_dashboard_claims.py` uses fail-closed host reader ports for displacement ProductClaim aggregation
- Lazy package fallbacks fail closed in standalone mode instead of silently importing Atlas package namespaces

Standalone adapters that require a host application fail closed until configured.

## Sync workflow

```bash
# Re-copy from atlas_brain into the scaffold (idempotent; safe to re-run)
bash scripts/sync_extracted_competitive_intelligence.sh

# Verify zero drift; exits non-zero if anything differs
bash scripts/validate_extracted_competitive_intelligence.sh
```

When you change a source file under `atlas_brain/`, run the sync afterward and commit the scaffold update in the same PR. The CI workflow at `.github/workflows/extracted_competitive_intelligence_checks.yml` enforces zero-drift on every PR that touches the scaffold.

Modules listed under `owned` in `manifest.json` are intentionally product-owned:
sync and byte-drift validation skip them, while ASCII and import checks still
cover them. This is the handoff path for moving a scaffolded module from Atlas
snapshot to extracted implementation.

The service-level `services/vendor_registry.py` module is product-owned: it
uses the extracted storage bridge and is no longer byte-synced from Atlas.
The first owned MCP modules are `vendor_registry.py`, `displacement.py`, and
`cross_vendor.py`; they are read-oriented surfaces with extracted-owned support
dependencies. `write_intelligence.py` is also product-owned: simple database
writes run locally, while deep runtime builders such as challenger briefs and
accounts-in-motion reports are explicit host ports defined in
`mcp/b2b/write_ports.py`.

`services/b2b/source_impact.py` and `services/scraping/capabilities.py` are now
product-owned as well. They provide the source impact ledger and static scrape
capability registry without importing Atlas runtime modules.

`services/b2b/challenger_dashboard_claims.py` is a product-owned host port for
direct-displacement ProductClaim aggregation. The mapped battle-card task keeps
its relative import: Atlas resolves the native aggregator, while standalone
competitive hosts register explicit claim readers.

`services/b2b_competitive_sets.py` is product-owned competitive-set planning
logic. Runtime preview helpers that need reasoning hashes, pool layers, or
scorecard fallback rows go through `services/b2b/competitive_set_ports.py`, so
standalone hosts can register explicit adapters instead of importing task
internals.

`autonomous/tasks/_b2b_cross_vendor_synthesis.py` is product-owned packet,
citation, contract-normalization, and lookup-reader logic. It depends on the
extracted semantic-cache facade for evidence hashes and no longer byte-syncs
from Atlas.

`autonomous/tasks/_b2b_batch_utils.py` is product-owned Anthropic batch
enablement and reconciliation helper logic. In standalone mode it resolves
auxiliary Anthropic LLM slots through `extracted_llm_infrastructure`.

`templates/email/vendor_briefing.py` is a product-owned customer-facing
renderer. It no longer imports runtime settings; hosts configure the fallback
witness highlight limit through an explicit renderer function.

`reasoning/cross_vendor_selection.py` is product-owned pure selection logic for
battles, categories, and asymmetry pairs. It is covered by extracted-package
behavior tests and is no longer byte-synced from Atlas.

The single-pass prompt modules under `reasoning/single_pass_prompts/` are
product-owned LLM contracts. They are covered by extracted-package contract
tests and are no longer byte-synced from Atlas.

`reasoning/ecosystem.py` is a product-owned host port for category-level
ecosystem analysis. The mapped battle-card task imports it through the package
relative path: Atlas resolves to its native analyzer, while standalone
competitive hosts register an explicit adapter.

## Local checks

```bash
bash scripts/run_extracted_competitive_intelligence_checks.sh
```

Runs six checks in sequence:

1. `validate_*.sh` — byte-diff mapped scaffold files vs source, excluding product-owned manifest entries
2. `check_ascii_python_*.sh` — every scaffolded `.py` is ASCII-only (true 0-based offsets on failure)
3. `check_extracted_competitive_intelligence_imports.py` — relative imports either resolve inside the scaffold or are listed in `import_debt_allowlist.txt` (resolver honors `level - 1` Python semantics)
4. `smoke_extracted_competitive_intelligence_imports.py` — every public module imports without raising
5. `smoke_extracted_competitive_intelligence_standalone.py` — standalone-mode substrate imports resolve to extracted-owned or extracted-LLM modules
6. Extracted pytest coverage — product-boundary behavior tests for cross-vendor selection, prompt contracts, and host ports

## Import debt

`import_debt_allowlist.txt` is **empty by design** — every scaffolded relative import resolves cleanly. The file is retained so future Phase 2/3 work can record intentional debt as the standalone substrate is built.

## Why a separate scaffold from `extracted_llm_infrastructure/`?

The Competitive Intel platform is a distinct product with a different audience: it sells to sales / RevOps teams, not ML platform teams. Customer-facing artifacts (battle cards, vendor briefings) are fundamentally different from cost-optimization infrastructure. Keeping the scaffolds separate lets either be priced, packaged, and shipped independently.

The Competitive Intel modules **consume** the LLM-infra package in standalone
mode through the local bridge modules. Deep task-level runtime seams still need
Phase 3 follow-up before the package is publishable without Atlas host
dependencies.

## Status

See `STATUS.md` for the per-file extraction state and remaining Phase 2 / Phase 3 work.
