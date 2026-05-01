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
| `autonomous/tasks/_b2b_cross_vendor_synthesis.py` | Cross-vendor packet builders |
| `services/b2b_competitive_sets.py` | Planner that scopes synthesis to a competitive set |
| `reasoning/cross_vendor_selection.py` | Selection logic for which vendor pairs deserve LLM budget |
| `reasoning/single_pass_prompts/cross_vendor_battle.py` | Single-pass cross-vendor battle prompt |
| `reasoning/single_pass_prompts/battle_card_reasoning.py` | Contracts-first battle card reasoning prompt |
| `templates/email/vendor_briefing.py` | HTML email template (Outlook-compatible) |
| `api/b2b_vendor_briefing.py` | REST endpoints for preview / send / approve / reject |

Plus 9 migrations: `095_b2b_vendor_registry.sql`, `099_displacement_edges_and_company_signals.sql`, `101_vendor_buyer_profiles.sql`, `147_displacement_velocity.sql`, `158_cross_vendor_conclusions.sql`, `245_cross_vendor_reasoning_synthesis.sql`, `261_b2b_competitive_sets.sql`, `262_b2b_competitive_set_runs.sql`, `263_b2b_competitive_set_run_constraints.sql`.

## What's out of scope (remaining Phase 3)

- Decoupling battle-card LLM calls so they consume `extracted_llm_infrastructure/` directly (LLM-infra extraction is in PR #40)
- API endpoint extraction beyond the briefing endpoints (`/b2b/win-loss`, dashboard endpoints stay in atlas_brain)
- Full runtime exercise without `atlas_brain` on `sys.path`; this slice adds the standalone substrate and smoke coverage, but deep task modules still carry Atlas-owned domain dependencies.

## Cross-product dependencies (acknowledged)

| Dependency | Status | Notes |
|---|---|---|
| **LLM Infrastructure** | extracted via PR #40 | `b2b_battle_cards.py:260` calls `pipelines.llm.call_llm_with_skill`; will rebase once PR #40 merges |
| **Evidence claims** | atlas-core | `services/b2b/evidence_claim_*.py` is shared with churn intel — keep central |
| **Campaign suppression** | injectable in standalone mode | Atlas bridge remains default; standalone mode uses a configured `SuppressionPolicy` |
| **Campaign sender (Resend)** | injectable in standalone mode | Atlas bridge remains default; standalone mode uses a configured campaign sender |
| **`_b2b_shared.py`** | atlas-core | Circular-import risk; not extracted |
| **`challenger_dashboard_claims.py`** | atlas-core | Bridge module aggregating displacement claims |

## Standalone toggle

Set `EXTRACTED_COMP_INTEL_STANDALONE=1` to route core substrate imports away from Atlas:

- `config.py` uses `extracted_competitive_intelligence._standalone.config`
- `storage/database.py` uses `extracted_llm_infrastructure.storage.database`
- `auth/dependencies.py` uses fail-closed standalone auth hooks
- `services/campaign_sender.py` and `autonomous/tasks/campaign_suppression.py` use injectable product-owned ports
- `services/protocols.py` and `pipelines/llm.py` use `extracted_llm_infrastructure`
- `services/scraping/sources.py` owns the source enum and classification sets locally
- MCP shared/server modules are extracted-owned and importable without the optional `mcp` package installed
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

## Local checks

```bash
bash scripts/run_extracted_competitive_intelligence_checks.sh
```

Runs five checks in sequence:

1. `validate_*.sh` — byte-diff scaffold vs source (with explicit missing-source reporting)
2. `check_ascii_python_*.sh` — every scaffolded `.py` is ASCII-only (true 0-based offsets on failure)
3. `check_extracted_competitive_intelligence_imports.py` — relative imports either resolve inside the scaffold or are listed in `import_debt_allowlist.txt` (resolver honors `level - 1` Python semantics)
4. `smoke_extracted_competitive_intelligence_imports.py` — every public module imports without raising
5. `smoke_extracted_competitive_intelligence_standalone.py` — standalone-mode substrate imports resolve to extracted-owned or extracted-LLM modules

## Import debt

`import_debt_allowlist.txt` is **empty by design** — every scaffolded relative import resolves cleanly. The file is retained so future Phase 2/3 work can record intentional debt as the standalone substrate is built.

## Why a separate scaffold from `extracted_llm_infrastructure/`?

The Competitive Intel platform is a distinct product with a different audience: it sells to sales / RevOps teams, not ML platform teams. Customer-facing artifacts (battle cards, vendor briefings) are fundamentally different from cost-optimization infrastructure. Keeping the scaffolds separate lets either be priced, packaged, and shipped independently.

The Competitive Intel modules **consume** the LLM-infra package at runtime (battle card overlays use `call_llm_with_skill`, vendor briefings use `clean_llm_output`), so the two extractions are sequenced: LLM-infra first (PR #40), then Competitive Intel here, then a Phase 2/3 follow-up to rewire LLM imports to consume the extracted package directly.

## Status

See `STATUS.md` for the per-file extraction state and remaining Phase 2 / Phase 3 work.
