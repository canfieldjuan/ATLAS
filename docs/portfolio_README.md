# Applied AI Portfolio — Atlas

**3 verifiable demos. Each runs end-to-end in under 5 minutes on a clean Postgres + Python box.**

I'm a backend / applied-AI engineer. Atlas is a multi-modal AI platform I designed and built around three reusable layers: an LLM gateway with cost tracking, a deterministic quality-gate engine, and an LLM-orchestrated content-generation pipeline. Below are the three demos a recruiter can run (or watch) to verify each layer.

If you have ~10 minutes total: watch the three videos. If you have ~30: run the demos yourself. If you have an afternoon and a technical reviewer: jump to the [technical appendix](portfolio_skill_demos.md).

---

## Demo 1 — LLM Campaign Generation Pipeline

**One-liner:** opportunities go in → LLM generates campaigns → reviewer queue + cost ledger come out.

**What this proves:**
- Async Python at production scale (asyncpg, 67K LOC, 118 modules in this layer alone)
- LLM orchestration with provider routing + token-level cost attribution
- Multi-stage pipeline: schema migrate → ingestion → generation → review → export → cost reconciliation

**Verify it:**

```bash
# Clean Postgres box, real Anthropic call. ~3 minutes wall-clock.

# 1. Schema migrate
python scripts/run_extracted_campaign_postgres_migrations.py --database-url $DB

# 2. Import seed opportunities
python scripts/run_extracted_campaign_postgres_import.py \
    --database-url $DB --opportunities seed_opportunities.jsonl

# 3. Generate campaigns (real LLM call, routed via extracted_llm_infrastructure)
python scripts/run_extracted_campaign_generation_postgres.py \
    --database-url $DB --reasoning-context reasoning_seed.json

# 4. Show generated drafts
psql $DB -c "select id, vendor_name, subject, body from b2b_campaigns limit 3;"

# 5. Show cost ledger entry — closes the loop
psql $DB -c "select model, prompt_tokens, completion_tokens, total_cost_usd
             from llm_usage order by created_at desc limit 3;"
```

**Code to look at (top 3):**
- `extracted_content_pipeline/campaign_generation.py` — orchestration core
- `extracted_content_pipeline/services/reasoning_provider_port.py` — Protocol-typed provider seam
- `extracted_llm_infrastructure/services/cost/` — cost ledger / cache savings / drift detection / budget gates

**Status:** ~85% standalone-usable per [cross-product audit](extraction/cross_product_audit_2026-05-04.md). Zero hard `atlas_brain` imports outside try/except bridges.

**🎬 Recording:** _(2 min — TODO: record + paste link)_

---

## Demo 2 — Quality Gate + Extraction Contract

**One-liner:** deterministic PASS / WARN / BLOCK validation, plus an AST-based regression guard that fails CI when an extracted package introduces a hard dependency on the host monolith.

**What this proves:**
- Pure-function policy engine (composable validation packs)
- AST-based static analysis (Python `ast` module, walks Import/ImportFrom nodes, allows `Try` / `If` / `FunctionDef` / `AsyncFunctionDef` ancestors as legitimate bridge patterns)
- CI-grade regression prevention (synthetic regression → fail-closed with file:line + remediation hint)

**Verify it:**

```bash
# Part A — Quality gate (3 lines)
python -c "
from extracted_quality_gate.source_quality_pack import evaluate_source_quality
report = evaluate_source_quality({
    'source_rows': [
        {'source_id': 'r1', 'witness_text': 'pricing increased 20%'},
        {'source_id': 'r2', 'witness_text': ''},   # gets suppressed
    ]
})
print('decision:', report.decision)         # WARN
for f in report.findings: print(' -', f)    # row-level finding for r2
"

# Part B — Forbidden-import guard, baseline pass
bash scripts/validate_extracted_competitive_intelligence.sh
# Validation passed: mapped files match
# forbid_hard_atlas_imports: clean

# Part C — Synthetic regression, fail-closed
echo "from atlas_brain.config import settings" > extracted_content_pipeline/_test_regression.py
bash scripts/validate_extracted_content_pipeline.sh
# rc=1
# extracted_content_pipeline/_test_regression.py:1: from atlas_brain.config import ...
rm extracted_content_pipeline/_test_regression.py
```

**Code to look at (top 3):**
- `extracted_quality_gate/source_quality_pack.py` — pure-function PASS/WARN/BLOCK
- `extracted/_shared/scripts/forbid_hard_atlas_imports.py` — 119-line AST scanner
- `tests/test_extracted_reasoning_core_domains.py` — 21 tests for the typed-envelope abstraction the guards protect

**Status:** Both shipped to main. Quality gate is a fully extracted package (`extracted_quality_gate/`); forbid scanner runs in CI on every PR.

**🎬 Recording:** _(90 sec — TODO: record + paste link)_

---

## Demo 3 — Review / Product Intelligence Extraction

**One-liner:** raw product reviews from web sources → structured business intelligence (pain points, buyer authority signals, competitor mentions, phrase-level evidence atoms).

**What this proves:**
- Multi-source web data acquisition at production scale (16 review sources, captcha + proxy + rate-limit + browser automation)
- Composable enrichment-policy framework (25 enrichment policy modules, each a pure function with typed input/output)
- Scoring layer: archetype classification + temporal evidence (velocity / slope / anomaly / percentile baselines)

**Verify it:**

```bash
# 1. Show source registry — 16 sources, capability matrix per source
python -c "
from atlas_brain.services.scraping.sources import list_sources
for s in list_sources(): print(f'{s.name}: {s.capabilities}')
"

# 2. Run a per-source raw-capture audit on a known vendor
python scripts/audit_capterra_raw_capture.py --vendor 'snowflake' --max 5
# Returns parsed reviews with structured fields

# 3. Run an enrichment policy chain on a single review row
python -c "
from atlas_brain.services.b2b.enrichment_row_runner import run_enrichment_row
result = run_enrichment_row(
    row={'source_id': 'capterra:snowflake#42', 'review_text': '...real review...'},
    policies=['pain', 'buyer_authority', 'phrase_metadata', 'domain'],
)
print('pain points:', result.pain_findings)
print('buyer signals:', result.buyer_authority)
print('budget used:', result.budget_used)
print('stage:', result.stage)
"

# 4. Score the enriched row against the archetype catalog
python -c "
from extracted_reasoning_core import score_archetypes
matches = score_archetypes(
    evidence={'pricing_mentions': 12, 'exec_change': True, 'review_volume': 30},
    limit=3,
)
for m in matches: print(m.archetype_id, m.score, m.label)
"
```

**Code to look at (top 3):**
- `atlas_brain/services/scraping/` — 19 modules covering 16 review sources
- `atlas_brain/services/b2b/enrichment_*.py` — 25 policy modules (pain / buyer_authority / phrase_metadata / domain / repair / ...)
- `extracted_reasoning_core/archetypes.py` + `temporal.py` — the scoring layer

**Status:** Scraping + enrichment live in `atlas_brain/`; reasoning-core scoring is fully extracted as `extracted_reasoning_core/`. Standalone-readiness varies by sub-layer (see [appendix](portfolio_skill_demos.md)).

**🎬 Recording:** _(3 min — TODO: record + paste link)_

---

## Skill tags (for ATS / keyword screens)

`Python 3.11` • `asyncio / asyncpg` • `FastAPI` • `Postgres + migrations` • `LLM orchestration (Anthropic / OpenAI / vLLM / llama.cpp / Ollama)` • `MCP (Model Context Protocol)` • `Protocol-based dependency injection` • `AST-based static analysis` • `Web scraping at scale (captcha / proxy / rate-limit / browser automation)` • `Composable policy engines` • `Token-level cost attribution` • `OAuth2 / BYOK key management` • `LangGraph multi-agent workflows` • `CI/CD architecture` • `multi-PR architectural extraction programs`

---

## How to use this portfolio

| You are a … | Start with |
|---|---|
| Recruiter screening for "applied AI / LLM platform" | Demo 1 + Demo 2 — they prove the technical chops in 5 minutes total |
| Hiring manager evaluating staff/principal level | All 3 demos + skim the [M5-α design](../extracted_reasoning_core/domains.py) + [M6 compatibility matrix](hybrid_reasoning_compatibility_matrix.md) |
| Tech screen interviewer | Run Demo 2 yourself — the AST scanner is the smallest, sharpest demonstration of architectural judgment |
| Founder evaluating for a build role | Demo 3 — proves I can take an open-ended "extract structured intelligence from messy text" problem from raw acquisition through scoring |

---

## Technical appendix

[`portfolio_skill_demos.md`](portfolio_skill_demos.md) — full breakdown of where reusable value lives across 8 axes (extraction contracts, source adapters, product/category normalization, enrichment logic, quality gates, scoring/aggregation, UI/API surfaces, demo-ready outputs). 495 lines. Read this if you're a hiring manager or reviewing for a senior+ role.

---

## Honest framing

- **Atlas is a working system, not a shipped commercial product.** Pieces work end-to-end (the demos above run); pieces are scaffolds (see per-package `STATUS.md` files).
- **Standalone-readiness varies.** `extracted_content_pipeline` is ~85% standalone; `extracted_competitive_intelligence` is ~32%. Numbers from `docs/extraction/cross_product_audit_2026-05-04.md`, not aspirational.
- **The architectural artifacts are themselves the portfolio.** `STATUS.md` scoreboards, the M1–M6 hybrid extraction program, the cross-product audit — these prove I can design, document, and execute multi-PR architectural shifts at scale, which is often what staff/principal interviews are actually testing for.

---

## Contact

_(TODO: name / email / linkedin / personal-site link)_
