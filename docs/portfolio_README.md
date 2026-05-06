# Applied AI Portfolio — Atlas

**3 verifiable demos. Each runs end-to-end on a clean Postgres + Python box.**

I'm a backend / applied-AI engineer. Atlas is a multi-modal AI platform I designed and built around three reusable layers: an LLM gateway with cost tracking, a deterministic quality-gate engine, and an LLM-orchestrated content-generation pipeline. Below are the three demos a recruiter can run (or watch) to verify each layer.

If you have ~10 minutes total: watch the three videos. If you have ~30: run the demos yourself. If you have an afternoon and a technical reviewer: jump to the [technical appendix](portfolio_skill_demos.md).

---

## Demo 1 — LLM Campaign Generation Pipeline

**One-liner:** opportunities go in → LLM generates campaigns → reviewer queue + cost ledger come out.

**What this proves:**
- Async Python at production scale (asyncpg, 64.5K LOC across the extracted content-pipeline package alone — see [cross-product audit](extraction/cross_product_audit_2026-05-04.md))
- LLM orchestration with provider routing + token-level cost attribution
- Multi-stage pipeline: schema migrate → ingestion → generation → review → export

**Verify it:**

```bash
# Clean Postgres box, real LLM call. Real CLI flags from
# extracted_content_pipeline/docs/host_install_runbook.md.
export DATABASE_URL=postgres://...

# 1. Apply the packaged extracted_content_pipeline migrations
python scripts/run_extracted_content_pipeline_migrations.py --database-url $DATABASE_URL

# 2. Import seed opportunities (CSV or JSON, --dry-run available)
python scripts/load_extracted_campaign_opportunities.py customer_opportunities.csv \
    --format csv --account-id acct_demo --database-url $DATABASE_URL

# 3. Generate campaigns (real LLM call routed via extracted_llm_infrastructure)
python scripts/run_extracted_campaign_generation_postgres.py \
    --account-id acct_demo --limit 5 --database-url $DATABASE_URL

# 4. Show the generated drafts
psql $DATABASE_URL -c "select id, vendor_name, channel, subject, body
                       from b2b_campaigns order by created_at desc limit 3;"

# 5. Show the cost ledger entry — closes the loop
psql $DATABASE_URL -c "select model_name, model_provider,
                              input_tokens, output_tokens, cost_usd
                       from llm_usage order by created_at desc limit 3;"
```

**Code to look at (top 3):**
- `extracted_content_pipeline/campaign_generation.py` — orchestration core
- `extracted_content_pipeline/services/reasoning_provider_port.py` — Protocol-typed provider seam
- `extracted_llm_infrastructure/services/cost/` — cost ledger / cache savings / drift detection / budget gates

**Status:** ~85% standalone-usable per [cross-product audit](extraction/cross_product_audit_2026-05-04.md). Zero hard `atlas_brain` imports outside try/except bridges. Full runbook in [`extracted_content_pipeline/docs/host_install_runbook.md`](../extracted_content_pipeline/docs/host_install_runbook.md).

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
# Part A — Quality gate. Real QualityInput / QualityReport API.
python -c "
from extracted_quality_gate.source_quality_pack import evaluate_source_quality
from extracted_quality_gate.types import QualityInput

# Two witness rows. The first is renderable; the second is a 'pain' witness
# whose polarity is positive, which the render gate suppresses.
report = evaluate_source_quality(QualityInput(
    artifact_type='b2b_review',
    context={'witnesses': [
        {
            'witness_id': 'w1',
            'grounding_status': 'grounded',
            'phrase_subject': 'subject_vendor',
            'phrase_role':    'primary_driver',
            'phrase_polarity':'negative',
            'witness_type':   'pain',
            'pain_confidence':'strong',
        },
        {
            'witness_id': 'w2',
            'grounding_status': 'grounded',
            'phrase_subject': 'subject_vendor',
            'phrase_role':    'primary_driver',
            'phrase_polarity':'positive',     # suppressed: positive polarity on a pain witness
            'witness_type':   'pain',
            'pain_confidence':'weak',
        },
    ]},
))
print('decision:', report.decision)              # WARN
for f in report.findings:
    print(' -', f.code, f.severity, f.message)
"

# Part B — Forbidden-import guard, baseline pass
bash scripts/validate_extracted_competitive_intelligence.sh
# Validation passed: mapped files match
# forbid_hard_atlas_imports: clean

# Part C — Synthetic regression, fail-closed
echo 'from atlas_brain.config import settings' > extracted_content_pipeline/_test_regression.py
bash scripts/validate_extracted_content_pipeline.sh
# rc=1
# extracted_content_pipeline/_test_regression.py:1: from atlas_brain.config import ...
rm extracted_content_pipeline/_test_regression.py
```

**Code to look at (top 3):**
- `extracted_quality_gate/source_quality_pack.py` — pure-function PASS/WARN/BLOCK with `QualityInput.context['witnesses']` shape
- `extracted/_shared/scripts/forbid_hard_atlas_imports.py` — 142-line AST scanner shared across packages
- `tests/test_extracted_reasoning_core_domains.py` — 21 tests for the typed-envelope abstraction the guards protect

**Status:** Both shipped to main. Quality gate is a fully extracted package (`extracted_quality_gate/`); forbid scanner runs in CI on every PR via `scripts/validate_extracted_*.sh`.

**🎬 Recording:** _(90 sec — TODO: record + paste link)_

---

## Demo 3 — Review / Product Intelligence Pipeline

**One-liner:** raw product reviews from web sources → structured LLM-ready payloads → archetype scoring against a fixed catalog.

**What this proves:**
- Multi-source web data acquisition at production scale (19 review sources, captcha + proxy + rate-limit + browser automation)
- Pure-function payload shaping (truncation policy, source-weight tagging, raw_metadata coercion)
- Scoring layer: archetype classification + temporal evidence (velocity / slope / anomaly / percentile baselines)

**Verify it:**

```bash
# 1. Show the canonical source enum + display names
python -c "
from atlas_brain.services.scraping.sources import (
    ReviewSource, ALL_SOURCES, SLUG_SOURCES, SEARCH_SOURCES, display_name,
)
print(f'{len(ALL_SOURCES)} review sources registered')
for s in sorted(ALL_SOURCES, key=lambda r: r.value):
    bucket = 'slug' if s in SLUG_SOURCES else ('search' if s in SEARCH_SOURCES else 'other')
    print(f'  {s.value:18s}  {display_name(s):20s}  ({bucket})')
"

# 2. Run a per-source raw-capture audit on a known vendor (real flags)
python scripts/audit_capterra_raw_capture.py --vendor-name 'snowflake' --pages 1
# Writes raw HTML + parsed reviewer-field audit under data/audits/capterra_raw/

# 3. Show the production payload shaper — raw row -> structured LLM-ready payload
python -c "
from atlas_brain.services.b2b.enrichment_domain import (
    build_classify_payload, smart_truncate,
)

raw_row = {
    'id': 'r1',
    'vendor_name': 'Snowflake',
    'product_name': 'Snowflake Data Cloud',
    'product_category': 'cloud_data_warehouse',
    'source': 'capterra',
    'content_type': 'review',
    'rating': 3.0, 'rating_max': 5,
    'summary':     'Pricing model surprised us',
    'review_text': 'Renewal came in 22% higher than the quote we signed. '
                   'Support said it was a tier change. We are evaluating BigQuery.',
    'pros':        'fast queries, good UI',
    'cons':        'cost, contract opacity',
    'reviewer_title':    'Director of Data Engineering',
    'reviewer_company':  'Acme Corp',
    'company_size_raw':  '1001-5000',
    'reviewer_industry': 'retail',
    'raw_metadata': {'source_weight': 0.9, 'source_type': 'review_site'},
}
payload = build_classify_payload(raw_row, truncate_length=3000, smart_truncate=smart_truncate)
import json; print(json.dumps(payload, indent=2))
"

# 4. Score the row against the archetype catalog
python -c "
from extracted_reasoning_core import score_archetypes
matches = score_archetypes(
    evidence={'pricing_mentions': 12, 'exec_change': True, 'review_volume': 30},
    limit=3,
)
for m in matches:
    print(m.archetype_id, round(m.score, 2), m.label, '|', m.risk_label)
"
```

**Code to look at (top 3):**
- `atlas_brain/services/scraping/` — 23 top-level modules + per-source parsers + universal adapters covering 19 review sources
- `atlas_brain/services/b2b/enrichment_*.py` — 34 enrichment modules (pain / buyer_authority / phrase_metadata / domain / repair / budget / persistence / …)
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

[`portfolio_skill_demos.md`](portfolio_skill_demos.md) — full breakdown of where reusable value lives across 8 axes (extraction contracts, source adapters, product/category normalization, enrichment logic, quality gates, scoring/aggregation, UI/API surfaces, demo-ready outputs). Read this if you're a hiring manager or reviewing for a senior+ role.

---

## Honest framing

- **Atlas is a working system, not a shipped commercial product.** Pieces work end-to-end (the demos above run); pieces are scaffolds (see per-package `STATUS.md` files).
- **Standalone-readiness varies.** `extracted_content_pipeline` is ~85% standalone; `extracted_competitive_intelligence` is ~32%. Numbers from `docs/extraction/cross_product_audit_2026-05-04.md`, not aspirational.
- **The architectural artifacts are themselves the portfolio.** `STATUS.md` scoreboards, the M1–M6 hybrid extraction program, the cross-product audit — these prove I can design, document, and execute multi-PR architectural shifts at scale, which is often what staff/principal interviews are actually testing for.

---

## Contact

**Juan Canfield** — [info@juancanfield.com](mailto:info@juancanfield.com) — [juancanfield.com](https://juancanfield.com)
