# Atlas Portfolio: Reusable Value, Mapped to Skills

**Purpose:** map the codebase along eight axes a recruiter can verify, and for each axis show **where the reusable value lives**, what skill it proves, and how to demo it.

**Why this lens:** "extracted N packages" doesn't translate to skills. "I designed the extraction contract pattern, the source-adapter Protocol, the enrichment policy framework, and the scoring engine" does — and each of those is a separately-demoable artifact.

**Honesty bar:** every percentage in this doc traces to `docs/extraction/cross_product_audit_2026-05-04.md` or to a `STATUS.md`. No aspirational claims.

---

## Axis 1 — Extraction Contracts

**The core architectural pattern that ties everything together.** Every extracted package has a manifest, a validator, and a host-port Protocol surface. Same pattern, four packages.

### Where the value lives

| Component | File | What it does |
|---|---|---|
| Manifests | `extracted_*/manifest.json` (×4) | Source → scaffold mappings for byte-snapshot validation |
| Validator (shared) | `extracted/_shared/scripts/validate_extracted.sh` | Drives the byte-equality check + import-debt audit |
| AST-based forbid scanner | `extracted/_shared/scripts/forbid_hard_atlas_imports.py` (M4, #207) | Fails CI if any file introduces a hard `atlas_brain` top-level import outside try/except/if-env-gate blocks |
| Host-port Protocols (one example) | `extracted_content_pipeline/campaign_ports.py` | `CampaignReasoningContextProvider`, `LLMClient`, `CampaignSender`, `CampaignAudit`, … |
| The M5-α typed envelope | `extracted_reasoning_core/domains.py` (#211) | `DomainReasoningResult[PayloadT]` + Subject/Producer/Consumer Protocols + domain registry |
| Compatibility matrix | `docs/hybrid_reasoning_compatibility_matrix.md` (#240) | Decision rubric for reuse vs rebuild on new domains |
| The plan itself | `docs/hybrid_extraction_plan_status_2026-05-05.md`, `hybrid_extraction_implementation_plan.md`, `hybrid_extraction_execution_board.md` | Multi-PR architectural program with explicit exit criteria |

### Reusable shape

This pattern transfers verbatim to **any monolith you'd extract from**:
- Manifest as the source of truth for "what's snapshotted vs owned"
- AST scanner as the regression guard
- `Protocol` types as the seam between host and extracted runtime
- Typed envelope (`DomainReasoningResult[PayloadT]`) for cross-domain composability

### Skill claim

"I can take a 350K-LOC codebase and design + execute a multi-PR extraction program with explicit decision rubrics, CI guardrails, and verifiable boundaries." Translates to: **staff/principal architect, platform engineer, eng-lead** roles where the job is to design boundaries, not just write features.

### Demo

```bash
# Show the byte-equality + forbidden-import guard run on every package
bash scripts/validate_extracted_competitive_intelligence.sh
bash scripts/validate_extracted_content_pipeline.sh
bash scripts/validate_extracted_llm_infrastructure.sh
# All output: "Validation passed" + "forbid_hard_atlas_imports: clean"

# Inject a synthetic regression and watch the AST scanner fail-close
echo "from atlas_brain.config import settings" > extracted_content_pipeline/_test.py
bash scripts/validate_extracted_content_pipeline.sh
# rc=1 with file:line of the violation
rm extracted_content_pipeline/_test.py

# Show the typed envelope + domain registry
python3 -m pytest tests/test_extracted_reasoning_core_domains.py -q
# 21 passed
```

---

## Axis 2 — Source Adapters

**Two layers:** application-data adapters (CRM, email, calendar, billing, scrape providers) and the **industrial-grade review-scraping stack** that's the real differentiator.

### Where the value lives

#### Application data adapters (Protocol-typed, swappable)

| Adapter | File | Implementations |
|---|---|---|
| CRM | `atlas_brain/services/crm_provider.py` + `extracted_*/services/crm_provider.py` | `DatabaseCRMProvider` (asyncpg, NocoDB-compatible) |
| Email | `atlas_brain/services/email_provider.py` | `GmailEmailProvider` (OAuth2) + `ResendEmailProvider` fallback + IMAP read for any provider |
| Apollo | `atlas_brain/services/apollo_provider.py` | Apollo.io B2B contact / company API |
| Calendar | `atlas_brain/services/calendar_provider.py` | Google Calendar (OAuth2) + CalDAV (Nextcloud / Apple / Fastmail / Proton) |
| Billing / cost reconciliation | `extracted_llm_infrastructure/services/cost/openai_billing.py`, `provider_cost_sync.py` | OpenAI billing API + per-provider token cost ledger |

#### Review-scraping stack (`atlas_brain/services/scraping/`, 23 top-level modules + parsers/ + universal/)

| Module | Role |
|---|---|
| `sources.py` | Canonical `ReviewSource` enum + classification sets — 19 sources (G2, Capterra, TrustRadius, Gartner, PeerSpot, GetApp, ProductHunt, Trustpilot, Reddit, HackerNews, GitHub, YouTube, StackOverflow, Quora, Twitter/X, RSS, Software Advice, SourceForge, Slashdot) |
| `serp_discovery.py` | SERP-driven target discovery |
| `target_planning.py`, `target_provisioning.py`, `target_validation.py` | Target lifecycle: plan / provision / validate |
| `eligibility.py`, `source_fit.py`, `source_yield.py` | Per-source scoring of "is this target worth scraping" |
| `client.py`, `browser.py`, `proxy.py`, `rate_limiter.py`, `captcha.py`, `web_unlocker_http.py` | Web Unlocker / browser automation infra |
| `parsers/` | Per-source HTML → structured-review parsers |
| `universal/` | Vendor-agnostic ingestion adapters |
| `capabilities.py` | Capability registry (which sources do which fields) |
| `relevance.py`, `profiles.py` | Post-scrape relevance / dedup, per-vendor profile cache |

Plus per-source raw-capture audit scripts: `scripts/audit_{capterra,g2,gartner,getapp,trustpilot}_raw_capture.py` — proves I can validate scrape quality at the source level.

### Reusable shape

The Protocol pattern (e.g. `class EmailProvider(Protocol): async def send(...)`) is portable to any provider category. The scraping stack's split — **discovery / planning / eligibility / fetching / parsing / capability** — generalizes to any web-data-acquisition product.

### Skill claim

"I built a multi-source web-data-acquisition pipeline at production scale (19 review sources, captcha + proxy + rate-limit + browser automation), and I designed adapter Protocols that let downstream consumers swap providers without code changes." Translates to: **data engineering, scraping/ingestion, integration engineer** roles.

### Demo

```bash
# Walk the canonical source enum + classification sets (no atlas runtime imports)
python -c "
from atlas_brain.services.scraping.sources import (
    ReviewSource, ALL_SOURCES, SLUG_SOURCES, SEARCH_SOURCES, display_name,
)
print(f'{len(ALL_SOURCES)} sources registered')
for s in sorted(ALL_SOURCES, key=lambda r: r.value):
    bucket = 'slug' if s in SLUG_SOURCES else ('search' if s in SEARCH_SOURCES else 'other')
    print(f'  {s.value:18s}  {display_name(s):20s}  ({bucket})')
"

# Run a per-source raw-capture audit (real flags from scripts/audit_capterra_raw_capture.py)
python scripts/audit_capterra_raw_capture.py --vendor-name 'snowflake' --pages 1
# Writes raw HTML + parsed reviewer-field audit under data/audits/capterra_raw/
```

---

## Axis 3 — Product / Category Normalization

**The hardest data-quality problem in B2B intelligence:** "Snowflake" vs "snowflake.com" vs "Snowflake Inc.". Atlas has a multi-layer answer.

### Where the value lives

| Component | File | What it does |
|---|---|---|
| Vendor registry | `atlas_brain/services/vendor_registry.py` | Canonical vendor names + aliases + tenant-scoped overrides |
| Fuzzy vendor search | `atlas_brain/mcp/b2b/vendor_registry.py` (registered as `fuzzy_vendor_search`, `fuzzy_company_search`) and `atlas_brain/api/b2b_dashboard.py` | Token-based + edit-distance matching with confidence scoring |
| Account resolver | `atlas_brain/services/b2b/account_resolver.py` | Match incoming review/scrape rows to canonical accounts |
| Category intelligence | `extracted_content_pipeline/campaign_postgres_seller_category_intelligence.py` (#209) | Amazon-seller broad-category snapshot refresh |
| Source impact registry | `atlas_brain/services/b2b/source_impact.py` (extracted to `extracted_quality_gate/`) | Per-source-field baseline coverage |

### Reusable shape

Same shape works for any "messy entity matching" problem:
- Canonical registry + alias table + tenant override
- Fuzzy match with explicit confidence band
- Resolver that returns `(canonical_id, confidence, alternates[])` rather than committing silently

### Skill claim

"I designed the entity-resolution layer for a B2B intelligence platform — fuzzy match, canonical IDs, tenant-scoped aliasing, confidence-banded outputs. The data-quality work that makes everything downstream actually work."

### Demo

```bash
# The fuzzy match is exposed two ways: as an MCP tool and via the dashboard API.
# Boot path A — MCP server (Claude Desktop / Cursor / any MCP client):
python -m atlas_brain.mcp.b2b_churn_server          # stdio
python -m atlas_brain.mcp.b2b_churn_server --sse    # SSE on port 8062

# Boot path B — REST via the dashboard router (atlas_brain/api/b2b_dashboard.py).
# After uvicorn is up:
curl -G "http://127.0.0.1:8001/api/v1/b2b/dashboard/vendors/fuzzy" \
     --data-urlencode "query=snowflake.com" | jq .
```

---

## Axis 4 — Enrichment Logic

**This is where the most code lives.** 34 `enrichment_*.py` modules in `atlas_brain/services/b2b/` (contracts, pure-function policies, runners, budgets, persistence), plus task-level orchestrators in `atlas_brain/autonomous/tasks/`.

### Where the value lives

#### Enrichment modules (composable, testable)

```
atlas_brain/services/b2b/enrichment_*.py  (34 modules)
├─ enrichment_contract.py             ← the entry-point Protocol
├─ enrichment_result_contract.py      ← the output dataclass
├─ enrichment_domain.py               ← shared coercers + build_classify_payload
├─ enrichment_buyer_authority.py      ← role-level + buyer-stage inference
├─ enrichment_pain_competition.py     ← pain categories + competitor mentions
├─ enrichment_phrase_metadata.py      ← phrase-level evidence atoms
├─ enrichment_derivation.py           ← composes derivation passes
├─ enrichment_urgency.py              ← timing / renewal / migration urgency
├─ enrichment_timeline.py             ← timeline anchor extraction
├─ enrichment_repair.py               ← retry / repair core
├─ enrichment_repair_policy.py        ← repair scheduling rules
├─ enrichment_validation.py           ← post-extraction validation
├─ enrichment_outcome_policy.py       ← terminal-state classifier
├─ enrichment_policy_pain.py          ← PAIN_PATTERNS regex catalog
├─ enrichment_policy_buyer_authority.py
├─ enrichment_policy_phrase_metadata.py
├─ enrichment_policy_validation.py
├─ enrichment_policy_low_fidelity.py
├─ enrichment_policy_repair.py
├─ enrichment_policy_timeline_budget.py
├─ enrichment_provider_calls.py       ← LLM call orchestration
├─ enrichment_budget.py               ← token/cost budget gate
├─ enrichment_notifications.py
├─ enrichment_row_runner.py           ← per-row execution
├─ enrichment_single_runner.py        ← single-review variant
├─ enrichment_stage_controller.py
├─ enrichment_stage_planner.py
├─ enrichment_stage_runs.py           ← multi-stage tracking
├─ enrichment_stage_support.py
├─ enrichment_task_runner.py          ← task-level coordination
├─ enrichment_task_ops.py
├─ enrichment_persistence.py
├─ enrichment_support.py
└─ enrichment_transport_support.py
```

#### Task-level orchestrators

| Task | File | Purpose |
|---|---|---|
| Deep enrichment | `atlas_brain/autonomous/tasks/deep_enrichment.py` | Heavy-weight pass over reviews / artifacts |
| B2B enrichment | `atlas_brain/autonomous/tasks/b2b_enrichment.py` | Vendor-pressure enrichment |
| Article enrichment | `atlas_brain/autonomous/tasks/article_enrichment.py` | News / content-page enrichment |
| Complaint enrichment | `atlas_brain/autonomous/tasks/complaint_enrichment.py` | Consumer-complaint extraction |
| Prospect enrichment | `atlas_brain/autonomous/tasks/prospect_enrichment.py` | Apollo/CRM contact enrichment |
| Vendor target enrichment | `atlas_brain/autonomous/tasks/vendor_target_enrichment.py` | Per-target metadata fetch |
| Repair pass | `atlas_brain/autonomous/tasks/b2b_enrichment_repair.py` | Retries failed enrichments |

### Reusable shape

The **policy / contract / runner / orchestrator** split is the reusable architecture:
- Each policy is a pure function with a typed input + typed result
- Contracts define what "successful enrichment" means per concern
- Runner composes policies in order, short-circuits on budget exhaustion
- Orchestrator schedules + persists + retries

### Skill claim

"I designed a composable enrichment-policy framework where each concern (pain, buyer authority, phrase metadata, domain signals, validation, repair) is a pure-function policy with typed inputs and outputs, executed by a budget-aware runner, with multi-stage tracking and repair passes." Translates to: **AI/ML platform engineer, data pipeline architect** roles.

### Demo

```bash
# Stage 1 — payload shaper: raw row -> structured LLM-ready payload (pure function)
python -c "
from atlas_brain.services.b2b.enrichment_domain import build_classify_payload, smart_truncate
import json

raw_row = {
    'id': 'r1', 'vendor_name': 'Snowflake',
    'product_name': 'Snowflake Data Cloud', 'product_category': 'cloud_data_warehouse',
    'source': 'capterra', 'content_type': 'review',
    'rating': 3.0, 'rating_max': 5,
    'summary': 'Pricing model surprised us',
    'review_text': 'Renewal came in 22% higher than the quote we signed.',
    'pros': 'fast queries', 'cons': 'cost, contract opacity',
    'reviewer_title': 'Director of Data Engineering',
    'reviewer_company': 'Acme Corp', 'company_size_raw': '1001-5000',
    'reviewer_industry': 'retail',
    'raw_metadata': {'source_weight': 0.9, 'source_type': 'review_site'},
}
print(json.dumps(
    build_classify_payload(raw_row, truncate_length=3000, smart_truncate=smart_truncate),
    indent=2,
))
"

# Stage 2 — pain-pattern catalog (pure regex policy module, no deps)
python -c "
from atlas_brain.services.b2b.enrichment_policy_pain import (
    PAIN_PATTERNS, KNOWN_PAIN_CATEGORIES, normalize_pain_category,
)
print('categories:', sorted(KNOWN_PAIN_CATEGORIES))
text = 'renewal price increase plus a billing dispute over the contract term'
for cat, pat in PAIN_PATTERNS.items():
    hits = pat.findall(text)
    if hits: print(f'  {cat:18s} -> {hits}')
"
```

The `build_classify_payload` helper is the production stage that maps raw scraped rows into the canonical LLM-input shape (with truncation, source-weight tagging, and field coercion); `enrichment_policy_pain.PAIN_PATTERNS` is the deterministic pain-category regex catalog the downstream `derive_pain_categories` policy walks. Both are pure functions you can demo without booting Postgres.

---

## Axis 5 — Quality Gates

**Already extracted as a standalone package.** The cleanest demo target in the codebase for "I built a deterministic policy engine."

### Where the value lives

| Component | File | What it does |
|---|---|---|
| Source-quality pack | `extracted_quality_gate/source_quality_pack.py` | Witness-render gate: PASS/WARN/BLOCK on row-level coverage |
| Safety gate | `extracted_quality_gate/safety_gate.py` | Safety-flag enforcement |
| Product-claim gate | `extracted_quality_gate/product_claim.py` | Suppression-reason policy for product claims |
| Witness-render gate (legacy entry) | `atlas_brain/services/b2b/witness_render_gate.py` | Thin re-export wrapper around the pack |
| Source-impact registry | `atlas_brain/services/b2b/source_impact.py` | Per-source coverage baselines |

### Reusable shape

The "**pack**" pattern is the gem here:
- A pack is a `(input_schema, policy, output_contract)` triple
- Policy is pure-function, deterministic
- Output: `QualityReport` with `decision: PASS|WARN|BLOCK` + `findings: list[Finding]`
- Composable: multiple packs can chain; aggregate decision is the strictest

This pattern transfers to any "validate this before letting it through" problem — content moderation, financial rules, compliance checks, schema validation.

### Skill claim

"I designed a composable policy-pack architecture for deterministic quality gates with explicit PASS/WARN/BLOCK decisions, row-level finding citations, and pluggable policy injection." Standalone-demoable in 5 lines of code.

### Demo

```bash
python -c "
from extracted_quality_gate.source_quality_pack import evaluate_source_quality
from extracted_quality_gate.types import QualityInput

# evaluate_source_quality reads input.context['witnesses'] and applies the
# witness-render gate to each row. A pain witness with positive polarity
# is suppressed; the report decision is WARN when at least one row is
# rendered AND at least one is suppressed.
report = evaluate_source_quality(QualityInput(
    artifact_type='b2b_review',
    context={'witnesses': [
        {'witness_id': 'w1', 'grounding_status': 'grounded',
         'phrase_subject': 'subject_vendor', 'phrase_role': 'primary_driver',
         'phrase_polarity': 'negative', 'witness_type': 'pain',
         'pain_confidence': 'strong'},
        {'witness_id': 'w2', 'grounding_status': 'grounded',
         'phrase_subject': 'subject_vendor', 'phrase_role': 'primary_driver',
         'phrase_polarity': 'positive',   # suppressed
         'witness_type': 'pain', 'pain_confidence': 'weak'},
    ]},
))
print('decision:', report.decision)
for f in report.findings:
    print(' -', f.code, f.severity, f.message)
"
# decision: WARN
#  - witness_suppressed:w2:polarity_not_renderable WARNING ...
```

---

## Axis 6 — Scoring / Aggregation

**The reasoning math.** Archetype scoring, temporal evidence (velocity / slope / anomaly / percentile baselines), tier-based pattern matching, displacement-edge scoring.

### Where the value lives

| Component | File | What it does |
|---|---|---|
| Archetype scorer | `extracted_reasoning_core/archetypes.py` | 10-archetype catalog + evidence-weighted scoring |
| Temporal evidence | `extracted_reasoning_core/temporal.py` | Velocity, long-term trend (30d/90d slope), volatility, z-score anomaly, category percentiles |
| Tiered pattern signals | `extracted_reasoning_core/tiers.py` | L1–L5 reasoning depth with per-tier signal building |
| Cross-vendor selection | `atlas_brain/reasoning/cross_vendor_selection.py` (extracted to `extracted_competitive_intelligence/reasoning/`) | Displacement-edge scoring + pair selection (graph-based) |
| Context aggregator | `atlas_brain/reasoning/context_aggregator.py` | Cross-section context build for synthesis |
| Pressure baselines | `atlas_brain/storage/migrations/080_b2b_alert_baselines.sql` + scoring code | Per-vendor pressure score baselines for spike detection |

### Reusable shape

The full scoring pipeline:
1. **Evidence input** — typed `EvidenceItem(source_type, source_id, text, metrics, metadata)`
2. **Temporal builder** — produces velocity / slope / anomaly / percentile features
3. **Archetype scorer** — matches features against a fixed catalog with explicit weights
4. **Tier classifier** — buckets into L1–L5 depth based on signal density
5. **Output** — `ArchetypeMatch(archetype_id, label, score, evidence_hits, missing_evidence, risk_label)`

This is **portable to any "score evidence against fixed patterns" problem** — fraud detection, anomaly detection, intent classification, churn prediction.

### Skill claim

"I designed and implemented the scoring layer for a vendor-displacement intelligence system: temporal-evidence builder (velocity / slope / anomaly / percentile), archetype-catalog scorer with explicit weights, tier-based depth classification, cross-vendor displacement-edge selection over a graph." Translates to: **applied ML, scoring/ranking engineer** roles.

### Demo

```bash
python -c "
from datetime import date, timedelta
from extracted_reasoning_core import score_archetypes, build_temporal_evidence

# Step 1: temporal features. build_temporal_evidence takes a sorted
# sequence of snapshot dicts (oldest-first) and returns velocities +
# long-term trends in-memory (no DB).
today = date.today()
snapshots = [
    {'vendor_name': 'Acme', 'snapshot_date': today - timedelta(days=i),
     'pricing_mentions': 12 - (i // 3), 'review_volume': 30 - i}
    for i in range(20, -1, -1)
]
temporal = build_temporal_evidence(snapshots)
print('snapshot_days:', temporal.snapshot_days,
      ' velocities:', len(temporal.velocities),
      ' trends:', len(temporal.trends))

# Step 2: archetype scoring against the catalog.
matches = score_archetypes(
    evidence={'pricing_mentions': 12, 'exec_change': True, 'review_volume': 30},
    limit=3,
)
for m in matches:
    print(m.archetype_id, round(m.score, 2), m.label, '|', m.risk_label)
"
```

---

## Axis 7 — UI / API Surfaces

**52 FastAPI router modules at the top level (plus nested `comms/`, `devices/`, `edge/`, `invoicing/`, `orchestrated/`, `query/` packages — 68 router files total) + 9 MCP servers + a React UI.** The demo-ready surface.

### Where the value lives

#### REST API (`atlas_brain/api/`, 52 top-level router modules + 6 nested router packages)

Selected highlights:

| Router | Purpose |
|---|---|
| `alerts.py` | Centralized alert API (list / ack / rules / test) |
| `b2b_dashboard.py`, `b2b_tenant_dashboard.py` | B2B intelligence dashboard |
| `b2b_evidence.py`, `b2b_reviews.py`, `b2b_vendor_briefing.py` | Evidence + reviews + briefing artifacts |
| `b2b_campaigns.py`, `seller_campaigns.py` | Campaign generation API |
| `llm_gateway.py`, `llm.py`, `ollama_compat.py`, `openai_compat.py` | LLM Gateway customer surface — OpenAI-compatible + Ollama-compatible endpoints |
| `auth.py`, `api_keys.py`, `billing.py`, `admin_costs.py` | Multi-tenant auth + per-account billing + cost admin |
| `vision.py`, `audio_events`, `voice` | Vision + audio + voice surfaces |
| `universal_scrape.py`, `b2b_scrape.py` | Scraping API (admin + tenant) |

#### MCP servers (`atlas_brain/mcp/`, 9 server modules, 130+ tools)

| Server | Tools | Port |
|---|---|---|
| CRM | 10 | 8056 |
| Email | 8 | 8057 |
| Twilio | 10 | 8058 |
| Calendar | 8 | 8059 |
| Invoicing | 15 | 8060 |
| Intelligence | 17 | 8061 |
| **B2B Churn** | **60+** | 8062 |
| Scraper | (admin) | — |
| Memory | (admin) | — |

Both stdio (Claude Desktop / Cursor) and SSE (HTTP + bearer auth) transports.

#### React UI (`atlas-ui/`)

Separate Next.js / React app for tenant-facing dashboards (vendor intelligence watchlists, accounts-in-motion feed, evidence drawers, …).

### Reusable shape

The pattern that's portable:
- FastAPI routers as the **edge surface**
- MCP servers as the **agent-tool surface**
- Both share the same service layer underneath (`atlas_brain/services/`)
- Both gate on the same auth / plan-tier / per-account scoping middleware

This is the right architecture for a multi-channel AI product (web app + agent integration + 3rd-party API).

### Skill claim

"I designed and implemented the multi-channel surface — 52 top-level FastAPI router modules (68 total including nested router packages) for the customer REST + admin API, 9 MCP servers exposing 130+ tools to agent clients, and the shared auth + plan-tier + per-account scoping that gates both. Plus a React/Next.js dashboard." Translates to: **full-stack, API platform, integration** roles.

### Demo

```bash
# Boot the stack
docker compose up -d postgres
uvicorn atlas_brain.main:app --port 8001 &

# Hit a route
curl http://127.0.0.1:8001/api/v1/health | jq .
curl http://127.0.0.1:8001/api/v1/alerts/stats | jq .

# Boot an MCP server
python -m atlas_brain.mcp.crm_server --sse  # port 8056

# Connect Claude Desktop / Cursor — tools appear in the UI
```

---

## Axis 8 — Demo-Ready Outputs

**The artifacts a recruiter can actually run end-to-end.**

### Where the value lives

| Output | File | What you get |
|---|---|---|
| Campaign generation runbook | `extracted_content_pipeline/docs/host_install_runbook.md` | 5-command end-to-end: migrate → import → generate → review → export |
| Campaign example | `extracted_content_pipeline/campaign_example.py` (+ `examples/`) | Importable Python example flow |
| Postgres-backed CLI runners | `scripts/run_extracted_campaign_generation_postgres.py`, `run_extracted_campaign_generation_example.py` | Real LLM call, real DB, real output |
| Standalone smoke scripts | `scripts/smoke_extracted_*_standalone.py` (×3 packages) | Proves zero-atlas-import boot |
| Hybrid-reasoning checks driver | `scripts/run_hybrid_reasoning_checks.sh` (#198) + JSON-report variant (#200) | Composable test-runner with structured output |
| Voice pipeline | `atlas_brain/voice/launcher.py` | "Hey Atlas" → ASR → LLM → TTS roundtrip |
| Alert system test trigger | `POST /api/v1/alerts/test` | Synthetic event → TTS + ntfy + DB row |
| Test suite as living documentation | `tests/test_extracted_*` | Each test is an executable usage example |

### Skill claim

"Every package has a smoke script + a runnable example + integration tests. The campaign-generation runbook is a 5-command end-to-end demo on a clean Postgres + Python box."

### The single best demo to record

If you make **one screen recording** for portfolio use, make it this one (real CLI flags from `extracted_content_pipeline/docs/host_install_runbook.md`):

```bash
# Clean Postgres box. ~3 minutes wall-clock. Real LLM call.
export DATABASE_URL=postgres://...

# 1. Apply the packaged extracted_content_pipeline migrations (5s)
python scripts/run_extracted_content_pipeline_migrations.py --database-url $DATABASE_URL

# 2. Import seed opportunities (CSV or JSON) (5s)
python scripts/load_extracted_campaign_opportunities.py customer_opportunities.csv \
    --format csv --account-id acct_demo --database-url $DATABASE_URL

# 3. Generate campaigns — real LLM call via extracted_llm_infrastructure (60-120s)
python scripts/run_extracted_campaign_generation_postgres.py \
    --account-id acct_demo --limit 5 --database-url $DATABASE_URL

# 4. Show the generated drafts (5s)
psql $DATABASE_URL -c "select id, vendor_name, channel, subject, body
                       from b2b_campaigns order by created_at desc limit 3;"

# 5. Show the cost ledger entry (5s)
psql $DATABASE_URL -c "select model_name, model_provider,
                              input_tokens, output_tokens, cost_usd
                       from llm_usage order by created_at desc limit 3;"
```

End-to-end proof of: schema design + ingestion + LLM orchestration + cost tracking + persistence — in 3 minutes.

---

## Recruiter persona index

| Persona | Lead with |
|---|---|
| **Backend / distributed systems** | Axis 1 (extraction contracts) + Axis 4 (enrichment policy framework) + Axis 7 (API surface, 52 top-level routers) |
| **AI / ML platform** | Axis 6 (scoring/aggregation) + Axis 5 (quality gates) + Axis 1's M5-α typed envelope |
| **Applied AI / agent / LLM** | Axis 7 (130+ MCP tools, OpenAI-compatible API) + voice pipeline + LLM Gateway |
| **Data engineering / scraping** | Axis 2 (scraping stack — 23 top-level modules + parsers/, 19 sources, captcha+proxy+rate-limit) + Axis 3 (entity resolution) |
| **Security / infra** | WiFi threat detection (`atlas_brain/security/wireless/`) + Axis 7 auth/api_keys/BYOK |
| **Staff / principal** | Axis 1 (extraction program, M1–M6) + the audit docs (`docs/extraction/*audit*.md`) — proves architectural-program execution at scale |

---

## Demo environment minimum

Single-box, no-GPU demo (covers Axes 1, 3, 4, 5, 6, 7, 8):
- Linux / macOS / WSL2, 16GB RAM, ~50GB disk
- Postgres 15+ (Docker fine)
- Python 3.11
- Optional: Anthropic API key for real LLM calls (else use Ollama with `qwen3:14b`)

GPU box (adds voice + vision):
- NVIDIA GPU with 24GB+ VRAM
- CUDA toolkit
- Mic + speakers

---

## Honest framing — what this portfolio is NOT

- **Not a single shipped commercial product.** Atlas is in active development; pieces work, pieces are scaffolds.
- **Not full standalone for every package.** `extracted_competitive_intelligence` is 32%; the reasoning *producer* is intentionally host-owned; voice pipeline is atlas-internal.
- **Not "I built this alone".** The codebase represents months of architectural work — recruiters will spot it, so be transparent about scope.

What it **is**: proof you can design + build + operate large-scale AI systems with explicit architectural boundaries and verifiable extraction contracts.

---

## Suggested next steps

1. **Pick the demo that maps to the role you're interviewing for.** Use the Recruiter Persona Index above.
2. **Record a 2-minute screen capture** of that demo running on a clean box. Upload to a private link.
3. **In the application**: drop the link + a 3-bullet skill claim from the matching axis. Don't dump the whole portfolio.
4. **For staff/principal interviews**: bring a printout of the cross-product audit (`docs/extraction/cross_product_audit_2026-05-04.md`) and the M5-α design doc (`extracted_reasoning_core/domains.py`'s docstring + the M6 compatibility matrix). They're the strongest architectural-judgment artifacts.
