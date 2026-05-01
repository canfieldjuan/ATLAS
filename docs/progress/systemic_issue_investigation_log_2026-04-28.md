# Systemic Issue Investigation Log - 2026-04-28

Scope: non-scraping Churn Signals / B2B product systems. Scraping UI and scraping pipeline work are explicitly out of scope for this audit because they are being worked separately.

Purpose: track places where recent bugs or patches look like symptoms of larger architectural drift rather than isolated defects. Each item should be investigated and fixed independently.

## Root Patterns

1. **Tenant boundary drift** - older global/internal routers remain mounted under `/api/v1` while newer tenant surfaces enforce auth/plan scope.
2. **UI-only safety gates** - React suppresses unsafe rows, but backend actions, exports, or automation can still act on the same unvalidated objects.
3. **Report/export-first leftovers** - downstream packaging surfaces still render legacy scored/synthesized fields instead of validated ProductClaim/EvidenceClaim envelopes.
4. **Legacy fallback ambiguity** - fallback behavior is useful for rollout, but absent validation payloads can mean either old cached data or a new backend regression.
5. **Lineage approximation** - several ProductClaim directness checks still use row-level grounding until Phase 9 lineage coverage is steady-state.
6. **Autonomous liveness opacity** - scheduled tasks can run on time while their domain-specific due queues are empty, stale, or stuck, producing clean task executions but no product work.
7. **Schema contract drift** - safety-critical gate fields are sometimes enforced by Pydantic response models and sometimes only implied by plain dicts plus optional TypeScript fields.

## Investigation Queue

| ID | Priority | Area | Systemic concern | Evidence | Next investigation |
| --- | --- | --- | --- | --- | --- |
| SYS-001 | High | B2B dashboard app auth | Legacy `/api/v1/b2b/dashboard` paths are hard-removed by middleware, and every legacy endpoint still registered as a `/api/v1/b2b/tenant` alias now requires `b2b_growth` before DB touch. This closes the optional-auth alias leak for company/opportunity queues, company-signal review actions, data corrections, reasoning/calibration, report PDF export, webhook integration controls, vendor/product intel reads, source/ops diagnostics, and search helpers. | `atlas_brain/main.py:740`, `atlas_brain/api/b2b_tenant_dashboard.py:4818`, `atlas_brain/api/b2b_dashboard.py:1933`, `atlas_brain/api/b2b_dashboard.py:3019`, `atlas_brain/api/b2b_dashboard.py:3609`, `atlas_brain/api/b2b_dashboard.py:3934`, `atlas_brain/api/b2b_dashboard.py:3990`, `atlas_brain/api/b2b_dashboard.py:4720`, `atlas_brain/api/b2b_dashboard.py:5546`, `atlas_brain/api/b2b_dashboard.py:6182`, `tests/test_b2b_access_boundaries.py:93`, `tests/test_b2b_access_boundaries.py:121`, `tests/test_b2b_access_boundaries.py:142`, `tests/test_b2b_access_boundaries.py:174`, `tests/test_b2b_access_boundaries.py:195` | Closed for tenant aliases. Remaining auth policy work moves to SYS-017: classify explicit tenant routes that already require authentication but may need stricter B2B plan tier/admin-role enforcement. |
| SYS-002 | High | Vendor target management | Vendor target list/create/read/update/claim/delete/report-generation now require auth, apply account-aware scoping, and are pinned in the access-boundary harness before DB touch. Legacy global reads remain explicit through `include_legacy_global`. | `atlas_brain/api/vendor_targets.py:167`, `atlas_brain/api/vendor_targets.py:242`, `atlas_brain/api/vendor_targets.py:400`, `atlas_brain/api/vendor_targets.py:626`, `tests/test_b2b_access_boundaries.py:34` | Keep legacy-global access explicit. If vendor targets become a paid tenant product rather than universal authenticated tooling, upgrade these routes from `require_auth` to `require_b2b_plan(...)` under SYS-017. |
| SYS-003 | High | CRM event ingestion and visibility | Generic CRM single/batch ingestion plus CRM event list/stats are now browser/operator APIs: they require auth, scope writes/reads by `account_id`, and are pinned in the access-boundary harness before DB touch. Native provider webhook endpoints still use session-style optional auth plus a manual user check. | `atlas_brain/api/b2b_crm_events.py:126`, `atlas_brain/api/b2b_crm_events.py:258`, `atlas_brain/api/b2b_crm_events.py:861`, `atlas_brain/api/b2b_crm_events.py:982`, `tests/test_b2b_access_boundaries.py:68`, `tests/test_b2b_crm_events.py:63` | Remaining work is native webhook identity: split HubSpot/Salesforce/Pipedrive ingestion behind signed provider secrets or account-scoped API keys so browser session auth is not required for real provider webhooks. |
| SYS-004 | High | Campaign generation action boundary | Opportunities UI gates `report_allowed`; backend fetchers now filter raw review and accounts-in-motion candidates on ACCOUNT ProductClaim `report_allowed` before generation, persisted campaign metadata carries the claim gate payload, detached replay storage fails closed when ProductClaim context is missing, API approve/queue/bulk actions require report-safe claim metadata before status changes, and campaign CSV export exposes claim gate columns. | `atlas-churn-ui/src/pages/Opportunities.tsx:601`, `atlas_brain/api/b2b_campaigns.py:1946`, `atlas_brain/autonomous/tasks/b2b_campaign_generation.py:3691`, `atlas_brain/autonomous/tasks/b2b_campaign_generation.py:4999` | Watch for any new campaign entrypoints that create/advance campaigns without `opportunity_claim` or `opportunity_claims` metadata. |
| SYS-021 | High | Campaign management access boundary | Campaign analytics, suppression mutations/list/check, review queues/candidates, and sequence controls lived in `b2b_campaigns.py` without a product-plan dependency, so SaaS auth enabled did not protect global funnel metrics, global suppression state, draft review queues, candidate summaries, or sequence state. | `atlas_brain/api/b2b_campaigns.py:162`, `atlas_brain/api/b2b_campaigns.py:933`, `atlas_brain/api/b2b_campaigns.py:1004`, `atlas_brain/api/b2b_campaigns.py:1058`, `atlas_brain/api/b2b_campaigns.py:1111`, `atlas_brain/api/b2b_campaigns.py:1321`, `atlas_brain/api/b2b_campaigns.py:1411`, `atlas_brain/api/b2b_campaigns.py:1460`, `atlas_brain/api/b2b_campaigns.py:1661`, `atlas_brain/api/b2b_campaigns.py:1786`, `atlas_brain/api/b2b_campaigns.py:1851`, `atlas_brain/api/b2b_campaigns.py:1895`, `atlas_brain/api/b2b_campaigns.py:1918`, `atlas_brain/autonomous/tasks/b2b_campaign_generation.py:1543`, `atlas_brain/autonomous/tasks/b2b_campaign_generation.py:3841`, `tests/test_b2b_access_boundaries.py:307`, `tests/test_b2b_access_boundaries.py:522`, `tests/test_b2b_campaigns_api.py:18`, `tests/test_b2b_campaigns_api.py:78`, `tests/test_b2b_campaign_generation.py:1395` | Campaign analytics now require `b2b_growth` and scope to the caller's tracked vendors; suppression mutations/list/check are plan-gated; review queues/candidates require `b2b_growth` and scope SQL/helper candidate selection to tracked vendors; sequence reads/writes require `b2b_growth` and enforce tracked-vendor ownership before returning data or updating sequence state. Remaining campaign reads (campaign list/stats/export optional-auth paths and campaign audit-log) still need plan and tenant-scope classification; the `campaign_suppressions` table remains a global blocklist by design, so future work should decide whether plan-gated global visibility is enough or account-scoped suppression state is required. |
| SYS-005 | High | Vendor briefing public/admin endpoints | Operator briefing controls now require auth: preview/generate/send-batch/list/review/export routes are pinned in the boundary harness, while gate/checkout/checkout-session/report-data remain intentionally public token/payment flows. | `atlas_brain/api/b2b_vendor_briefing.py:255`, `atlas_brain/api/b2b_vendor_briefing.py:279`, `atlas_brain/api/b2b_vendor_briefing.py:702`, `atlas_brain/api/b2b_vendor_briefing.py:716`, `tests/test_b2b_access_boundaries.py:34` | If public briefing flow expands, split public/operator routers so auth inheritance is obvious. Separately, audit public `report-data` packaging under SYS-013 before expanding public access. |
| SYS-006 | High | Admin cost/system API | Admin cost/system telemetry now has router-level admin-role enforcement, not just authentication. Anonymous requests still reject before DB touch, and authenticated member users now receive 403 before DB touch. | `atlas_brain/api/admin_costs.py:38`, `atlas_brain/api/admin_costs.py:41`, `atlas_brain/api/admin_costs.py:663`, `tests/test_b2b_access_boundaries.py:34`, `tests/test_admin_costs.py:969` | Closed for role boundary. If this surface becomes platform-operator-only rather than account-admin-visible, add a distinct platform-admin claim; current `AuthUser.is_admin` intentionally collapses owner/admin roles with the stored admin flag. Keep scraping-summary fixture failures tracked outside this non-scraping audit. |
| SYS-007 | High | Amazon Seller campaign API | Amazon Seller target CRUD, campaign generation, campaign listing, intelligence listing, and refresh now inherit router-level `require_auth`; the access-boundary harness pins representative target and generation routes. | `atlas_brain/api/seller_campaigns.py:26`, `atlas_brain/api/seller_campaigns.py:29`, `atlas_brain/api/seller_campaigns.py:133`, `atlas_brain/api/seller_campaigns.py:319`, `tests/test_b2b_access_boundaries.py:34` | Classify Seller as internal tooling vs tenant product. If it becomes a paid product surface, upgrade from `require_auth` to the appropriate `require_b2b_plan(...)` gate under SYS-017. |
| SYS-008 | High | High-intent export | CSV export now shapes rows through the same `opportunity_claim` helper as the UI, defaults to report-safe rows only, and exposes `report_safe_only=false` for explicit raw/audit exports with readiness columns. | `atlas_brain/api/b2b_dashboard.py:6364` | Add tenant-route integration assertion when that harness is next touched; otherwise monitor for downstream CSV consumers that call raw export and ignore readiness columns. |
| SYS-009 | High | Generic report renderers | Generic renderer dangerous sections now fail closed for cross-vendor battles, objection handlers, recommended plays, talk tracks, and weakness analysis at both top-level and nested `MixedObjectCard` paths unless every rendered item carries report-safe ProductClaim context. Legacy/unvalidated sections show a gate fallback instead of winner/play/recommendation language. | `atlas-churn-ui/src/components/report-renderers/StructuredReportData.tsx:62`, `atlas-churn-ui/src/components/report-renderers/StructuredReportData.tsx:638`, `atlas-churn-ui/src/components/report-renderers/StructuredReportData.tsx:670`, `atlas-churn-ui/src/components/report-renderers/StructuredReportData.tsx:745`, `atlas-churn-ui/src/components/report-renderers/StructuredReportData.test.tsx:204` | Continue with specialized-renderer coverage in SYS-018 and any future generic dangerous keys discovered in report payloads. |
| SYS-018 | High | Specialized report renderer residuals | Patch 6 gated head-to-head and battle-card displacement reasoning. Specialized challenger/battle/weekly/displacement report surfaces now fail closed for weakness rows, playbook talk tracks/recommended plays, category-council winner language, cross-vendor battles, top displacement targets, market leaderboards, and top battles unless each row/block carries report-safe ProductClaim context. The named specialized-renderer residuals from this sweep are closed. | `atlas-churn-ui/src/components/report-renderers/SpecializedReportData.tsx:462`, `atlas-churn-ui/src/components/report-renderers/SpecializedReportData.tsx:582`, `atlas-churn-ui/src/components/report-renderers/SpecializedReportData.tsx:800`, `atlas-churn-ui/src/components/report-renderers/SpecializedReportData.tsx:1306`, `atlas-churn-ui/src/components/report-renderers/SpecializedReportData.tsx:1421`, `atlas-churn-ui/src/components/report-renderers/SpecializedReportData.tsx:2042`, `atlas-churn-ui/src/components/report-renderers/SpecializedReportData.tsx:2812`, `atlas-churn-ui/src/components/report-renderers/SpecializedReportData.tsx:2863` | Continue monitoring new specialized report fields. Any future report section with winner/play/recommendation/displacement semantics should require ProductClaim gate context or render as legacy/unvalidated. |
| SYS-010 | Medium | Vendor Workspace claim coverage | DM churn and price complaint cards are ProductClaim-gated. Uncontracted Vendor Workspace fields (NPS proxy, churn intent count, slow-burn metrics, reasoning intelligence, and reasoning highlights) now render with visible `Legacy/unvalidated` labels instead of looking claim-validated. | `atlas-churn-ui/src/pages/VendorDetail.tsx:1225`, `atlas-churn-ui/src/pages/VendorDetail.tsx:1257`, `atlas-churn-ui/src/pages/VendorDetail.tsx:1268`, `atlas-churn-ui/src/pages/VendorDetail.tsx:1388`, `atlas-churn-ui/src/pages/VendorDetail.tsx:1562`, `atlas_brain/api/b2b_vendor_claims.py:189` | Keep these labels until real VENDOR-scope ProductClaim aggregators exist for the corresponding fields; when added, replace the labels with ProductClaim gates rather than silent raw renders. |
| SYS-011 | Medium | ProductClaim policy registry | Production ProductClaim builders can now require a registered `(scope, claim_type)` policy. Missing policy raises instead of silently falling back to permissive defaults, and current VENDOR/ACCOUNT/COMPETITOR_PAIR aggregators use the strict path. | `atlas_brain/services/b2b/product_claim.py:134`, `atlas_brain/services/b2b/product_claim.py:152`, `atlas_brain/services/b2b/product_claim.py:448`, `atlas_brain/services/b2b/vendor_dashboard_claims.py:236`, `atlas_brain/services/b2b/account_opportunity_claims.py:183`, `atlas_brain/services/b2b/challenger_dashboard_claims.py:425`, `tests/test_product_claim_contract.py:754` | Keep explicit policy overrides only for tests/audit scenarios. Future customer-facing aggregators should set `require_registered_policy=True` or resolve via `get_registered_policy()` before building claims. |
| SYS-012 | Medium | Direct evidence lineage | `use_claim_lineage_for_direct_evidence` is still default-off, leaving row-level grounding as a directness approximation until the Phase 9 soak is complete. | `atlas_brain/services/b2b/product_claim.py:116`, `atlas_brain/services/b2b/vendor_dashboard_claims.py:236`, `atlas_brain/services/b2b/vendor_dashboard_claims.py:411`, `docs/progress/product_claim_contract_plan_2026-04-26.md:535` | Re-run after the May 5 soak verification. If pass criteria hold, canary-flip claim lineage and resolve the v2.5/v3 DM-churn directness path. |
| SYS-013 | Medium | Public report-data package | Public briefing `report-data` now strips dangerous report/profile sections from the API payload unless they carry report-safe ProductClaim context. Cached briefing data and intelligence reports drop unsafe battles/plays/weakness-style sections; public product profiles keep summary/category but withhold unvalidated strengths/weaknesses/comparison lists. | `atlas_brain/api/b2b_vendor_briefing.py:227`, `atlas_brain/api/b2b_vendor_briefing.py:725`, `atlas_brain/api/b2b_vendor_briefing.py:759`, `atlas_brain/api/b2b_vendor_briefing.py:791`, `tests/test_b2b_vendor_briefing.py:240` | Keep public API packaging fail-closed. If more public report sections are added, either include report-safe ProductClaim context or add the section key to the public dangerous-key strip list. |
| SYS-014 | Low/Medium | Evidence lazy-on-read fallback | Evidence detail GET can compute and persist grounding status as a migration fallback, and the fallback path is now pinned to emit `lazy_grounding_fallback_hit` whenever it updates a pending witness. | `atlas_brain/api/b2b_evidence.py:382`, `atlas_brain/api/b2b_evidence.py:397`, `atlas_brain/api/b2b_evidence.py:428`, `tests/test_truthful_artifact_routes.py:663` | Monitor `lazy_grounding_fallback_hit`; steady-state should be zero. Any recurring hits mean the write-path classifier or backfill is missing rows. |
| SYS-015 | Low/Medium | ACCOUNT source-review count coupling | `source_review_count` is now computed explicitly from row source-review ids / quote evidence and passed into ACCOUNT ProductClaim serialization instead of being inferred from `claim.sample_size`. | `atlas_brain/services/b2b/account_opportunity_claims.py:97`, `atlas_brain/services/b2b/account_opportunity_claims.py:105`, `atlas_brain/services/b2b/account_opportunity_claims.py:171`, `atlas_brain/autonomous/tasks/b2b_campaign_generation.py:3977`, `tests/test_account_opportunity_claims.py:36` | Keep `sample_size` and `source_review_count` semantically separate when ACCOUNT denominator semantics evolve. |
| SYS-016 | Medium | API prefix fallback | The frontend no longer retries `/api/v1/...` requests against `/api/...` on 404. Shared API and auth clients now fail on the production-effective mount path, with tests pinning no hidden fallback retry. | `atlas-churn-ui/src/api/client.ts:151`, `atlas-churn-ui/src/api/client.test.ts:112`, `atlas-churn-ui/src/auth/AuthContext.tsx:45`, `atlas-churn-ui/src/auth/AuthContext.test.tsx:167` | If a legacy non-v1 route is still needed, add an explicit per-route client path or a narrow allowlist with tests; do not restore a global fallback. |
| SYS-017 | Medium | B2B product/plan gate split | Closed for explicit tenant dashboard routes: the `/b2b/tenant` router now has zero local routes still depending directly on `require_auth`; every paid B2B tenant product route uses `require_b2b_plan("b2b_growth")` before repo/DB/service touch. This covers product controls, report generation/delivery, exports, CRM push, dispositions, and read/analytics pages. | `atlas_brain/api/b2b_tenant_dashboard.py:1177`, `atlas_brain/api/b2b_tenant_dashboard.py:1767`, `atlas_brain/api/b2b_tenant_dashboard.py:3513`, `atlas_brain/api/b2b_tenant_dashboard.py:4384`, `atlas_brain/api/b2b_tenant_dashboard.py:4659`, `tests/test_b2b_access_boundaries.py:160`, `tests/test_b2b_access_boundaries.py:376` | Closed. Future explicit tenant routes should use `require_b2b_plan(...)` by default unless they are intentionally universal account/profile routes, and the access-boundary matrix should pin both anonymous 401 and underplan 403 before DB touch. |
| SYS-019 | Medium | Autonomous competitive-set liveness | `b2b_reasoning_synthesis` no longer returns a bare `No due competitive sets` skip. The competitive-set repository now exposes due-queue health with active/scheduled/missing-interval/running/stale-running/due/not-due counts and the scheduled task attaches that payload to skip results. | `atlas_brain/storage/repositories/competitive_set.py:340`, `atlas_brain/autonomous/tasks/b2b_reasoning_synthesis.py:1262`, `atlas_brain/autonomous/tasks/b2b_reasoning_synthesis.py:1266`, `tests/test_b2b_competitive_sets.py:906`, `tests/test_b2b_competitive_sets.py:1121` | Next step is policy, not visibility: decide whether stale `last_run_status='running'` rows should be auto-reset after the reported threshold or only alerted for operator cleanup. |
| SYS-020 | Medium | Safety envelope schema contracts | ProductClaim claim endpoints, evidence witness list/detail, and high-intent opportunity rows now use Pydantic response models that require safety-bearing gate fields. Missing `render_allowed` / `report_allowed` fails at the API schema boundary instead of relying on React to fail closed. | `atlas_brain/api/b2b_evidence.py:173`, `atlas_brain/api/b2b_evidence.py:244`, `atlas_brain/api/b2b_dashboard.py:151`, `atlas_brain/api/b2b_dashboard.py:1891`, `tests/test_truthful_artifact_routes.py:346`, `tests/test_b2b_dashboard_accounts_in_motion.py:2587` | Keep future safety-bearing response shapes behind strict API models. Frontend optionality is allowed only at the row wrapper level for explicit legacy/fail-closed rollout semantics. |

## Access Boundary Deep Dive - 2026-04-28

The boundary problem is broader than anonymous access. There are four separate route classes currently blended together:

| Route class | Intended behavior | Current risk |
| --- | --- | --- |
| Public sales flow | No session auth, but protected by gate token, checkout session, signed webhook secret, or rate limit. | Mixed into operator routers, so public intent is hard to distinguish from accidental unauthenticated access. |
| Tenant app read/write | Requires authenticated B2B user and account scope. | Many older routes use `optional_auth`; unauthenticated means global/unscoped instead of denied. |
| Operator/admin tooling | Requires admin or internal network. | Some admin/seller APIs have no auth dependency while mounted under `/api/v1`. |
| Webhooks | Requires provider signature or account-scoped API key, not browser session auth. | Generic CRM ingestion accepts unauthenticated account-null events; native endpoints use optional auth plus manual `if not user` checks. |

### Concrete Route-Level Findings

1. `vendor_targets` should be treated as tenant-owned app state, not global mutable state. Unauthenticated create/update/delete/report-generation is the highest-risk direct mutation path in this pass.
2. `b2b_crm_events` needs a webhook-auth design. Browser/session auth is wrong for provider webhooks, but unauthenticated account-null writes are also wrong.
3. `b2b_vendor_briefing` needs router separation. `/gate`, `/checkout`, `/checkout-session`, and `/report-data` are intentionally public/token/payment flows; `/preview`, `/generate`, `/send-batch`, and list/export/review controls are operator flows.
4. `admin_costs` should be assumed internal-only until proven otherwise. It exposes cost, run, error, task, model, and system-resource telemetry.
5. `seller_campaigns` has no auth boundary. If this is a dormant/internal product, it should still be guarded before more campaign/outreach work builds on it.
6. `b2b_tenant_dashboard` is better than old routers on anonymous auth, but still needs a product/tier audit because most routes use `require_auth`.
7. Frontend route protection is not a backend control. `ProtectedRoute` and `usePlanGate` improve UX but do not protect direct API calls.

### Test Surface Notes

- `tests/test_b2b_vendor_claims_api_live.py` and `tests/test_b2b_challenger_claims_api_live.py` already pin the desired production-effective auth pattern: mount the aggregate API router under `/api/v1` and assert unauthenticated requests get 401.
- `tests/test_pipeline_visibility_api.py` has a simple `TestClient` unauthorized-route pattern that can be copied for route-boundary tests.
- `tests/test_b2b_crm_events.py`, `tests/test_seller_campaigns.py`, and `tests/test_admin_costs.py` currently exercise these routers without auth requirements, so boundary changes will need deliberate test updates rather than incidental breakage.
- Frontend tests for `ProtectedRoute` / `usePlanGate` are useful UX guards, but they should not be counted as API authorization coverage.

### Recommended First Fix Slice

Patch A should be a route-boundary test harness before behavior changes:

1. Add a focused `tests/test_b2b_access_boundaries.py`.
2. Mount `atlas_brain.api.router` under `/api/v1` so tests exercise production-effective paths.
3. Assert unauthenticated requests to tenant/operator endpoints return 401 for:
   - `/api/v1/b2b/vendor-targets`
   - `/api/v1/b2b/vendor-targets/{id}`
   - `/api/v1/b2b/briefings/preview`
   - `/api/v1/b2b/briefings/generate`
   - `/api/v1/b2b/briefings/send-batch`
   - `/api/v1/admin/costs/summary`
   - `/api/v1/seller/targets`
   - `/api/v1/seller/campaigns/generate`
4. Keep explicitly public routes out of the failing set:
   - `/api/v1/b2b/briefings/gate`
   - `/api/v1/b2b/briefings/checkout`
   - `/api/v1/b2b/briefings/checkout-session`
   - `/api/v1/b2b/briefings/report-data`
5. Mark CRM ingestion separately because it needs webhook/API-key semantics, not a simple session-auth conversion.

Patch B should then convert the smallest high-risk router. Best candidate: `seller_campaigns`, because it has no public-token flows to preserve and no tenant-global legacy semantics to untangle. Second candidate: `vendor_targets`, because it is product-critical but needs a legacy-global claim migration policy.

### Implementation Update - Access Boundary Patch A/B1

Implemented the first boundary slice:

1. Added `tests/test_b2b_access_boundaries.py`, mounting the aggregate API router under `/api/v1` and asserting unauthenticated requests return 401 before data/service access. The harness monkeypatches DB pool access to raise, so it pins auth-before-DB ordering, not just final response status.
2. Converted vendor-target CRUD and report-generation endpoints from `optional_auth` to `require_auth`.
3. Converted briefing operator endpoints (`preview`, `generate`, `send-batch`, list) from `optional_auth` to `require_auth`.
4. Added router-level `require_auth` to `admin_costs`.
5. Added router-level `require_auth` to `seller_campaigns`.
6. Left public briefing gate/checkout/report-data routes unauthenticated by design.

Verification:

- `pytest tests/test_b2b_access_boundaries.py` -> 20 passed.
- `pytest tests/test_b2b_access_boundaries.py tests/test_admin_costs.py -q -k 'not scraping'` -> 30 passed, 3 deselected.
- `pytest tests/test_b2b_vendor_briefing.py::test_generate_briefing_trims_vendor_name_and_blank_email tests/test_b2b_vendor_briefing.py::test_briefing_gate_rejects_blank_email_before_db_touch tests/test_b2b_vendor_briefing.py::test_vendor_checkout_trims_customer_email_before_stripe tests/test_seller_campaigns.py -q` -> 10 passed.
- `python -m py_compile` on touched API/test files passed.

Known unrelated verification note: full `tests/test_admin_costs.py` currently has two scraping-summary fixture failures around `saved_calls_today`; scraping is out of scope for this audit and those failures were not introduced by the auth-boundary change.

## Action / Export Inheritance Deep Dive - 2026-04-28

The opportunity surface has the same root pattern Patch 6 fixed for reports: the UI consumes a validated envelope, while adjacent non-UI surfaces still consume the legacy row.

### High-Intent Row Paths

| Surface | Data path | ProductClaim status |
| --- | --- | --- |
| UI list | `list_high_intent` -> `read_high_intent_companies` -> `_shape_high_intent_company_payload` -> `attach_account_opportunity_claim` | Gated. |
| Tenant lead detail / dashboard widgets | Calls `read_high_intent_companies`, then local shaping varies by endpoint | Needs per-endpoint audit. |
| CSV export | `export_high_intent` -> `read_high_intent_companies` -> `_shape_high_intent_company_payload` -> readiness CSV columns | Gated metadata exported. |
| Tenant CSV export | `export_tenant_high_intent` -> legacy `export_high_intent` | Inherits fixed export behavior. |
| Campaign generation | `generate_campaigns` -> `_fetch_opportunities` / `_fetch_accounts_in_motion_opportunities` -> ACCOUNT ProductClaim -> `report_allowed` filter | Action-gated before LLM generation. |

### Concrete Findings

1. `list_high_intent` is the correct reference path because it attaches `opportunity_claim` before returning rows.
2. `export_high_intent` previously bypassed the shaping function and serialized `company`, `vendor`, `urgency`, `pain`, `buying_stage`, and related fields directly.
3. `export_tenant_high_intent` enforces B2B product access before delegating; it inherits the fixed legacy export function.
4. Campaign generation uses two raw candidate sources:
   - `_fetch_opportunities` from `read_campaign_opportunities`.
   - `_fetch_accounts_in_motion_opportunities` from persisted `accounts_in_motion` report rows.
5. Both campaign candidate sources now attach `opportunity_claim` and filter on `report_allowed`.
6. `force` currently bypasses dedup/briefing gates. It must not become a bypass for ProductClaim safety; safety gates should be non-overridable unless a separate internal-only override is created and audited.

### Implementation Status

First action/export inheritance slice is patched:

1. `export_high_intent` now shapes every row through `_shape_high_intent_company_payload`, defaults to `report_safe_only=true`, and includes `opportunity_claim_id`, render/report booleans, confidence, evidence posture, suppression reason, supporting count, direct evidence count, and source review count when `report_safe_only=false` is explicitly requested.
2. Tenant export inherits the same behavior and forwards the `report_safe_only` flag because it delegates to `export_high_intent`.
3. `_fetch_opportunities` maps campaign rows into the ACCOUNT ProductClaim shape, attaches `opportunity_claim`, and skips candidates where `report_allowed` is false.
4. `_fetch_accounts_in_motion_opportunities` applies the same non-overridable `report_allowed` filter to persisted accounts-in-motion rows before campaign generation can use them.
5. Current ACCOUNT v1 remains conservative: single-review rows can render in UI but do not publish to campaign generation until witness-dedup/corroboration lifts confidence above low.
6. Campaign storage metadata now persists `opportunity_claim`, a compact `opportunity_claim_gate`, and aggregated `opportunity_claims` lists for vendor/challenger campaign modes.
7. Detached batch replay now requires report-safe ProductClaim gate context before storing a replayed campaign. Legacy queued items without claim metadata fail before DB insert instead of bypassing the fetcher gate.
8. Campaign approval paths now enforce report-safe ProductClaim metadata before status changes: single approve, single queue-send, bulk approve, and bulk queue-send.
9. Campaign queue CSV export now includes ProductClaim gate columns from stored metadata, including claim id, count, render/report booleans, confidence, evidence posture, and suppression reason.

### Recommended Fix Shape

Create one server-side opportunity contract helper and make every surface use it:

1. `shape_account_opportunity_rows(pool, rows, *, as_of_date, window_days, user)` returns rows with nested `opportunity_claim`.
2. UI list keeps using render semantics.
3. CSV export defaults to report-safe rows only. Operators can request raw/audit exports with `report_safe_only=false`, in which case claim readiness columns make monitor-only and suppressed rows explicit.
4. Campaign generation filters on `opportunity_claim.report_allowed === true` before LLM generation or persistence.
5. Campaign rows persist the claim payload / compact gate metadata through the existing `metadata` JSONB, so downstream audit can prove why a campaign was allowed.
6. Replay/batch storage fails closed if the replay payload lacks a report-safe single `opportunity_claim` or all-report-safe aggregate `opportunity_claims`.
7. API approve/queue actions fail closed if stored campaign metadata lacks a report-safe full claim, compact claim gate, or all-report-safe aggregate claim list.
8. Campaign CSV export surfaces the same stored gate metadata so operator exports can distinguish report-safe, monitor-only, and legacy rows.

### Test Targets

- Export test: high-intent CSV defaults to report-safe rows only.
- Export test: `report_safe_only=false` includes ProductClaim readiness columns for both render-safe monitor rows and suppressed rows.
- Campaign fetcher test: raw review opportunity with a non-report-safe ACCOUNT claim is filtered before campaign generation.
- Campaign fetcher test: report-safe ACCOUNT claim still flows through and carries the serialized claim payload.
- Autonomous accounts-in-motion test: persisted report account with no report-safe claim is skipped before campaign generation.
- Campaign metadata test: persisted metadata includes both single-row `opportunity_claim` and aggregate `opportunity_claims` gate payloads.
- Campaign replay test: replay storage raises before DB touch when ProductClaim gate context is missing.
- Campaign API tests: approve, queue-send, bulk approve, and bulk queue-send block legacy campaigns missing ProductClaim gate metadata before update.
- Campaign export test: campaign CSV includes ProductClaim gate columns from stored metadata.
- Tenant alias follow-up: `/api/v1/b2b/tenant/export/high-intent` delegates to the fixed export function; add an integration assertion when the tenant test harness is next touched.

## Report Renderer Contract Deep Dive - 2026-04-28

Patch 6 moved the canonical Microsoft-365-style winner-call paths onto ProductClaim gates, but the report renderer system still has parallel legacy surfaces. This is a systemic coverage issue, not a single component bug: report view models were built before ProductClaim existed, so some sections now inherit the contract while neighboring sections still render raw synthesized fields.

### Renderer Surface Map

| Surface | Current behavior | Risk |
| --- | --- | --- |
| Challenger brief head-to-head | Uses `head_to_head.product_claim` and `ProductClaimGate`. | Good reference path. |
| Battle-card displacement reasoning | Uses `displacement_reasoning.product_claim_gate` and field-level gate messages. | Good reference path. |
| Generic cross-vendor battles | `BattleList` renders `winner`, `loser`, `durability`, `conclusion`, and `key_insights` directly. | Recreates the exact winner-call class Patch 6 fixed, but through generic report fields. |
| Generic weakness analysis | Renders `weakness`, `evidence`, `customer_quote`, `winning_position`, and `recommendation` directly. | Incumbent weakness and challenger proof language can publish without evidence posture. |
| Generic recommended plays | Renders `play`, `key_message`, `target_segment`, and `timing` directly. | Actionable guidance can bypass opportunity/report safety gates. |
| Specialized category council | Renders category `winner`, `loser`, `confidence`, `durability`, and conclusion directly in multiple sections. | "Winner" semantics survive outside the head-to-head gate. |
| Specialized cross-vendor battles | Renders `battle.winner`, `battle.loser`, confidence, durability, conclusion, and citations directly. | ProductClaim-gated head-to-head can be suppressed while sibling battle cards still say who won. |
| Specialized top displacement targets | Renders competitor names and mention counts from `top_displacement_targets`. | Direct displacement can look validated based on mention counts rather than claim lineage. |
| Displacement report detail | Renders market losers/winners, net flow, top destination/source, and drivers directly. | Aggregate displacement leaderboard can imply confirmed winners without pair-level claim gates. |
| Public report page | Reads `briefing.top_displacement_targets` into customer-facing report content. | Public packaged report can inherit legacy displacement rows even if internal UI suppresses them. |

### Why This Is Systemic

The codebase now has two report-rendering contracts:

1. **Validated contract:** render only from ProductClaim-backed fields, fail closed on missing gate context, and preserve suppression metadata.
2. **Legacy structured contract:** parse whatever report JSON contains and render dangerous field names with attractive badges.

The second contract was reasonable when reports were the primary product artifact. It is now the wrong default because dashboards are the truth layer and reports should inherit validated objects. Any generic renderer that treats `winner`, `weakness`, `recommendation`, `top_destination`, or `market_winner` as plain display fields can bypass the dashboard contract.

### Recommended Fix Shape

Do not patch one label at a time. Add a report-claim coverage layer:

1. Define a dangerous report semantic allowlist: `winner`, `loser`, `market_winner`, `top_destination`, `top_source`, `recommended_plays`, `weakness_analysis`, `cross_vendor_battles`, `top_displacement_targets`, `category_council`.
2. For each semantic, decide one of:
   - `ProductClaim required`
   - `EvidenceClaim required`
   - `Legacy allowed but must label "Unvalidated / legacy"`
   - `Internal-only; do not render in customer reports`
3. Update `StructuredReportValue` so dangerous keys fail closed unless the payload includes gate metadata.
4. Update specialized view models so every winner/play/displacement section exposes a gate envelope or an explicit legacy state.
5. Add renderer tests that a payload with `winner` but no gate context does not render a green winner badge.

### First Slice Candidate

Start with `StructuredReportData.tsx`, not the specialized renderer:

1. Add a small `DangerousStructuredFieldGate` that wraps `cross_vendor_battles`, `weakness_analysis`, `vendor_weaknesses`, and `recommended_plays`.
2. If a payload item lacks ProductClaim/gate metadata, render a muted "Legacy report field - validation unavailable" block instead of winner/recommendation language.
3. Keep raw structured tables for non-dangerous keys unchanged.
4. Pin with tests using the existing `StructuredReportData.test.tsx` fixtures.

This closes the broadest bypass first. Specialized residuals can then move one section at a time using the same gate vocabulary.

## Autonomous Liveness Deep Dive - 2026-04-28

The recent "No due competitive sets" symptom is not primarily a scheduler bug. The scheduler can fire exactly on time and still do no B2B reasoning work if the domain queue is empty or filtered out. That makes it a liveness observability problem: task success does not prove product coverage.

### Current Due Predicate

`CompetitiveSetRepository.list_due_scheduled()` only returns rows where all of these are true:

1. `active = TRUE`
2. `refresh_mode = 'scheduled'`
3. `refresh_interval_hours IS NOT NULL`
4. `last_run_status IS NULL OR last_run_status != 'running'`
5. `COALESCE(last_run_at, last_success_at, created_at) <= NOW() - refresh_interval_hours`

`b2b_reasoning_synthesis` then returns `{"_skip_synthesis": "No due competitive sets"}` when that list is empty. That skip can be operationally correct for a single day, but it is dangerous as a repeated steady state because the scheduled task execution still looks clean while the product receives no new reasoning coverage.

### Liveness Gaps

1. Competitive sets default to `refresh_mode="manual"` with `refresh_interval_hours=None`, so newly-created sets are invisible to scheduled synthesis unless explicitly activated.
2. The scheduler's orphan cleanup updates `task_executions.status='failed'`, but it does not repair `b2b_competitive_sets.last_run_status='running'`.
3. `mark_run_started()` sets `last_run_at=NOW()` and `last_run_status='running'`; a process crash before `mark_run_completed()` can leave the set permanently excluded by the due predicate.
4. The Phase 9 soak activation runbook fixes one cohort manually, but the underlying system still lacks a standing "why is my due queue empty?" diagnostic.
5. Scheduled task health currently answers "did the cron fire?" more than "did the domain queue produce product work?"

### Recommended Fix Shape

Add a domain-liveness audit for competitive-set synthesis:

1. Query counts by eligibility bucket:
   - active scheduled with interval and due
   - active scheduled with interval but not due
   - active scheduled but stuck `running`
   - active manual
   - inactive
   - scheduled with missing interval, which should be impossible under the DB constraint but worth checking
2. Emit that summary whenever `b2b_reasoning_synthesis` returns `No due competitive sets`.
3. Alert only on repeated or impossible states, not on a single normal empty day:
   - zero active scheduled rows for more than one scheduled cycle
   - any `last_run_status='running'` older than the task timeout plus slack
   - no due rows for N cycles while active scheduled rows exist and at least one should have matured
4. Add an operator repair path for stale-running competitive sets, separate from task execution cleanup.
5. Persist the due-queue summary into the task execution result so future audits can distinguish "cron fired and no work was due" from "cron fired but product configuration is broken."

### Test Targets

- Repository test: a stale `last_run_status='running'` row is excluded from `list_due_scheduled`.
- Liveness helper test: rows are classified into mutually exclusive eligibility buckets.
- Task test: repeated `No due competitive sets` returns/records the eligibility summary.
- Repair test: stale-running reset changes only rows older than the configured threshold.

## Schema Contract Deep Dive - 2026-04-28

The newer ProductClaim endpoints show the right pattern: backend response models define every gate field as required, and frontend types mirror that shape. Older or adjacent safety surfaces still rely on plain dicts, so missing gate fields are only caught if each React consumer remembers to fail closed.

### Reference Pattern

`b2b_vendor_claims.py` and `b2b_challenger_claims.py` are the good contract:

1. Pydantic response model exists at the route boundary.
2. `render_allowed` and `report_allowed` are required booleans.
3. `suppression_reason` is nullable but present.
4. The serializer is centralized.
5. The frontend `VendorClaim` type has required gate fields.

This is the shape new safety-bearing APIs should follow.

### Drift Pattern

The drift appears where the payload is a larger row and the claim/gate is nested or appended:

1. Evidence witness list/detail endpoints return plain dicts. They append `render_allowed`, `report_allowed`, `suppression_reason`, `evidence_posture`, and `confidence` via `apply_witness_render_gate`, but no response model enforces that those fields are present.
2. High-intent opportunity rows expose `opportunity_claim?: AccountOpportunityClaim | null` in TypeScript. That optionality is currently justified for rollout/cache tolerance, but it is not enforced by a backend response model that says fresh rows must carry the claim.
3. Report view models convert embedded product-claim payloads from untyped report JSON. The converters validate enums, but the original API payload is still a cached report blob rather than a response model with gate guarantees.
4. `fetchWithApiFallback` can mask a route registration drift by retrying a different prefix, so schema and route drift can arrive together.

### Recommended Fix Shape

Treat safety envelopes as first-class schemas:

1. Add Pydantic response models for evidence witness list/detail rows with required witness gate fields.
2. Add a Pydantic high-intent row response model that makes `opportunity_claim` required for fresh API rows. If legacy tolerance is still needed, use an explicit wrapper state such as `claim_status: "validated" | "legacy" | "validation_unavailable"`.
3. Keep frontend types aligned with backend models:
   - gate fields required inside validated envelopes
   - row-level legacy nullable fields allowed only when the UI has a matching fail-closed state
4. Add route tests that intentionally remove a required gate field from a fixture/model and fail before the response reaches React.
5. Stop adding new safety fields to plain dicts without a response-model test.

### Test Targets

- Evidence API test: list/detail response always contains non-null `render_allowed`, `report_allowed`, `evidence_posture`, and `confidence` for every witness row.
- Opportunity API test: fresh high-intent rows always contain `opportunity_claim`; missing claim maps to explicit validation-unavailable state rather than omission.
- Type test or compile guard: `render_allowed?:` / `report_allowed?:` should not appear inside validated claim envelopes.
- Route-prefix test: ProductClaim endpoints remain pinned to `/api/v1` without client fallback masking.

## Initial Prioritization

1. **Access boundary sweep:** SYS-001, SYS-002, SYS-003, SYS-005, SYS-006, SYS-007, SYS-016, SYS-017.
2. **Action/export inheritance:** SYS-004, SYS-008.
3. **Report/UI contract completion:** SYS-009, SYS-010, SYS-013, SYS-018.
4. **Autonomous liveness:** SYS-019.
5. **Schema contracts:** SYS-020.
6. **Claim substrate hardening:** SYS-011, SYS-012, SYS-014, SYS-015.

## Clean / Lower-Suspicion Areas From This Pass

- Evidence list/detail/trace rendering now generally fails closed on `render_allowed === true`; remaining concern is operational fallback monitoring, not React fail-open behavior.
- Newer tenant dashboard endpoints do not appear anonymously fail-open, but they still need product/tier enforcement review because many use `require_auth` instead of `require_b2b_plan`.
- Win/Loss Predictor is now internally honest after nullable probability gates and compare-response hardening; the remaining concern is customer-facing naming if it is externalized.
