# PR-CSV-Owner-Lane-Vertical

## Why this slice exists

CSV-first product-gap reporting can only be buyer-safe if ATLAS is precise about which claims come from customer text, which come from safe routing/index metadata, and which require follow-up evidence. The current report can count repeated questions and estimate handling cost, but it does not preserve enough safe CSV routing metadata to say "this repeated login gap routes to Auth / Product UX" without pretending to know the exact UI root cause. This vertical proves the first routeable product-friction deliverable for repeated CSV-backed gaps such as "Where is the login button?"

This PR is over the 400 LOC soft cap because the slice is intentionally cross-surface: normalization, grouped FAQ item metadata, paid report-model/export fields, delivery email copy, generated contract artifacts, ATLAS hosted-result smoke coverage, and focused regression tests must land together or downstream consumers cannot rely on the new fields.

## Scope (this PR)

Ownership lane: deflection/csv-owner-lane
Slice phase: Vertical slice

1. Preserve safe Zendesk/CSV routing metadata (`group`, `assignee`, `tags`, `brand`, `organization`, `product_area`, `custom_product_area`) during ticket normalization while keeping internal/private note filtering unchanged.
2. Add `routing_signals` and `evidence_tier` to paid action items and generated hosted report-model contracts.
3. Route `owner_lane` deterministically from customer/topic wording first, then non-identity routing signals, with first-slice mappings for auth, billing, reporting, and admin/access.
4. Update email next-action rows and ATLAS hosted result-page smoke rendering to show owner/evidence information without source IDs, raw evidence, `top_evidence`, or exact UI-root-cause claims.
5. Add focused tests for CSV routing metadata, index-only evidence tier, auth owner-lane routing, email privacy, and hosted result-page paid/free boundaries.

### Review Contract

Acceptance criteria:
- CSV rows with `Group` and `Tags` preserve those values as metadata/routing signals, not as customer evidence text.
- Index-only CSV rows produce `csv_index_metadata_only`; public customer text produces `csv_customer_text`; scoped public resolution evidence can produce `csv_full_thread_resolution_evidence`.
- Repeated login/auth rows route to `Auth / Product UX` and keep cost at `ticket_count * 13.50`.
- Buyer-safe surfaces show owner lane, evidence tier, routing cues, repeat count, and estimated handling cost without raw quotes, source IDs, private/internal text, `top_evidence`, or exact UI-path/root-cause wording.

Affected surfaces:
- CSV support-ticket normalization and FAQ grouping.
- Paid report model/export contract.
- Email delivery summary.
- ATLAS hosted-result smoke page and generated frontend contract artifacts.

Risk areas:
- Private/internal note leakage.
- Overclaiming product root cause from CSV metadata.
- Contract drift between Python producer and generated frontend artifacts.
- Owner-lane fallback changing established report rows without useful metadata.

Reviewer rules triggered: R1, R2, R4, R8, R10, R13, R14.

### Files touched

- `plans/PR-CSV-Owner-Lane-Vertical.md` - this plan.
- `extracted_content_pipeline/support_ticket_input_package.py` - preserve safe routing metadata and ticket evidence tier.
- `extracted_content_pipeline/ticket_faq_markdown.py` - roll routing signals and item evidence tier into grouped FAQ items.
- `extracted_content_pipeline/faq_deflection_report.py` - expose paid action/detail fields, strengthen owner-lane mapping, and update hosted contract shapes.
- `extracted_content_pipeline/deflection_report_access.py` - backfill legacy owner metadata and trim hosted routing projection to buyer-safe labels.
- `atlas_brain/content_ops_deflection_delivery.py` - include owner/evidence tier in email action rows.
- `portfolio-ui/src/types/deflectionReportModel.ts` - regenerated ATLAS report-model TypeScript contract.
- `portfolio-ui/api/content-ops/deflection/report-model-contract.js` - regenerated hosted report-model contract.
- `portfolio-ui/api/content-ops/deflection/result-page.js` - render owner/evidence/cost product-gap cards without exact root-cause claims.
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs` - paid/free hosted result-page boundary coverage.
- `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs` - keep package-script enrollment expectations current.
- `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs` - hosted proxy smoke copy expectation for Product Gap cards.
- `docs/frontend/content_ops_faq_report_example.json` - document FAQ item evidence tier in the core frontend example.
- `docs/frontend/content_ops_faq_deflection_report_example.json` - refresh generated deflection report-model example with owner/evidence/routing fields.
- `docs/extraction/validation/content_ops_faq_search_route_contract_handoff.md` - document generated detail-item evidence tier shape.
- `scripts/check_content_ops_faq_search_route_contract.py` - validate generated contract evidence tier shape.
- `scripts/generate_deflection_frontend_contract_types.py` - generator type mappings for evidence tier and routing signals.
- `scripts/smoke_content_ops_faq_search_concurrency.py` - seed detail-route smoke items with evidence tier.
- `tests/test_atlas_billing_content_ops_deflection_paid_flow.py` - paid hosted projection coverage for routing signal privacy.
- `tests/test_smoke_content_ops_support_ticket_package.py` - CSV metadata and evidence-tier ingestion coverage.
- `tests/test_check_content_ops_faq_search_route_contract.py` - contract checker coverage for detail-item evidence tier.
- `tests/test_content_ops_deflection_report.py` - owner-lane, routing signal, cost, and report-model coverage.
- `tests/test_atlas_content_ops_deflection_delivery.py` - email owner/evidence and privacy coverage.
- `tests/test_extracted_content_deflection_submit.py` - paid report-model route projection coverage for routing metadata.
- `tests/test_extracted_support_ticket_input_package.py` - normalized source-material routing metadata coverage.
- `tests/test_generate_deflection_frontend_contract_types.py` - generated contract shape coverage.
- `tests/test_smoke_content_ops_faq_search_route_concurrency.py` - route detail fixture evidence tier coverage.

## Mechanism

`support_ticket_input_package` normalizes a fixed allowlist of routing/index metadata from CSV rows and computes a package-level and row-level `support_ticket_evidence_tier`. It keeps the existing private/internal exclusions and strips internal-only markers before rows become public source material.

`ticket_faq_markdown` rolls grouped rows into a deterministic `routing_signals` object and chooses the strongest item evidence tier. `faq_deflection_report` admits those fields into paid action rows, generated hosted safe-field shapes, and owner-lane routing. Owner routing uses customer/topic wording first, then non-identity routing labels, and falls back to `Unknown` only when no useful signal exists.

Email rows render only buyer-safe summaries: question, owner lane, evidence tier, count, cost, and action. The ATLAS hosted-result page smoke renders product-gap cards as routeable product friction, not exact UI root-cause proof.

## Intentional

- No Zendesk API fields are required in this slice.
- No Jira template, monthly normalization, richer department taxonomy, or exact UI path/root-cause claim is included.
- `routing_signals` is metadata for routing and ownership, not evidence text used to quote customers.
- Generated frontend artifacts are committed because downstream portfolio contract drift checks consume them directly.
- The diff is above the soft cap because landing only one lane would leave a contract field unused or a buyer surface unable to prove the deliverable.

## Deferred

- Atlas-portfolio live buyer page rendering lands in the companion `atlas-portfolio` PR.
- Jira bug template generation remains follow-up work after the owner-lane vertical is proven.
- Monthly normalization remains follow-up work; this slice preserves the existing `ticket_count * 13.50` estimate.
- Richer department taxonomy remains follow-up work after the first deterministic lane mapping is validated.

Parked hardening: none.

## Verification

- Targeted pytest suite passed, 387 tests: `tests/test_check_content_ops_faq_search_route_contract.py`, `tests/test_extracted_support_ticket_input_package.py`, `tests/test_smoke_content_ops_support_ticket_package.py`, `tests/test_content_ops_deflection_report.py`, `tests/test_atlas_content_ops_deflection_delivery.py`, and `tests/test_generate_deflection_frontend_contract_types.py`.
- `python scripts/generate_deflection_frontend_contract_types.py --check` - passed; generated snapshot/report-model artifacts are current.
- `node portfolio-ui/scripts/faq-deflection-result-page.test.mjs` - passed.
- `node portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs` - passed.
- `python -m pytest tests/test_content_ops_deflection_report.py tests/test_generate_deflection_frontend_contract_types.py tests/test_check_content_ops_faq_search_route_contract.py tests/test_smoke_content_ops_faq_search_concurrency.py tests/test_smoke_content_ops_faq_search_route_concurrency.py -q` - passed, 377 tests.
- `python -m pytest tests/test_content_ops_faq_report_contract_docs.py -q` - passed, 5 tests.
- `python -m pytest tests/test_extracted_content_deflection_submit.py -q` - passed, 79 tests.
- `python -m pytest tests/test_content_ops_faq_report_contract_docs.py tests/test_extracted_content_deflection_submit.py tests/test_content_ops_deflection_report.py tests/test_generate_deflection_frontend_contract_types.py tests/test_check_content_ops_faq_search_route_contract.py tests/test_smoke_content_ops_faq_search_concurrency.py tests/test_smoke_content_ops_faq_search_route_concurrency.py -q` - passed, 461 tests.
- `node portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs` - passed.
- `rg "button is buried|Account Settings -> Preferences -> Billing|Account Settings > Preferences > Billing" extracted_content_pipeline atlas_brain portfolio-ui tests` - no exact UI-path/root-cause claim remains in the touched owner-lane surfaces.
- Partial local reproduction of `scripts/run_extracted_pipeline_checks.sh` found and fixed stale docs/report-model route assertions; full local completion is blocked by Windows-only symlink, newline, and path-separator differences that do not reproduce on Linux CI.
- Not run: Bash wrapper gauntlets (`scripts/local_pr_review.sh`, `scripts/validate_extracted_content_pipeline.sh`, `scripts/check_ascii_python.sh`) because this Windows runtime has no `bash` executable.

## Estimated diff size

| Area | Approx LOC |
|---|---:|
| Plan | ~80 |
| Producer/report/email/result-page code | ~452 |
| Generated contracts | ~240 |
| Tests/docs/checkers | ~567 |
| Total | ~1262 additions / ~64 deletions |
