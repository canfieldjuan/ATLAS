# PR-Product-Gaps-Action-Context

## Why this slice exists

#1849 proved the first owner-lane vertical, but #1845 still has a gap between
"routeable owner/cost card" and "ready to hand to a product/support lead." The
root cause is that the action-row contract still carries mostly FAQ-shaped
fields: it has owner lane, evidence tier, routing signals, cost, and
representative phrasing, but it does not package those into an explicit
product-gap summary, period/cost-confidence labels, or a Jira-ready action
handoff. Buyers can see where a repeat routes, but they still have to assemble
the product-gap task by hand.

This change fixes that root for the first action-context layer by deriving the
missing context from already-scrubbed report fields and adding it to the
ATLAS-owned report-model contract. It is intentionally additive to
`deflection.v1`; no existing owner/cost/customer-wording fields are duplicated
without a named use.

This PR is over the 400 LOC soft cap because the report-model contract is now
single-source/generated: adding one hosted-safe nested action object updates the
Python contract metadata, generated TypeScript/API artifacts, and the focused
contract tests in the same slice. Splitting those would recreate the
dual-map drift this lane just removed.

## Scope (this PR)

Ownership lane: deflection/product-gaps-report-shape
Slice phase: Vertical slice

1. Add the remaining first-pass #1845 action-context fields to report action
   rows: `product_gap_summary`, `customer_vocabulary`, `cost_period`,
   `cost_confidence`, and `jira_template`.
2. Derive `customer_vocabulary` from the existing representative
   phrasing/customer wording path instead of creating a parallel raw customer
   text source.
3. Label the existing per-gap benchmark cost as batch-period,
   benchmark-confidence cost context; do not introduce monthly normalization in
   this PR.
4. Surface at least one buyer-visible payoff in email/result-page Product Gap
   rows so this is not a model-only slice.
5. Keep free/locked snapshots unchanged; new fields stay out of the free teaser
   unless explicitly hosted-safe through the contract.

### Review Contract

Acceptance criteria:
- Action rows expose `product_gap_summary`, `customer_vocabulary`,
  `cost_period`, `cost_confidence`, and `jira_template` through the
  ATLAS-owned report-model contract metadata and regenerated JS/TS artifacts.
- The new fields are populated from existing safe action-row ingredients:
  question, owner lane, evidence tier, representative phrasing, ticket count,
  estimated support cost, and recommended action.
- `product_gap_summary` does not claim exact UI root cause or screen path.
- `customer_vocabulary` is bounded and scrub-compatible; no raw `top_evidence`,
  source IDs, or private/internal text is surfaced.
- Email or paid result-page rendering uses the new context for an unlocked
  buyer-visible Product Gap row/card, while locked/free output stays unchanged.
- Generated contract drift check and focused report/email/result-page tests
  pass.

Affected surfaces:
- `extracted_content_pipeline/faq_deflection_report.py` action-row producer and
  report-model contract metadata.
- Generated report-model artifacts in `portfolio-ui/`.
- ATLAS delivery email and hosted result-page Product Gap card.
- Focused report, delivery, generator, and result-page tests.

Risk areas:
- PII leakage from customer-worded fields.
- Overclaiming root cause from CSV metadata.
- Hand-editing generated consumers instead of regenerating from the ATLAS
  contract.
- Duplicating existing confidence/cost/customer-wording semantics instead of
  labeling them.

Reviewer rules triggered: R1, R2, R4, R8, R10, R13, R14.

### Files touched

- `atlas_brain/content_ops_deflection_delivery.py`
- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Product-Gaps-Action-Context.md`
- `portfolio-ui/api/content-ops/deflection/report-model-contract.js`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`
- `portfolio-ui/src/types/deflectionReportModel.ts`
- `scripts/generate_deflection_frontend_contract_types.py`
- `tests/test_atlas_content_ops_deflection_delivery.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_generate_deflection_frontend_contract_types.py`

## Mechanism

The report producer will build action-context fields inside `_action_item` from
the same normalized values it already emits:

- `product_gap_summary`: a short, non-root-cause sentence naming the repeated
  owner lane, repeat count, evidence tier, and benchmark handling cost. It is
  omitted for non-repeated low-confidence rows.
- `customer_vocabulary`: a bounded list derived from
  `_action_representative_phrasing`.
- `cost_period`: a scalar label for the existing per-batch estimate, such as
  `batch_upload`.
- `cost_confidence`: a scalar label reflecting benchmark-only cost basis and
  evidence strength.
- `jira_template`: a structured object with title, description, owner lane,
  impact, evidence tier, repeat count, cost, customer vocabulary, and next
  action.

The new fields go into `_REPORT_ACTION_ITEM_FIELDS`, hosted-safe fields only
where buyer-safe, optional projected fields for `deflection.v1` compatibility,
nested object metadata for `jira_template`, and generated frontend contract artifacts via
`scripts/generate_deflection_frontend_contract_types.py`. Email and the
paid result-page card will read the generated/report-model fields rather than
maintaining a hand-authored parallel map.

## Intentional

- No monthly-normalized cost math in this slice; `cost_period` labels the
  existing batch estimate so it is honest without changing economics.
- No exact root-cause copy. Product-gap summaries say repeated friction/routes,
  not "the button is buried" or specific screen paths.
- `customer_vocabulary` is derived from existing representative phrasing rather
  than a new raw-text extraction path.
- #1853 remains paused/downstream and untouched.
- Review follow-up: net-new action-context fields are optional for persisted
  `deflection.v1` reports, and the producer example fixture is regenerated with
  the same source as the docs parity test.

## Deferred

- Monthly normalization and richer cost-period math remain follow-up work after
  this explicit label lands.
- Richer department taxonomy remains deferred from #1849.
- Cross-repo `atlas-portfolio` consumption remains a separate slice if this
  ATLAS-owned artifact changes external repo expectations.
- #1853 demo-derive remains paused until #1843/#1845/#1846/#1847 settle.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_content_ops_deflection_report.py::test_csv_product_gap_owner_lane_vertical_routes_login_gap tests/test_atlas_content_ops_deflection_delivery.py::test_delivery_worker_renders_model_backed_email_summary tests/test_generate_deflection_frontend_contract_types.py::test_deflection_report_model_types_include_backend_projection_fields tests/test_generate_deflection_frontend_contract_types.py::test_deflection_report_model_types_publish_hosted_safe_allowlists tests/test_generate_deflection_frontend_contract_types.py::test_deflection_report_model_types_publish_hosted_field_shapes tests/test_generate_deflection_frontend_contract_types.py::test_deflection_report_model_api_contract_publishes_hosted_safe_allowlists -q` - passed, 6 tests.
- Generator check with `python` running
  `scripts/generate_deflection_frontend_contract_types.py --check` - passed;
  generated artifacts are current.
- `python -m pytest tests/test_content_ops_deflection_report.py tests/test_atlas_content_ops_deflection_delivery.py tests/test_generate_deflection_frontend_contract_types.py -q` - passed, 214 tests.
- `python -m pytest tests/test_extracted_content_deflection_submit.py tests/test_check_content_ops_faq_search_route_contract.py -q` - passed, 161 tests.
- `node portfolio-ui/scripts/faq-deflection-result-page.test.mjs` - passed.
- `python -m pytest tests/test_content_ops_deflection_report.py::test_csv_product_gap_owner_lane_vertical_routes_login_gap tests/test_content_ops_deflection_report.py::test_product_gap_summary_does_not_copy_root_cause_or_screen_path_question tests/test_content_ops_deflection_report.py::test_product_gap_summary_is_omitted_for_non_repeated_low_confidence_rows tests/test_content_ops_deflection_report.py::test_deflection_report_projection_separates_paid_and_hosted_action_fields tests/test_generate_deflection_frontend_contract_types.py::test_deflection_report_model_types_include_backend_projection_fields tests/test_atlas_content_ops_deflection_delivery.py::test_delivery_worker_renders_model_backed_email_summary tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_deflection_example_matches_producer_shape -q` - passed, 7 tests.
- `python -m pytest tests/test_content_ops_deflection_report.py tests/test_atlas_content_ops_deflection_delivery.py tests/test_generate_deflection_frontend_contract_types.py tests/test_content_ops_faq_report_contract_docs.py -q` - passed, 221 tests.
- Extracted gauntlet with `bash` running
  `scripts/run_extracted_pipeline_checks.sh` - passed, 4999 passed, 15 skipped.
- Pending before push: local PR review via `scripts/push_pr.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/content_ops_deflection_delivery.py` | 55 |
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 138 |
| `extracted_content_pipeline/faq_deflection_report.py` | 164 |
| `plans/PR-Product-Gaps-Action-Context.md` | 174 |
| `portfolio-ui/api/content-ops/deflection/report-model-contract.js` | 144 |
| `portfolio-ui/api/content-ops/deflection/result-page.js` | 49 |
| `portfolio-ui/scripts/faq-deflection-result-page.test.mjs` | 43 |
| `portfolio-ui/src/types/deflectionReportModel.ts` | 262 |
| `scripts/generate_deflection_frontend_contract_types.py` | 5 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 16 |
| `tests/test_content_ops_deflection_report.py` | 122 |
| `tests/test_generate_deflection_frontend_contract_types.py` | 34 |
| **Total** | **1206** |
