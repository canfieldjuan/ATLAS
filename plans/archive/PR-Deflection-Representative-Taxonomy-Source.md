# PR-Deflection-Representative-Taxonomy-Source

## Why this slice exists

PR #1521 made representative source-policy labels PII-safe by admitting
structured `product`/`issue` values only when they match a known taxonomy
allowlist. The merge review accepted that as the right privacy tradeoff for the
CFPB-first path, but flagged the coupling: the generic FAQ renderer now carries
23 CFPB consumer-finance terms internally, so non-finance tenants cannot supply
their own safe taxonomy and the standalone renderer is no longer domain-neutral.

This slice moves that allowlist to an explicit request/config surface while
preserving the current support-ticket package behavior. The renderer should
validate structured fields against terms supplied by the caller, the support
ticket input package should provide the CFPB default for the current launch
path, and callers with no taxonomy should fail closed to the generic topic
fallback instead of silently trusting arbitrary upload fields.

The diff is slightly over the 400 LOC target because the review fix must wire
the same setting through both halves of the contract: backend catalog/execution
and the hosted UI that renders advertised FAQ configuration inputs. Splitting
those would leave CI red and the new setting technically present but unusable
from the hosted form.

## Scope (this PR)

Ownership lane: deflection/clustering
Slice phase: Production hardening

1. Add a `faq_representative_taxonomy_terms` input/config path alongside
   `faq_documentation_terms`.
2. Move the CFPB representative taxonomy terms out of the generic FAQ renderer
   and into the support-ticket input package as the current launch default.
3. Thread representative taxonomy terms through hosted generation planning,
   execution dispatch, the service config, and the low-level renderer.
4. Keep the renderer fail-closed when no representative taxonomy terms or
   documentation terms are supplied.
5. Add tests proving custom taxonomy injection works for non-CFPB fields, no
   taxonomy falls back generically, the support-ticket package supplies the
   CFPB default, hosted execution carries the new config, and the hosted UI
   exposes the advertised input.

### Files touched

- `atlas-intel-ui/scripts/content-ops-faq-configuration-inputs.test.mjs`
- `atlas-intel-ui/src/domain/contentOps/faqConfigurationInputs.ts`
- `atlas-intel-ui/src/domain/contentOps/index.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/faq_deflection_report.py`
- `extracted_content_pipeline/generation_plan.py`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `extracted_content_pipeline/ticket_faq_input_contract.py`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Deflection-Representative-Taxonomy-Source.md`
- `tests/test_extracted_content_control_surface_api.py`
- `tests/test_extracted_content_generation_plan.py`
- `tests/test_extracted_content_ops_execution.py`
- `tests/test_extracted_support_ticket_input_package.py`
- `tests/test_extracted_ticket_faq_markdown.py`

### Review Contract

- Acceptance criteria:
  - [ ] The generic FAQ renderer no longer owns a CFPB-specific taxonomy
        allowlist as its implicit default.
  - [ ] The support-ticket input package emits the CFPB representative taxonomy
        terms for the current CFPB-first launch path.
  - [ ] A caller can inject non-CFPB representative taxonomy terms and get
        distinct representative labels from matching structured fields.
  - [ ] A caller with no documentation terms and no representative taxonomy
        terms keeps the generic topic fallback for structured fields.
  - [ ] Hosted generation plan and execution dispatch preserve the new config
        from request inputs to the FAQ service call.
  - [ ] The hosted Content Ops new-run form renders and edits the advertised
        representative taxonomy input when the backend catalog includes it.
  - [ ] Existing documentation-term behavior and vocabulary-gap behavior remain
        unchanged.
- Affected surfaces: extracted FAQ renderer API, hosted content generation plan
  config, support-ticket input package outputs, hosted Content Ops FAQ config
  UI, and tests.
- Risk areas: request/config compatibility, privacy fail-closed behavior,
  product path parity after moving the CFPB terms, UI discoverability, and CI
  enrollment.
- Reviewer rules triggered: R1, R2, R5, R9, R10, R12, R13, R14.

## Mechanism

Introduce a named input key for representative taxonomy terms and parse it with
the same decoded-input tolerant sequence helpers already used for documentation
terms. The plan config carries the terms to the FAQ service, and the service
passes them to the renderer.

The renderer builds the safe structured-field allowlist from the supplied
representative taxonomy terms rather than a module-local CFPB constant. The
existing safe-label path still combines documentation terms with validated
structured context, still requires repeated evidence tokens, and still rejects
email/long-number shaped terms. With an empty taxonomy, structured fields
contribute nothing, so arbitrary upload values cannot become headings.

The support-ticket input package owns the current CFPB default by adding
`faq_representative_taxonomy_terms` to generated request inputs. That keeps the
paid support-ticket path behavior from #1521 while making the renderer reusable
for other tenant/product taxonomies.

The hosted new-run form uses the same FAQ configuration contract list as the
backend catalog. Adding the representative taxonomy key there makes the field
visible and editable without raw JSON editing.

## Intentional

- **No tenant registry in this PR.** This is a transport/config seam; a future
  tenant settings slice can choose where per-tenant taxonomy terms live.
- **CFPB default remains in the support-ticket package.** It is still a launch
  default for the current data shape, but it no longer lives in the generic
  renderer.
- **No label wording polish.** Clipped labels such as stopword-stripped CFPB
  phrases remain a separate product-polish follow-up.

## Deferred

- Per-tenant/per-upload taxonomy management outside the support-ticket package.
- Safe-vocabulary collision disambiguation from `HARDENING.md`.
- Customer-wording FAQ heading PII hardening from `HARDENING.md`.

Parked hardening: `Safe-vocabulary representative label collisions render
duplicate FAQ headings`; `Customer-wording FAQ headings can publish raw PII`.

## Verification

- Command passed: python -m py_compile extracted_content_pipeline/ticket_faq_input_contract.py extracted_content_pipeline/support_ticket_input_package.py extracted_content_pipeline/ticket_faq_markdown.py extracted_content_pipeline/faq_deflection_report.py extracted_content_pipeline/generation_plan.py extracted_content_pipeline/content_ops_execution.py tests/test_extracted_support_ticket_input_package.py tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py tests/test_extracted_content_control_surface_api.py.
- Command passed: pytest tests/test_extracted_support_ticket_input_package.py tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py tests/test_extracted_content_control_surface_api.py -q -- 572 passed, 1 skipped.
- Command passed: cd atlas-intel-ui && npm run test:content-ops-faq-configuration-inputs -- 4 passed.
- Command passed: cd atlas-intel-ui && npm run build.
- Command passed: bash scripts/run_extracted_pipeline_checks.sh -- 4042 passed, 10 skipped, 1 warning.
- Pending before push: bash scripts/local_pr_review.sh --current-pr-body-file tmp/deflection_representative_taxonomy_source_pr_body.md.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas-intel-ui/scripts/content-ops-faq-configuration-inputs.test.mjs` | 19 |
| `atlas-intel-ui/src/domain/contentOps/faqConfigurationInputs.ts` | 4 |
| `atlas-intel-ui/src/domain/contentOps/index.ts` | 1 |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | 69 |
| `extracted_content_pipeline/content_ops_execution.py` | 8 |
| `extracted_content_pipeline/faq_deflection_report.py` | 2 |
| `extracted_content_pipeline/generation_plan.py` | 8 |
| `extracted_content_pipeline/support_ticket_input_package.py` | 28 |
| `extracted_content_pipeline/ticket_faq_input_contract.py` | 8 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 70 |
| `plans/PR-Deflection-Representative-Taxonomy-Source.md` | 158 |
| `tests/test_extracted_content_control_surface_api.py` | 8 |
| `tests/test_extracted_content_generation_plan.py` | 2 |
| `tests/test_extracted_content_ops_execution.py` | 4 |
| `tests/test_extracted_support_ticket_input_package.py` | 4 |
| `tests/test_extracted_ticket_faq_markdown.py` | 39 |
| **Total** | **432** |
