## Why this slice exists

`PR-Deflection-Question-Evidence-Scoping` fixed the cross-resolution bleed that
large SaaS validation exposed, and the follow-up copy slices made the proven
drafts buyer-readable. The remaining safety gap is explicit enforcement: the
generator now scopes resolution rows correctly, but the output checks do not
fail closed if a future change marks an item as `resolution_evidence` while its
resolution rows are outside the selected question scope.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection-backend

Slice phase: Production hardening

1. Add item-level resolution evidence scope metadata for generated FAQ items.
2. Add an output check, `resolution_evidence_scoped`, that fails when a proven
   item is not scoped to its own resolution evidence.
3. Update CLI diagnostics and contract fixtures for the additional output
   check.
4. Add focused positive and negative tests, including a direct malformed-item
   fixture proving the checker failure branch fires.
5. Update FAQ search/detail contract fixtures so the route checker accepts the
   producer's new item metadata.
6. Address review feedback by failing closed when resolution evidence exists
   but the selected question has no evidence scope to compare.

### Files touched

- `extracted_content_pipeline/ticket_faq_markdown.py`
- `scripts/build_extracted_ticket_faq_markdown.py`
- `scripts/check_content_ops_faq_search_route_contract.py`
- `scripts/smoke_content_ops_faq_output_proof.py`
- `scripts/smoke_content_ops_faq_search_concurrency.py`
- `docs/extraction/validation/content_ops_faq_search_route_contract_handoff.md`
- `tests/test_extracted_ticket_faq_markdown.py`
- `tests/test_content_ops_faq_report_contract_docs.py`
- `tests/test_content_ops_faq_saas_demo_corpus.py`
- `tests/test_check_content_ops_faq_search_route_contract.py`
- `tests/test_extracted_content_ops_execution_smoke.py`
- `tests/test_atlas_content_ops_input_provider.py`
- `tests/test_smoke_content_ops_cfpb_faq_markdown.py`
- `tests/test_smoke_content_ops_faq_lifecycle.py`
- `tests/test_smoke_content_ops_faq_output_proof.py`
- `tests/test_smoke_content_ops_faq_scale_run.py`
- `tests/test_smoke_content_ops_faq_search_route_concurrency.py`
- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `docs/frontend/content_ops_faq_report_example.json`
- `plans/PR-Deflection-Evidence-Scope-Gate.md`

## Mechanism

The item builder already computes the selected question row, mixed-scope state,
and scoped resolution rows before producing steps. This slice records a compact
`resolution_evidence_scope` status on each item:

- `scoped` for resolution-backed drafts whose resolution rows match the selected
  question scope.
- `not_applicable` for review-needed drafts without resolution evidence.
- fail-closed statuses such as `mixed_evidence_scope`, `missing_question_scope`,
  or `scope_mismatch` for malformed proven states.

The output-check builder then adds `resolution_evidence_scoped`, which requires every
`resolution_evidence` item to carry the `scoped` status. The CLI and output
proof smoke include the new check in their fail-closed checks, and
`_output_check_hint()` gets a specific message for the new check.

Because item detail payloads now expose a new compact status field, the FAQ
search route contract checker admits `resolution_evidence_scope` as an expected
string field and the handoff doc names it in the detail item shape.

## Intentional

- This does not change grouping, ranking, answer copy, paywall behavior, or
  macro writeback. It adds an explicit detector for a safety invariant that the
  current generator already satisfies.
- The negative fixture tests the checker directly with a malformed item instead
  of weakening the generator to create an invalid state.
- A generator-level negative fixture covers the source-policy fallback case:
  resolution evidence without a scoped selected question now yields
  `missing_question_scope` and fails `resolution_evidence_scoped`.
- The new item metadata is compact status-only data; it does not expose
  resolution text, raw evidence, or Markdown in snapshot payloads.
- The broader fixture updates are shape synchronization only; they keep
  existing support-ticket, CFPB, execution, lifecycle, and search-route tests on
  the same output-check contract.

## Deferred

- Follow-up slice: richer final help-center prose over verified resolution
  evidence.
- Considered hardening: the existing SaaS demo preflight dotenv entry in
  `HARDENING.md` is unrelated to this detector and remains parked.
- Parked hardening: none.

## Verification

- `pytest tests/test_extracted_ticket_faq_markdown.py -q` -- 154 passed.
- `pytest tests/test_extracted_ticket_faq_markdown.py tests/test_content_ops_faq_report_contract_docs.py tests/test_content_ops_deflection_report.py tests/test_smoke_content_ops_faq_output_proof.py -q` -- 184 passed.
- `pytest tests/test_content_ops_faq_saas_demo_corpus.py tests/test_check_content_ops_faq_search_route_contract.py tests/test_smoke_content_ops_faq_scale_run.py tests/test_smoke_content_ops_faq_lifecycle.py tests/test_smoke_content_ops_cfpb_faq_markdown.py tests/test_extracted_content_ops_execution_smoke.py tests/test_atlas_content_ops_input_provider.py tests/test_extracted_ticket_faq_markdown.py tests/test_content_ops_faq_report_contract_docs.py tests/test_content_ops_deflection_report.py tests/test_smoke_content_ops_faq_output_proof.py -q` -- 379 passed, 1 warning.
- `pytest tests/test_smoke_content_ops_faq_search_concurrency.py tests/test_smoke_content_ops_faq_search_route_concurrency.py tests/test_extracted_ticket_faq_markdown.py tests/test_check_content_ops_faq_search_route_contract.py tests/test_smoke_content_ops_faq_output_proof.py -q` -- 342 passed.
- Direct 420-row SaaS sample generator check -- 8 items, 7 proven,
  `resolution_evidence_scoped=True`, no unscoped proven answers.
- `scripts/validate_extracted_content_pipeline.sh` run with bash -- passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -- passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -- passed.
- `scripts/check_ascii_python.sh` run with bash -- passed.
- Full extracted mirror with live deflection env vars blanked before
  `scripts/run_extracted_pipeline_checks.sh` -- 2900 passed, 10 skipped, 1
  known local failure in
  `tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py::test_script_preflight_uses_atlas_db_settings_fallback`
  from the parked SaaS demo preflight dotenv hardening entry.
- `scripts/local_pr_review.sh` with the current PR body -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| `extracted_content_pipeline/ticket_faq_markdown.py` | ~45 |
| CLI/search contract scripts | ~15 |
| Tests and contract docs | ~100 |
| Frontend JSON examples | ~10 |
| Plan doc | ~100 |
| **Total** | **~285** |

Under the 400 LOC soft cap.
