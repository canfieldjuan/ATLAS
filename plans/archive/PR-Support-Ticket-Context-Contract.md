# Support Ticket Context Contract

## Why this slice exists

#966 closed the live Haiku blog gate validation, but the reviewer flagged a
real drift risk: the support-ticket data-context markers are duplicated across
the executor, smoke harness, and evaluator. Those strings decide whether the
support-ticket gate runs and whether undated uploads are allowed to mention
calendar windows or cadence claims, so drift here can create false passes.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider
Slice phase: Production hardening

1. Add one shared support-ticket context contract for canonical topic type,
   source marker, uploaded-source periods, count keys, and cluster keys.
2. Update the support-ticket executor, input package, blog gate predicate, live
   smoke payload, and generated-content evaluator to consume the shared
   contract instead of retyping the same markers.
3. Add focused tests that prove the uploaded-ticket predicate and support-ticket
   context predicate still fire from each marker and reject lookalikes.
4. Enroll the new contract test in the extracted pipeline runner and workflow
   path filters so CI gates it.

### Files touched

- `extracted_content_pipeline/support_ticket_context_contract.py`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `extracted_content_pipeline/blog_generation.py`
- `extracted_content_pipeline/support_ticket_generated_content_eval.py`
- `scripts/smoke_content_ops_live_generation.py`
- `tests/test_support_ticket_context_contract.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Support-Ticket-Context-Contract.md`
- existing focused tests as needed

## Mechanism

The new contract module owns constants such as
`SUPPORT_TICKET_TOPIC_TYPE`, `SUPPORT_TICKET_SOURCE`,
`UPLOADED_SUPPORT_TICKETS_SOURCE_PERIOD`, and
`UPLOADED_TICKETS_REVIEW_PERIOD`, plus small predicates:

```python
is_uploaded_ticket_context(context)
is_support_ticket_context(context)
```

Callers keep their current behavior but import the same contract, so changing a
canonical marker later updates the executor, smoke harness, and evaluator
together.

## Intentional

- This does not change generated copy, prompt rules, LLM routing, or the FAQ
  generator's output shape. It only consolidates the support-ticket context
  markers used by this lane.
- The broader cadence-regex false-positive trade-off from #966 remains
  intentional. This slice keeps the same detection behavior.

## Deferred

- Future PR: adapt the contract if the FAQ output schema changes from the
  current support-ticket input package shape to a richer FAQ article shape.
- Parked hardening: none.

## Verification

Ran locally:

- `python -m pytest tests/test_support_ticket_context_contract.py tests/test_evaluate_support_ticket_generated_content.py tests/test_extracted_blog_generation.py tests/test_extracted_content_ops_execution.py tests/test_smoke_content_ops_live_generation.py tests/test_extracted_support_ticket_input_package.py -q`
  - 209 passed
- `python extracted_content_pipeline/support_ticket_generated_content_eval.py --help`
  - passed
- `python -m py_compile extracted_content_pipeline/support_ticket_context_contract.py extracted_content_pipeline/content_ops_execution.py extracted_content_pipeline/blog_generation.py extracted_content_pipeline/support_ticket_generated_content_eval.py extracted_content_pipeline/support_ticket_input_package.py scripts/smoke_content_ops_live_generation.py tests/test_support_ticket_context_contract.py`
  - passed
- `python scripts/audit_extracted_pipeline_ci_enrollment.py`
  - passed; 116 matching tests are enrolled
- `python -m pytest tests/test_support_ticket_context_contract.py tests/test_extracted_pipeline_route_ci_contract.py tests/test_audit_extracted_pipeline_ci_enrollment.py -q`
  - 42 passed
- `python -m pytest tests/test_support_ticket_context_contract.py tests/test_evaluate_support_ticket_generated_content.py tests/test_extracted_blog_generation.py tests/test_extracted_content_ops_execution.py tests/test_smoke_content_ops_live_generation.py tests/test_extracted_support_ticket_input_package.py tests/test_extracted_pipeline_route_ci_contract.py tests/test_audit_extracted_pipeline_ci_enrollment.py -q`
  - 221 passed
- `bash scripts/local_pr_review.sh`
  - passed

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Shared contract | ~70 |
| Caller wiring | ~80 |
| Focused tests | ~95 |
| CI enrollment | ~5 |
| Plan doc | ~85 |
| **Total** | **~335** |
