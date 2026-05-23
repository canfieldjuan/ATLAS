# PR-Content-Ops-FAQ-Vocab-Gap-IO-Smoke

## Why this slice exists

The FAQ vocabulary-gap feature is now wired through generator logic, hosted
execution config, UI controls, execute summaries, and catalog input contracts.
Before continuing feature work, we need a thin input/output proof that the real
hosted execute route accepts the vocabulary-gap inputs and returns an FAQ output
with vocabulary mappings.

This slice pauses feature expansion and tests the real request-to-result flow.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-io-tests

1. Add a focused hosted execute-route test for `faq_markdown`.
2. Send `faq_documentation_terms` and `faq_vocabulary_gap_rules` through the
   request `inputs` object.
3. Assert the plan config preserves those inputs and the execution output
   includes generated FAQ items, source IDs, output checks, Markdown, and the
   expected term mapping.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Vocab-Gap-IO-Smoke.md` | Plan doc for this input/output smoke slice. |
| `tests/test_extracted_content_ops_live_execute_harness.py` | Adds the hosted FAQ vocabulary-gap execute-route test. |

## Mechanism

The existing live execute harness tests already call the mounted
`/content-ops/execute` route directly. This slice adds a FAQ-only route test
with host-injected `ContentOpsExecutionServices(faq_markdown=TicketFAQMarkdownService())`.

The payload uses inline support-ticket source material plus the same input keys
the UI sends:

```python
"faq_documentation_terms": ["Single sign-on setup"],
"faq_vocabulary_gap_rules": [["SSO", "single sign-on"]],
```

The assertion inspects both sides of the flow:

- `plan.steps[0].config` proves request inputs reached the execution plan.
- `steps[0].result.items[0].term_mappings` proves the generated output used
  those inputs.

## Intentional

- No generator behavior changes. This is an input/output proof over existing
  behavior.
- No new script or fixture file. The source material is small enough to live in
  the test.
- No UI test in this slice; PR-Content-Ops-FAQ-Catalog-Driven-Inputs already
  covered catalog display precedence.

## Deferred

- Larger-upload hosted smoke coverage remains a future slice. This test proves
  the vocabulary-gap input/output seam, not 1,000-row scale behavior.
- CLI/result JSON coverage is already present in the FAQ generator tests and is
  not duplicated here.
- Current `HARDENING.md` entries were scanned; no root hardening items are
  parked for this FAQ lane.

## Verification

- `python -m pytest tests/test_extracted_content_ops_live_execute_harness.py::test_live_execute_route_accepts_faq_vocabulary_gap_inputs -q` - passed, 1 test.
- `python -m pytest tests/test_extracted_content_ops_live_execute_harness.py -q` - passed, 3 tests.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed, 0 findings.
- `bash scripts/check_ascii_python.sh` - passed.
- `git diff --check` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - passed, 1780 tests, 1 skipped, 1 existing `torch`/`pynvml` warning.
- `bash scripts/local_pr_review.sh origin/main --allow-dirty` - passed.
- Reviewer readability comment update: focused hosted FAQ IO pytest still
  passed, 1 test; `git diff --check` passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| Hosted execute-route test | ~70 |
| **Total** | ~150 |
