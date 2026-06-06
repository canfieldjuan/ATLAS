# PR-Content-Ops-FAQ-Hosted-Bundle-IO-Smoke

## Why this slice exists

PR-Content-Ops-FAQ-Hosted-Bulk-IO-Smoke proved the hosted execute route can run
1,000 direct FAQ source rows through the real generator, and separately proved
the API input validator accepts 1,000 rows in the one-level
`source_material.support_tickets` bundle shape. The remaining test gap is the
real generator path for that bundle shape: a visitor or UI adapter can send the
catalog-style bundle form and still get a grounded FAQ artifact at 1,000 rows.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-io-tests

1. Add a hosted execute smoke that sends
   `inputs.source_material.support_tickets` with 1,000 support-ticket rows to
   the real `TicketFAQMarkdownService`.
2. Assert the output preserves the 1,000-row source count, passes output checks,
   and retains first/last source IDs in the generated FAQ item.

### Files touched

- `plans/PR-Content-Ops-FAQ-Hosted-Bundle-IO-Smoke.md`
- `tests/test_extracted_content_ops_live_execute_harness.py`

## Mechanism

The new test mirrors the direct-list bulk smoke but wraps the rows in the
one-level bundle shape:

```python
"source_material": {"support_tickets": source_rows}
```

The route should normalize that bundle into FAQ rows, pass it through
`TicketFAQMarkdownService`, and report 1,000 ticket sources in the compact
result payload.

## Intentional

- This is test-only. The code path should already work after the validator and
  generator fixes in the prior slices; the slice locks the behavior rather than
  changing production code.
- The test uses in-process route execution instead of a real HTTP server so the
  assertion stays deterministic and cheap while still exercising the hosted
  execute route contract.

## Deferred

- Real browser file-upload coverage remains deferred until the UI upload path is
  the active slice; this test covers the API payload shape the UI adapter should
  produce.
- Real database lifecycle coverage remains in the existing smoke scripts rather
  than CI unit tests to avoid making the package suite depend on local database
  availability.

## Verification

- `pytest tests/test_extracted_content_ops_live_execute_harness.py -q` - 5 passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - 1818 passed, 1 skipped.
- `bash scripts/local_pr_review.sh --allow-dirty` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~65 |
| Hosted bundle route test | ~55 |
| **Total** | **~120** |

Actual diff: +118 / -0.
