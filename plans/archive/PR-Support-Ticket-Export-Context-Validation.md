# Support Ticket Export Context Validation

## Why this slice exists

The support-ticket CSV live smoke can now generate landing-page and blog-post drafts
from the shared support-ticket input package. The next validation gap is the saved
draft export artifact: a run could pass because generation returned a saved id while
the exported row lost the package-derived support-ticket context we need to inspect.

This slice keeps the proof in the smoke harness, where operators already request
`--support-ticket-csv` plus `--export-saved-draft`.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: Functional validation

1. Add support-ticket export-context validation to the live generation smoke.
2. Validate landing-page exports against `metadata.source_context`.
3. Validate blog-post exports against `data_context`.
4. Cover passing and failing export-context cases for both supported outputs.

### Files touched

- `scripts/smoke_content_ops_live_generation.py`
- `tests/test_smoke_content_ops_live_generation.py`
- `plans/PR-Support-Ticket-Export-Context-Validation.md`

## Mechanism

When the smoke runs in support-ticket CSV mode and `--export-saved-draft` is set,
the script compares the exported saved draft row against the package-enriched
payload inputs after generation. It checks durable context fields such as source
row count, included ticket row count, question-like row count, and top ticket
clusters. Landing-page export validation also checks skipped/truncated counts
because `metadata.source_context` receives the provider source context directly.
Blog export validation also checks source period because blog rows carry that
value in `data_context`.

Landing-page export rows are validated at `metadata.source_context`. Blog-post
export rows are validated at `data_context`, where the blueprint context is already
expected to survive.

## Intentional

- Validation only runs for support-ticket CSV mode plus saved-draft export.
- The smoke remains read-only after generation; it does not mutate drafts to fix
  missing context.
- The helper validates the exported row shape instead of the in-memory generation
  result because the exported artifact is what operators inspect.

## Deferred

- Parked hardening: none added by this slice.
- Existing parked hardening `FAQSCALE-1` remains owned by
  `content-ops/faq-generation-scale`; this slice does not touch FAQ generation or
  hosted large-upload backpressure.

## Verification

- `python -m pytest tests/test_smoke_content_ops_live_generation.py::test_live_generation_smoke_validates_support_ticket_landing_export_context tests/test_smoke_content_ops_live_generation.py::test_live_generation_smoke_fails_when_support_ticket_landing_export_context_missing tests/test_smoke_content_ops_live_generation.py::test_live_generation_smoke_validates_support_ticket_blog_export_context tests/test_smoke_content_ops_live_generation.py::test_live_generation_smoke_fails_when_support_ticket_blog_export_context_drifts -q`
  - 4 passed.
- `python -m pytest tests/test_smoke_content_ops_live_generation.py -q`
  - 28 passed.
- `python -m pytest tests/test_extracted_support_ticket_input_package.py tests/test_extracted_support_ticket_input_provider.py tests/test_smoke_content_ops_support_ticket_package.py tests/test_support_ticket_provider_landing_blog_execute.py -q`
  - 44 passed.
- Py compile for `scripts/smoke_content_ops_live_generation.py`
  - Passed.
- Local PR review wrapper
  - Passed.
- Review comment fix: narrowed blog export expected context to the fields the
  seeded support-ticket blog blueprint actually persists.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| Smoke helper | ~90 |
| Tests | ~205 |
| **Total** | **~370** |

Target under 400 LOC. This is a focused smoke/test change with no production
route or schema changes.
