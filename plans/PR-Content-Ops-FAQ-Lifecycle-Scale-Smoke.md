# PR-Content-Ops-FAQ-Lifecycle-Scale-Smoke

## Why this slice exists

The FAQ product now has 1,000-row proof for hosted direct input, hosted bundle
input, and file-backed CLI/artifact generation. The next confidence gap is
lifecycle behavior after generation: the artifact must persist as a reviewable
FAQ draft, export as a draft, update review status, and export again after the
status change without losing scale metadata.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-io-tests

1. Add a 1,000-row JSON support-ticket bundle lifecycle smoke test using the
   existing lifecycle script and fake async pool.
2. Assert generation persists one FAQ draft with 1,000 source rows and 1,000
   ticket sources.
3. Assert draft and reviewed exports preserve source counts, rendered item
   source coverage, and review status.

### Files touched

- `plans/PR-Content-Ops-FAQ-Lifecycle-Scale-Smoke.md`
- `tests/test_smoke_content_ops_faq_lifecycle.py`

## Mechanism

The test writes a temporary JSON file:

```json
{"support_tickets": [{... 1000 rows ...}]}
```

It then runs `run_faq_lifecycle_smoke(...)`, which uses the production
source-file loader, `TicketFAQMarkdownService`, `PostgresTicketFAQRepository`,
`export_ticket_faq_drafts(...)`, and status-update path. The test fakes only the
async pool so CI does not require a live database.

## Intentional

- This is test-only. The lifecycle script and repository path already exist; the
  slice locks scale behavior across that path.
- The fake pool remains the right CI boundary. Real database execution stays in
  smoke/manual environments where `EXTRACTED_DATABASE_URL` and migrations are
  available.

## Deferred

- A live database 1,000-row lifecycle run is deferred to the manual smoke lane;
  it should use the same script with a real `--database-url`.
- Browser upload coverage remains deferred until the UI upload path is active.

## Verification

- `pytest tests/test_smoke_content_ops_faq_lifecycle.py -q` - 5 passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - 1820 passed, 1 skipped.
- `bash scripts/local_pr_review.sh --allow-dirty` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~65 |
| Lifecycle 1,000-row scale test | ~55 |
| **Total** | **~120** |
