# FAQ Source ID Execute Smoke

## Why this slice exists

#1118 made the New Run UI write selected FAQ report IDs into
`inputs.source_faq_ids`, and #1116 made the Atlas input provider load those IDs
through a tenant-scoped FAQ repository. The remaining proof is the execute
route: a selected saved FAQ ID should load the draft, normalize it as
support-ticket-derived source material, and feed landing/blog generation with
the selected FAQ's customer wording and resolution evidence.

This slice closes that validation gap without adding another product surface.

## Scope (this PR)

Ownership lane: content-ops/faq-output-ingestion
Slice phase: Functional validation

1. Add execute-route smoke coverage for `inputs.source_faq_ids` on both
   `landing_page` and `blog_post` outputs.
2. Use the existing Atlas input-provider repository factory seam to supply a
   saved `TicketFAQDraft` fixture and assert tenant-scoped `get_draft(...)`
   lookup.
3. Assert the selected FAQ output reaches the generated landing/blog context
   through derived fields: FAQ question, draft-ID trace, customer wording, and
   resolution evidence.

### Files touched

- `tests/test_support_ticket_provider_landing_blog_execute.py`
- `plans/PR-FAQ-Source-ID-Execute-Smoke.md`

## Mechanism

The test creates the real Content Ops control-surface router with
`build_content_ops_input_provider(pool_provider=..., faq_repository_factory=...)`
and the existing capture services used by support-ticket landing/blog execute
tests. A small fake FAQ repository returns a `TicketFAQDraft` from the supplied
pool and records the `(faq_id, account_id)` lookup.

The execute request sends only:

```python
{"outputs": [output], "inputs": {"source_faq_ids": [FAQ_DRAFT_ID]}}
```

No inline `source_material` is provided. Passing the test proves the selected ID
path itself supplies the source rows that execute needs.

## Intentional

- This is a route-level smoke using the repository seam, not a live Postgres
  integration. The Postgres repository SQL contract is already covered in the
  FAQ repository tests; this slice proves the Content Ops execute wiring that
  #1118 exposed to users.
- No production code changes are expected. If the smoke fails, the failing
  integration point will be fixed in this slice.
- The test covers both landing and blog outputs because the UI exposes saved
  FAQ selection for both.
- Local review's cross-layer caller hints are from new test-helper names
  (`_FAQRepo`, `_saved_faq_draft`) colliding with similar helper names in other
  tests; no production symbol or shared function changes in this slice.

## Deferred

- Future PR: hosted/live Postgres execute proof if operators need an
  environment-backed runbook artifact rather than in-process route coverage.
- Future PR: richer saved-FAQ picker with search/status filters if operators
  need more than the recent list.
- Parked hardening: none.

## Verification

Run locally:

- Command: python -m pytest tests/test_support_ticket_provider_landing_blog_execute.py::test_selected_faq_id_feeds_execute_context -q
  - 2 passed
- Command: python -m pytest tests/test_support_ticket_provider_landing_blog_execute.py tests/test_atlas_content_ops_input_provider.py -q
  - 32 passed, 1 warning
- Command: python -m py_compile tests/test_support_ticket_provider_landing_blog_execute.py
  - passed
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-source-id-execute-smoke.md
  - passed

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Execute-route smoke test | ~136 |
| Plan doc | ~88 |
| **Total** | **~224** |
