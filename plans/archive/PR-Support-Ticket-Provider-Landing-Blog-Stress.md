# PR-Support-Ticket-Provider-Landing-Blog-Stress

## Why this slice exists

The support-ticket input provider now feeds FAQ Markdown, landing-page, and
blog-post generation. The happy path and live smoke are covered, but the route
handoff still needs robust testing for larger host-loaded ticket exports and
parallel requests before we claim the path can survive real customer files.

This slice stays in the support-ticket provider lane. It does not change FAQ
generation, hosted upload persistence, or generated content quality.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: Robust testing

1. Add offline route-level stress coverage for loader-backed support-ticket
   source material.
2. Prove preview/plan/execute keep provider diagnostics accurate for large
   loaded inputs.
3. Prove landing-page and blog-post execute calls receive bounded provider-built
   inputs under concurrent requests.

### Files touched

- `plans/PR-Support-Ticket-Provider-Landing-Blog-Stress.md`
- `tests/test_support_ticket_provider_landing_blog_execute.py`

## Mechanism

The tests use `SupportTicketInputProvider(source_material_loader=...)` instead
of inline `inputs.source_material`, because inline requests over 1,000 rows are
already rejected before provider packaging. The loader returns reusable
support-ticket-shaped rows at 1,000, 10,000, and 50,000 rows. The provider keeps
only the first 1,000 rows, reports `ticket_rows_truncated` when applicable, and
the route surfaces bounded diagnostics on preview, plan, and execute responses.

Execute tests use deterministic fake landing-page and blog-post services so the
stress probe does not call an LLM, database, or external provider.

## Intentional

- No live LLM stress: this tests route/provider/generation handoff, not model
  throughput or vendor rate limits.
- No hosted file-upload changes: persisted upload lookup and background-job
  policy are separate ingestion/host concerns.
- No FAQ generator changes: FAQ generation is owned by the parallel FAQ
  session.

## Deferred

- Future PR: hosted upload/background execution policy for large customer files
  once persisted support-ticket uploads are wired into the provider loader.
- Future PR: generated content quality evaluation using real support-ticket
  datasets after route survivability is proven.
- Parked hardening: none. `HARDENING.md` was scanned; the current FAQ scale
  entry is owned by the FAQ generation lane and is not required for this
  support-ticket provider handoff stress slice.

## Verification

- `python -m pytest tests/test_support_ticket_provider_landing_blog_execute.py -q`
  - passed, 9 tests.
- `python -m pytest tests/test_extracted_support_ticket_input_provider.py tests/test_atlas_content_ops_input_provider.py -q`
  - passed, 24 tests, 1 existing environment warning from `torch`/`pynvml`.
- `scripts/local_pr_review.sh --allow-dirty`
  - passed.
- `scripts/local_pr_review.sh`
  - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| Route stress tests | ~215 |
| **Total** | **~290** |
