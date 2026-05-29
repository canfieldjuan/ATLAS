# FAQ Output By ID Selection

## Why this slice exists

#1114 lets Content Ops reuse a saved FAQ report when the caller passes the full
FAQ draft payload as `inputs.source_material`. That proves the ingestion path,
but it still makes the caller carry the whole saved report around.

The next thin product path is tenant-scoped selection by saved FAQ draft ID:
when a request names saved FAQ IDs in `inputs`, the Atlas host provider should
fetch those drafts for the authenticated account and hand their payloads to the
same FAQ-output source adapter. This keeps the database and authorization
concern at the host boundary instead of teaching blog or landing generators how
to read FAQ storage.

## Scope (this PR)

Ownership lane: content-ops/faq-output-ingestion
Slice phase: Vertical slice

1. Add `inputs.source_faq_ids` support to the Atlas Content Ops input provider.
2. Fetch selected FAQ drafts through `PostgresTicketFAQRepository.get_draft(...)`
   using the existing tenant `TenantScope`.
3. Route loaded draft payloads through the existing support-ticket/FAQ-output
   package path.
4. Wire the hosted API mount with the existing DB pool provider.
5. Add focused tests for tenant-scoped repository lookup, missing IDs, and the
   API preview route.

### Files touched

- `atlas_brain/_content_ops_input_provider.py`
- `atlas_brain/api/__init__.py`
- `tests/test_atlas_content_ops_input_provider.py`
- `plans/PR-FAQ-Output-By-ID-Selection.md`

## Mechanism

The host input provider gains optional repository wiring:

```python
build_content_ops_input_provider(pool_provider=get_db_pool)
```

When `inputs.source_faq_ids` contains one or more IDs, the provider resolves the
pool, builds a `PostgresTicketFAQRepository`, and calls `get_draft(faq_id,
scope=scope)` for each ID. Drafts found under that tenant are converted with
`TicketFAQDraft.as_dict()` and passed as source material to the existing
`SupportTicketInputProvider`.

Missing IDs are reported as provider warnings and do not leak cross-tenant
detail. If no selected draft is found and no other support-ticket material is
present, the provider returns a warning-bearing noop package so preview/plan can
surface the selection problem without silently running on unrelated inputs.

## Intentional

- This does not add the UI picker. It only makes the hosted request contract
  capable of selecting saved FAQ reports by ID.
- This does not add a new top-level request field. IDs live under `inputs` so
  the existing bounded request model can carry them.
- This does not fetch by ID in the extracted package. Tenant-scoped database
  reads remain host-owned; extracted code continues consuming already-loaded
  source material.

## Deferred

- Future PR: add Intel UI controls that let a user pick saved FAQ reports and
  send `inputs.source_faq_ids`.
- Future PR: support deliberate mixed inline `source_material` plus
  `source_faq_ids` composition if the UI needs both in one run.
- Future PR: add execute-route smoke coverage once the UI/API picker is in
  place and representative saved drafts exist in the test fixture layer.
- Parked hardening: none.

## Verification

Ran locally:

- Command: python -m pytest tests/test_atlas_content_ops_input_provider.py -q
  - 18 passed, 1 warning
- Command: python -m py_compile atlas_brain/_content_ops_input_provider.py tests/test_atlas_content_ops_input_provider.py
  - passed
- Command: git diff --check
  - passed
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-output-by-id-selection.md
  - passed

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Host provider by-ID loader | ~120 |
| Hosted API wiring | ~5 |
| Focused tests | ~120 |
| Plan doc | ~90 |
| **Total** | **~335** |
