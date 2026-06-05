# Content Ops Source Bundle Ingest

## Why This Slice Exists

AI Content Ops source-row ingest supports many individual row types, but a
customer bundle JSON object with multiple collections only loads the first
recognized list key. A host export that contains reviews and support tickets in
one file can silently drop the later collections.

## Scope

- Flatten multiple recognized source collections from one JSON object.
- Support nested `sources` objects that group collections under one account.
- Inherit safe top-level account/vendor metadata into child rows without
  overriding child fields.
- Add focused source-adapter tests and refresh the active coordination row.

## Mechanism

`campaign_source_adapters._load_source_rows()` will delegate JSON object loading
to a small bundle helper. The helper collects all recognized row-list keys,
recurses into nested source bundle objects, and merges safe scalar parent fields
into child mapping rows.

## Intentional

- No CLI flag changes.
- No target-mode or generation behavior changes.
- No attempt to define a full customer-bundle schema.

## Deferred

- Explicit versioned source-bundle schema.
- Per-collection source-type overrides.
- Frontend bundle upload UX.

## Verification

- Focused source-adapter tests.
- Python compile check for touched source-adapter files.
- Git diff whitespace check.
- Local PR review wrapper.

### Files Touched

- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/campaign_source_adapters.py`
- `tests/test_extracted_campaign_source_adapters.py`
- `plans/PR-Content-Ops-Source-Bundle-Ingest.md`

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Source bundle loader helper | ~70 |
| Tests | ~105 |
| Coordination | ~5 |
| Plan doc | ~45 |
| **Total** | ~225 |
