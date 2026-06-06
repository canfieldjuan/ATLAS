# Post-586 Content Ops Closeout

## Why this slice exists

PR #586 merged the Content Ops CSV ingestion file loader, but the coordination
ledger still showed that work as pending. That stale claim makes later sessions
avoid a hot zone that is no longer active.

## Scope (this PR)

Clear the post-merge coordination state for Content Ops ingestion file loading.

### Files touched

- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Post-586-Content-Ops-Closeout.md`

## Mechanism

- Remove the pending CSV ingestion UI row from the in-flight ledger.
- Move `extracted_content_pipeline` state from PR #585 / pending to PR #586 / no
  active PRs.
- Record that hosted ingestion file loading now covers JSON, JSONL, NDJSON, and
  CSV.

## Intentional

This is a docs-only coordination closeout. It does not touch frontend code,
pipeline runtime code, tests, or the hosted ingestion contract.

## Deferred

Further Content Ops ingestion slices should wait for a real host source export
fixture or a concrete extracted_reasoning_core productization need. The active
backlog does not currently justify speculative source adapters.

## Verification

- `git diff --check`
- Local PR review hook

## Estimated diff size

| Area | Estimate |
|---|---:|
| Coordination docs | 2 line edits |
| Plan doc | ~40 lines |
| **Total** | ~55 |
