# PR-Content-Ops-FAQ-Scale-Smoke-Observability

## Why this slice exists

PR-Content-Ops-FAQ-Scale-Smoke made FAQ scale runs repeatable for any JSON,
JSONL, or CSV upload, but operators still have to inspect multiple artifacts to
answer the first debugging questions: how long did the run take, which files
were actually written, and whether a failure came from output checks or a hard
CLI error. Large uploads need that visibility in the summary artifact itself.

This slice adds a compact observability layer to the existing scale-smoke
wrapper so failure capture scales with upload size without changing FAQ
generation behavior.

## Scope (this PR)

1. Add elapsed-time metadata to the summary artifact.
2. Add artifact existence and byte-size metadata without removing the existing
   artifact path fields.
3. Add a compact failure summary that distinguishes output-check failures from
   hard CLI errors.
4. Extend focused scale-smoke tests for success, fail-closed, and hard-failure
   observability.

### Files touched

- `plans/PR-Content-Ops-FAQ-Scale-Smoke-Observability.md`
- `scripts/smoke_content_ops_faq_scale_run.py`
- `tests/test_smoke_content_ops_faq_scale_run.py`

## Mechanism

The wrapper will time the subprocess call with `time.monotonic()`, write stdout
and stderr as it does today, then build a summary with two additional fields:

- `timing`: elapsed seconds for the subprocess run.
- `artifact_details`: per-artifact path, existence, and byte count.
- `failure`: `null` on success, otherwise a compact object with the exit code,
  result status, failed output checks when available, and a bounded stderr tail.

The existing `artifacts` path/null map stays in place so callers that already
read those fields do not need to change.

## Intentional

- No FAQ generator behavior changes.
- No source-row counting is added for arbitrary CSV/JSON/JSONL here; the result
  JSON remains the source of truth for loaded ticket/source counts.
- The stderr copy in `failure` is bounded so a large upload cannot duplicate an
  unbounded log into the summary JSON.

## Deferred

- Raw-archive source-density reporting remains CFPB/exporter-specific work.
- A hosted artifact viewer remains separate from this CLI summary enrichment.

## Verification

- Focused scale-smoke + FAQ Markdown CLI tests passed, 76 tests.
- Full extracted pipeline wrapper passed, including 1,583 extracted Content Ops
  tests and 295 reasoning-core tests.
- Local PR review passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | 71 |
| Scale-smoke wrapper | 90 |
| Tests | 22 |
| **Total** | **183** |
