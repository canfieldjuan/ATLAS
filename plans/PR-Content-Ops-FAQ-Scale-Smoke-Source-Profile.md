# PR-Content-Ops-FAQ-Scale-Smoke-Source-Profile

## Why this slice exists

The 1,000-row FAQ run proved the generator can process large batches quickly,
but it also showed that source density matters: a raw archive can scan many
rows before finding enough usable ticket narratives. PR-Content-Ops-FAQ-Scale-
Smoke-Observability made failures visible after generation. Operators still
need a quick preflight profile that says how many upload rows were seen, how
many normalized into usable source opportunities, and which adapter warnings
explain skipped rows.

This keeps large-upload confidence grounded in the actual source file instead
of only the generated FAQ result.

## Scope (this PR)

1. Add an `input_profile` block to the FAQ scale-smoke summary.
2. Count raw rows for common CSV, JSONL, JSON array, and common JSON bundle
   shapes where available.
3. Use the public source adapter to report normalized usable row count and
   warning counts by code.
4. Preserve existing smoke exit behavior and FAQ generation behavior.
5. Extend focused scale-smoke tests for source-profile success and hard-failure
   visibility.

### Files touched

- `plans/PR-Content-Ops-FAQ-Scale-Smoke-Source-Profile.md`
- `scripts/smoke_content_ops_faq_scale_run.py`
- `tests/test_smoke_content_ops_faq_scale_run.py`

## Mechanism

The wrapper will compute a bounded preflight profile before launching the FAQ
Markdown CLI. The profile uses standard parsing only for raw-row counts, then
calls `load_source_campaign_opportunities_from_file` with the same source
format, text limit, and default fields the CLI receives. Adapter warnings are
summarized by code with a small sample for debugging.

If preflight parsing fails, the wrapper records `input_profile.status="error"`
and still runs the existing CLI so the CLI remains the source of truth for exit
behavior and stderr artifacts.

## Intentional

- No FAQ generator behavior changes.
- No fail-closed gate is added for density. This is visibility only.
- Raw-row counting is best-effort for arbitrary JSON objects; unsupported bundle
  shapes report `raw_row_count=null` instead of guessing.
- The full warning list is not duplicated into the summary; the profile keeps a
  bounded sample and counts by code.

## Deferred

- CFPB raw-archive scan count reporting remains exporter-specific work.
- Hosted UI display for source-profile diagnostics remains separate.

## Verification

- Focused scale-smoke + FAQ Markdown CLI tests passed, 78 tests.
- Full extracted pipeline wrapper passed, including 1,587 extracted Content Ops
  tests and 295 reasoning-core tests.
- Local PR review passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | 73 |
| Scale-smoke wrapper | 117 |
| Tests | 50 |
| **Total** | **240** |
