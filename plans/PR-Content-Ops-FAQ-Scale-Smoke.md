# PR-Content-Ops-FAQ-Scale-Smoke

## Why this slice exists

PR-Content-Ops-FAQ-Result-Output made the FAQ CLI write structured diagnostics,
but operators still need to assemble output paths, stderr capture, and exit-code
metadata by hand for each 500-row, 1,000-row, or larger upload. A small wrapper
should make scale runs repeatable without changing FAQ generation behavior.

## Scope (this PR)

1. Add a standalone FAQ scale-smoke wrapper that calls the existing Markdown
   builder with `--result-output`.
2. Write a standard artifact directory containing Markdown, result JSON,
   stdout, stderr, and a run summary.
3. Preserve fail-closed exit behavior while still leaving diagnostics on disk.
4. Add focused tests for successful, failed, format, and default smoke runs.

### Files touched

- `scripts/smoke_content_ops_faq_scale_run.py`
- `tests/test_smoke_content_ops_faq_scale_run.py`
- `plans/PR-Content-Ops-FAQ-Scale-Smoke.md`

## Mechanism

The new wrapper is intentionally thin. It shells out to
`scripts/build_extracted_ticket_faq_markdown.py`, passes through the common FAQ
size/configuration flags, and always writes:

Markdown, result JSON, stdout text, stderr text, and run-summary JSON artifacts.

The summary records the command, exit code, artifact paths, source path, and the
compact result JSON when present.

## Intentional

- No FAQ generation behavior changes.
- No CFPB-specific branch. The wrapper accepts any JSON, JSONL, or CSV source
  upload the FAQ CLI can already load.
- The wrapper uses the existing CLI as the source of truth instead of importing
  generator internals again.

## Deferred

- A later slice can add archive-specific extraction helpers if we want one
  command from raw CFPB archive to FAQ artifacts.
- A hosted UI for smoke artifacts remains separate from this CLI utility.

## Verification

- Focused scale-smoke + FAQ Markdown CLI tests passed, 75 tests.
- Manual 1,000-row CFPB scale-smoke run passed. Standard artifacts were written
  under `/tmp/content_ops_faq_scale_smoke_1000`, with `exit_code=0`,
  `source_count=1000`, `ticket_source_count=1000`, `generated=12`, and no
  failed output checks.
- Extracted gauntlet passed: manifest sync, reasoning import guard,
  standalone audit with 0 Atlas runtime import findings, and ASCII Python check.
- Full extracted pipeline wrapper passed, including 1,583 extracted Content Ops
  tests and 295 reasoning-core tests.
- Local PR review passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | 70 |
| Scale-smoke wrapper | 154 |
| Tests | 175 |
| **Total** | **399** |
