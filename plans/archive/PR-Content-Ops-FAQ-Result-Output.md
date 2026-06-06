# PR-Content-Ops-FAQ-Result-Output

## Why this slice exists

The 1,000-row CFPB FAQ run proved the generator can process large uploads, but
the failure evidence was collected manually after reruns. Future 500-row,
1,000-row, or larger uploads should leave a structured result artifact even when
`--require-output-checks` exits nonzero, so operators can see which check failed
and where to inspect first.

## Scope (this PR)

1. Add a `--result-output` option to the standalone FAQ Markdown CLI.
2. Write a compact JSON run artifact on both successful and fail-closed runs.
3. Include small diagnostics for failed checks, item distribution, question
   source distribution, warnings, and output paths.
4. Add focused CLI tests for success and failure result artifacts.

### Files touched

- `scripts/build_extracted_ticket_faq_markdown.py`
- `tests/test_extracted_ticket_faq_markdown.py`
- `plans/PR-Content-Ops-FAQ-Result-Output.md`

## Mechanism

The CLI already receives a `TicketFAQMarkdownResult` before it decides whether
`--require-output-checks` should fail. This slice serializes a compact run
summary immediately after generation and before raising `SystemExit` for failed
checks.

The artifact intentionally excludes full Markdown body text. It records counts,
output checks, failed checks, warning summaries, item summaries, question source
counts, ticket-count distribution, CLI config, input path, and output path.

## Intentional

- No FAQ generation behavior changes.
- No new runtime dependency or persistence dependency.
- The result artifact is compact by default so very large uploads do not produce
  duplicate Markdown-sized JSON.
- The CLI still exits nonzero when `--require-output-checks` fails.

## Deferred

- A later slice can add a dedicated scale-smoke wrapper that extracts source
  rows and calls this CLI with a standard artifact directory.
- A later slice can surface these diagnostics in a hosted review UI if operators
  need browser access to run artifacts.

## Verification

- Focused FAQ Markdown pytest - passed, 62 tests.
- Manual 1,000-row CFPB CLI check with `--result-output` - passed. Result JSON
  reported `source_count=1000`, `ticket_source_count=1000`, `generated=12`,
  no failed output checks, and `rendered_ticket_source_count=1000`.
- Extracted gauntlet passed: manifest sync, reasoning import guard,
  standalone audit with 0 Atlas runtime import findings, and ASCII Python check.
- Full extracted pipeline wrapper - passed, including 1,581 extracted Content
  Ops tests and 295 reasoning-core tests.
- Local PR review - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | 72 |
| FAQ CLI | 132 |
| Tests | 49 |
| **Total** | **251** |
