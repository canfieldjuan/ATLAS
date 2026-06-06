# PR-Blog-GEO-Repair-Loop

## Why this slice exists

PR-Blog-GEO-Quality-Gate blocks generated blog drafts that miss the
draft-level GEO contract. Blocking is useful, but the generator should get one
targeted chance to fix a parsed draft before the row is skipped.

This slice adds a save-time repair prompt for blog drafts that parse cleanly
but fail the blog quality gate.

## Scope (this PR)

1. Reuse the existing parse-retry budget for one quality repair attempt.
2. Add a targeted quality-repair prompt for blog quality blockers.
3. Preserve accumulated generation usage across the initial draft and repair.
4. Record repair-attempt metadata on saved blog drafts.
5. Add tests for successful repair and no-budget blocking behavior.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-GEO-Repair-Loop.md` | Plan doc for this slice. |
| `extracted_content_pipeline/blog_generation.py` | Add quality repair prompt and retry path. |
| `tests/test_extracted_blog_generation.py` | Cover repair success and no-budget block behavior. |

## Mechanism

The generator already has a parse retry budget. This PR spends any unused retry
budget on a quality repair. If the first response parses successfully but fails
the quality gate, and at least one retry remains, the service asks the model to
return the full blog JSON again while fixing the listed blockers.

If parsing already consumed the retry budget, the quality failure still blocks
without another LLM call.

## Intentional

- No new default budget multiplier.
- No repair loop for unparseable responses beyond the existing parse retry.
- No frontend or publish-level GEO checks.
- No changes to non-blog asset generators.

## Deferred

- Add publish-level GEO verification for public blog routes.
- Add richer blocker-specific repair instructions if failure diagnostics become
  more granular.
- Expose repair-attempt counts in review/export surfaces if operators need it.

## Verification

- Focused blog-generation tests passed.
- Touched-module Python compile check passed.
- Whitespace diff check passed.
- Extracted pipeline check suite passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~60 |
| Blog generation | ~160 |
| Tests | ~90 |
| **Total** | **~310** |
