# Fable 5 PR 1935-1941 Review Lessons

Date captured: 2026-07-02

## Purpose

This note preserves the review lessons from the autonomous Fable 5 arc covering
PRs 1935-1941 so future builder sessions can reuse the patterns before opening
another PR. The source analysis reviewed the PR bodies and review-resolution
summaries for:

- #1935 -- Reddit Listening S1: watchlist config and keyword scoring.
- #1936 -- Local MCP eval live runbook.
- #1937 -- Reddit Listening S2: SQLite store and state model.
- #1938 -- Reddit Listening S3: Markdown digest and CLI.
- #1939 -- Reddit Listening S4: PRAW read-only poller.
- #1940 -- Reddit Listening S5: reply tracker read path.
- #1941 -- Reddit Listening S6: deletion-compliance purge.

## What Fable should keep doing

1. Fix review findings at the root-cause level, not only at the cited input.
   Good examples from the arc include removing score-rounding from ranking,
   making stale observations update nothing, wrapping store failures at the
   boundary, and normalizing Reddit ids at the producer/consumer seam.
2. Keep thin-slice sequencing. The arc moved from config/scoring, to state, to
   digest/CLI, to read-only polling, to reply tracking, to deletion compliance.
   That made each review focused and made deferrals understandable.
3. Prefer real-boundary tests over mocks. The strongest tests exercised real
   parsers, file I/O, store APIs, CLI paths, and PRAW constructor seams while
   stubbing only external transport.
4. Document intentional trade-offs in the PR body and plan. Reviewers could
   distinguish real gaps from intentional choices such as TOML over YAML,
   JSON-in-TEXT over join tables, PRAW over hand-rolled HTTP, and manual-run
   scope over scheduling.

## Review findings that should become pre-push checks

### Exact-type validation

Avoid Python truthiness and loose equality at boundaries. The arc included
examples where `1.0 == 1`, truthy strings, `NaN`, `inf`, and string-typed
numeric metrics reached contracts that needed strict values. Validators should
check exact bool/int/finite-float shapes and include negative fixtures for each
invalid type.

### Producer/consumer shape fidelity

Consumer tests must include at least one fixture produced by the real upstream
mapper. The S6 purge risk came from tests that used fullname-shaped Reddit ids
while the real poller stored bare ids. Any cross-module id needs one canonical
representation and tests on both the write and read sides.

### Shared validation across entry paths

Do not validate a knob in only one entry path. If a value can enter through
settings, environment, CLI flags, function params, SQL writes, or a live adapter,
centralize the bounds and prove each path rejects the same bad inputs.

### Boundary-wide error contracts

Wrap the full external boundary, not only the first line that failed locally.
For file/database/network/API code, tests should cover open/connect/import,
auth, fetch/read, and write failures so raw exceptions do not leak past the
operator-facing contract.

### Output and artifact safety

Sanitize every external interpolation at the render boundary, not only the most
obvious field. Destructive cleanup must operate only on owned generated names,
be tested against unrelated neighbor files, and derive retry state from
persisted facts rather than transient run state.

### Persist lifecycle state

Do not infer durable lifecycle state from absence in a limited API window. If a
state transition matters later, persist it at the event boundary. Examples from
the arc include own-submission status, last activity, dormancy, and purge
tombstones.

### Narrow conflict handling

Avoid broad `INSERT OR IGNORE` for replay safety. Suppress only the exact unique
conflict that represents a replay; let integrity failures, malformed ids, and
foreign-key violations surface.

## Deferrals that were acceptable

The Reddit arc correctly deferred work that would have expanded beyond the
manual, read-only/local-tool scope: scheduling, delivery, live Reddit credential
smokes, LLM judge-fit, unread badges, threading-depth UX, and own-content
deletion-on-request. Those deferrals were acceptable because each PR still
proved the slice it claimed.

## Deferrals to avoid repeating

The MCP eval runbook waived strict `passed` typing, per-run output files, and
redaction of tool-error details from grade errors. Treat that class as unsafe to
repeat for any live or tenant-adjacent runbook. If operators can run it against
real data, stale aggregation and raw error-detail leakage are part of the safety
contract, not optional polish.

## Patterns to avoid altogether

- Using Python truthiness as validation.
- Assuming one validated entry path means all entry paths are safe.
- Writing consumer fixtures by hand without a real-producer fixture.
- Sanitizing only the field currently under review instead of every rendered
  external value.
- Treating absence from a limited API result as inactivity or deletion without
  proving query shape and lifecycle semantics.
- Using broad `INSERT OR IGNORE` as a replay mechanism.
- Running cleanup globs over user-configured directories without an owned-name
  allowlist.
- Shipping a live runbook that can leak raw tenant/customer/tool details because
  outputs are gitignored.

## Issue-ready comment

Copy this section into the active GitHub issue if direct issue mutation is not
available from the local checkout.

```markdown
## Fable 5 PR 1935-1941 review lessons

Persisted locally in `docs/fable5_pr_1935_1941_review_lessons.md`.

### Keep
- Root-cause fixes over symptom patches.
- Thin slices with explicit deferrals.
- Real-boundary tests that mock transport only.
- PR bodies that explain intentional trade-offs.

### Add before the next autonomous arc
- Exact-type negative fixtures for bool/int/finite-float boundaries.
- Producer-fidelity fixtures for every cross-module id or payload shape.
- Shared validators for settings/env/CLI/function/SQL/live-adapter entry paths.
- Boundary-wide failure tests for open/connect/import/auth/fetch/write stages.
- Render-boundary escaping for every external field.
- Persisted lifecycle markers instead of absence-based inference.
- Narrow conflict handling instead of broad `INSERT OR IGNORE`.
- Destructive cleanup constrained to owned generated filenames.

### Do not repeat
- Waiving strict typing, per-run output isolation, or raw error redaction for any
  live or tenant-adjacent runbook.
- Letting reviewers be the first adversarial fixture harness.
```
