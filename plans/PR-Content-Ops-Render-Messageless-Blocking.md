# PR-Content-Ops-Render-Messageless-Blocking

## Why this slice exists

#1490 (slice 8) merged at its pre-fix head, so a review fix it advertised never
reached main: the verdict renderer's `_comment_lines` skips any message-less
comment (`if not message: continue`), but the MCP parser keeps a message-less
*blocking* comment, which drives the verdict to `revision_required` and is
counted in the Reasons line. So a message-less blocking comment flips the verdict
yet is invisible in the Objections section -- if it is the only comment, the
marketer reads "Decision: revision_required / Reasons: - 1 blocking comment(s)"
with no objection shown anywhere, contradicting slice 8's own "a blocking
comment is marked, not hidden" invariant. This lands the fix the resolved #1490
review thread already records.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Production hardening

1. `_comment_lines`: render a message-less *blocking* comment with a
   "(no message provided)" placeholder instead of dropping it; keep dropping
   message-less non-blocking comments (decoration).

### Files touched

- `atlas_brain/mcp/content_ops_marketer_verify_chatgpt_adapter_server.py`
- `plans/PR-Content-Ops-Render-Messageless-Blocking.md`
- `tests/test_mcp_content_ops_marketer_verify.py`

### Review Contract

Acceptance criteria:
- A message-less blocking comment renders as `[<category>] [BLOCKING] (no message
  provided)` (and its evidence if any).
- A message-less non-blocking comment is still skipped.
- No other render behavior changes.

Affected surfaces: the ChatGPT adapter's `_comment_lines` only.

Risk areas: none beyond the single branch.

Reviewer rules triggered: R1, R2, R5, R10, R14.

## Mechanism

In `_comment_lines`, the empty-message guard now branches on `blocking`: a
non-blocking message-less comment is skipped as before, but a blocking one is
kept with `message = "(no message provided)"` so the existing
`[<category>] [BLOCKING] ...` line still renders.

## Intentional

- **Placeholder, not the raw empty string.** An empty message would render a
  dangling `[compliance] [BLOCKING] ` line; the placeholder makes the absence
  explicit and honest.
- **Non-blocking message-less comments stay dropped.** They are decoration and
  did not affect the verdict, so hiding them is correct.

## Deferred

- Tenant calibration-library reader port (from #1489).
- Corroboration surfacing (from #1488).
- Parked hardening: none new.

## Verification

- Reviewer rules triggered: R1, R2, R5, R10, R14.
- Passed: pytest tests/test_mcp_content_ops_marketer_verify.py -- 53 passed
  (2 new fixtures: message-less blocking placeholder, message-less non-blocking drop).
- Passed: bash scripts/check_ascii_python.sh -- ASCII check passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/mcp/content_ops_marketer_verify_chatgpt_adapter_server.py` | 7 |
| `plans/PR-Content-Ops-Render-Messageless-Blocking.md` | 75 |
| `tests/test_mcp_content_ops_marketer_verify.py` | 29 |
| **Total** | **111** |
