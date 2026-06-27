# PR-Deflection-Delta-Delivery-Summary

## Why this slice exists

#1316's Report Delta path now has stable action identity, a pure
`deflection_delta.v1` comparator, persistence, paid API/MCP read surfaces,
monthly generation, cap/overflow telemetry, and source-window baseline
selection. The remaining D4 work is customer delivery, but sending a recurring
delta email is too large and too customer-visible for one jump.

Root cause: the delivery layer has no delivery-safe summary contract for a
persisted delta. The only rendered delta copy today is the MCP document text,
which is useful for a tool response but not a reusable email/customer delivery
primitive. This PR fixes the root at the delivery-copy boundary by adding a
pure, bounded renderer over the already allowlisted paid delta payload. It does
not send emails or change automation; it gives the next D4 slice a tested
copy primitive to compose.

This branch also archives the just-merged #1861 plan as the AGENTS.md teardown
step.

Diff budget note: the product code is a small pure renderer, but the slice lands
slightly over the soft cap because the privacy-boundary tests include positive,
negative, cap, and HTML-escaping fixtures, and the branch folds in the #1861
plan archive move.

## Scope (this PR)

Ownership lane: issue-1316/deflection-delta-delivery-summary
Slice phase: Vertical slice

1. Add a pure delivery-summary renderer for `deflection_delta_read_payload(...)`
   output.
2. Render only bounded aggregate counts, support-cost movement, source windows,
   and the top action rows already present in the allowlisted delta payload.
3. Add tests proving useful positive/negative movement copy and that raw
   evidence/source IDs cannot appear in the delivery summary.
4. Archive the merged #1861 plan and refresh the plans index.

### Review Contract

Acceptance criteria:
- The renderer accepts the paid delta read payload shape and returns a stable
  subject/text/html summary that can be used by a future email sender.
- The summary highlights new, resolved, growing, still-unresolved, and
  support-cost movement without requiring a live DB, sender, or scheduler.
- The top row list is capped and uses only allowlisted customer-facing row
  fields; raw evidence, source IDs, representative phrasing arrays, and full
  artifacts never enter the copy.
- Existing paid report delivery behavior is unchanged; no queued delivery, cron,
  customer email send, result page, or entitlement behavior changes in this
  slice.

Affected surfaces:
- `atlas_brain.content_ops_deflection_delivery`
- `tests.test_atlas_content_ops_deflection_delivery`
- `plans/`

Risk areas:
- Accidentally widening the email surface from paid delta summaries into raw
  ticket/evidence payloads.
- Writing customer-facing copy that overclaims savings when the support-cost
  delta is positive, negative, zero, or missing.
- Coupling the renderer to MCP-specific text instead of the shared allowlisted
  read payload.

Reviewer rules triggered: R1, R2, R6, R8, R10, R14.

### Files touched

- `atlas_brain/content_ops_deflection_delivery.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Delta-Delivery-Summary.md`
- `plans/archive/PR-Deflection-Delta-Source-Window-Baseline.md`
- `tests/test_atlas_content_ops_deflection_delivery.py`

## Mechanism

- Add small frozen dataclasses for the delivery delta summary and capped action
  rows.
- Parse the already-safe `deflection_delta_read_payload(...)` dictionary rather
  than the raw persisted artifact or raw `content_ops_deflection_deltas.delta`
  value.
- Render subject, plaintext, and HTML through shared formatter helpers. Row
  copy includes question, owner lane, status movement, ticket-count movement,
  and support-cost movement only when present.
- Escape HTML and cap rows to keep the future email surface bounded and
  deterministic.

## Intentional

- This is not a send path. The existing paid report delivery queue and the
  monthly delta automation remain unchanged.
- This does not add result-page UI or portfolio changes; the paid API/MCP read
  surfaces already expose the underlying delta.
- The renderer uses the paid delta read payload, not raw artifacts, because the
  allowlist is the privacy boundary.

## Deferred

- Wiring the summary into a monthly Report Delta delivery email.
- Customer-facing result-page delta UI.
- Entitlement/billing changes for a monthly Report Delta subscription.
- Macro-writeback or report-delta upsell flows.

Parked hardening: none.

## Verification

- Focused delivery pytest for `tests/test_atlas_content_ops_deflection_delivery.py`
  -- 23 passed.
- Python byte-compile check for `atlas_brain/content_ops_deflection_delivery.py`
  -- passed.
- Plan sync check for `plans/PR-Deflection-Delta-Delivery-Summary.md`
  -- passed.
- Body-wired local PR review -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/content_ops_deflection_delivery.py` | 219 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-Delta-Delivery-Summary.md` | 127 |
| `plans/archive/PR-Deflection-Delta-Source-Window-Baseline.md` | 0 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 121 |
| **Total** | **470** |
