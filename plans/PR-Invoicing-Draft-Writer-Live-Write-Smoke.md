# Invoicing Draft Writer Live Write Smoke

## Why this slice exists

The draft-writer connector now works through ChatGPT and #629 fixed the
read-after-create invoice-number lookup bug. The remaining verification gap is
operational: we have a no-mutation OAuth e2e check that lists tools, but no
repeatable host-facing smoke that proves the public connector can create a
safe draft and then read it back.

The live test we ran manually created an intentionally blocked draft invoice
(no email, zero subtotal) and verified that the connector did not expose send
or payment tools. This slice turns that manual write check into an explicit
operator command.

This slice exceeds the soft 400 LOC budget because the safe live-write command
needs its own validation contract: explicit write acknowledgement, idempotent
draft creation, invoice-number readback, pending-draft blocker checks, tests,
and operator docs. Splitting those apart would ship a write-capable script
without the reviewable safety harness in the same PR.

## Scope

1. Add a live write smoke script for the draft-writer OAuth connector.
2. Require an explicit acknowledgement flag before any draft is created.
3. Create or reuse one idempotent blocked test draft.
4. Verify readback by invoice number and visibility in `list_pending_drafts`.
5. Document where the live write smoke fits after the no-mutation e2e smoke.
6. Add unit tests for the smoke script's safety and validation behavior.

### Files touched

- `scripts/check_invoicing_draft_writer_live_write.py`
- `tests/test_check_invoicing_draft_writer_live_write.py`
- `docs/INVOICING_MCP_WRITE_ACCESS_GUARDRAILS.md`
- `docs/MCP_CHATGPT_OAUTH_ROLLOUT_RUNBOOK.md`
- `plans/PR-Invoicing-Draft-Writer-Live-Write-Smoke.md`

## Mechanism

The new script reuses the existing OAuth helper functions from
`scripts/check_invoicing_draft_writer_oauth_e2e.py` for client registration,
operator approval, token exchange, and public MCP transport setup. After the
operator passes `--create-blocked-draft`, the script calls
`create_draft_invoice` with a default idempotency key, test customer name, no
email address, and a zero-dollar line item. That shape is intentionally blocked
from sending by `no_email` and warned by `subtotal_zero`.

The script then calls `get_invoice` with the returned invoice number and calls
`list_pending_drafts` with the smoke business context. The command exits nonzero
if create/read/list disagree, if the draft is not blocked, or if the metadata
marker is missing.

## Intentional

- This is a live write smoke, not part of the default no-mutation e2e check.
  The explicit `--create-blocked-draft` flag prevents accidental writes.
- The smoke is idempotent by default, so repeated runs reuse the same daily test
  draft instead of creating unbounded duplicates.
- The script does not void or delete the test draft. The draft remains blocked
  and visible for operator audit.
- No connector tool-surface change. This validates the deployed surface only.

## Deferred

- A separate cleanup/void operator command can be added later if blocked smoke
  drafts become noisy.
- Extending the same live-write pattern to other MCP write connectors remains a
  separate connector-by-connector rollout.

## Verification

- Focused pytest for the new smoke script and OAuth helper: 22 passed.
- Python compile check for the new script and tests: passed.
- Git whitespace check: passed.
- Live public smoke against the draft-writer connector: reused
  `INV-2026-May-0185`, confirmed `no_email` blocker and `subtotal_zero`
  warning.
- Local PR review bundle in advisory dirty mode: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Live write smoke script | ~365 |
| Unit tests | ~210 |
| Operator docs | ~30 |
| Plan doc | ~90 |
| **Total** | ~695 |
