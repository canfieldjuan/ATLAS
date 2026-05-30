# PR-FAQ-Macro-Writeback-Live-Smoke

## Why this slice exists

The FAQ macro writeback path now has tenant credentials, provider wiring,
publish UI, durable attempt history, and drawer history display. The remaining
deferred validation gap is an operator-safe live Zendesk smoke. That smoke
must not run accidentally: publishing creates or updates real Zendesk macros,
so the harness needs explicit confirmation and fail-closed preflight checks
before any network write can happen.

This is intentionally over the 400 LOC soft cap because the same slice needs
both the guarded live-write harness and focused negative fixtures for every
branch that prevents accidental Zendesk writes.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Functional validation

1. Add a guarded operator smoke script for one existing approved FAQ Markdown
   draft and one tenant account.
2. Require explicit live-write confirmation, tenant credentials, a publishable
   draft, and an optional expected Zendesk base URL before invoking the publish
   service.
3. Return machine-readable `skipped` / `not_run_reason` payloads for preflight
   failures instead of silently passing.
4. Cover the no-network preflight branches and fake successful publish path
   with focused tests.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Live-Smoke.md` -- slice plan.
- `scripts/smoke_content_ops_faq_macro_live_zendesk.py` -- guarded operator
  live smoke.
- `tests/test_faq_macro_writeback_live_zendesk_smoke.py` -- preflight and
  fake publish coverage.

## Mechanism

The script is invoked only with explicit operator inputs:

```bash
python scripts/smoke_content_ops_faq_macro_live_zendesk.py \
  --database-url "$DATABASE_URL" \
  --account-id "$ACCOUNT_ID" \
  --faq-id "$FAQ_DRAFT_ID" \
  --expected-zendesk-base-url "https://sandbox.zendesk.com" \
  --confirm-live-zendesk-write \
  --json
```

It builds tenant scope from `--account-id`, resolves tenant Zendesk
credentials through the same host provider used by the generated-asset publish
route, checks the selected FAQ draft has at least one publishable verified
macro, then calls `FAQMacroWritebackPublishService` with the Postgres FAQ
repository, Zendesk provider, and attempt-history repository. The Zendesk
provider receives a static wrapper around the already-validated credentials so
the optional base-url guard applies to the actual network write. Missing
confirmation, missing credentials, unexpected Zendesk endpoint, missing draft,
or no publishable macros return a non-zero skipped payload before any Zendesk
transport call.

## Intentional

- No new API route or scheduler. This is an operator-run smoke harness, not
  automatic production behavior.
- No environment reads for secrets. The script requires `--database-url`; live
  Zendesk credentials come from tenant credential storage through existing
  host wiring.
- No test creates real Zendesk macros. Unit tests use fake repositories and
  providers to prove preflight and service wiring.
- The test file is kept out of the extracted-checks filename patterns because
  this smoke validates host/operator wiring and imports host modules.

## Deferred

- A future robust-testing slice can add an ephemeral sandbox fixture that
  creates, publishes, and cleans up a dedicated FAQ draft when safe Zendesk
  sandbox data is available.
- Parked hardening: none.

## Verification

- `python -m pytest tests/test_faq_macro_writeback_live_zendesk_smoke.py -q`
  (7 passed)
- `python -m py_compile scripts/smoke_content_ops_faq_macro_live_zendesk.py tests/test_faq_macro_writeback_live_zendesk_smoke.py`
- `python scripts/audit_plan_doc.py plans/PR-FAQ-Macro-Writeback-Live-Smoke.md`
- `python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Macro-Writeback-Live-Smoke.md`
- `python scripts/audit_extracted_pipeline_ci_enrollment.py`
- `bash scripts/check_ascii_python.sh`
- `git diff --cached --check`
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-macro-writeback-live-smoke.md`

## Estimated diff size

| Area | Estimate |
|---|---:|
| Plan | ~110 |
| Smoke script | ~275 |
| Tests | ~325 |
| Total | ~710 |

The estimate is above the 400 LOC soft cap because the live-write guard needs
focused negative fixtures for each branch that prevents accidental Zendesk
writes, and splitting the tests from the operator harness would leave the
safety claim unenforced.
