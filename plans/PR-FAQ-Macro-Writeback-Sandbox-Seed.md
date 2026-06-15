# PR-FAQ-Macro-Writeback-Sandbox-Seed

## Why this slice exists

The macro writeback lane already has a guarded live Zendesk smoke, but it assumes
an approved, publishable FAQ draft already exists in Postgres. Now that we have
a Zendesk test API available, the remaining operator friction is bootstrapping a
safe draft for the smoke without hand-editing the database or depending on a
real customer FAQ.

This slice adds only the missing sandbox seed step. It does not write to
Zendesk. The existing smoke remains the only command that creates or updates
Zendesk macros, and it still requires `--confirm-live-zendesk-write` plus the
expected Zendesk base URL guard.

This PR is over the 400 LOC soft cap because the operator helper, fail-closed
write guards, preview validation, CI enrollment, and focused tests for each
error branch need to ship together. Splitting the tests or workflow enrollment
from the script would create an unprotected live-validation helper.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Functional validation

1. Add an operator-run script that inserts one disposable approved FAQ draft for
   macro writeback live-smoke validation.
2. Require explicit confirmation before the script writes to Postgres.
3. Build the seed draft through the existing `TicketFAQDraft` shape and
   `PostgresTicketFAQRepository`, then approve it through the repository status
   path so the existing macro preview and publish service see a normal draft.
4. Emit machine-readable JSON with the created FAQ draft id, publishable macro
   count, and the exact next live Zendesk smoke command.
5. Do not call Zendesk or create macros in this slice.

### Files touched

- `.github/workflows/atlas_content_ops_macro_writeback_checks.yml`
- `plans/PR-FAQ-Macro-Writeback-Sandbox-Seed.md`
- `scripts/seed_faq_macro_writeback_live_smoke_draft.py`
- `tests/test_seed_faq_macro_writeback_live_smoke_draft.py`

### Review Contract

- Acceptance criteria:
  - [ ] Missing confirmation exits with a skipped payload before any database
        pool is opened.
  - [ ] With confirmation, the script saves exactly one FAQ draft scoped to the
        requested account and marks it `approved` through the repository.
  - [ ] The seeded draft produces at least one publishable macro through
        `build_macro_writeback_preview`.
  - [ ] The JSON output includes the created `faq_id` and a next-step command
        for `scripts/smoke_content_ops_faq_macro_live_zendesk.py`.
  - [ ] The script never imports or calls the Zendesk transport/provider.
- Affected surfaces: operator scripts, macro-writeback live validation tests,
  Postgres draft seeding path.
- Risk areas: accidental writes, tenant scope correctness, live Zendesk safety,
  and drift from the existing publish path.
- Reviewer rules triggered: R1, R2, R3, R6, R8, R10, R12, R14.

## Mechanism

The new script accepts `--database-url`, `--account-id`, optional seed labels,
and `--confirm-create-draft`. Without confirmation, it returns a skipped JSON
payload and exits before opening a database pool.

When confirmed, the script opens a small asyncpg pool, builds one
`TicketFAQDraft` with a single `resolution_evidence` FAQ item, saves it through
`PostgresTicketFAQRepository.save_drafts`, and then calls `update_status` to mark
the saved draft `approved`. The script validates the resulting draft shape with
`build_macro_writeback_preview` using the saved id before printing the next
operator command for the already-existing guarded Zendesk smoke.

The helper does not publish anything to Zendesk. It prepares the local Atlas
state needed for the existing live smoke to do the actual API write under its
own confirmation and base-url guard.

## Intentional

- No live Zendesk call. This is the seed half only; the existing live smoke owns
  the external write.
- No cleanup/delete path in this PR. The seed draft is harmless local Atlas
  state, and cleanup can be added after the first live test proves the path.
- No new API route or UI. This is operator validation tooling for the sandbox
  proof.
- No credential reads. Zendesk credentials remain handled by the existing smoke
  and tenant credential provider.

## Deferred

- Run the existing live Zendesk smoke against the seeded draft once the operator
  chooses the test tenant/account and expected Zendesk base URL.
- Future robust-testing slice: optional cleanup helper that deletes the seeded
  draft and macro mapping after the sandbox write is verified.
- Future robust-testing slice: create/update/delete a dedicated Zendesk macro
  fixture when the test account lifecycle policy is settled.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_seed_faq_macro_writeback_live_smoke_draft.py -q`
  - Result: 7 passed.
- `python -m pytest tests/test_autonomous_faq_macro_writeback_scheduled_publish.py tests/test_seed_faq_macro_writeback_live_smoke_draft.py tests/test_scheduler.py::TestDefaults::test_content_ops_faq_macro_writeback_default_seed_respects_opt_in tests/test_scheduler.py::TestDefaults::test_content_ops_faq_macro_writeback_default_notifies_skipped_summary tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_notify_opt_in_uses_fallback_message tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_notify_failure_is_nonfatal tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_without_notify_opt_in_keeps_existing_response_shape -q`
  - Result: 19 passed, 1 warning.
- Python compile check for the seed script and test module.
  - Result: passed.
- `python scripts/sync_pr_plan.py plans/PR-FAQ-Macro-Writeback-Sandbox-Seed.md --check`
  - Result: passed.
- Pending before push: local PR review via `scripts/push_pr.sh`

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_macro_writeback_checks.yml` | 5 |
| `plans/PR-FAQ-Macro-Writeback-Sandbox-Seed.md` | 120 |
| `scripts/seed_faq_macro_writeback_live_smoke_draft.py` | 311 |
| `tests/test_seed_faq_macro_writeback_live_smoke_draft.py` | 212 |
| **Total** | **648** |
