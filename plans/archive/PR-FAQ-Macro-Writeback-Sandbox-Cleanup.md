# PR-FAQ-Macro-Writeback-Sandbox-Cleanup

## Why this slice exists

PR #1595 made the first real sandbox Zendesk macro write one guarded command
away. That command can create two kinds of cleanup work: an external Zendesk
macro and local Atlas rows for the seeded FAQ draft, macro mapping, publish
attempts, and search projection. Without an explicit cleanup command, operators
must either leave sandbox artifacts around or hand-edit Postgres after a live
proof.

This slice adds the missing reversible path for the sandbox proof. Cleanup is
intentionally narrow: one account, one explicit `faq_id`, preview by default,
and execute only with a live Zendesk delete confirmation plus expected base URL.
It deletes external macros first and deletes the FAQ draft last, relying on the
existing `ON DELETE CASCADE` relationships for local mapping, attempt, and
search rows.

This PR is over the 400 LOC soft cap because the command is live-delete
capable. The preview result, live-delete guards, external-before-local ordering,
partial-failure behavior, and CI enrollment need to ship together with focused
negative fixtures.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Vertical slice

1. Add an operator-run cleanup script for one sandbox FAQ macro writeback proof.
2. Preview draft, macro mapping, publish attempt, and search projection counts
   for a tenant-scoped `faq_id` without deleting anything by default.
3. Require `--execute`, `--confirm-live-zendesk-delete`, and
   `--expected-zendesk-base-url` before deleting mapped external Zendesk macros
   or local FAQ rows.
4. Delete mapped Zendesk macros before deleting the FAQ draft; if any external
   delete fails, leave local rows in place for retry/audit.
5. Delete only the explicit tenant-scoped FAQ draft row; local mapping,
   attempt, and search projection cleanup relies on existing database cascades.
6. Let the shared Zendesk HTTP transport treat successful empty-body DELETE
   responses as an empty payload instead of trying to parse JSON.
7. Fail closed before any delete when a mapping row is present without a usable
   external Zendesk macro id.
8. Restrict cleanup mappings to Zendesk rows, treat Zendesk 404 deletes as
   idempotent success on retry, and fail closed when a stored macro URL points
   at a different Zendesk base.

### Files touched

- `.github/workflows/atlas_content_ops_macro_writeback_checks.yml`
- `extracted_content_pipeline/faq_macro_writeback_zendesk.py`
- `plans/PR-FAQ-Macro-Writeback-Sandbox-Cleanup.md`
- `scripts/cleanup_content_ops_faq_macro_sandbox.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_cleanup_content_ops_faq_macro_sandbox.py`
- `tests/test_extracted_ticket_faq_macro_writeback_zendesk.py`

### Review Contract

- Acceptance criteria:
  - [ ] Preview mode opens the database, returns scoped counts and mapped
        external ids, and performs no Zendesk or local deletes.
  - [ ] Missing `--execute` never calls the Zendesk transport and never deletes
        local rows.
  - [ ] Execute mode requires `--confirm-live-zendesk-delete` and a non-empty
        expected Zendesk base URL before any database pool is opened.
  - [ ] Execute mode resolves credentials and rejects unexpected Zendesk base
        URLs before any delete.
  - [ ] External Zendesk macro deletes run before the local FAQ row delete.
  - [ ] A mapping row with a missing `external_id` returns a non-green artifact
        and leaves local rows in place.
  - [ ] Non-Zendesk mappings are not included in the Zendesk cleanup snapshot.
  - [ ] A retry where Zendesk reports a mapped macro as already missing can
        still finish local cleanup.
  - [ ] A mapping whose stored URL points at a different Zendesk base returns a
        non-green artifact before any delete.
  - [ ] External delete failure leaves local rows in place and returns a
        non-green artifact.
  - [ ] Local delete is tenant-scoped by `faq_id` and `account_id`.
- Affected surfaces: operator scripts, macro-writeback live validation tests,
  macro-writeback CI workflow.
- Risk areas: accidental live Zendesk deletes, broad tenant cleanup, partial
  cleanup false-greens, and deleting local audit state before external cleanup.
- Reviewer rules triggered: R1, R2, R3, R6, R8, R10, R12, R14.

## Mechanism

The cleanup script accepts `--database-url`, `--account-id`, and `--faq-id`.
Without `--execute`, it queries the tenant-scoped FAQ row plus dependent local
row counts and returns a JSON preview artifact. Preview mode does not resolve
Zendesk credentials and does not call the Zendesk transport.

With `--execute`, the script first checks `--confirm-live-zendesk-delete` and
`--expected-zendesk-base-url` before opening a pool. It loads tenant Zendesk
credentials through the existing macro-writeback credential provider, verifies
the normalized base URL against the expected URL, then deletes each mapped
external Zendesk macro id with the existing Zendesk transport boundary. Only
after all external deletes succeed does it delete the tenant-scoped
`ticket_faq_markdown` row. The existing database foreign keys cascade local
`ticket_faq_search_documents`, `ticket_faq_macro_writebacks`, and
`ticket_faq_macro_publish_attempts` rows.

The script reuses `ZendeskHTTPMacroTransport`. That transport now returns an
empty mapping for successful empty-body responses so a normal Zendesk DELETE
success path does not fail during JSON parsing.

Before resolving credentials or deleting anything, execute mode also checks
that every local macro mapping has a usable external id. Pending or
persist-failed mappings are not safe to cascade-delete because they may be the
only local evidence needed to reconcile an orphaned live Zendesk macro.

The mapping lookup is restricted to `ZENDESK_PLATFORM`, since this cleanup
command only knows how to delete Zendesk macros. Before delete, each stored
macro URL base is compared with the expected/current Zendesk base; mismatches
fail closed instead of deleting from the current credentials' instance. During
retry after a partial external cleanup, Zendesk 404/not-found deletes are
treated as already cleaned so the command can continue to the local cascade.

## Intentional

- No broad delete by account, target id, or age. The command accepts one
  explicit `faq_id` so cleanup cannot sweep unrelated customer artifacts.
- No new Zendesk HTTP client. The command uses the existing
  `ZendeskHTTPMacroTransport` boundary and credentials shape.
- No cleanup built into the E2E smoke wrapper. The write proof and cleanup proof
  remain separate explicit operator actions.
- No deletion when the external Zendesk delete fails. Local rows stay available
  for retry and audit.

## Deferred

- Run the cleanup command against the first live sandbox artifact after the
  write proof is captured.
- Future robust-testing slice: batch cleanup by a manifest of explicit FAQ ids
  if repeated sandbox runs become common.
- Future product slice: expose sandbox/live proof cleanup status in operator
  docs or UI if this flow graduates beyond validation tooling.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_cleanup_content_ops_faq_macro_sandbox.py -q`
  - 11 passed.
- `python -m pytest tests/test_extracted_ticket_faq_macro_writeback_zendesk.py -q`
  - 14 passed.
- `python -m pytest tests/test_autonomous_faq_macro_writeback_scheduled_publish.py tests/test_cleanup_content_ops_faq_macro_sandbox.py tests/test_extracted_ticket_faq_macro_writeback_zendesk.py tests/test_faq_macro_writeback_sandbox_e2e_smoke.py tests/test_seed_faq_macro_writeback_live_smoke_draft.py tests/test_scheduler.py::TestDefaults::test_content_ops_faq_macro_writeback_default_seed_respects_opt_in tests/test_scheduler.py::TestDefaults::test_content_ops_faq_macro_writeback_default_notifies_skipped_summary tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_notify_opt_in_uses_fallback_message tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_notify_failure_is_nonfatal tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_without_notify_opt_in_keeps_existing_response_shape -q`
  - 54 passed, 1 warning.
- Python compile check for the modified Python files.
  - Passed.
- `python scripts/sync_pr_plan.py plans/PR-FAQ-Macro-Writeback-Sandbox-Cleanup.md --check`
  - Passed.
- Pending before push: local PR review via `scripts/push_pr.sh`

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_macro_writeback_checks.yml` | 8 |
| `extracted_content_pipeline/faq_macro_writeback_zendesk.py` | 2 |
| `plans/PR-FAQ-Macro-Writeback-Sandbox-Cleanup.md` | 165 |
| `scripts/cleanup_content_ops_faq_macro_sandbox.py` | 536 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_cleanup_content_ops_faq_macro_sandbox.py` | 483 |
| `tests/test_extracted_ticket_faq_macro_writeback_zendesk.py` | 92 |
| **Total** | **1287** |
