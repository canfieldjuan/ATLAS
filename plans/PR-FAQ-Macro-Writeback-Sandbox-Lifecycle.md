# PR-FAQ-Macro-Writeback-Sandbox-Lifecycle

## Why this slice exists

PR #1595 made sandbox macro writeback a one-command write proof, and PR #1597
made cleanup a one-command delete proof. The remaining validation gap is that a
real operator run is still a manual sequence: run E2E, copy the returned
`faq_id`, run cleanup, then preserve two separate artifacts. That is exactly
where live-proof mistakes happen: cleaning the wrong FAQ id, losing the write
artifact, or reporting a green write while cleanup failed.

This slice adds the thinnest lifecycle wrapper around the two existing guarded
commands. It does not publish or delete directly; it composes the existing
write wrapper and cleanup command into one artifact that proves the sandbox
macro lifecycle from seed/write through cleanup.

This PR may exceed the 400 LOC soft cap because the wrapper is both live-write
and live-delete capable. The guard matrix, stage handoff, failure envelopes,
cleanup skip semantics, and CI enrollment need to ship with focused tests.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Vertical slice

1. Add an operator-run sandbox lifecycle wrapper for one FAQ macro writeback
   proof.
2. Require create, live Zendesk write, and live Zendesk delete confirmations
   plus a non-empty expected Zendesk base URL before opening a database pool.
3. Reuse the existing sandbox E2E write wrapper, then pass its returned
   `faq_id` to the existing sandbox cleanup command.
4. Emit one JSON artifact with `write` and `cleanup` stage payloads, top-level
   `faq_id`, and top-level success/error state.
5. Stop before cleanup if the write stage fails or does not return a usable
   `faq_id`.
6. Surface cleanup failure as a non-green lifecycle artifact without rewriting
   the cleanup payload.
7. Do not add a new Zendesk provider, transport, API route, scheduler, or UI.

### Files touched

- `.github/workflows/atlas_content_ops_macro_writeback_checks.yml`
- `plans/PR-FAQ-Macro-Writeback-Sandbox-Lifecycle.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `scripts/smoke_content_ops_faq_macro_sandbox_lifecycle.py`
- `tests/test_faq_macro_writeback_sandbox_lifecycle.py`

### Review Contract

- Acceptance criteria:
  - [ ] Missing create confirmation exits skipped before any database pool is
        opened.
  - [ ] Missing live-write confirmation exits skipped before any database pool
        is opened.
  - [ ] Missing live-delete confirmation exits skipped before any database pool
        is opened.
  - [ ] Missing expected Zendesk base URL exits skipped before any database
        pool is opened.
  - [ ] The wrapper calls the existing E2E write stage first, then calls the
        existing cleanup stage with the returned `faq_id`.
  - [ ] Write-stage failure stops before cleanup and returns a non-green
        lifecycle artifact.
  - [ ] Cleanup-stage failure is surfaced as non-green with the cleanup payload
        preserved.
  - [ ] The wrapper source does not import Zendesk transport/provider classes.
- Affected surfaces: operator scripts, macro-writeback live validation tests,
  macro-writeback CI workflow.
- Risk areas: accidental live Zendesk writes/deletes, wrong FAQ id handoff,
  false-green lifecycle artifacts, and scope drift into provider logic.
- Reviewer rules triggered: R1, R2, R3, R6, R8, R10, R12, R14.

## Mechanism

The lifecycle wrapper is a coordinator around two existing scripts:

1. `scripts/smoke_content_ops_faq_macro_sandbox_e2e.py`
2. `scripts/cleanup_content_ops_faq_macro_sandbox.py`

It accepts the operator inputs needed for both stages: database URL, account
id, optional user id, expected Zendesk base URL, seed question/answer text, and
three explicit confirmation flags. If any live guard is missing, it returns a
skipped JSON payload before opening a database pool.

When all guards are present, the wrapper opens one asyncpg pool, calls the E2E
write stage, reads the returned `faq_id`, then builds a cleanup argument
namespace with that `faq_id` and calls the cleanup stage. The wrapper does not
inspect Zendesk internals or directly call Zendesk; it only carries the stage
payloads into one combined artifact and returns a non-zero code when either
stage fails.

## Intentional

- No new Zendesk transport or provider. The write and delete behavior remains
  owned by the existing E2E and cleanup scripts.
- No automatic live execution. The wrapper is still operator-run and guarded by
  explicit confirmations.
- No UI, API route, or scheduler. This is a validation harness for the sandbox
  proof lane.
- No broad cleanup. The cleanup stage receives only the `faq_id` returned by
  the write stage.

## Deferred

- Run the lifecycle wrapper against the Zendesk test account and attach the
  sanitized artifact to the lane issue or proof log once the operator selects
  credentials/account id.
- Future robust-testing slice: manifest-based batch cleanup if repeated
  sandbox lifecycle runs become common.
- Future product slice: decide whether this validation harness graduates into
  operator UI/status.

Parked hardening: none.

## Verification

- python -m pytest tests/test_faq_macro_writeback_sandbox_lifecycle.py -q
  - 11 passed.
- python -m pytest tests/test_autonomous_faq_macro_writeback_scheduled_publish.py tests/test_cleanup_content_ops_faq_macro_sandbox.py tests/test_extracted_ticket_faq_macro_writeback_zendesk.py tests/test_faq_macro_writeback_sandbox_e2e_smoke.py tests/test_faq_macro_writeback_sandbox_lifecycle.py tests/test_seed_faq_macro_writeback_live_smoke_draft.py tests/test_scheduler.py::TestDefaults::test_content_ops_faq_macro_writeback_default_seed_respects_opt_in tests/test_scheduler.py::TestDefaults::test_content_ops_faq_macro_writeback_default_notifies_skipped_summary tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_notify_opt_in_uses_fallback_message tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_notify_failure_is_nonfatal tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_without_notify_opt_in_keeps_existing_response_shape -q
  - 65 passed, 1 warning.
- python -m py_compile scripts/smoke_content_ops_faq_macro_sandbox_lifecycle.py tests/test_faq_macro_writeback_sandbox_lifecycle.py
- python scripts/audit_extracted_pipeline_ci_enrollment.py
  - OK: 182 matching tests are enrolled.
- bash scripts/check_ascii_python.sh
  - ASCII check passed for extracted_content_pipeline Python files.
- python scripts/sync_pr_plan.py plans/PR-FAQ-Macro-Writeback-Sandbox-Lifecycle.md --check
  - plan already in sync.
- Pending before push: local PR review via `scripts/push_pr.sh`

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_macro_writeback_checks.yml` | 5 |
| `plans/PR-FAQ-Macro-Writeback-Sandbox-Lifecycle.md` | 138 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `scripts/smoke_content_ops_faq_macro_sandbox_lifecycle.py` | 297 |
| `tests/test_faq_macro_writeback_sandbox_lifecycle.py` | 345 |
| **Total** | **786** |
