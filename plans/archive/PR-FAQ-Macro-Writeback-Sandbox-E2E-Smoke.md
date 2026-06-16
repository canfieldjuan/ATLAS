# PR-FAQ-Macro-Writeback-Sandbox-E2E-Smoke

## Why this slice exists

PR #1593 closed the first sandbox gap by adding a safe helper that creates one
approved FAQ draft for macro writeback validation. The next product proof is
the actual end-to-end sandbox write: seed a disposable approved draft, publish
it through the existing guarded Zendesk live smoke, and emit one JSON artifact
that shows both stages.

This slice exists because running two separate commands is still easy to botch:
the operator can seed one draft, copy the wrong FAQ id, omit the expected
Zendesk base URL guard, or lose the publish artifact. A thin wrapper gives us a
single reproducible proof path while keeping all real publish behavior in the
existing seed and live-smoke scripts.

This PR is over the 400 LOC soft cap because the wrapper is live-write capable.
The fail-closed branches, stage handoff, failure envelopes, CI enrollment, and
tests need to ship together; splitting the safety tests from the wrapper would
leave the live validation path under-proven.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Vertical slice

1. Add an operator-run sandbox E2E smoke wrapper for FAQ macro writeback.
2. Require `--confirm-create-draft`, `--confirm-live-zendesk-write`, and a
   non-empty `--expected-zendesk-base-url` before opening a database pool.
3. Reuse the #1593 seed helper to create one approved draft, then pass the
   returned `faq_id` into the existing guarded live Zendesk smoke.
4. Emit one JSON artifact with `seed` and `live_smoke` stage payloads,
   top-level `faq_id`, and top-level success/error state.
5. Do not add a new Zendesk provider, transport, API route, scheduler, or UI.

### Review Contract

- Acceptance criteria:
  - [ ] Missing create confirmation exits skipped before any database pool is
        opened.
  - [ ] Missing live-write confirmation exits skipped before any database pool
        is opened.
  - [ ] Missing expected Zendesk base URL exits skipped before any database
        pool is opened.
  - [ ] The wrapper calls the seed stage first, then calls the existing live
        smoke with the seeded `faq_id`.
  - [ ] Seed-stage failure stops before the live smoke.
  - [ ] Live-stage failure is surfaced in the combined artifact without
        rewriting the live-smoke payload.
  - [ ] The wrapper source does not import Zendesk transport/provider classes.
- Affected surfaces: operator scripts, macro-writeback live validation tests,
  macro-writeback CI workflow.
- Risk areas: accidental Zendesk writes, wrong tenant or FAQ handoff,
  false-green live artifacts, and scope drift into provider logic.
- Reviewer rules triggered: R1, R2, R3, R6, R8, R10, R12, R14.

### Files touched

- `.github/workflows/atlas_content_ops_macro_writeback_checks.yml`
- `plans/PR-FAQ-Macro-Writeback-Sandbox-E2E-Smoke.md`
- `scripts/smoke_content_ops_faq_macro_sandbox_e2e.py`
- `tests/test_faq_macro_writeback_sandbox_e2e_smoke.py`

## Mechanism

The wrapper is a coordinator around two existing scripts:

1. `scripts/seed_faq_macro_writeback_live_smoke_draft.py`
2. `scripts/smoke_content_ops_faq_macro_live_zendesk.py`

It accepts the same operator inputs needed for both stages: database URL,
account id, optional user id, expected Zendesk base URL, the seed question and
answer text, and both explicit confirmation flags. If any live-write guard is
missing, it returns a skipped JSON payload before opening a database pool.

When all guards are present, the wrapper opens one asyncpg pool, calls the seed
stage, reads the returned `faq_id`, then builds a live-smoke argument namespace
with that `faq_id` and calls the existing live-smoke stage. The wrapper does
not inspect or alter Zendesk publish internals; it only carries the stage
payloads into one combined artifact and returns the live-smoke exit code when
the publish stage fails.

## Intentional

- No new Zendesk transport or provider. The live write remains owned by
  `smoke_content_ops_faq_macro_live_zendesk.py` and the existing publish
  service.
- No cleanup/delete path in this slice. The purpose is to prove the sandbox
  write path first; cleanup policy depends on how the test Zendesk account
  should retain evidence.
- No API route, UI, or scheduler. This is an operator proof harness.
- The expected Zendesk base URL is mandatory for this combined live-write path
  even though the lower-level live smoke accepts it as optional. The wrapper is
  specifically for sandbox proof, not generic publishing.

## Deferred

- Run the wrapper against the Zendesk test account and attach the sanitized
  artifact to the lane issue once credentials and account id are selected.
- Future robust-testing slice: cleanup helper for seeded FAQ draft, macro
  mapping, and sandbox macro lifecycle.
- Future product slice: expose publish proof/history ergonomically if sandbox
  validation shows the flow is ready for operators.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_faq_macro_writeback_sandbox_e2e_smoke.py -q`
  - Result: 10 passed.
- `python -m pytest tests/test_autonomous_faq_macro_writeback_scheduled_publish.py tests/test_faq_macro_writeback_sandbox_e2e_smoke.py tests/test_seed_faq_macro_writeback_live_smoke_draft.py tests/test_scheduler.py::TestDefaults::test_content_ops_faq_macro_writeback_default_seed_respects_opt_in tests/test_scheduler.py::TestDefaults::test_content_ops_faq_macro_writeback_default_notifies_skipped_summary tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_notify_opt_in_uses_fallback_message tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_notify_failure_is_nonfatal tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_without_notify_opt_in_keeps_existing_response_shape -q`
  - Result: 29 passed, 1 warning.
- Python compile check for the sandbox wrapper and test module.
  - Result: passed.
- `python scripts/sync_pr_plan.py plans/PR-FAQ-Macro-Writeback-Sandbox-E2E-Smoke.md --check`
  - Result: passed.
- Pending before push: local PR review via `scripts/push_pr.sh`

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_macro_writeback_checks.yml` | 5 |
| `plans/PR-FAQ-Macro-Writeback-Sandbox-E2E-Smoke.md` | 127 |
| `scripts/smoke_content_ops_faq_macro_sandbox_e2e.py` | 289 |
| `tests/test_faq_macro_writeback_sandbox_e2e_smoke.py` | 298 |
| **Total** | **719** |
