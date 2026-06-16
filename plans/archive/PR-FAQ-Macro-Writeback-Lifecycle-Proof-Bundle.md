# PR-FAQ-Macro-Writeback-Lifecycle-Proof-Bundle

## Why this slice exists

PR #1599 made the sandbox FAQ macro lifecycle one guarded command, and PR #1601
added a fail-closed checker plus sanitized Markdown proof summary for the JSON
artifact. The remaining operator gap is that the real proof still requires two
manual commands: run lifecycle, then remember to run the checker against the
right output file. That creates the same class of live-proof mistake we have
been removing in this lane: a raw lifecycle artifact can be captured without
the sanitized proof summary, or a summary can be generated from the wrong file.

This slice bundles the two existing pieces without adding new live behavior.
The lifecycle wrapper can now write the raw JSON artifact and, when requested,
write the sanitized proof summary from the same in-memory lifecycle payload.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Vertical slice

1. Add an optional `--summary-output` argument to the sandbox lifecycle wrapper.
2. After the lifecycle payload is produced, validate that exact payload through
   the existing lifecycle artifact checker and write the sanitized Markdown
   summary to `--summary-output`.
3. Preserve existing raw JSON behavior: stdout by default, `--output` when
   provided.
4. Keep preflight skips before any database pool is opened.
5. If the lifecycle stage returns success but the checker rejects the payload,
   return non-zero with a `proof_summary_validation_failed` error instead of
   false-greening the run.
6. Do not add new Zendesk, Postgres, provider, scheduler, API, or UI behavior.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Lifecycle-Proof-Bundle.md`
- `scripts/smoke_content_ops_faq_macro_sandbox_lifecycle.py`
- `tests/test_faq_macro_writeback_sandbox_lifecycle.py`

### Review Contract

- Acceptance criteria:
  - [ ] A successful lifecycle run with `--summary-output` writes raw JSON and
        a sanitized Markdown proof summary from the same payload.
  - [ ] Preflight skips with `--summary-output` still skip before opening a
        database pool and write a failure summary.
  - [ ] A lifecycle failure writes a failure summary without masking the
        lifecycle exit code.
  - [ ] If a green lifecycle payload fails proof validation, the wrapper
        returns non-zero and records `proof_summary_validation_failed`.
  - [ ] The summary excludes raw seed FAQ copy and operational `next_command`
        text by delegating to the existing checker.
  - [ ] The wrapper still does not define Zendesk transport/provider classes.
- Affected surfaces: operator lifecycle script and macro-writeback lifecycle
  tests.
- Risk areas: false-green proof bundles, summary/raw artifact mismatch, and
  scope drift into live provider behavior.
- Reviewer rules triggered: R1, R2, R3, R6, R8, R10, R14.

## Mechanism

The lifecycle wrapper dynamically loads
`scripts/check_content_ops_faq_macro_lifecycle_artifact.py` alongside the
existing E2E and cleanup scripts. The wrapper still produces the lifecycle
payload exactly once. If `--summary-output` is provided, it passes the in-memory
payload to `validate_lifecycle_artifact`, renders the sanitized Markdown via
`render_summary`, and writes that Markdown to the requested path.

If the lifecycle exit code is already non-zero, the wrapper keeps that exit code
and writes a failure summary. If the lifecycle exit code is zero but validation
fails, the wrapper changes the exit code to non-zero and adds the checker
errors to the lifecycle JSON payload under `proof_summary_errors`.

## Intentional

- No new validation rules. The proof bundle delegates validation and Markdown
  rendering to the checker from PR #1601.
- No automatic live execution. The existing lifecycle confirmation flags remain
  required.
- No raw JSON redump inside the Markdown summary. The summary stays sanitized by
  construction because the checker owns rendering.
- No UI or route. This is still an operator-run proof command.

## Deferred

- Run the proof bundle against the Zendesk test account and attach the sanitized
  summary to the lane issue or proof log.
- Future robust-testing slice: batch proof bundle validation if repeated
  lifecycle artifacts become common.

Parked hardening: none.

## Verification

- python -m pytest tests/test_faq_macro_writeback_sandbox_lifecycle.py -q
  - 15 passed.
- python -m pytest tests/test_autonomous_faq_macro_writeback_scheduled_publish.py tests/test_check_content_ops_faq_macro_lifecycle_artifact.py tests/test_cleanup_content_ops_faq_macro_sandbox.py tests/test_extracted_ticket_faq_macro_writeback_zendesk.py tests/test_faq_macro_writeback_sandbox_e2e_smoke.py tests/test_faq_macro_writeback_sandbox_lifecycle.py tests/test_seed_faq_macro_writeback_live_smoke_draft.py tests/test_scheduler.py::TestDefaults::test_content_ops_faq_macro_writeback_default_seed_respects_opt_in tests/test_scheduler.py::TestDefaults::test_content_ops_faq_macro_writeback_default_notifies_skipped_summary tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_notify_opt_in_uses_fallback_message tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_notify_failure_is_nonfatal tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_without_notify_opt_in_keeps_existing_response_shape -q
  - 102 passed, 1 warning.
- python -m py_compile scripts/smoke_content_ops_faq_macro_sandbox_lifecycle.py tests/test_faq_macro_writeback_sandbox_lifecycle.py
- python scripts/audit_extracted_pipeline_ci_enrollment.py
  - OK: 183 matching tests are enrolled.
- bash scripts/check_ascii_python.sh
  - ASCII check passed for extracted_content_pipeline Python files.
- python scripts/sync_pr_plan.py plans/PR-FAQ-Macro-Writeback-Lifecycle-Proof-Bundle.md --check
  - plan already in sync.
- Pending before push: local PR review via `scripts/push_pr.sh`

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-FAQ-Macro-Writeback-Lifecycle-Proof-Bundle.md` | 115 |
| `scripts/smoke_content_ops_faq_macro_sandbox_lifecycle.py` | 40 |
| `tests/test_faq_macro_writeback_sandbox_lifecycle.py` | 206 |
| **Total** | **361** |
