# PR-FAQ-Macro-Writeback-Proof-Directory

## Why this slice exists

PR #1602 made the sandbox lifecycle command write a raw artifact and sanitized
Markdown proof summary from the same in-memory payload. The remaining operator
footgun is path selection: the live proof command still asks the operator to
hand-type two related paths. A typo can leave the raw JSON in one location and
the shareable summary somewhere else, or accidentally overwrite a stale proof
from a previous run.

This slice adds the narrow proof-directory mode before the live Zendesk proof
run. It does not write to Zendesk, Postgres, or GitHub by itself. It makes the
existing operator command choose the artifact and summary filenames together,
fail closed on conflicting output arguments, and refuse stale proof-directory
contents before any database pool or Zendesk operation can begin.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Vertical slice

1. Add `--proof-dir` to the sandbox lifecycle wrapper as a convenience mode for
   writing both lifecycle proof files.
2. In `--proof-dir` mode, write the lifecycle artifact and Markdown summary in
   that directory by wiring the existing `--output` and `--summary-output`
   behavior.
3. Reject `--proof-dir` combined with explicit `--output` or
   `--summary-output` so one invocation has one output contract.
4. Refuse an existing proof directory that already contains files before
   preflight/database/Zendesk work begins.
5. Preserve existing stdout, `--output`, and `--summary-output` behavior when
   `--proof-dir` is not used.
6. Do not add live Zendesk execution, provider behavior, cleanup behavior, API,
   scheduler, or UI changes.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Proof-Directory.md`
- `scripts/smoke_content_ops_faq_macro_sandbox_lifecycle.py`
- `tests/test_faq_macro_writeback_sandbox_lifecycle.py`

### Review Contract

- Acceptance criteria:
  - [ ] `--proof-dir` writes raw JSON and sanitized Markdown proof files from
        the same lifecycle payload.
  - [ ] `--proof-dir` cannot be combined with explicit `--output` or
        `--summary-output`.
  - [ ] A non-empty proof directory is rejected before opening a database pool
        or running the write stage.
  - [ ] Existing output modes keep their current behavior.
  - [ ] Preflight skip behavior remains before pool creation.
  - [ ] The wrapper still does not define Zendesk transport/provider classes.
- Affected surfaces: operator lifecycle script and macro-writeback lifecycle
  tests.
- Risk areas: accidental stale proof reuse, output path mismatch, and scope
  drift into live provider behavior.
- Reviewer rules triggered: R1, R2, R3, R6, R8, R10, R14.

## Mechanism

Argument normalization runs immediately after parsing. If `--proof-dir` is set,
the wrapper refuses explicit output paths, checks the directory contents before
any preflight/database work, creates the directory when safe, and fills
`args.output` and `args.summary_output` with the paired proof-file paths.

The rest of the lifecycle flow remains the PR #1602 path: the wrapper produces
one lifecycle payload, validates the same payload through the artifact checker
when summary output is requested, writes the sanitized Markdown summary, and
then writes the raw JSON artifact.

## Intentional

- No automatic live run. The existing confirmation flags and credential
  requirements remain unchanged.
- No timestamped filenames. A single proof directory is simpler to reference
  and safer to review; stale contents are rejected instead of silently creating
  another variant.
- No GitHub issue upload. Attaching the sanitized summary is still an operator
  step after the live proof run succeeds.

## Deferred

- Run the proof-directory command against the Zendesk test account and attach
  the sanitized summary to the lane tracker/proof log.

Parked hardening: none.

## Verification

- python -m pytest tests/test_faq_macro_writeback_sandbox_lifecycle.py -q
  - 19 passed.
- python -m pytest tests/test_autonomous_faq_macro_writeback_scheduled_publish.py tests/test_check_content_ops_faq_macro_lifecycle_artifact.py tests/test_cleanup_content_ops_faq_macro_sandbox.py tests/test_extracted_ticket_faq_macro_writeback_zendesk.py tests/test_faq_macro_writeback_sandbox_e2e_smoke.py tests/test_faq_macro_writeback_sandbox_lifecycle.py tests/test_seed_faq_macro_writeback_live_smoke_draft.py tests/test_scheduler.py::TestDefaults::test_content_ops_faq_macro_writeback_default_seed_respects_opt_in tests/test_scheduler.py::TestDefaults::test_content_ops_faq_macro_writeback_default_notifies_skipped_summary tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_notify_opt_in_uses_fallback_message tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_notify_failure_is_nonfatal tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_without_notify_opt_in_keeps_existing_response_shape -q
  - 106 passed, 1 warning.
- python -m py_compile scripts/smoke_content_ops_faq_macro_sandbox_lifecycle.py tests/test_faq_macro_writeback_sandbox_lifecycle.py
- python scripts/audit_extracted_pipeline_ci_enrollment.py
  - OK: 183 matching tests are enrolled.
- bash scripts/check_ascii_python.sh
  - ASCII check passed for extracted_content_pipeline Python files.
- python scripts/sync_pr_plan.py plans/PR-FAQ-Macro-Writeback-Proof-Directory.md --check
  - plan already in sync.
- Pending before push: local PR review via `scripts/push_pr.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-FAQ-Macro-Writeback-Proof-Directory.md` | 112 |
| `scripts/smoke_content_ops_faq_macro_sandbox_lifecycle.py` | 26 |
| `tests/test_faq_macro_writeback_sandbox_lifecycle.py` | 85 |
| **Total** | **223** |
