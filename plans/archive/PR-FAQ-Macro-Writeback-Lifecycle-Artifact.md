# PR-FAQ-Macro-Writeback-Lifecycle-Artifact

## Why this slice exists

PR #1599 made the sandbox macro lifecycle one guarded command, but the
operator still needs a safe way to decide whether the resulting JSON artifact
is the proof we meant to capture. The raw artifact can include nested stage
payloads, operational command text, seed copy, and partial-lifecycle states.
Attaching that raw JSON directly to an issue or proof log makes review noisy and
increases the chance that a write-ok/cleanup-failed run is mistaken for a clean
proof.

This slice adds the thinnest post-run artifact gate: read one lifecycle JSON
artifact, fail closed unless the write and cleanup stages both completed, and
emit a sanitized proof summary that carries only IDs, stage/status fields, base
URL, and counts. It does not run Zendesk or Postgres; it validates the artifact
that the already-merged lifecycle wrapper produced.

This PR exceeds the 400 LOC soft cap because the checker is mostly failure
detection logic. The malformed JSON, partial lifecycle states, FAQ-id mismatch,
external-delete proof, summary-redaction, CI enrollment, and no-live-client
guards need to ship with the checker rather than in a later proof-only PR.
Review on #1601 found same-class false-green holes in the first checker version:
truthy string success flags, blank external delete ids, incomplete write stages,
missing or mismatched live-smoke proof, cross-instance base URLs, and non-UTF-8
input crashes. This update fixes those at the artifact gate and expands the
negative fixture matrix so the checker proves those failures fire.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Vertical slice

1. Add an operator-run lifecycle artifact checker for the sandbox FAQ macro
   writeback proof.
2. Require a complete, non-skipped lifecycle artifact: top-level `ok`, stage
   `complete`, non-empty `faq_id`, matching stage FAQ ids, strict JSON boolean
   success flags, write success, live-smoke success, and cleanup success.
3. Require matching non-empty Zendesk base URLs across every stage that records
   one.
4. Fail closed on partial or misleading states: top-level skipped, cleanup
   skipped, write failure, cleanup failure, missing delete count, missing
   external delete proof, blank external delete id, or mismatched FAQ ids.
5. Emit a JSON validation result by default and optionally write a sanitized
   Markdown proof summary.
6. Keep the summary free of seed question/answer text, raw `next_command`
   values, and nested raw payload dumps.
7. Do not add Zendesk, Postgres, scheduler, API, or UI behavior.

### Files touched

- `.github/workflows/atlas_content_ops_macro_writeback_checks.yml`
- `plans/PR-FAQ-Macro-Writeback-Lifecycle-Artifact.md`
- `scripts/check_content_ops_faq_macro_lifecycle_artifact.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_check_content_ops_faq_macro_lifecycle_artifact.py`

### Review Contract

- Acceptance criteria:
  - [ ] Missing or malformed JSON returns non-zero with a specific error.
  - [ ] A complete lifecycle artifact returns green and exposes a sanitized
        summary payload.
  - [ ] Top-level skipped and cleanup-skipped states fail closed.
  - [ ] String-typed success flags fail closed instead of passing by
        truthiness.
  - [ ] Write or cleanup failure fails closed without hiding the stage errors.
  - [ ] Incomplete write stage and missing live-smoke proof fail closed.
  - [ ] Mismatched top-level/write/cleanup/live-smoke FAQ ids fail closed.
  - [ ] Mismatched non-empty Zendesk base URLs fail closed.
  - [ ] Missing, blank-id, or failed external delete proof fails closed.
  - [ ] The optional Markdown proof summary excludes raw seed FAQ copy and
        operational `next_command` text.
  - [ ] The checker source does not import Zendesk or Postgres clients.
- Affected surfaces: operator scripts, macro-writeback validation tests, and
  macro-writeback CI enrollment.
- Risk areas: false-green proof artifacts, leaking raw seed/operator text in a
  summary, and scope drift into live Zendesk/Postgres behavior.
- Reviewer rules triggered: R1, R2, R3, R6, R8, R10, R12, R14.

## Mechanism

The checker reads a JSON file from `--artifact`, validates only the lifecycle
artifact envelope and the nested write/cleanup status fields, then returns a
small validation payload. The payload includes `ok`, `errors`, `account_id`,
`faq_id`, `zendesk_base_url`, stage names, write publishable count, cleanup
deleted count, and external delete ids/statuses.

When `--summary-output` is provided, the checker writes a Markdown proof summary
from the sanitized validation payload. The summary intentionally does not render
raw nested `write`, `cleanup`, `seed`, or `live_smoke` payloads. It also does not
render seed FAQ question/answer copy or command strings such as `next_command`.

## Intentional

- No live Zendesk or Postgres access. This slice validates the artifact after an
  operator-run lifecycle command; it does not trigger the lifecycle itself.
- No raw artifact redump in the Markdown summary. The raw JSON remains available
  locally if needed, but the shareable proof should be small and low-risk.
- No broad schema model. The checker validates only the fields needed to prove
  a clean sandbox write/delete lifecycle.

## Deferred

- Run the lifecycle wrapper against the Zendesk test account and validate the
  produced artifact with this checker.
- Future robust-testing slice: batch manifest validation if repeated lifecycle
  artifacts become common.

Parked hardening: none.

## Verification

- python -m pytest tests/test_check_content_ops_faq_macro_lifecycle_artifact.py -q
  - 33 passed.
- python -m pytest tests/test_autonomous_faq_macro_writeback_scheduled_publish.py tests/test_check_content_ops_faq_macro_lifecycle_artifact.py tests/test_cleanup_content_ops_faq_macro_sandbox.py tests/test_extracted_ticket_faq_macro_writeback_zendesk.py tests/test_faq_macro_writeback_sandbox_e2e_smoke.py tests/test_faq_macro_writeback_sandbox_lifecycle.py tests/test_seed_faq_macro_writeback_live_smoke_draft.py tests/test_scheduler.py::TestDefaults::test_content_ops_faq_macro_writeback_default_seed_respects_opt_in tests/test_scheduler.py::TestDefaults::test_content_ops_faq_macro_writeback_default_notifies_skipped_summary tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_notify_opt_in_uses_fallback_message tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_notify_failure_is_nonfatal tests/test_synthesis.py::TestRunBuiltinSynthesis::test_skip_synthesis_without_notify_opt_in_keeps_existing_response_shape -q
  - 98 passed, 1 warning.
- python -m py_compile scripts/check_content_ops_faq_macro_lifecycle_artifact.py tests/test_check_content_ops_faq_macro_lifecycle_artifact.py
- python scripts/audit_extracted_pipeline_ci_enrollment.py
  - OK: 183 matching tests are enrolled.
- bash scripts/check_ascii_python.sh
  - ASCII check passed for extracted_content_pipeline Python files.
- python scripts/sync_pr_plan.py plans/PR-FAQ-Macro-Writeback-Lifecycle-Artifact.md --check
  - plan already in sync.
- Pending before push: local PR review via `scripts/push_pr.sh`

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_macro_writeback_checks.yml` | 5 |
| `plans/PR-FAQ-Macro-Writeback-Lifecycle-Artifact.md` | 136 |
| `scripts/check_content_ops_faq_macro_lifecycle_artifact.py` | 263 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_check_content_ops_faq_macro_lifecycle_artifact.py` | 303 |
| **Total** | **708** |
