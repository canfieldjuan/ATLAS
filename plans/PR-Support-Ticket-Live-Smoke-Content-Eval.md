# Support Ticket Live Smoke Content Eval

## Why this slice exists

PR #953 added a standalone deterministic evaluator for saved-draft exports. That
lets operators check whether generated landing/blog copy actually uses the
support-ticket facts that survived export, but it is still a separate manual
step after the live smoke.

The next product slice is to wire that evaluator into the live generation smoke
as an optional flag, so one command can generate, export, validate source
context, and fail on obvious generated-copy drift.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: Product polish

1. Add `--evaluate-generated-content` to the live Content Ops generation smoke.
2. Require that flag to run with `--support-ticket-csv` and
   `--export-saved-draft`, because the evaluator depends on support-ticket
   export context.
3. Run the deterministic evaluator against the in-memory saved-draft export
   after source-context export validation passes.
4. Add the evaluator result to the smoke result JSON.
5. Fail the smoke when the evaluator returns errors.
6. Add focused tests for passing evaluation, failing evaluation, and missing
   required flags.

### Files touched

- `plans/PR-Support-Ticket-Live-Smoke-Content-Eval.md`
- `scripts/smoke_content_ops_live_generation.py`
- `tests/test_smoke_content_ops_live_generation.py`

## Mechanism

The smoke keeps the generated-content evaluator optional. When
`--evaluate-generated-content` is present, the smoke first ensures
`--support-ticket-csv` and `--export-saved-draft` are also present. After
generation returns saved draft ids and the existing export/context checks pass,
the smoke calls `evaluate_support_ticket_generated_content(saved_draft_export,
output=<landing_page|blog_post>)`.

The evaluator payload is stored as `generated_content_evaluation` in the smoke
result. Any evaluator errors are prefixed and appended to the smoke errors so
CLI callers get a non-zero exit and a readable failure reason.

## Intentional

- No prompt or generator changes. This only wires the existing evaluator into
  the operator smoke path.
- No FAQ generator changes. FAQ-owned work remains separate.
- The evaluator does not run unless explicitly requested.
- The smoke reuses the in-memory export result; it does not reread the JSON file
  it just wrote.
- Local review caller hints are generic name collisions (`_parse_args`,
  test-local `_evaluate`) across unrelated scripts/tests; this slice only
  changes `scripts/smoke_content_ops_live_generation.py` behavior and covers it
  with focused smoke tests.

## Deferred

- Future PR: run this flag in a recorded live Haiku validation and archive the
  evaluator output.
- Future PR: decide whether the flag should become default for support-ticket
  live smokes after we see live output quality over several runs.
- Parked hardening: none added by this slice.

## Verification

- `python -m pytest tests/test_smoke_content_ops_live_generation.py::test_live_generation_smoke_real_generated_content_evaluator_imports tests/test_smoke_content_ops_live_generation.py::test_live_generation_smoke_evaluates_support_ticket_landing_export_content tests/test_smoke_content_ops_live_generation.py::test_live_generation_smoke_fails_when_generated_content_evaluation_fails tests/test_smoke_content_ops_live_generation.py::test_live_generation_smoke_requires_export_for_generated_content_evaluation tests/test_smoke_content_ops_live_generation.py::test_live_generation_smoke_requires_support_ticket_csv_for_content_evaluation -q`
  - 5 passed.
- `python -m pytest tests/test_smoke_content_ops_live_generation.py -q`
  - 33 passed.
- `python -m pytest tests/test_smoke_content_ops_live_generation.py tests/test_evaluate_support_ticket_generated_content.py -q`
  - 42 passed.
- Py compile for `scripts/smoke_content_ops_live_generation.py`,
  `tests/test_smoke_content_ops_live_generation.py`, and
  `scripts/evaluate_support_ticket_generated_content.py`
  - Passed.
- Local PR review wrapper
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~90 |
| Smoke script | ~90 |
| Tests | ~220 |
| **Total** | **~400** |
