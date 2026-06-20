# PR-Deflection-Snapshot-Report-Drift-Test

## Why this slice exists

The free deflection snapshot and the paid `deflection.v1` report model are
sibling projections of the same FAQ result: `build_deflection_snapshot()` reads
the FAQ items/summary directly, and `build_deflection_report_model()` builds its
sections from the same items. Neither derives from the other, so a report-shape
change (reordering, renaming, or re-ranking questions; adding a data field) can
silently desync the snapshot, and a stray field can cross the paywall boundary
without any test going red.

This slice adds a deterministic drift guard that pins the snapshot to the report
and reuses the submit smoke's canonical forbidden-field detector, so the leak
half cannot fork its own denylist. It is the cheap bridge agreed before any
registry-level snapshot/report unification: it does not remove the double-edit,
but it converts "we forgot to update the snapshot" from a silent paywall leak
into a failing test. It is intentionally test-only.

The estimate is slightly over the 400 LOC soft cap. The overage is indivisible:
the in-test fixture, the positive derivation assertions, and the failure-first
negative leak cases are one guarantee -- a guard that asserts coverage without
also proving (via injected leaks) that its own detector bites would be a false
sense of safety, so splitting the negatives into a follow-up would ship an
unverified guard. The CI-enrollment edit must also land in the same PR, because
a drift guard that does not run in CI protects nothing.

## Scope (this PR)

Ownership lane: content-ops/deflection-full-report-qa
Slice phase: Production hardening

1. Add `tests/test_deflection_snapshot_report_drift.py`.
2. Assert the snapshot's ranked coverage equals the report model's
   `ranked_questions` (no more, no fewer) so add/drop drift fails in both
   directions.
3. Assert snapshot summary counts, top-question rows, and locked-question rows
   are derived from the report model / artifact summary.
4. Assert the single teaser `full_answer` maps to a genuinely scoped
   `resolution_evidence` `question_details` row, and previews withhold body.
5. Reuse `scripts/smoke_content_ops_deflection_submit_handoff.py`'s
   `_forbidden_key_paths` / `_validate_snapshot` for the leak assertions instead
   of declaring a second denylist.
6. Include failure-first negative tests proving the detector flags an injected
   `source_ids` leak in `top_questions` and an `answer` body leak in a teaser
   preview.
7. Enroll the new test in CI by adding it to the path filters and pytest
   invocation of `.github/workflows/atlas_content_ops_deflection_report_checks.yml`.

### Files touched

- `.github/workflows/atlas_content_ops_deflection_report_checks.yml`
- `plans/PR-Deflection-Snapshot-Report-Drift-Test.md`
- `tests/test_deflection_snapshot_report_drift.py`

### Review Contract

Acceptance criteria:

- [ ] Snapshot ranked-question set equals the report model ranked set.
- [ ] Snapshot summary/top/locked fields equal their report-derived sources.
- [ ] Teaser full answer is a scoped resolution-evidence report row; previews
      carry `body_withheld` and no answer/steps keys.
- [ ] No forbidden paid field appears outside `$.teaser.full_answer`, asserted
      via the smoke's canonical detector.
- [ ] Negative tests fail closed when a leak is injected.

Affected surfaces: deflection snapshot/report contract tests only.

Risk areas: coupling the test to the smoke detector (intended single source of
truth); fixture realism; teaser-selection placement.

Reviewer rules triggered: R1, R2, R10, R14.

## Mechanism

The test builds one artifact from an in-test fixture (two scoped proven items +
two review items) and projects the snapshot at `top_n=2` so top_questions,
locked_questions, the teaser full answer, and a preview are all exercised. It
indexes the report model sections by id and rows by rank, then asserts the
snapshot fields equal the report-derived values. The leak half loads the submit
smoke via `importlib` (registering it in `sys.modules` first so its frozen
dataclasses resolve during class creation) and calls the same detector the
production submit smoke uses.

## Intentional

- Test-only. No producer/contract change; the snapshot and report builders are
  unchanged.
- Reuses the smoke detector rather than re-listing forbidden keys, so the guard
  cannot drift from the enforced paywall boundary.
- Uses an explicit in-test fixture, not a network/live artifact or any
  production environment data.
- Asserts the contracted teaser exception precisely (scoped resolution evidence
  only); it does not loosen the boundary.

## Deferred

- Registry-level unification (a `snapshot`/`free` surface plus per-section
  field-visibility so the snapshot is derived from the report model) remains a
  separate slice, to be done after the resolution-report shape is settled.

Parked hardening: none.

## Verification

- Command: `python -m pytest tests/test_deflection_snapshot_report_drift.py -q`
  -- 10 passed.
- Drift-catch proof: injecting `source_ids` into the snapshot top-question
  projection turns `test_snapshot_has_no_forbidden_paid_fields` red
  (`$.top_questions[0].source_ids`); reverting restores green.
- CI enrollment: the exact workflow pytest invocation (with the new file added)
  -- 45 passed, 3 skipped (pre-existing live-runner skips).
- Command: `python -m pytest tests/test_deflection_snapshot_report_drift.py
  tests/test_content_ops_deflection_report.py
  tests/test_smoke_content_ops_deflection_submit_handoff.py -q` -- 125 passed.
- ASCII gate: `scripts/check_ascii_python.sh` -- passed.
- Workflow YAML: `yaml.safe_load` of the edited workflow -- OK.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_deflection_report_checks.yml` | 3 |
| `plans/PR-Deflection-Snapshot-Report-Drift-Test.md` | 128 |
| `tests/test_deflection_snapshot_report_drift.py` | 286 |
| **Total** | **417** |
