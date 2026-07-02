# PR-Fable5-Lessons-Index

## Why this slice exists

The original lessons-doc PR (#1943 as first opened) was reviewed with two
blockers -- wrong base branch and documentation-only "codification" -- and
its branch was subsequently repurposed into the retired-failure-mode
investigation brief that merged instead. The lessons synthesis therefore
never landed. This slice lands it in the shape the review demanded: an
INDEX that maps every lesson to its enforcement status (gate / audit /
tracked waiver / investigation), which is now possible because the
enforcement exists (#1944 diff-budget gate, #1945 required-check audit,
the pre-existing reconciliation gates, and the merged detection-layer
brief). The doc's standing rule forbids adding prose-only lessons.

## Scope (this PR)

Ownership lane: Workflow/process
Slice phase: Workflow/process

1. `docs/fable5_pr_1935_1941_review_lessons.md` -- the lessons index:
   7-row lesson-to-enforcement map, judgment practices, deferral
   boundaries (#1936/#1942 class), positive results, and the add-a-row
   contract.
2. Housekeeping: archive the merged required-check plan to
   `plans/archive/PR-Diff-Budget-Required-Check.md` and regenerate
   `plans/INDEX.md` (arc convention: rides the next slice).

### Review Contract

- Acceptance: every lesson row carries ENFORCED/OPEN/TRACKED/
  INVESTIGATION with a named artifact; enforced rows name gates that
  exist on main; no acceptance criterion claims a test that does not
  exist (the exact defect Copilot caught on #1945's plan).
- Reviewer rules triggered: none auto-triggered (documentation-only
  diff); R14 applies to reviewer verdict handling as usual.

### Files touched

- `plans/PR-Fable5-Lessons-Index.md`
- `docs/fable5_pr_1935_1941_review_lessons.md`
- `plans/INDEX.md`
- `plans/archive/PR-Diff-Budget-Required-Check.md`

## Mechanism

Documentation-only. The index derives from the recorded audits (#1934
comments, #1940/#1941 reviews, #1944/#1945 reconciliations) and links
each row to on-main artifacts, so drift is checkable by reading the
named files.

## Intentional

- **Same filename as the original attempt**: the #1943 review referenced
  this path; history and links stay coherent.
- **No new gates in this slice**: the two OPEN rows (producer-fidelity
  fixtures, negatives-presence check) are named follow-up slices, not
  smuggled implementations.

## Deferred

Codification slice 2 (producer-fidelity fixture factory); the
negatives-presence check; the repo-wide trusted-base workflow slice
(#1944 waiver 18). Parked hardening: none.

## Verification

- Every ENFORCED row's artifact exists on main:
  `ls .github/workflows/diff_budget.yml scripts/check_diff_budget.py
  scripts/check_required_status_checks.py
  scripts/check_ai_reconciliation_live.py
  docs/retired_failure_mode_detection_layer.md` -- all present.
- `python scripts/archive_plans.py index` -- INDEX regenerated.
- Documentation-only: no runtime tests required (no .py changes).

## Estimated diff size

| File | LOC |
|---|---:|
| **Total** | **~160** |

Under the 400 cap; no override needed.
