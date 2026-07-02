# PR-Diff-Budget-Required-Check

## Why this slice exists

The diff-budget gate merged in #1944 runs on every PR but is not yet a
required status check, so a red can still be merged past. Making it binding
has two halves: the operator's branch-protection toggle (repo settings,
admin-only), and the repo-side audit expectation. This slice lands the
audit half: `diff-budget` joins `DEFAULT_REQUIRED_CONTEXTS` in the
branch-protection checker, so the weekly/manual `Branch Protection Required
Checks` audit fails if the toggle is missing or ever removed -- the setting
itself becomes drift-detectable, same as the security contexts.

## Scope (this PR)

Ownership lane: Workflow/process
Slice phase: Vertical slice

1. `scripts/check_required_status_checks.py` -- add `diff-budget` to
   `DEFAULT_REQUIRED_CONTEXTS` (pinned to the GitHub Actions app like the
   existing contexts).
2. `tests/test_security_guardrails_workflow.py` -- the five fixtures that
   enumerate the default set gain the new context, including the
   missing-context failure ordering.
3. `docs/SECURITY_GUARDRAILS.md` -- the branch-protection sentence names
   `diff-budget`.
4. Housekeeping: archive the merged `plans/PR-Diff-Budget-Gate.md` and
   regenerate `plans/INDEX.md` (arc convention: rides the next slice).

### Review Contract

- Acceptance: audit fails when `diff-budget` is absent from the live
  payload or present only as a legacy/unpinned context; passes when pinned
  to the GitHub Actions app; docs test asserts the doc names the gitleaks
  checks (unchanged) and the doc paragraph now lists diff-budget.
- Reviewer rules triggered: R10 (gate predicate expectation change),
  R12 (tests already enrolled via test_security_guardrails_workflow.py in
  the pre-push-audit list).

### Files touched

- `plans/PR-Diff-Budget-Required-Check.md`
- `scripts/check_required_status_checks.py`
- `tests/test_security_guardrails_workflow.py`
- `docs/SECURITY_GUARDRAILS.md`
- `plans/INDEX.md`
- `plans/archive/PR-Diff-Budget-Gate.md`

## Mechanism

`DEFAULT_REQUIRED_CONTEXTS` drives both `missing_required_contexts` and the
app-id-pinned `required_status_check_failures`; adding one context makes the
live audit require it with zero logic changes. The operator toggle remains a
repo-settings action; until it is flipped, the next audit run reports the
missing context -- which is the desired signal, not a defect.

## Intentional

- **Audit-first ordering**: landing the expectation before the toggle means
  the audit documents the gap until the operator flips it; that is the
  point of a drift detector.
- **No workflow edits**: the branch-protection workflow's path filters
  already cover the checker script and its test file.

## Deferred

Repo-wide trusted-base-ref execution for gate scripts (#1944 waiver 18);
producer-fidelity fixture factory (codification slice 2). Parked
hardening: none.

## Verification

- `python -m pytest tests/test_security_guardrails_workflow.py
  tests/test_check_diff_budget.py -q` -- 76 passed.
- `python scripts/archive_plans.py index` -- INDEX regenerated.
- ASCII scan on touched .py files: clean.

## Estimated diff size

| File | LOC |
|---|---:|
| **Total** | **~120** |

Well under the 400 cap the gate itself now enforces.
