# PR-Diff-Budget-Gate

## Why this slice exists

The Fable 5 arc review (#1934, PRs 1935-1941) found the 400 LOC budget
exceeded 6/6 times with after-the-fact justifications; the #1943 review
concluded lessons must be codified as gates, not prose. This slice lands
gate 1: CI fails any PR whose added lines exceed the budget unless the
body carries a reasoned `Diff-budget override:` marker. Contract unchanged
(AGENTS.md already requires justified overage): model-follows-prompt
becomes CI-enforced, mirroring the reconciliation gate.

## Scope (this PR)

Ownership lane: Workflow/process
Slice phase: Vertical slice

1. `scripts/check_diff_budget.py` -- pure `evaluate()` core + gh fetch;
   exit contract 0/1/2 mirroring the reconciliation gate.
2. `tests/test_check_diff_budget.py` -- under/at/over budget, marker
   variants, placeholder rejection, prose non-markers, CLI offline exits.
3. `.github/workflows/diff_budget.yml` -- pull_request incl. `edited` so
   fixing the body re-runs the check.

### Review Contract

- Acceptance: <= 400 passes; > 400 without marker fails naming count and
  fix; reasoned override passes echoing the reason; placeholders, copied
  <templates>, and fenced examples fail; gh failure or missing additions
  exits 2. One existing file touched: pre_push_audit.yml (CI enrollment).
  Decision core is pure; the gh subprocess fetch is guarded and tested.
- Reviewer rules triggered: R2, R10 (new gate predicate / evaluator),
  R12 (tests enrolled in the pre-push-audit PR workflow).

### Files touched

- `plans/PR-Diff-Budget-Gate.md`
- `scripts/check_diff_budget.py`
- `tests/test_check_diff_budget.py`
- `.github/workflows/diff_budget.yml`
- `.github/workflows/pre_push_audit.yml`

## Mechanism

`evaluate(additions, body, budget)` -> (exit_code, messages). Over budget:
no line-anchored `Diff-budget override:` marker or a placeholder reason ->
fail with instructions; substantive reason -> pass, echoing overage +
reason so every override is a logged, countable event. The CLI fetches
additions/body via `gh pr view --json` into the pure core. Additions only
are counted: deleting code is never penalized.

## Intentional

- **Marker over path-exemptions**: docs/tests/plans count -- the arc's
  every-time justification was "tests and plan dominate"; exempting them
  re-opens that hole. Soft cap stays 400 (operator decision to change);
  required-check enrollment is a repo-settings action for the operator.

## Deferred

Producer-fidelity fixture factory (codification slice 2 from #1943);
adversarial-negatives presence check; per-path budget shaping only if the
simple total proves too blunt. Parked hardening: none.

## Verification

- `python -m pytest tests/test_check_diff_budget.py -q` -- 53 passed
  (incl. review-wave probes: fenced/indented/blockquoted/list-nested
  code markers ignored, <=3-space real marker honored, blockquoted
  markers never count, copied template / punctuation-only / punctuated
  and dash-suffixed placeholders rejected, early placeholder does not
  shadow a later real reason, non-object JSON root and missing/
  non-numeric additions raise).
- Offline smokes: 200 -> exit 0; 900 -> exit 1 with instructions.
- ASCII scan clean; maturity-sweep ratchet green.

## Estimated diff size

| File | LOC |
|---|---:|
| **Total** | **~520** |

The initial slice shipped at exactly 400. The Codex+Copilot wave-1 fixes (fence
stripping, fail-closed additions, template/punctuation rejection, any-match
marker scan, CI enrollment) and their mandated regression tests pushed it
over; the PR body carries a
reasoned Diff-budget override -- the mechanism working as designed on its
own PR, with both paths (fenced example ignored, real marker honored)
exercised live.
