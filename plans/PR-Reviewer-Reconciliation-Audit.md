# PR-Reviewer-Reconciliation-Audit

## Why this slice exists

Slice S1 (PR #1330, merged) added the AI-finding reconciliation rule to
`docs/REVIEWER_RULES.md` and `AGENTS.md` section 4a.1 as reviewer *discipline*.
This slice S2 of the review-workflow redesign (issue #1328) makes the
reconciliation record mechanically checkable -- the operating-model gap (b)
fix in its enforceable form. Without a gate, "fixed or waived before LGTM"
is just another line a tired reviewer can skip; S1's own Deferred section
names this audit as the next step.

Diff budget: ~470 LOC, over the 400 soft cap; the plan doc is ~128 of that and
the audit+fixtures (grown by the review-round hardening of the section parser
and resolution markers) are the irreducible substance. Net tooling prose
excluding the plan doc is ~345 LOC.

## Scope (this PR)

Ownership lane: dev-workflow/review-contract
Slice phase: Workflow/process

1. Add `scripts/audit_ai_reconciliation.py`: parse a PR body file and fail
   closed when its "AI reconciliation" record is internally unresolved (a
   finding marked neither fixed nor waived, a waiver with no reason, or a
   section present with no resolution marker). Optional `--require` also fails
   when the section is absent.
2. Add `tests/test_audit_ai_reconciliation.py`: failure-proving fixtures for
   each detection branch plus a lookalike-rejection and the CLI exit-code
   contract, per AGENTS.md section 3i.
3. Wire the audit into `scripts/local_pr_review.sh` (after the cross-session
   drift check, reusing the existing PR-body-file plumbing) and enroll the new
   test in `.github/workflows/pre_push_audit.yml` in the same PR (section 3e).

### Review Contract

- Acceptance criteria:
  - [ ] `scripts/audit_ai_reconciliation.py` fails (exit 1) on an unresolved
        record, a reason-less waiver, and a resolution-less section; passes on
        a resolved record or (without `--require`) an absent section.
  - [ ] A prose mention of "reconciliation" that is not a heading is not
        treated as the section (no false trigger).
  - [ ] The audit runs inside `scripts/local_pr_review.sh` and the fixture test
        runs in `.github/workflows/pre_push_audit.yml`.
  - [ ] Existing `tests/test_local_pr_review.py` still passes with the new
        bundle check.
- Affected surfaces: dev workflow tooling only (one new script, one new test,
  one bundle edit, one workflow edit). No product API/DB/auth/frontend surface.
- Risk areas: a false-positive that blocks a legitimate PR. Mitigated by
  defaulting to non-`--require` (an absent section passes) and by the
  near-miss/lookalike fixtures; the regexes only fire on explicit unresolved
  markers.
- Reviewer rules triggered: R2 (the audit is a detector -- failure branches
  proven), R10 (maintainable), R12 (the new test is CI-enrolled this PR).

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `plans/INDEX.md`
- `plans/PR-Reviewer-Reconciliation-Audit.md`
- `plans/archive/PR-Reviewer-Rules-Contract.md`
- `scripts/audit_ai_reconciliation.py`
- `scripts/local_pr_review.sh`
- `tests/test_audit_ai_reconciliation.py`

## Mechanism

`scripts/audit_ai_reconciliation.py` resolves a PR body file from
`--current-pr-body-file` or `ATLAS_CURRENT_PR_BODY_FILE` (mirroring
`scripts/audit_pr_session_drift.py`), anchors the reconciliation record on a
heading-like line (`## AI reconciliation`, `**AI reconciliation**`, or a line
starting with it) so prose mentions do not match, and bounds the record at the
next same-or-higher-level heading so subheadings (`### Codex`) stay inside it.
Within the record it fails closed on an unresolved marker (`fixed or waived:
no`, `findings still open`, ...), a waiver line with no rationale, or -- when a
record exists -- the absence of any resolution marker (`all fixed or waived:
yes`, `no findings`, `nothing to reconcile`; a bare `no findings waived` is
deliberately not a resolution marker). Local
tooling cannot read live GitHub bot threads (the bundle has no `gh`), so this
enforces the half that is checkable from the body: a recorded reconciliation
can be trusted. The check is added to `scripts/local_pr_review.sh` next to the
drift audit and passes the body file when one is set; `reconciliation_errors`
and `extract_section` are unit-covered directly.

## Intentional

- Default is non-`--require`: a PR opened before any bot has reviewed has no
  findings to reconcile, so an absent section passes. The audit only fails on a
  *malformed or unresolved* record -- it makes a recorded reconciliation
  trustworthy rather than forcing the section prematurely. This matches gap
  (b)'s honesty that there is no full mechanical fix.
- This branch also carries the section-1g teardown for S1: the merged plan
  `plans/PR-Reviewer-Rules-Contract.md` is moved to `plans/archive/` and
  `plans/INDEX.md` is regenerated. AGENTS.md section 1g explicitly sanctions
  folding that move into the next branch off main.

## Deferred

- S3: a path-to-rule trigger audit that derives required rule IDs from the diff
  and fails when the plan's triggered-rules line omits one.
- CI-side reconciliation: a workflow step that fetches the live Codex/Copilot
  threads (needs `gh`/API) and fails when a recorded reconciliation omits a
  real open finding. The local audit only validates the recorded body.
- S4: a reviewer-metrics summarizer over `REVIEW_MISSES.md`.

Parked hardening: none.

## Verification

- `tests/test_audit_ai_reconciliation.py` -- 14 fixtures pass (each detection
  branch, lookalike + near-miss rejection, subheading retention, CLI exit
  codes 0/1/2).
- `tests/test_local_pr_review.py` -- unchanged suite still passes with the new
  bundle check.
- `scripts/audit_ai_reconciliation.py` run on a resolved vs unresolved body --
  exit 0 vs exit 1.
- `scripts/check_ascii_python.sh` -- ASCII-clean.
- `scripts/local_pr_review.sh` -- full mechanical bundle green before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 2 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Reviewer-Reconciliation-Audit.md` | 131 |
| `plans/archive/PR-Reviewer-Rules-Contract.md` | 0 |
| `scripts/audit_ai_reconciliation.py` | 179 |
| `scripts/local_pr_review.sh` | 12 |
| `tests/test_audit_ai_reconciliation.py` | 148 |
| **Total** | **475** |
