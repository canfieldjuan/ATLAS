# PR-Trusted-Base-Gate-Execution

## Why this slice exists

Closes #1944 waiver 18, named there as the priority follow-up: every PR gate
executes its own scripts from the PR merge ref, so a PR that edits a gate
script (or the gate's workflow file) is judged by its own edited gate and can
self-pass. Root cause: the gates were built on `pull_request` checkouts,
which materialize the untrusted merge ref; nothing in the repo ran gate code
from the base. The repo already carries the reviewed fix pattern -- the
Gitleaks baseline growth guard runs on `pull_request_target` with a
trusted-base checkout -- but it was never applied to the other gates. This
slice applies that existing pattern to the three API-driven gates
(diff-budget, live-reconciliation, pr-body-contract); patching just one gate
or re-describing the risk in docs would treat the symptom.

Diff-budget overage (589 added lines vs the 400 cap): the workflow posture
audit ITSELF gates pull_request_target adoption behind its allowlist plus
guard-shape check, so the three workflow conversions, the allowlist
expansion, its failure-branch tests, and the review-events job split are one
indivisible slice -- shipping the workflows without the allowlist leaves CI
red; shipping the allowlist without the workflows allowlists nothing.

## Scope (this PR)

Ownership lane: Workflow/process
Slice phase: Vertical slice

1. `.github/workflows/diff_budget.yml`,
   `.github/workflows/ai_reconciliation_live.yml`, and
   `.github/workflows/pr_body_contract.yml`: convert to `pull_request_target` with a checkout of
   `github.event.pull_request.base.sha` (the Gitleaks-guard shape, checkout
   action SHA-pinned the same way). Scripts and workflow logic now come
   from the trusted base on every run.
2. `scripts/audit_pr_body.py`: the plan-doc-exists check gains a
   `--plan-git-ref` mode (git cat-file against the fetched PR head ref),
   because new plan docs arrive WITH the PR and a base checkout cannot see
   them. Pure core takes an injectable `plan_exists` callable; the git
   boundary fails closed (unresolvable ref -> exit 2, never a silent pass).
3. `tests/test_audit_pr_body.py`: failure-branch fixtures for the new mode
   (plan present at ref, missing at ref, unresolvable ref, callable
   injection).
4. `scripts/audit_workflow_security_posture.py`: the pull_request_target
   allowlist generalizes from the single Gitleaks-guard tuple to the four
   audited gate jobs -- allowlisting is necessary but not sufficient; each
   entry must still match the guard shape (if-guard + SHA-pinned base-SHA
   checkout). live-reconciliation splits into the required target-only job
   plus an advisory review-events job (same base-ref script) so bot-comment
   re-runs survive the guard shape.
5. `.github/workflows/ai_reconciliation_review_retrigger.yml` (review round
   2): review-event runs resolve the workflow file PR-side and cannot be
   trusted to refresh the required context, so this default-branch
   `workflow_run` follow-up re-runs the latest `pull_request_target` run
   for the same head SHA whenever a review event lands -- the required
   `live-reconciliation` context stays fresh through trusted code.
   Loop-safe: reruns arrive back with event == pull_request_target and are
   ignored.
6. `docs/SECURITY_GUARDRAILS.md`: document which gates run trusted-base and
   the residual PR-ref surface (named below).

### Files touched

- `.github/workflows/diff_budget.yml`
- `.github/workflows/ai_reconciliation_live.yml`
- `.github/workflows/pr_body_contract.yml`
- `.github/workflows/ai_reconciliation_review_retrigger.yml`
- `scripts/audit_pr_body.py`
- `scripts/audit_workflow_security_posture.py`
- `tests/test_audit_pr_body.py`
- `tests/test_audit_workflow_security_posture.py`
- `docs/SECURITY_GUARDRAILS.md`
- `plans/PR-Trusted-Base-Gate-Execution.md`

### Review Contract

Acceptance criteria (reviewer checks one-by-one):

1. All three converted workflows trigger on `pull_request_target` (plus the
   existing review events for live-reconciliation), check out
   `github.event.pull_request.base.sha`, and never execute PR-ref code.
2. Check-run names (`diff-budget`, `live-reconciliation`,
   `pr-body-contract`) are unchanged, so #1945's required-check audit
   expectations still match.
3. `audit_pr_body.py --plan-git-ref` fails closed: a plan doc missing at
   the head ref is a contract failure naming the ref; an unresolvable ref
   exits 2 (infra), never 0.
4. The default (no flag) filesystem behavior is byte-for-byte unchanged --
   local usage and the pre-push bundle are unaffected.
5. The posture audit passes with the three gates reported as allowed
   guard-shaped jobs, and an allowlisted gate that drops the if-guard or
   the pinned checkout still errors (tested).
6. `python -m pytest tests/test_audit_pr_body.py
   tests/test_audit_workflow_security_posture.py -q` passes, including the
   new failure branches.

Affected surfaces: three gate workflows, one gate script + its tests, one
security doc. No product code.

Risk areas: `pull_request_target` grants base-context tokens -- safe here
because the jobs run only base-ref code and keep read-only permissions, but
any FUTURE edit that executes PR-ref content in these jobs is a privilege
escalation; the guardrails doc says so explicitly. Migration window: see
Intentional.

Reviewer rules triggered: R2, R10 (gate predicate change in
`scripts/audit_pr_body.py` -> failure-branch fixtures per AGENTS.md 3h/3i,
plus the audit-script row), R14 (checked-out PR-head verification).

## Mechanism

`pull_request_target` events resolve the workflow file from the BASE branch,
and the explicit `base.sha` checkout materializes base-ref scripts, so after
this merges, neither a gate script edit nor a gate workflow edit in a PR
changes what judges that PR. diff-budget and live-reconciliation read
everything they need from the GitHub API (additions, body, review threads)
and need no PR tree at all. pr-body-contract additionally fetches
`pull/<n>/head` as a named ref and asks git (`ls-tree`, requiring a
regular-file blob -- a tree or symlink at the plan path is not a plan doc)
whether the plan doc exists there -- inspection of untrusted content, never
execution.

## Intentional

- **Migration window:** on THIS PR the three gates do not run at all --
  `pull_request` events consult the merge-ref workflows (now
  target-only) and `pull_request_target` consults main's pre-conversion
  files. First effective runs happen on the next PR after merge. The other
  gates (pre-push-audit and reddit/tooling suites) still cover this PR.
- **Residual PR-ref surface, named not hidden:** pre-push-audit and the
  maturity sweeps still run PR-ref scripts -- they operate on the PR tree
  by design (diff audits, tooling pytest) and need a different treatment;
  review events resolve workflow files PR-side, so the required
  `live-reconciliation` context is fed only by trusted target runs, and
  the default-branch review-retrigger follow-up re-runs the latest target
  run on every review event (keeping the required context fresh without
  ever trusting PR-side code). Consistent with the #1944 threat model:
  friction and a logged decision for honest-but-hasty authors, not
  adversary-proofing.
- Check names, triggers' type lists, and script invocations are otherwise
  unchanged -- this slice moves WHERE gate code comes from, not what it does.

## Deferred

- Trusted-base treatment for the tree-dependent gates (pre-push-audit,
  maturity sweeps): needs a scripts-from-base / tree-from-PR split and is
  its own slice.
- Factory adoption in the remaining reddit test files (#1947 Deferred).

Parked hardening: none.

## Verification

Commands run from the repo root:

- `python -m pytest tests/test_audit_pr_body.py
  tests/test_audit_workflow_security_posture.py -q` -- pass count recorded
  in the PR body, including the new failure branches on both scripts.
- `python scripts/audit_workflow_security_posture.py .github/workflows` --
  passed; the three converted gates report as allowed guard-shaped jobs.
- `bash scripts/local_pr_review.sh --current-pr-body-file <pr-body.md>` --
  all checks PASS.
- `python scripts/check_diff_budget.py --additions 589 --body-file
  <pr-body.md>` -- within the 400 budget.
- Post-merge, first PR: confirm the three checks report from
  `pull_request_target` runs (workflow run event visible in the check-run
  URL), then flip/keep branch-protection required contexts.

## Estimated diff size

| File | LOC (added) |
|---|---:|
| workflows (4) | 129 |
| `scripts/audit_pr_body.py` | 74 |
| `scripts/audit_workflow_security_posture.py` | 13 |
| `tests/test_audit_pr_body.py` | 110 |
| `tests/test_audit_workflow_security_posture.py` | 68 |
| `docs/SECURITY_GUARDRAILS.md` | 16 |
| `plans/PR-Trusted-Base-Gate-Execution.md` | 179 |
| **Total** | **589** |
