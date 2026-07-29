# PR-Review-Contract-PreAdmission

## Why this slice exists

#2104/#2035 call for process gates to move from advisory/hidden checks toward
independently visible branch-protection contexts. #2119 and #2123 added the
actual `session-lane` and `plan-admission` producers after each workflow/job
tuple was pre-admitted by the trusted-base workflow-security auditor.

The next producer in the same lane is Review Contract. A PR cannot safely add a
new `pull_request_target` workflow and its workflow-security allowlist entry in
the same diff, because base-owned `pre-push-audit` evaluates PR workflow files
with the base branch auditor. This slice pre-admits the future
review_contract.yml / review-contract workflow only as one canonical normalized
YAML string so the next producer slice can add exactly that workflow without
weakening base-owned workflow-security review.

### Problem-derived contract

- Root cause: the base workflow-security posture auditor does not yet know the
  future review_contract.yml / review-contract workflow/job pair, so the later
  producer PR would be rejected before its concrete trusted-base workflow can be
  audited; a bare tuple seed is too open for this gate because it admits later
  unsafe steps and write permissions once the first trusted-base checkout
  matches.
- Correct fix must touch/change: add the future workflow/job tuple to
  `ALLOWED_PULL_REQUEST_TARGET_JOBS`, but require review_contract.yml to match
  one canonical normalized YAML workflow string before the tuple is accepted;
  add regression coverage proving the canonical workflow passes and a drifted
  workflow with unsafe write permissions/later merge command fails; document the
  pre-admission/bootstrap split; archive the merged #2123 plan by name and
  refresh `plans/INDEX.md`.
- Must not change: do not add the future review_contract.yml workflow file, do not
  mutate branch protection, do not mark Review Contract required, do not change
  plan-doc or reviewer-rule audit semantics, do not change `plan-admission`,
  `session-lane`, `claude-review`, `live-reconciliation`, product behavior,
  protected S6/content/dependabot/local-model lanes, or weaken
  `pre-push-audit`.
- Convergence decision: the earlier per-field exact-spec predicate over-scoped
  this pre-admission and produced non-converging review rounds. This PR uses the
  other §3k.2 convergence-safe option: compare the whole future workflow as one
  canonical normalized YAML string, leaving no per-dimension predicate for the
  next round to probe.

## Scope (this PR)

Ownership lane: dev-workflow/process-gate-enrollment
Slice phase: Workflow/process

1. Add the future Review Contract workflow/job pair to the workflow-security
   posture allowlist, gated by a whole-workflow canonical YAML match.
2. Add regression coverage proving that canonical workflow is accepted and a
   drifted unsafe workflow is rejected.
3. Document that this is pre-admission only; the actual producer and branch
   protection enrollment remain separate #2104 follow-ups.
4. Move the merged #2123 plan into `plans/archive/` and refresh the plan index.

### Review Contract

- Acceptance criteria:
  - [ ] The only runtime code change is the workflow-security allowlist adding
        review_contract.yml / review-contract plus the whole-workflow canonical
        YAML matcher for that future file.
  - [ ] The regression test proves the canonical workflow is accepted and unsafe
        workflow drift is rejected.
  - [ ] This PR does not add the future Review Contract workflow file.
  - [ ] Docs describe the context as future/pre-admitted, not currently running
        or required.
  - [ ] The #2123 plan is moved to `plans/archive/` by name and the index is
        refreshed; no bulk archive sweep runs.
- Reachability proof: workflow-security tests exercise the real
  `scripts/audit_workflow_security_posture.py` allowlist and shared guard-shape
  enforcement. This is a pre-admission slice, so there is no new live workflow
  entrypoint yet.
- Affected surfaces: workflow security posture allowlist/tests, security docs,
  and plan archive housekeeping.
- Risk areas: accidentally adding a workflow, changing unrelated trusted-base
  predicates, letting the canonical YAML drift from the later producer, documenting
  Review Contract as required before it exists, or changing Review Contract audit
  semantics instead of only admitting the future producer shape.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Files touched

- `docs/SECURITY_GUARDRAILS.md`
- `plans/INDEX.md`
- `plans/PR-Review-Contract-PreAdmission.md`
- `plans/archive/PR-Plan-Admission-Producer.md`
- `scripts/audit_workflow_security_posture.py`
- `tests/test_audit_workflow_security_posture.py`

## Mechanism

`scripts/audit_workflow_security_posture.py` keeps a tuple allowlist of the only
jobs allowed to run on `pull_request_target`. Most allowlisted jobs must match
the shared trusted-base guard shape: job-level
`if: github.event_name == 'pull_request_target'`, a first step using the
SHA-pinned `actions/checkout` action, and checkout ref
`${{ github.event.pull_request.base.sha }}`.

This PR adds the future review_contract.yml / review-contract tuple, but that
tuple is accepted only when the whole workflow file matches
`REVIEW_CONTRACT_CANONICAL_WORKFLOW` after line-trimming normalization. The
canonical workflow includes read-only permissions, full-history trusted-base
checkout, pinned Python setup, PR-head materialization as data, base-owned
plan/reviewer-rule audits rooted at the PR worktree, and zero-plan no-op
behavior. Any extra job, permission drift, step reordering, or appended command
changes the normalized workflow text and is rejected.

## Intentional

- This PR intentionally does not add the Review Contract workflow.
- This PR intentionally avoids per-field workflow predicates; the whole
  normalized workflow is the fail-closed admission unit.
- This PR does not change `scripts/audit_plan_doc.py` or
  `scripts/audit_review_rules_triggered.py`; their semantics are not part of
  this pre-admission slice.
- This PR does not change branch protection or required-check settings.

## Deferred

- #2104 next producer slice: add the actual trusted-base Review Contract
  workflow using existing plan-doc/reviewer-rule audits.
- #2104 enrollment slice: require Review Contract only after producer burn-in
  and a REST branch-protection patch that preserves every existing context.
- #2104 reviewer slice: replace or constrain the forgeable `claude-review`
  publisher before requiring it.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_audit_workflow_security_posture.py -q` — 19 passed.
- `python scripts/audit_workflow_security_posture.py .github/workflows` — passed with existing mutable-action warnings and expected trusted-base allowlist warnings.
- `python scripts/sync_pr_plan.py plans/PR-Review-Contract-PreAdmission.md --check` — passed after sync.
- `python scripts/audit_pr_body.py /tmp/atlas-review-contract-preadmission-pr-body.md` — passed.
- Plan/code consistency audit — passed.
- `git diff --check` — passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/SECURITY_GUARDRAILS.md` | 12 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Review-Contract-PreAdmission.md` | 150 |
| `plans/archive/PR-Plan-Admission-Producer.md` | 0 |
| `scripts/audit_workflow_security_posture.py` | 90 |
| `tests/test_audit_workflow_security_posture.py` | 30 |
| **Total** | **285** |
